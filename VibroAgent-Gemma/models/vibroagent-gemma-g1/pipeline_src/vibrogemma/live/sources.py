from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, Callable, Protocol

import numpy as np

from ..exceptions import LiveAcquisitionError
from ..signal.units import acceleration_to_g

ChunkCallback = Callable[[str, np.ndarray, float, np.ndarray | None, np.ndarray | None, int], None]


class LiveSource(Protocol):
    def start(self, callback: ChunkCallback) -> None: ...
    def stop(self) -> None: ...


class ReplaySource:
    """Continuous six-board replay source for integration and regression testing."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate_hz: float | None = None,
        chunk_seconds: float = 0.25,
        real_time: bool = False,
    ) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float32)
            if values.ndim == 5:
                values = values.reshape(values.shape[0], 6, 3, -1)
                values = np.concatenate([window for window in values], axis=-1)
            if values.ndim != 3 or values.shape[:2] != (6, 3):
                raise LiveAcquisitionError("Replay NPZ values must have shape [6,3,samples]")
            self.values = values
            if sample_rate_hz is None:
                if "sample_rate_hz" not in archive:
                    raise LiveAcquisitionError("Replay source requires sample_rate_hz")
                sample_rate_hz = float(np.asarray(archive["sample_rate_hz"]).item())
        self.sample_rate_hz = float(sample_rate_hz)
        self.chunk_samples = max(1, int(round(chunk_seconds * self.sample_rate_hz)))
        self.real_time = real_time
        self.stop_event = Event()
        self.finished_event = Event()
        self.thread: Thread | None = None

    def start(self, callback: ChunkCallback) -> None:
        if self.thread is not None:
            raise LiveAcquisitionError("Replay source already started")
        self.stop_event.clear()
        self.finished_event.clear()
        self.thread = Thread(target=self._run, args=(callback,), name="VibroGemmaReplay", daemon=True)
        self.thread.start()

    def _run(self, callback: ChunkCallback) -> None:
        start_wall = time.time()
        try:
            for start in range(0, self.values.shape[-1], self.chunk_samples):
                if self.stop_event.is_set():
                    break
                end = min(start + self.chunk_samples, self.values.shape[-1])
                chunk_start = start_wall + start / self.sample_rate_hz
                for board_index in range(6):
                    callback(
                        str(board_index),
                        self.values[board_index, :, start:end],
                        chunk_start,
                        None,
                        None,
                        0,
                    )
                if self.real_time:
                    target = start_wall + end / self.sample_rate_hz
                    self.stop_event.wait(max(0.0, target - time.time()))
        finally:
            self.finished_event.set()

    @property
    def finished(self) -> bool:
        return self.finished_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None


@dataclass(slots=True)
class _ComponentDecoder:
    component_name: str
    dimensions: int
    dtype: np.dtype
    sensitivity: float
    unit: str
    usb_dps: int

    def decode(self, payload: bytes) -> np.ndarray:
        raw = np.frombuffer(payload, dtype=self.dtype)
        complete = (raw.size // self.dimensions) * self.dimensions
        if complete == 0:
            return np.empty((3, 0), dtype=np.float32)
        raw = raw[:complete].reshape(-1, self.dimensions).T.astype(np.float32)
        physical = raw * self.sensitivity
        physical_g = acceleration_to_g(physical, self.unit)
        if self.dimensions != 3:
            raise LiveAcquisitionError(
                f"{self.component_name}: expected three genuine axes, got dimensions={self.dimensions}"
            )
        return physical_g


class StdatalogUSBSource:
    """One-owner, multi-device USB source using stdatalog-pysdk HSDLink.

    The class never creates one HSDLink instance per board. One native owner is
    shared by six polling threads, while each device has an independent packet
    counter and decoder.
    """

    def __init__(self, live_config: dict[str, Any]) -> None:
        self.config = live_config
        self.sensitivity_unit = live_config.get("sensitivity_unit")
        self.stop_event = Event()
        self.threads: list[Thread] = []
        self.hsd: Any = None
        self.started_devices: list[int] = []
        self.callback: ChunkCallback | None = None
        self.decoders: dict[int, _ComponentDecoder] = {}
        self.previous_counters: dict[int, int | None] = {}

    def start(self, callback: ChunkCallback) -> None:
        if self.hsd is not None:
            raise LiveAcquisitionError("STDATALOG source already started")
        try:
            from stdatalog_core.HSD_link.HSDLink import HSDLink
        except ImportError as exc:
            raise LiveAcquisitionError(
                "stdatalog-pysdk is not installed; install the project with the live extra"
            ) from exc
        self.callback = callback
        factory = HSDLink()
        self.hsd = factory.create_hsd_link(
            dev_com_type=str(self.config.get("dev_com_type", "st_hsd")),
            acquisition_folder=str(self.config.get("acquisition_folder", "events/raw")),
        )
        if self.hsd is None:
            raise LiveAcquisitionError("HSDLink factory returned no USB link")
        device_ids = [
            int(self.config["reference_device_id"]),
            *[int(value) for value in self.config["target_device_ids"]],
        ]
        if len(device_ids) != 6 or len(set(device_ids)) != 6:
            raise LiveAcquisitionError("Runtime configuration must identify six unique devices")
        pattern = str(self.config.get("sensor_component_pattern", "iis3dwb_acc")).lower()
        for device_id in device_ids:
            status_payload = self.hsd.get_device_status(device_id)
            device_status = self._extract_device_status(status_payload, device_id)
            component_name, component_status = self._find_component(device_status, pattern)
            configured_rate = float(self.config["original_sample_rate_hz"])
            reported_rate = _resolve_reported_sample_rate(component_status)
            if reported_rate is not None and not np.isclose(reported_rate, configured_rate, rtol=0.0, atol=1e-3):
                raise LiveAcquisitionError(
                    f"Device {device_id}: configured sample rate {configured_rate:g} Hz does not match "
                    f"stdatalog metadata {reported_rate:g} Hz"
                )
            data_type = component_status.get("data_type")
            if data_type is None:
                raise LiveAcquisitionError(f"Device {device_id} component lacks data_type")
            dtype = self._numpy_dtype(str(data_type))
            dimensions = int(component_status.get("dim", 0))
            sensitivity = float(component_status.get("sensitivity", 1.0))
            unit = _resolve_sample_unit(component_status, self.sensitivity_unit)
            if not unit:
                raise LiveAcquisitionError(
                    f"Device {device_id}: acceleration unit is absent; set live.sensitivity_unit explicitly"
                )
            usb_dps = int(component_status.get("usb_dps", 0))
            if usb_dps <= 0:
                raise LiveAcquisitionError(f"Device {device_id}: component lacks a positive usb_dps")
            self.decoders[device_id] = _ComponentDecoder(
                component_name=component_name,
                dimensions=dimensions,
                dtype=dtype,
                sensitivity=sensitivity,
                unit=unit,
                usb_dps=usb_dps,
            )
            self.previous_counters[device_id] = None

        self.stop_event.clear()
        for device_id in device_ids:
            self.hsd.start_log(device_id, interface=1)
            self.started_devices.append(device_id)
            thread = Thread(
                target=self._poll_device,
                args=(device_id,),
                name=f"STDATALOG-{device_id}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _poll_device(self, device_id: int) -> None:
        assert self.hsd is not None and self.callback is not None
        decoder = self.decoders[device_id]
        packet_span = decoder.usb_dps + 4
        board_id = str(device_id)
        while not self.stop_event.wait(0.005):
            result = self.hsd.get_sensor_data(device_id, decoder.component_name)
            if result is None:
                continue
            _size, blob = result
            if not blob:
                continue
            packet_count = len(blob) // packet_span
            for packet_index in range(packet_count):
                base = packet_index * packet_span
                counter = struct.unpack("=i", blob[base : base + 4])[0]
                payload = blob[base + 4 : base + packet_span]
                previous = self.previous_counters[device_id]
                loss_count = 0
                if previous is not None:
                    delta = counter - previous
                    if counter != 0 and delta != decoder.usb_dps:
                        loss_count = max(0, delta // decoder.usb_dps - 1)
                self.previous_counters[device_id] = counter
                values = decoder.decode(payload)
                if values.shape[1] == 0:
                    continue
                duration = values.shape[1] / float(self.config["original_sample_rate_hz"])
                start_time = time.time() - duration
                self.callback(board_id, values, start_time, None, None, loss_count)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3)
        self.threads.clear()
        if self.hsd is not None:
            for device_id in self.started_devices:
                try:
                    self.hsd.stop_log(device_id)
                except Exception:
                    pass
            try:
                self.hsd.close()
            except Exception:
                pass
        self.started_devices.clear()
        self.hsd = None

    @staticmethod
    def _extract_device_status(payload: Any, device_id: int) -> dict[str, Any]:
        if isinstance(payload, dict) and "devices" in payload:
            devices = payload["devices"]
            if isinstance(devices, list):
                return devices[device_id]
            return devices[str(device_id)] if str(device_id) in devices else devices[device_id]
        if isinstance(payload, dict):
            return payload
        raise LiveAcquisitionError(f"Unexpected device-status payload for device {device_id}")

    @staticmethod
    def _find_component(device_status: dict[str, Any], pattern: str) -> tuple[str, dict[str, Any]]:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for component in device_status.get("components", []):
            if not isinstance(component, dict) or len(component) != 1:
                continue
            name = next(iter(component))
            status = component[name]
            if pattern in name.lower() and bool(status.get("enable", True)):
                candidates.append((name, status))
        if len(candidates) != 1:
            raise LiveAcquisitionError(
                f"Expected one enabled component matching {pattern!r}, found {[name for name, _ in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _numpy_dtype(data_type: str) -> np.dtype:
        normalized = data_type.lower().replace("_t", "").replace(" ", "")
        mapping = {
            "int8": np.dtype("<i1"),
            "uint8": np.dtype("<u1"),
            "int16": np.dtype("<i2"),
            "uint16": np.dtype("<u2"),
            "int32": np.dtype("<i4"),
            "uint32": np.dtype("<u4"),
            "float": np.dtype("<f4"),
            "float32": np.dtype("<f4"),
            "double": np.dtype("<f8"),
            "float64": np.dtype("<f8"),
        }
        if normalized not in mapping:
            raise LiveAcquisitionError(f"Unsupported stdatalog data_type: {data_type}")
        return mapping[normalized]


def _resolve_sample_unit(component_status: dict[str, Any], configured: str | None) -> str | None:
    for key in ("unit", "sensitivity_unit", "measurement_unit", "unit_symbol"):
        value = component_status.get(key)
        if isinstance(value, dict):
            value = value.get("value") or value.get("val") or value.get("unit")
        if value:
            return str(value)
    return str(configured) if configured else None


def _resolve_reported_sample_rate(component_status: dict[str, Any]) -> float | None:
    for key in ("odr", "sample_rate_hz", "sampling_frequency", "sampling_rate"):
        value = component_status.get(key)
        if isinstance(value, dict):
            value = value.get("value", value.get("val"))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None
