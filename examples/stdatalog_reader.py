"""Minimal STDatalog v2 ``.dat`` reader for IIS3DWB acceleration streams.

The STWIN.box FP-SNS-DATALOG2 firmware writes each enabled sensor stream as
``<component>.dat`` next to ``device_config.json``.  For a component whose
status carries ``samples_per_ts = S``, ``dim = C`` and ``data_type = int16``,
the file is a sequence of fixed-size frames:

    [ S * C * int16 samples ][ 8-byte little-endian float64 timestamp ]

(the layout used by the acquisition logger's native data-ready callbacks;
``timestamp_frame_bytes = samples_per_ts * dim * dtype_len + 8``).

This reader needs only numpy — no STMicroelectronics SDK, no USB device —
so the recorded example acquisitions in ``examples/`` can be decoded on any
machine.  Physical units: raw int16 counts * ``sensitivity`` = acceleration
in g (sensitivity comes from the same component status).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPE_LEN = {"int16": 2, "int32": 4, "float": 4, "float32": 4}


@dataclass
class SensorStream:
    """One decoded sensor stream in physical units."""

    component: str
    data: np.ndarray          # (N, dim) float64, acceleration in g
    timestamps: np.ndarray    # (frames,) float64, one per frame of samples_per_ts
    fs_hz: float              # measured output data rate from device_config
    sensitivity: float
    samples_per_ts: int
    dim: int

    @property
    def duration_s(self) -> float:
        return self.data.shape[0] / self.fs_hz

    def axis(self, name: str) -> np.ndarray:
        index = {"x": 0, "y": 1, "z": 2}[name]
        if index >= self.dim:
            raise ValueError(f"stream has {self.dim} axes, no {name!r}")
        return self.data[:, index]


def component_status(device_config: dict, component: str) -> dict:
    """Return the status object for one component from device_config.json."""
    for device in device_config.get("devices", []):
        for entry in device.get("components", []):
            if component in entry:
                return entry[component]
    raise KeyError(f"component {component!r} not found in device_config")


def read_acquisition(folder: str | Path,
                     component: str = "iis3dwb_acc") -> SensorStream:
    """Decode ``<folder>/<component>.dat`` using ``<folder>/device_config.json``."""
    root = Path(folder)
    config = json.loads((root / "device_config.json").read_text(encoding="utf-8"))
    status = component_status(config, component)
    if not status.get("enable", False):
        raise ValueError(f"{component}: component is not enabled in device_config")
    data_type = str(status["data_type"])
    if data_type not in _DTYPE_LEN:
        raise ValueError(f"{component}: unsupported data_type {data_type!r}")
    samples_per_ts = int(status["samples_per_ts"])
    dim = int(status["dim"])
    sensitivity = float(status["sensitivity"])
    fs_hz = float(status.get("measodr") or status.get("odr"))
    if not (samples_per_ts > 0 and dim > 0 and fs_hz > 0):
        raise ValueError(
            f"{component}: invalid stream parameters "
            f"(samples_per_ts={samples_per_ts}, dim={dim}, fs={fs_hz})")

    dtype_len = _DTYPE_LEN[data_type]
    frame_bytes = samples_per_ts * dim * dtype_len + 8
    blob = (root / f"{component}.dat").read_bytes()
    n_frames = len(blob) // frame_bytes
    if n_frames == 0:
        raise ValueError(f"{component}.dat holds no complete frame "
                         f"({len(blob)} bytes < {frame_bytes})")
    trimmed = np.frombuffer(blob[: n_frames * frame_bytes], dtype=np.uint8)
    frames = trimmed.reshape(n_frames, frame_bytes)
    sample_bytes = frames[:, : samples_per_ts * dim * dtype_len]
    counts = sample_bytes.reshape(-1).view(np.int16 if data_type == "int16"
                                           else np.int32)
    data = counts.astype(np.float64).reshape(-1, dim) * sensitivity
    timestamps = frames[:, samples_per_ts * dim * dtype_len:].copy().view(np.float64)[:, 0]
    return SensorStream(
        component=component,
        data=data,
        timestamps=timestamps,
        fs_hz=fs_hz,
        sensitivity=sensitivity,
        samples_per_ts=samples_per_ts,
        dim=dim,
    )
