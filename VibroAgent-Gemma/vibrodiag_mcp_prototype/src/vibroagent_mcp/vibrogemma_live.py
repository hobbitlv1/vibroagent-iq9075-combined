"""Live six-board VibroGemma decision path for the existing webchat panel."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .board_reader_process import (
    board_reader_processes_enabled,
    read_sdk_vibrometer_window_via_board_process,
)
from .errors import format_exception_for_response
from .model_client import ModelClient
from .sdk_vibrometer import read_sdk_vibrometer_window
from .sensor_registry import SensorRegistry

SLOTS = ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
CLASSES = {
    "normal",
    "unknown_anomaly",
    "data_invalid",
}
SEVERITIES = {"none", "advisory"}
MODEL_WINDOW_S = 10.0
CAPTURE_HISTORY_S = 11.5
CAPTURE_READ_S = 14.0
G1_DECISION_MODE = "parallel_binary_global_plus_joint_target_set_v1"
FACTORIZED_DECISION_MODE = "factorized_global_plus_independent_targets_v1"
G1_LABEL_CONTRACT = {
    "classes": ["normal", "unknown_anomaly"],
    "quality_class": "data_invalid",
    "severities": ["none", "advisory"],
}
_LUMO_TRAINING_FIXTURES = {
    "target_3": (
        "10_DAM4_010",
        "target_3_DAM4_010_520_530s.npz",
        "cd1e068e1d060210f426f1a0049a97bf359865d7957ef9e06b9c1d647de73402",
    ),
    "target_5": (
        "08_DAM6_010",
        "target_5_DAM6_010_520_530s.npz",
        "d2c612767ad51ab711192d661e14a7422c5fbb652b695759dba1895d21b56150",
    ),
}
_LUMO_TRAINING_PANEL = {
    "baseline": ("accel04", "ML4", 5.95),
    "target_1": ("accel02", "ML2", 8.00),
    "target_2": ("accel05", "ML5", 5.00),
    "target_3": ("accel06", "ML6", 4.00),
    "target_4": ("accel08", "ML8", 2.00),
    "target_5": ("accel09", "ML9", 1.00),
}


@dataclass(frozen=True)
class _CapturedBoard:
    values: np.ndarray
    sampling_rate_hz: float
    metadata: dict[str, Any]
    approximate_end_utc: datetime | None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _lumo_training_fixture(
    source_text: str,
    expected_sha256: str,
) -> tuple[dict[str, np.ndarray], float, dict[str, np.ndarray]]:
    """Load one verified 520-530 s LUMO training window once per web process."""
    source = Path(source_text).resolve()
    actual_sha256 = _file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"LUMO training source checksum mismatch: {actual_sha256}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as compact:
            names = [str(value) for value in compact["channel_names"]]
            values = np.asarray(compact["data"], dtype=np.float32)
            rate = _safe_float(compact["sample_rate_hz"])
        start = 0
    else:
        from scipy.io import loadmat

        dat = loadmat(source, simplify_cells=True).get("Dat")
        if not isinstance(dat, dict):
            raise ValueError("LUMO training source has no Dat record")
        names = [str(value) for value in np.atleast_1d(dat.get("ChannelNames"))]
        values = np.asarray(dat.get("Data"), dtype=np.float32)
        rate = _safe_float(dat.get("Fs"))
        start = int(round(520.0 * rate)) if rate is not None else 0
    columns = {name: index for index, name in enumerate(names)}
    if values.ndim != 2 or rate is None or rate <= 0:
        raise ValueError("LUMO training source has invalid Data/Fs fields")
    count = int(round(MODEL_WINDOW_S * rate))
    stop = start + count
    if stop > values.shape[0]:
        raise ValueError("LUMO training source is shorter than the selected window")

    arrays: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    for slot, (sensor, _level, elevation_m) in _LUMO_TRAINING_PANEL.items():
        required = (f"{sensor}x", f"{sensor}y")
        if any(channel not in columns for channel in required):
            raise ValueError(f"LUMO training source is missing {required}")
        board = np.zeros((3, count), dtype=np.float32)
        board[0] = values[start:stop, columns[required[0]]]
        board[1] = values[start:stop, columns[required[1]]]
        arrays[slot] = board
        positions[slot] = np.asarray((0.0, 0.0, elevation_m / 8.95), dtype=np.float32)
    return arrays, rate, positions


def _lumo_training_replay(target: str = "target_3") -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    try:
        condition, filename, expected_sha256 = _LUMO_TRAINING_FIXTURES[target]
    except KeyError as exc:
        raise ValueError(f"unsupported LUMO training target: {target}") from exc
    configured = os.environ.get("VIBRO_LUMO_TRAINING_SOURCE", "").strip()
    source = (
        Path(configured).expanduser().resolve()
        if configured
        else _project_root()
        / "data"
        / "external"
        / "lumo"
        / condition
        / filename
    )
    cached_arrays, rate, cached_positions = _lumo_training_fixture(
        str(source), expected_sha256
    )
    arrays = {slot: values.copy() for slot, values in cached_arrays.items()}
    positions = {slot: values.copy() for slot, values in cached_positions.items()}
    rates = dict.fromkeys(SLOTS, rate)
    capture_by_slot = {
        slot: {
            "data_is_current": True,
            "data_file": str(source),
            "data_provenance": "public_training_split_replay",
            "source_dataset": "lumo",
            "source_sensor_name": sensor,
            "source_measurement_level": level,
            "timestamp_monotonic": True,
            "timestamp_warning": None,
            "clock_skew_ms": 0.0,
            "clipped_fraction": 0.0,
            "packet_loss_count": 0,
            "selected_timestamp_start_s": 520.0,
            "selected_timestamp_end_s": 530.0,
            "quality_notes": ["public_dataset", "training_split_replay", "missing_z_axis_masked"],
        }
        for slot, (sensor, level, _elevation_m) in _LUMO_TRAINING_PANEL.items()
    }
    now = datetime.now(timezone.utc)
    capture_metadata = {
        "capture_started_utc": now.isoformat(),
        "capture_completed_utc": now.isoformat(),
        "window_start_utc": (now - timedelta(seconds=MODEL_WINDOW_S)).isoformat(),
        "window_end_utc": now.isoformat(),
        "window_time_source": "lumo_training_split_replay",
        "cross_board_alignment_verified": True,
        "cross_board_trim_s": dict.fromkeys(SLOTS, 0.0),
        "phase_synchronized": False,
        "synchronization_verification_id": "lumo_shared_dat_matrix_v1",
        "geometry_valid": True,
        "geometry_source": "lumo_publisher_geometry",
        "source_dataset": "lumo",
        "source_split": "train",
        "source_fixture": f"{condition}/{filename}@520-530s",
        "episode_provenance": {
            "dataset_id": "lumo",
            "provenance_type": "public",
            "source_files": [str(source)],
            "source_checksums": [expected_sha256],
            "licence": "local LUMO benchmark copy; inference test only",
            "licence_accepted": True,
            "structure_id": "lumo_lattice_mast",
            "session_id": condition,
            "experiment_id": f"{condition}::{Path(filename).stem}",
            "split_group": f"lumo|lumo_lattice_mast|{condition}",
            "derived_operations": ["window:520.0:530.0", "missing_z_axis_masked"],
            "local_training_data": False,
        },
        "synthetic_injection": {
            "active": True,
            "test_only": True,
            "kind": "lumo_training_replay",
            "source_dataset": "lumo",
            "source_split": "train",
        },
    }
    axis_masks = {slot: np.asarray((True, True, False)) for slot in SLOTS}
    return arrays, rates, capture_by_slot, capture_metadata, positions, axis_masks


def _lumo_overlay_gain() -> float:
    try:
        gain = float(os.environ.get("VIBRO_LUMO_OVERLAY_GAIN", "1.0"))
    except ValueError:
        gain = 1.0
    return min(10.0, max(0.05, gain))


@lru_cache(maxsize=4)
def _lumo_overlay_templates(
    target: str,
    sample_rate_hz: float,
    sample_count: int,
    gain: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, rates, _capture_by_slot, capture, positions, axis_masks = (
        _lumo_training_replay(target)
    )
    templates: dict[str, np.ndarray] = {}
    for slot in SLOTS:
        overlay_rate_hz = max(
            sample_rate_hz,
            sample_count / MODEL_WINDOW_S,
        )
        resampled = _resample_board_window(
            arrays[slot],
            source_rate_hz=rates[slot],
            target_rate_hz=overlay_rate_hz,
        )[:, -sample_count:]
        if resampled.shape[1] != sample_count:
            raise ValueError(
                f"{slot}: overlay produced {resampled.shape[1]} samples, expected {sample_count}"
            )
        templates[slot] = (
            resampled - np.mean(resampled, axis=1, keepdims=True)
        ).astype(np.float32) * gain
    return templates, {
        "active": True,
        "test_only": True,
        "kind": "lumo_live_overlay",
        "selected_target": target,
        "source_dataset": "lumo",
        "source_split": "train",
        "source_fixture": capture["source_fixture"],
        "gain": gain,
        "live_weight": 1.0 / (1.0 + gain),
        "template_weight": gain / (1.0 + gain),
        "amplitude_normalized_blend": True,
        "live_residual_amplitude_matched": True,
        "overlay_phase_synchronized": True,
        "live_capture_preserved": True,
        "phase_synchronized": False,
        "synchronization_verification_id": None,
        "geometry_source": "lumo_publisher_geometry",
        "sensor_positions": {
            slot: positions[slot].tolist() for slot in SLOTS
        },
        "axis_masks": {
            slot: axis_masks[slot].tolist() for slot in SLOTS
        },
        "episode_provenance": capture["episode_provenance"],
    }


def _apply_lumo_live_overlay(
    arrays: dict[str, np.ndarray],
    *,
    sample_rate_hz: float,
    target: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    live_sample_count = arrays["baseline"].shape[1]
    if any(values.shape != (3, live_sample_count) for values in arrays.values()):
        raise ValueError("live overlay requires six aligned XYZ windows")
    _source_arrays, source_rates, *_rest = _lumo_training_replay(target)
    overlay_rate_hz = float(source_rates["baseline"])
    sample_count = int(round(MODEL_WINDOW_S * overlay_rate_hz))
    live_arrays = {
        slot: _resample_board_window(
            arrays[slot],
            source_rate_hz=sample_rate_hz,
            target_rate_hz=overlay_rate_hz,
        )
        for slot in SLOTS
    }
    gain = _lumo_overlay_gain()
    templates, metadata = _lumo_overlay_templates(
        target,
        round(overlay_rate_hz, 6),
        sample_count,
        gain,
    )
    hybrid: dict[str, np.ndarray] = {}
    for slot in SLOTS:
        hybrid[slot] = _blend_lumo_board(live_arrays[slot], templates[slot], gain)
    metadata = {**metadata, "encoder_sample_rate_hz": overlay_rate_hz}
    return hybrid, metadata, overlay_rate_hz


def _blend_lumo_board(live, template, gain):
    """Blend XYZ before axis selection, norm, graph reduction or spectrum."""
    live = np.asarray(live, dtype=np.float32)
    valid = np.isfinite(live).all(axis=0)
    clean = np.where(np.isfinite(live), live, 0.0)
    residual = clean - (clean[:, valid].mean(axis=1, keepdims=True) if valid.any() else 0.0)
    template_rms = np.sqrt(np.mean((template / gain) ** 2, axis=1, keepdims=True))
    live_rms = np.sqrt(np.mean(residual[:, valid] ** 2, axis=1, keepdims=True)) if valid.any() else np.zeros((3, 1))
    scale = np.divide(template_rms, live_rms, out=np.zeros_like(template_rms), where=live_rms > 1e-12)
    hybrid = (residual * scale + template) / (1.0 + gain)
    hybrid[2] = live[2]  # LUMO has no measured Z axis.
    hybrid[:, ~valid] = np.nan
    return hybrid


def _resample_board_window(
    values: np.ndarray,
    *,
    source_rate_hz: float,
    target_rate_hz: float,
    duration_s: float = 10.0,
) -> np.ndarray:
    """Align one board's measured clock to the reference-board clock."""
    from math import gcd
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import resample_poly

    if not all(np.isfinite(rate) and rate > 0 for rate in (source_rate_hz, target_rate_hz, duration_s)):
        raise ValueError("sample rates and duration must be finite and positive")
    source_count = int(round(duration_s * source_rate_hz))
    target_count = int(round(duration_s * target_rate_hz))
    if source_count < 2 or target_count < 2:
        raise ValueError("resampling needs at least two input and output samples")
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < source_count:
        raise ValueError(f"expected [3, >= {source_count}] values, got {values.shape}")
    values = np.asarray(values[:, -source_count:], dtype=np.float32)
    if source_count == target_count and abs(source_rate_hz - target_rate_hz) <= 1e-6:
        return values

    valid = np.isfinite(values).all(axis=0)
    clean = np.where(np.isfinite(values), values, 0.0)
    divisor = gcd(source_count, target_count)
    up, down = target_count // divisor, source_count // divisor
    # Reverse to anchor both clocks at the newest sample, as the SDK tail reads do.
    aligned = resample_poly(clean[:, ::-1], up, down, axis=1, padtype="line")[:, ::-1].copy()
    # Invalid input contaminates the FIR support, not only its nearest output sample.
    radius = int(np.ceil(10 * max(up, down) / up))
    invalid = maximum_filter1d((~valid)[::-1], size=2 * radius + 1, mode="constant", cval=0)
    indices = np.minimum(source_count - 1, np.rint(np.arange(target_count) * down / up).astype(int))
    aligned[:, invalid[indices][::-1]] = np.nan
    return aligned


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _compact_axis_metadata(window: Any) -> dict[str, Any]:
    metadata = dict(getattr(window, "metadata", {}) or {})
    keep = {
        "acquisition_folder",
        "data_file",
        "data_file_age_s",
        "data_file_modified_at_utc",
        "data_is_current",
        "full_scale_g",
        "measured_odr_hz",
        "configured_odr_hz",
        "sampling_rate_hz",
        "sdk_latest_timestamp_gap_s",
        "sdk_read_strategy",
        "sensitivity",
        "synthetic",
        "synthetic_injection",
        "data_provenance",
        "timestamp_mode",
        "timestamp_warning",
    }
    return {key: metadata.get(key) for key in keep if key in metadata}


def _window_end_utc(window: Any, selected_timestamp_end_s: float | None) -> datetime | None:
    metadata = dict(getattr(window, "metadata", {}) or {})
    modified = _parse_utc(metadata.get("data_file_modified_at_utc"))
    gap_s = _safe_float(metadata.get("sdk_latest_timestamp_gap_s"))
    if modified is None or gap_s is None or selected_timestamp_end_s is None:
        return None
    gap_s = max(0.0, gap_s)
    timestamps = getattr(window, "timestamps_s", None)
    decoded_end_s = None
    if timestamps is not None and np.asarray(timestamps).size:
        decoded_end_s = float(np.asarray(timestamps, dtype=np.float64)[-1]) + 1.0 / float(window.sampling_rate_hz)
    selected_lag_s = (
        max(0.0, decoded_end_s - selected_timestamp_end_s)
        if decoded_end_s is not None and selected_timestamp_end_s is not None
        else None
    )
    if selected_lag_s is None:
        return None
    return modified - timedelta(seconds=gap_s + selected_lag_s)


def _device_identity(registry: SensorRegistry) -> dict[str, Any]:
    configured = os.environ.get("VIBRO_LIVE_DEVICE_MANIFEST", "").strip()
    required = _truthy("VIBRO_REQUIRE_DEVICE_IDENTITY", default=False)
    folders = [Path(registry.sensors[slot].acquisition_folder or "").resolve() for slot in SLOTS]
    candidate = Path(configured).expanduser().resolve() if configured else (
        Path(os.path.commonpath([str(path) for path in folders])) / "vibroagent_live_devices.json"
    )
    if not candidate.is_file():
        if required or configured:
            raise ValueError(f"live device identity manifest is missing: {candidate}")
        return {"verified": False, "reason": "manifest_unavailable"}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise ValueError("live device identity manifest has no devices list")
    by_slot = {
        str(item.get("sensor_id")): item
        for item in devices
        if isinstance(item, dict) and str(item.get("sensor_id") or "")
    }
    mapping: dict[str, dict[str, Any]] = {}
    serials: set[str] = set()
    device_ids: set[int] = set()
    for slot in SLOTS:
        item = by_slot.get(slot)
        if item is None:
            raise ValueError(f"live device identity manifest is missing {slot}")
        folder = Path(str(item.get("folder") or "")).expanduser().resolve()
        if folder != folders[SLOTS.index(slot)]:
            raise ValueError(f"{slot}: registry folder does not match the logger identity manifest")
        serial = str(item.get("serial") or "").strip()
        if not serial or serial.lower() == "unknown" or serial in serials:
            raise ValueError(f"{slot}: invalid or duplicate physical board serial")
        try:
            device_id = int(item["device_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{slot}: invalid logger device id") from exc
        if device_id in device_ids:
            raise ValueError(f"{slot}: duplicate logger device id")
        configured_sensor = registry.sensors[slot].hsd_sensor_name
        if configured_sensor not in (item.get("logged_sensors") or []):
            raise ValueError(f"{slot}: configured sensor is not bound to the logged physical board")
        serials.add(serial)
        device_ids.add(device_id)
        mapping[slot] = {"serial": serial, "device_id": device_id, "folder": str(folder)}
    return {
        "verified": True,
        "manifest_file": candidate.name,
        "manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "created_at_unix": payload.get("created_at_unix"),
        "mapping": mapping,
    }


def _source_bindings(registry: SensorRegistry) -> dict[str, dict[str, str]]:
    """Validate that every logical slot is wired to a distinct live source."""
    bindings: dict[str, dict[str, str]] = {}
    owners: dict[tuple[str, str], str] = {}
    for slot in SLOTS:
        sensor = registry.sensors[slot]
        raw_folder = str(sensor.acquisition_folder or "").strip()
        if not raw_folder:
            raise ValueError(f"{slot}: acquisition_folder is not configured")
        folder = str(Path(raw_folder).expanduser().resolve())
        sensor_name = str(sensor.hsd_sensor_name or "").strip()
        if not sensor_name:
            raise ValueError(f"{slot}: hsd_sensor_name is not configured")
        key = (folder, sensor_name)
        previous = owners.get(key)
        if previous is not None:
            raise ValueError(
                f"{slot}: live source duplicates {previous} ({folder}, {sensor_name})"
            )
        owners[key] = slot
        bindings[slot] = {"acquisition_folder": folder, "sensor_name": sensor_name}
    return bindings


def _capture_window_fingerprints(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    """Hash each resampled XYZ window and reject byte-identical board feeds."""
    fingerprints: dict[str, str] = {}
    owners: dict[str, str] = {}
    for slot in SLOTS:
        values = np.ascontiguousarray(np.asarray(arrays[slot], dtype="<f4"))
        digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        previous = owners.get(digest)
        if previous is not None:
            raise ValueError(
                f"{slot}: captured XYZ window is byte-identical to {previous}; "
                "check board-to-folder routing"
            )
        owners[digest] = slot
        fingerprints[slot] = digest
    return fingerprints


def _clipping_abs_g(window: Any) -> float | None:
    metadata = dict(getattr(window, "metadata", {}) or {})
    sensitivity = _safe_float(metadata.get("sensitivity"))
    if sensitivity is not None and sensitivity > 0:
        # IIS3DWB is signed int16; STDATALOG has already multiplied counts by
        # sensitivity because the read uses raw_data=False.
        return sensitivity * 32767.0
    full_scale = _safe_float(metadata.get("full_scale_g"))
    return full_scale if full_scale is not None and full_scale >= 8.0 else None


def _capture_board(
    slot: str,
    sensor: Any,
    reader: Any,
    *,
    start_time_s: float | None = None,
    require_current: bool = True,
) -> _CapturedBoard:
    fixed_window = start_time_s is not None
    read_duration_s = MODEL_WINDOW_S + 0.25 if fixed_window else CAPTURE_READ_S
    history_s = MODEL_WINDOW_S if fixed_window else CAPTURE_HISTORY_S
    windows: dict[str, Any] = {}
    rates: list[float] = []
    for axis in ("x", "y", "z"):
        window = reader(
            machine_id=slot,
            acquisition_folder=sensor.acquisition_folder,
            sensor_name=sensor.hsd_sensor_name,
            axis=axis,
            start_time_s=start_time_s,
            duration_s=read_duration_s,
            require_current=require_current,
        )
        windows[axis] = window
        rates.append(float(window.sampling_rate_hz))
    if max(rates) - min(rates) > 1e-6:
        raise ValueError(f"{slot}: XYZ sampling rates differ: {rates}")
    rate = rates[0]
    history_count = int(round(history_s * rate))
    timestamps_by_axis: dict[str, np.ndarray] = {}
    timestamp_warning_by_axis: dict[str, str | None] = {}
    for axis, window in windows.items():
        timestamps = getattr(window, "timestamps_s", None)
        warning = str((window.metadata or {}).get("timestamp_warning") or "").strip() or None
        timestamp_warning_by_axis[axis] = warning
        if timestamps is None:
            timestamps_by_axis = {}
            break
        array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if array.size != np.asarray(window.signal).size or array.size < history_count or not np.isfinite(array).all() or np.any(np.diff(array) <= 0):
            timestamps_by_axis = {}
            break
        timestamps_by_axis[axis] = array

    timestamp_epoch_normalized = False
    if len(timestamps_by_axis) == 3 and not any(timestamp_warning_by_axis.values()):
        raw_ends = {
            axis: float(timestamps[-1]) + 1.0 / rate
            for axis, timestamps in timestamps_by_axis.items()
        }
        if max(raw_ends.values()) - min(raw_ends.values()) > CAPTURE_HISTORY_S - MODEL_WINDOW_S:
            wall_ends = {
                axis: _window_end_utc(windows[axis], raw_ends[axis])
                for axis in ("x", "y", "z")
            }
            if all(value is not None for value in wall_ends.values()):
                for axis in ("x", "y", "z"):
                    timestamps_by_axis[axis] = timestamps_by_axis[axis] + (
                        wall_ends[axis].timestamp() - raw_ends[axis]
                    )
                timestamp_epoch_normalized = True
            else:
                timestamps_by_axis = {}

    selected: list[np.ndarray] = []
    selected_timestamps: list[np.ndarray] = []
    selected_timestamp_start_s: float | None = None
    selected_timestamp_end_s: float | None = None
    alignment_mode = "sample_tail_unverified"
    if fixed_window:
        for axis in ("x", "y", "z"):
            values = np.asarray(windows[axis].signal, dtype=np.float32)
            if values.size < history_count:
                raise ValueError(f"{slot}: {axis} is shorter than {history_s:g} seconds")
            selected.append(values[:history_count])
            if axis in timestamps_by_axis:
                selected_timestamps.append(timestamps_by_axis[axis][:history_count])
        selected_timestamp_start_s = float(start_time_s)
        selected_timestamp_end_s = selected_timestamp_start_s + MODEL_WINDOW_S
        alignment_mode = "fixed_start_time_xyz"
    elif len(timestamps_by_axis) == 3 and not any(timestamp_warning_by_axis.values()):
        selected_timestamp_end_s = min(
            float(timestamps[-1]) + 1.0 / rate for timestamps in timestamps_by_axis.values()
        )
        for axis in ("x", "y", "z"):
            timestamps = timestamps_by_axis[axis]
            stop = int(np.searchsorted(timestamps, selected_timestamp_end_s, side="left"))
            start = stop - history_count
            if start < 0:
                raise ValueError(f"{slot}: {axis} lacks {history_s:g} seconds before the common XYZ end")
            selected.append(np.asarray(windows[axis].signal[start:stop], dtype=np.float32))
            selected_timestamps.append(timestamps[start:stop])
        selected_timestamp_start_s = selected_timestamp_end_s - history_s
        alignment_mode = "sdk_timestamp_common_xyz_end"
    else:
        for axis in ("x", "y", "z"):
            values = np.asarray(windows[axis].signal, dtype=np.float32)
            if values.size < history_count:
                raise ValueError(f"{slot}: {axis} is shorter than {history_s:g} seconds")
            selected.append(values[-history_count:])

    if fixed_window and selected_timestamp_end_s is not None:
        approximate_end_utc = datetime.fromtimestamp(selected_timestamp_end_s, tz=timezone.utc)
    elif timestamp_epoch_normalized and selected_timestamp_end_s is not None:
        approximate_end_utc = datetime.fromtimestamp(selected_timestamp_end_s, tz=timezone.utc)
    else:
        end_candidates = [
            _window_end_utc(window, selected_timestamp_end_s) for window in windows.values()
        ]
        end_candidates = [value for value in end_candidates if value is not None]
        approximate_end_utc = min(end_candidates) if len(end_candidates) == 3 else None
    warnings = sorted({value for value in timestamp_warning_by_axis.values() if value})
    packet_gap_counts = [
        int(np.sum(np.diff(timestamps) > (1.5 / rate))) for timestamps in selected_timestamps
    ]
    clipping_by_axis = {
        axis: _clipping_abs_g(window) for axis, window in windows.items()
    }
    metadata = {
        "alignment_mode": alignment_mode,
        "axis_metadata": {axis: _compact_axis_metadata(window) for axis, window in windows.items()},
        "data_is_current": all(bool((window.metadata or {}).get("data_is_current")) for window in windows.values()),
        "native_sampling_rate_hz": rate,
        "selected_timestamp_start_s": selected_timestamp_start_s,
        "selected_timestamp_end_s": selected_timestamp_end_s,
        "timestamp_monotonic": len(timestamps_by_axis) == 3 and not warnings,
        "timestamp_warning": "; ".join(warnings) or None,
        "quality_notes": [
            f"capture_alignment={alignment_mode}",
            *(["timestamp_epoch_normalized_from_file_mtime"] if timestamp_epoch_normalized else []),
        ],
        "timestamp_epoch_normalized": timestamp_epoch_normalized,
        "clock_skew_ms": None,
        "clipped_fraction": None,
        "clipping_abs_g_by_axis": clipping_by_axis,
        "packet_loss_count": max(packet_gap_counts) if len(packet_gap_counts) == 3 else None,
    }
    synthetic_injections = [
        (window.metadata or {}).get("synthetic_injection")
        for window in windows.values()
        if (window.metadata or {}).get("synthetic_injection")
    ]
    if synthetic_injections:
        metadata["synthetic"] = True
        metadata["data_provenance"] = "synthetic_injected_live_capture"
        metadata["synthetic_injection"] = dict(synthetic_injections[0])
    return _CapturedBoard(
        values=np.stack(selected),
        sampling_rate_hz=rate,
        metadata=metadata,
        approximate_end_utc=approximate_end_utc,
    )


def _finalize_capture(
    captured: dict[str, _CapturedBoard],
    *,
    capture_started_utc: datetime,
    capture_completed_utc: datetime,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    usable_ends = [item.approximate_end_utc for item in captured.values()]
    aligned = all(value is not None for value in usable_ends) and all(
        item.metadata.get("timestamp_monotonic") is True
        and item.metadata.get("alignment_mode") in {"sdk_timestamp_common_xyz_end", "fixed_start_time_xyz"}
        for item in captured.values()
    )
    common_end_utc = min(value for value in usable_ends if value is not None) if aligned else capture_completed_utc
    proposed_shifts = {
        slot: (
            max(0.0, (captured[slot].approximate_end_utc - common_end_utc).total_seconds())
            if aligned and captured[slot].approximate_end_utc is not None
            else 0.0
        )
        for slot in SLOTS
    }
    if aligned:
        aligned = all(
            captured[slot].values.shape[1]
            - int(round(proposed_shifts[slot] * captured[slot].sampling_rate_hz))
            >= int(round(MODEL_WINDOW_S * captured[slot].sampling_rate_hz))
            for slot in SLOTS
        )
    arrays: dict[str, np.ndarray] = {}
    native_rates: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    alignment_shifts: dict[str, float | None] = {}
    for slot in SLOTS:
        item = captured[slot]
        rate = item.sampling_rate_hz
        count = int(round(MODEL_WINDOW_S * rate))
        shift_s = proposed_shifts[slot] if aligned else 0.0
        end = item.values.shape[1] - int(round(shift_s * rate))
        start = end - count
        if start < 0:
            raise ValueError(f"{slot}: captured XYZ history is shorter than {MODEL_WINDOW_S:g} seconds")
        selected_values = item.values[:, start:end]
        arrays[slot] = selected_values
        native_rates[slot] = rate
        slot_metadata = dict(item.metadata)
        slot_metadata["cross_board_trim_s"] = shift_s if aligned else None
        actual_shift_s = (item.values.shape[1] - end) / rate
        slot_metadata["clock_skew_ms"] = (
            (shift_s - actual_shift_s) * 1000.0 if aligned else None
        )
        slot_metadata["clock_skew_source"] = (
            "common_end_rounding_residual" if aligned else "unavailable"
        )
        from .vibration_quality import raw_window_quality
        slot_metadata.update(raw_window_quality(selected_values, clipping_by_axis=slot_metadata.get("clipping_abs_g_by_axis")))
        if slot_metadata.get("selected_timestamp_end_s") is not None:
            selected_end = float(slot_metadata["selected_timestamp_end_s"]) - shift_s
            slot_metadata["selected_timestamp_end_s"] = selected_end
            slot_metadata["selected_timestamp_start_s"] = selected_end - MODEL_WINDOW_S
        slot_metadata["cross_board_alignment"] = (
            "file_mtime_and_sdk_gap_common_end" if aligned else "unverified_per_board_tail"
        )
        if not aligned:
            slot_metadata["timestamp_monotonic"] = False
            prior_warning = str(slot_metadata.get("timestamp_warning") or "").strip()
            slot_metadata["timestamp_warning"] = "; ".join(
                item for item in (prior_warning, "cross_board_common_end_unavailable") if item
            )
            slot_metadata.setdefault("quality_notes", []).append(
                "cross_board_alignment_unavailable"
            )
        metadata[slot] = slot_metadata
        alignment_shifts[slot] = shift_s if aligned else None

    if not aligned:
        common_end_utc = capture_completed_utc
    capture_metadata = {
        "capture_started_utc": capture_started_utc.isoformat(),
        "capture_completed_utc": capture_completed_utc.isoformat(),
        "window_start_utc": (common_end_utc - timedelta(seconds=MODEL_WINDOW_S)).isoformat(),
        "window_end_utc": common_end_utc.isoformat(),
        "window_time_source": (
            "file_mtime_minus_sdk_gap_common_end" if aligned else "capture_completion_upper_bound"
        ),
        "cross_board_alignment_verified": bool(aligned),
        "cross_board_trim_s": alignment_shifts,
        "phase_synchronized": False,
    }
    return arrays, native_rates, metadata, capture_metadata


def _arrays_sha256(arrays: dict[str, np.ndarray], sampling_rate_hz: float) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(sampling_rate_hz, dtype="<f8").tobytes())
    for slot in SLOTS:
        digest.update(slot.encode("ascii"))
        digest.update(np.asarray(arrays[slot], dtype="<f4").tobytes())
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle_decision_mode(bundle: Path) -> str:
    try:
        config = json.loads((bundle / "vibrogemma_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return G1_DECISION_MODE
    generation = config.get("generation") or {}
    if generation.get("structured_set_decoding"):
        return G1_DECISION_MODE
    if generation.get("factorized_decoding"):
        return FACTORIZED_DECISION_MODE
    return "joint_target_record"


def _bundle_model_identity(bundle: Path) -> dict[str, str]:
    manifest_path = bundle / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    checkpoint = str((manifest.get("geniex") or {}).get("gguf") or "")
    entries = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("files") or []
        if isinstance(item, dict)
    }
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": entries.get(checkpoint, ""),
        "bundle_version": str(manifest.get("bundle_version") or ""),
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "decision_mode": _bundle_decision_mode(bundle),
    }


def _replace_result_measurements(
    result: dict[str, Any],
    *,
    registry: SensorRegistry,
    arrays: dict[str, np.ndarray],
    sampling_rate_hz: float,
    native_rates: dict[str, float],
    capture_by_slot: dict[str, dict[str, Any]],
    capture_metadata: dict[str, Any],
    episode_id: str,
) -> None:
    from .llm_sensor_agent import SensorWindow, build_building_metrics, compare_building_metrics

    rate = int(round(sampling_rate_hz))
    offline_replay = bool(capture_metadata.get("offline_replay"))
    metrics_by_slot: dict[str, dict[str, Any]] = {}
    for slot in SLOTS:
        capture = capture_by_slot[slot]
        timestamp_start = _safe_float(capture.get("selected_timestamp_start_s"))
        timestamps = (
            timestamp_start + np.arange(arrays[slot].shape[1], dtype=np.float64) / sampling_rate_hz
            if timestamp_start is not None
            else None
        )
        quality_flags: list[str] = []
        if not offline_replay and capture.get("data_is_current") is False:
            quality_flags.append("stale_data")
        if capture.get("timestamp_monotonic") is False:
            quality_flags.append("timestamp_quality_issue")
        source_metadata = {
            "source": "vibrogemma_authoritative_xyz_episode",
            "episode_id": episode_id,
            "acquisition_folder": registry.sensors[slot].acquisition_folder,
            "sensor_name": registry.sensors[slot].hsd_sensor_name,
            "axis": "norm_from_xyz",
            "sampling_rate_hz": rate,
            "native_sampling_rate_hz": native_rates[slot],
            "start_time_s": timestamp_start,
            "requested_duration_s": MODEL_WINDOW_S,
            "data_is_current": capture.get("data_is_current"),
            "quality_flags": quality_flags,
            "capture": capture,
            "window_start_utc": capture_metadata["window_start_utc"],
            "window_end_utc": capture_metadata["window_end_utc"],
            "window_time_source": capture_metadata["window_time_source"],
        }
        window = SensorWindow(
            config=registry.sensors[slot],
            signal=np.linalg.norm(arrays[slot], axis=0).astype(np.float64),
            timestamps_s=timestamps,
            sampling_rate_hz=rate,
            metadata=source_metadata,
        )
        metrics = build_building_metrics(window_id=episode_id, window=window, duration_s=MODEL_WINDOW_S)
        metrics["source_metadata"] = source_metadata
        metrics_by_slot[slot] = metrics

    baseline_metrics = metrics_by_slot["baseline"]
    result["window_id"] = episode_id
    result["baseline_sensor_id"] = "baseline"
    result["window"] = {
        "schema_version": result.get("schema_version"),
        "analysis_domain": result.get("analysis_domain"),
        "window_id": episode_id,
        "episode_id": episode_id,
        "start_time_s": _safe_float(capture_by_slot["baseline"].get("selected_timestamp_start_s")),
        "start_utc": capture_metadata["window_start_utc"],
        "end_utc": capture_metadata["window_end_utc"],
        "time_source": capture_metadata["window_time_source"],
        "duration_s": MODEL_WINDOW_S,
        "mode": "replay" if offline_replay else "live",
        "require_current": not offline_replay,
    }
    result["baseline_metrics"] = baseline_metrics
    result["sensor_metrics"] = [metrics_by_slot[slot] for slot in SLOTS[1:]]
    result["all_sensor_metrics"] = [metrics_by_slot[slot] for slot in SLOTS]

    existing_reports = {
        str(report.get("sensor_id")): report
        for report in result.get("sensor_agent_reports") or []
        if isinstance(report, dict)
    }
    reports: list[dict[str, Any]] = []
    for slot in SLOTS[1:]:
        report = existing_reports.get(slot, {})
        comparison = compare_building_metrics(metrics_by_slot[slot], baseline_metrics)
        report.update(
            {
                "schema_version": result.get("schema_version"),
                "analysis_domain": result.get("analysis_domain"),
                "window_id": episode_id,
                "sensor_id": slot,
                "baseline_sensor_id": "baseline",
                "location": registry.sensors[slot].location,
                "role": "target",
                "axis": "norm_from_xyz",
                "sampling_rate_hz": rate,
                "duration_s": MODEL_WINDOW_S,
                "quality_flags": sorted(
                    set(metrics_by_slot[slot].get("quality_flags") or [])
                    | set(baseline_metrics.get("quality_flags") or [])
                ),
                "metrics": metrics_by_slot[slot],
                "baseline_comparison": comparison,
                "evidence": [],
            }
        )
        for inherited in ("python_rule_label", "small_agent_confidence"):
            report.pop(inherited, None)
        reports.append(report)
    result["sensor_agent_reports"] = reports
    result.pop("history_summary", None)
    result.pop("reference_history", None)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sensor_config(value: str | None) -> Path:
    selected = value or os.environ.get("VIBRO_BUILDING_SENSOR_CONFIG") or "config/sensors.live.yaml"
    path = Path(selected).expanduser()
    return path.resolve() if path.is_absolute() else (_project_root() / path).resolve()


def _truthy(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_model_result(
    data: Any,
    *,
    expected_model_identity: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("model response has the wrong schema_version")
    required = {"window", "system_state", "targets", "explanation", "quality", "model", "evidence"}
    object_fields = {"window", "quality", "model", "evidence"}
    if not required <= data.keys() or not all(isinstance(data[key], dict) for key in object_fields):
        raise ValueError("model response does not match alert.schema.json")
    if not isinstance(data["explanation"], str):
        raise ValueError("model response explanation must be text")
    if data["system_state"] not in {"normal", "advisory", "data_invalid"}:
        raise ValueError("model response has an unsupported system_state")
    window = data["window"]
    if not {
        "start_utc",
        "duration_seconds",
        "effective_duration_seconds",
        "episode_id",
    } <= window.keys():
        raise ValueError("model response window is incomplete")
    if float(window["duration_seconds"]) != MODEL_WINDOW_S or float(window["effective_duration_seconds"]) <= 0:
        raise ValueError("model response window duration is invalid")
    model = data["model"]
    if not {
        "checkpoint",
        "raw_record",
        "explanation_source",
        "explanation_grounded",
        "decision_mode",
        "label_contract",
        "global_target_consistent",
    } <= model.keys():
        raise ValueError("model response metadata is incomplete")
    if not isinstance(model["explanation_grounded"], bool):
        raise ValueError("model response explanation_grounded must be boolean")
    expected_decision_mode = str(
        (expected_model_identity or {}).get("decision_mode") or G1_DECISION_MODE
    )
    if model["decision_mode"] != expected_decision_mode:
        active = "G1" if expected_decision_mode == G1_DECISION_MODE else "bundle"
        raise ValueError(f"model response does not use the active {active} decision mode")
    if model["label_contract"] != G1_LABEL_CONTRACT:
        raise ValueError("model response does not use the active G1 binary label contract")
    for key, expected in (expected_model_identity or {}).items():
        if expected and model.get(key) != expected:
            raise ValueError(f"model response {key} does not match the active bundle")
    global_class = model.get("global_class")
    global_severity = model.get("global_severity")
    if global_class not in {"normal", "unknown_anomaly"}:
        raise ValueError("model response has an invalid global class")
    if global_severity != ("none" if global_class == "normal" else "advisory"):
        raise ValueError("model response has an invalid global class/severity tuple")
    if model.get("global_choice_probability") is not None or model.get("global_confidence_operational") is not False:
        raise ValueError("model response incorrectly claims operational global confidence")
    targets = data.get("targets")
    if not isinstance(targets, list) or len(targets) != 5:
        raise ValueError("model response must contain five targets")
    for index, target in enumerate(targets, 1):
        required_target = {
            "sensor_id",
            "affected",
            "class",
            "severity",
            "choice_probability",
            "raw_choice_probability",
            "confidence_operational",
        }
        if not isinstance(target, dict) or set(target) != required_target:
            raise ValueError(f"target_{index} does not match the active G1 target contract")
        if not isinstance(target, dict) or target.get("sensor_id") != f"target_{index}":
            raise ValueError("model response target order is invalid")
        if target.get("class") not in CLASSES or target.get("severity") not in SEVERITIES:
            raise ValueError(f"target_{index} has an unsupported class or severity")
        affected = target.get("affected")
        if not isinstance(affected, bool):
            raise ValueError(f"target_{index}.affected must be boolean")
        if target["choice_probability"] is not None or target["confidence_operational"] is not False:
            raise ValueError(f"target_{index} incorrectly claims operational confidence")
        raw_probability = target["raw_choice_probability"]
        if raw_probability is not None and (
            not isinstance(raw_probability, (int, float))
            or not np.isfinite(float(raw_probability))
            or not 0.0 <= float(raw_probability) <= 1.0
        ):
            raise ValueError(f"target_{index}.raw_choice_probability must be null or within [0,1]")
        if target["class"] == "normal" and (affected or target["severity"] != "none"):
            raise ValueError(f"target_{index} has an invalid normal tuple")
        if target["class"] == "data_invalid" and (affected or target["severity"] != "advisory"):
            raise ValueError(f"target_{index} has an invalid data_invalid tuple")
        if target["class"] not in {"normal", "data_invalid"} and not affected:
            raise ValueError(f"target_{index} anomaly must be affected")
        if target["class"] == "unknown_anomaly" and target["severity"] != "advisory":
            raise ValueError(f"target_{index} anomaly must have advisory severity")
    has_invalid = any(target["class"] == "data_invalid" for target in targets)
    target_anomaly = any(target["affected"] for target in targets)
    expected_consistency = (global_class == "unknown_anomaly") == target_anomaly
    if (
        not isinstance(model["global_target_consistent"], bool)
        or model["global_target_consistent"] is not expected_consistency
    ):
        raise ValueError("model global_target_consistent disagrees with its global and target decisions")
    has_anomaly = global_class == "unknown_anomaly" or target_anomaly
    expected_state = "data_invalid" if has_invalid else ("advisory" if has_anomaly else "normal")
    if data["system_state"] != expected_state:
        raise ValueError("model system_state disagrees with the global and target decisions")
    return targets


def _panel_sensor_label(target: dict[str, Any]) -> str:
    if target["class"] == "normal":
        return "normal"
    if target["class"] == "data_invalid":
        return "data_invalid"
    return "unknown_anomaly"


def _panel_network_label(targets: list[dict[str, Any]]) -> str:
    if any(target["class"] == "data_invalid" for target in targets):
        return "data_invalid"
    affected = [target for target in targets if target["affected"]]
    if not affected:
        return "normal"
    return "unknown_anomaly"


def _panel_system_label(model_result: dict[str, Any]) -> str:
    state = str(model_result.get("system_state") or "").strip().lower()
    if state == "data_invalid":
        return "data_invalid"
    if state == "normal":
        return "normal"
    return "unknown_anomaly"


def apply_vibrogemma_monitor_decision(
    result: dict[str, Any],
    *,
    model_base_url: str,
    model_api_key: str,
    model: str,
    model_timeout_s: float,
    config_path: str | None = None,
    process_reader: bool = False,
    start_time_s: float | None = None,
    require_current: bool = True,
    replay_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    decision_started = time.monotonic()
    project = _project_root()
    bundle = Path(
        os.environ.get("VIBROGEMMA_BUNDLE")
        or project.parent / "models" / "vibroagent-gemma-g1"
    )

    def failed(reason: str) -> dict[str, Any]:
        targets = [
            {
                "sensor_id": slot,
                "affected": False,
                "class": "data_invalid",
                "severity": "advisory",
                "choice_probability": None,
                "raw_choice_probability": None,
                "confidence_operational": False,
            }
            for slot in SLOTS[1:]
        ]
        for report in result.get("sensor_agent_reports") or []:
            if not isinstance(report, dict) or report.get("sensor_id") not in SLOTS[1:]:
                continue
            report["small_agent_label"] = "data_invalid"
            report["quality_flags"] = sorted(
                set(report.get("quality_flags") or [])
                | {"data_invalid", "vibrogemma_deployment_failure"}
            )
            report["model_metadata"] = {
                "decision_source": "vibrogemma_deployment_guard",
                "model_used": False,
                "error": reason,
            }
            report["evidence"] = []
            for inherited in ("python_rule_label", "small_agent_confidence"):
                report.pop(inherited, None)

        inherited_assessment = result.get("network_assessment") or {}
        assessment = {
            key: inherited_assessment.get(key)
            for key in ("schema_version", "analysis_domain", "window_id", "baseline_sensor_id")
            if inherited_assessment.get(key) is not None
        }
        window_id = str(result.get("window_id") or assessment.get("window_id") or "").strip()
        explanation = (
            "Agent check unavailable: the live VibroGemma deployment did not complete "
            f"({reason}). No normal verdict was emitted."
        )
        assessment.update(
            {
                "window_id": window_id or None,
                "baseline_sensor_id": "baseline",
                "main_agent_label": "data_invalid",
                "network_label": "data_invalid",
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": ["data_invalid", "vibrogemma_deployment_failure"],
                "descriptor_text": "Agent check unavailable",
                "label_uncertain": True,
                "localization_status": "deployment_unavailable",
                "localization_verified": False,
                "geometry_valid": False,
                "evidence": [],
                "explanation": explanation,
                "model_metadata": {
                    "decision_source": "vibrogemma_deployment_guard",
                    "model_used": False,
                    "error": reason,
                },
            }
        )
        result["network_assessment"] = assessment
        result["main_agent_explanation"] = explanation
        result["main_agent_explanation_source"] = "vibrogemma_deployment_guard"
        result.pop("_vibrogemma_replay_input", None)
        result.pop("_vibrogemma_replay_input_sha256", None)
        result.pop("_vibrogemma_encoder_replay_input", None)
        result["model_metadata"] = {
            "agent_mode": "vibrogemma_live",
            "monitor_llm_model_used": False,
            "monitor_llm_fallback_reason": reason,
            "fallbacks_used": [],
            "vibrogemma_monitor": {
                "targets": targets,
                "trained_label_contract": dict(G1_LABEL_CONTRACT),
                "decision_mode": _bundle_decision_mode(bundle),
                "global_class": "data_invalid",
                "global_severity": "advisory",
                "global_target_consistent": True,
                "localization_status": "deployment_unavailable",
                "localization_verified": False,
                "deployment_error": reason,
                "episode_id": window_id or None,
            },
            "total_duration_s": round(time.monotonic() - decision_started, 3),
        }
        return result

    try:
        registry = SensorRegistry(_sensor_config(config_path))
    except Exception as exc:
        return failed(f"sensor_registry_failed:{format_exception_for_response(exc)}")
    sensors = registry.sensors
    if registry.baseline_sensor_id != "baseline" or not set(SLOTS).issubset(sensors):
        return failed(f"sensor_set_mismatch:{sorted(sensors)}")
    replay_targets = {
        label.removeprefix("lumo:")
        for label in replay_labels
        if label.startswith("lumo:")
    }
    if len(replay_targets) > 1 or any(target not in _LUMO_TRAINING_FIXTURES for target in replay_targets):
        return failed(f"invalid_replay_labels:{sorted(replay_labels)}")
    scheduled_lumo_target = next(iter(replay_targets), "")
    configured_lumo_target = os.environ.get("VIBRO_LUMO_TRAINING_INJECTION", "").strip().lower()
    lumo_training_target = (
        "target_3"
        if configured_lumo_target in {"1", "true", "yes", "on"}
        else configured_lumo_target
    )
    lumo_training_target = scheduled_lumo_target or lumo_training_target
    lumo_training_test = lumo_training_target in _LUMO_TRAINING_FIXTURES
    try:
        source_bindings = _source_bindings(registry)
    except Exception as exc:
        return failed(f"sensor_source_failed:{format_exception_for_response(exc)}")
    if start_time_s is not None:
        device_identity = {"verified": False, "reason": "immutable_offline_recording"}
    else:
        try:
            device_identity = _device_identity(registry)
        except Exception as exc:
            return failed(f"device_identity_failed:{format_exception_for_response(exc)}")

    reader = (
        read_sdk_vibrometer_window_via_board_process
        if board_reader_processes_enabled(default=process_reader)
        else read_sdk_vibrometer_window
    )
    episode_id = str(result.get("window_id") or "").strip() or (
        "vibrogemma_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    capture_started_utc = datetime.now(timezone.utc)
    read_started = time.monotonic()
    try:
        position_overrides: dict[str, np.ndarray] = {}
        axis_masks: dict[str, np.ndarray] = {}
        captured: dict[str, _CapturedBoard] = {}
        with ThreadPoolExecutor(max_workers=len(SLOTS), thread_name_prefix="vibrogemma-read") as executor:
            for slot, captured_board in zip(
                SLOTS,
                executor.map(
                    lambda slot: _capture_board(
                        slot,
                        sensors[slot],
                        reader,
                        start_time_s=start_time_s,
                        require_current=require_current,
                    ),
                    SLOTS,
                ),
                strict=True,
            ):
                captured[slot] = captured_board
        capture_completed_utc = datetime.now(timezone.utc)
        arrays, native_rates, capture_by_slot, capture_metadata = _finalize_capture(
            captured,
            capture_started_utc=capture_started_utc,
            capture_completed_utc=capture_completed_utc,
        )
        geometry_valid = all(sensors[slot].sensor_position is not None for slot in SLOTS)
        capture_metadata.update(
            {
                "episode_id": episode_id,
                "sensor_config_path": str(registry.config_path),
                "sensor_config_sha256": hashlib.sha256(registry.config_path.read_bytes()).hexdigest(),
                "geometry_valid": geometry_valid,
                "geometry_source_by_slot": {
                    slot: ("sensor_registry" if sensors[slot].sensor_position is not None else "unconfigured")
                    for slot in SLOTS
                },
                "orientation_source_by_slot": {
                    slot: ("sensor_registry" if sensors[slot].orientation_matrix is not None else "unconfigured")
                    for slot in SLOTS
                },
                "device_identity": device_identity,
                "source_bindings": source_bindings,
            }
        )
        if start_time_s is not None:
            capture_metadata.update(
                {
                    "offline_replay": True,
                    "replay_start_time_s": float(start_time_s),
                    "replay_labels": list(replay_labels),
                    "window_time_source": "offline_replay_manifest",
                }
            )
        synthetic_by_slot = {
            slot: capture_by_slot[slot]["synthetic_injection"]
            for slot in SLOTS
            if capture_by_slot[slot].get("synthetic_injection")
        }
        if synthetic_by_slot:
            capture_metadata["synthetic_injection"] = {
                "active": True,
                "test_only": True,
                "targets": synthetic_by_slot,
            }
        reference_rate = native_rates["baseline"]
        arrays = {
            slot: _resample_board_window(
                arrays[slot],
                source_rate_hz=native_rates[slot],
                target_rate_hz=reference_rate,
            )
            for slot in SLOTS
        }
        if lumo_training_test:
            arrays, overlay, reference_rate = _apply_lumo_live_overlay(
                arrays,
                sample_rate_hz=reference_rate,
                target=lumo_training_target,
            )
            position_overrides = {
                slot: np.asarray(overlay["sensor_positions"][slot], dtype=np.float32)
                for slot in SLOTS
            }
            axis_masks = {
                slot: np.asarray(overlay["axis_masks"][slot], dtype=bool)
                for slot in SLOTS
            }
            if start_time_s is not None:
                overlay.update(
                    {
                        "kind": "lumo_offline_replay",
                        "scheduled": True,
                        "replay_start_time_s": float(start_time_s),
                    }
                )
            data_provenance = (
                "recorded_dat_with_lumo_training_overlay"
                if start_time_s is not None
                else "live_capture_with_lumo_training_overlay"
            )
            capture_metadata["synthetic_injection"] = overlay
            capture_metadata["data_provenance"] = data_provenance
            capture_metadata["phase_synchronized"] = False
            capture_metadata["synchronization_verification_id"] = overlay[
                "synchronization_verification_id"
            ]
            capture_metadata["geometry_valid"] = True
            capture_metadata["geometry_source"] = overlay["geometry_source"]
            capture_metadata["geometry_source_by_slot"] = dict.fromkeys(
                SLOTS, overlay["geometry_source"]
            )
            capture_metadata["episode_provenance"] = overlay["episode_provenance"]
            for slot in SLOTS:
                capture_by_slot[slot]["synthetic_injection"] = {
                    **overlay,
                    "overlay_slot": slot,
                }
                capture_by_slot[slot]["data_provenance"] = data_provenance
        capture_metadata["resampled_window_sha256_by_slot"] = _capture_window_fingerprints(arrays)
        rates = dict.fromkeys(SLOTS, reference_rate)
    except Exception as exc:
        return failed(f"live_read_failed:{format_exception_for_response(exc)}")
    read_duration_s = time.monotonic() - read_started

    encoder_started = time.monotonic()
    try:
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from scripts.live_vibrogemma_worker import encode_live_window

        encoder_payload: dict[str, Any] = {
            **arrays,
            **{f"fs_hz__{slot}": np.float64(rate) for slot, rate in rates.items()},
            **{f"metadata__{slot}": capture_by_slot[slot] for slot in SLOTS},
            **{f"axis_mask__{slot}": mask for slot, mask in axis_masks.items()},
            "episode_id": episode_id,
            "window_start_utc": capture_metadata["window_start_utc"],
            "window_end_utc": capture_metadata["window_end_utc"],
            "window_time_source": capture_metadata["window_time_source"],
            "capture_metadata": capture_metadata,
        }
        for slot in SLOTS:
            position = position_overrides.get(slot)
            if position is None and sensors[slot].sensor_position is not None:
                position = np.asarray(sensors[slot].sensor_position, dtype=np.float32)
            if position is not None:
                encoder_payload[f"sensor_position__{slot}"] = np.asarray(
                    position, dtype=np.float32
                )
            if not position_overrides and sensors[slot].orientation_matrix is not None:
                encoder_payload[f"orientation_matrix__{slot}"] = np.asarray(
                    sensors[slot].orientation_matrix, dtype=np.float32
                )
        vibration = encode_live_window(
            encoder_payload,
            bundle,
            amplitude_calibrated=_truthy("VIBROGEMMA_AMPLITUDE_CALIBRATED", default=True),
        )
    except Exception as exc:
        return failed(f"encoder_failed:{format_exception_for_response(exc)}")
    encoder_duration_s = time.monotonic() - encoder_started
    vibration_payload_sha256 = _payload_sha256(vibration)
    episode_arrays_sha256 = _arrays_sha256(arrays, reference_rate)

    client = ModelClient(
        base_url=model_base_url,
        api_key=model_api_key,
        model=model,
        base_url_env="MAIN_AGENT_BASE_URL",
        api_key_env="MAIN_AGENT_API_KEY",
        model_env="MAIN_AGENT_MODEL",
        role="vibrogemma_live",
        timeout_s=model_timeout_s,
        max_tokens=1024,
    )
    model_started = time.monotonic()
    model_result = client.call_json_model(
        "Classify the current six-board vibration episode.",
        extra_body={"vibration": vibration},
    )
    model_duration_s = time.monotonic() - model_started
    if not model_result.ok:
        return failed(model_result.error or "model_call_failed")
    try:
        targets = _validate_model_result(
            model_result.data,
            expected_model_identity=_bundle_model_identity(bundle),
        )
    except ValueError as exc:
        return failed(f"model_output_invalid:{exc}")
    if str((model_result.data.get("window") or {}).get("episode_id") or "") != episode_id:
        return failed("model_output_invalid:model response episode_id does not match the captured episode")
    _replace_result_measurements(
        result,
        registry=registry,
        arrays=arrays,
        sampling_rate_hz=reference_rate,
        native_rates=native_rates,
        capture_by_slot=capture_by_slot,
        capture_metadata=capture_metadata,
        episode_id=episode_id,
    )
    model_info = model_result.data.get("model") or {}
    encoder_replay_metadata = {
        "schema": "vibroagent-encoder-input-v1",
        "episode_id": episode_id,
        "sampling_rate_hz": reference_rate,
        "sampling_rate_hz_by_slot": rates,
        "native_sampling_rate_hz_by_slot": native_rates,
        "capture": capture_metadata,
        "capture_by_slot": capture_by_slot,
        "episode_arrays_sha256": episode_arrays_sha256,
        "encoder_artifacts": (vibration.get("metadata") or {}).get("artifact_sha256") or {},
        "model_artifacts": {
            key: model_info.get(key)
            for key in (
                "checkpoint",
                "checkpoint_sha256",
                "bundle_version",
                "bundle_manifest_sha256",
            )
            if model_info.get(key) is not None
        },
    }
    result["_vibrogemma_encoder_replay_input"] = {
        "arrays": {slot: arrays[slot] for slot in SLOTS},
        "metadata": encoder_replay_metadata,
    }
    result["_vibrogemma_replay_input"] = vibration
    result["_vibrogemma_replay_input_sha256"] = vibration_payload_sha256
    by_id = {target["sensor_id"]: target for target in targets}
    for report in result.get("sensor_agent_reports") or []:
        target = by_id.get(str(report.get("sensor_id") or ""))
        if target:
            report["small_agent_label"] = _panel_sensor_label(target)
            report["model_metadata"] = {
                "decision_source": "vibrogemma_encoder_gemma",
                "model_used": True,
                "vibrogemma_class": target["class"],
                "vibrogemma_severity": target["severity"],
            }
            report["evidence"] = []

    network_label = _panel_system_label(model_result.data)
    affected = [target["sensor_id"] for target in targets if target["affected"]]
    significant = [
        target["sensor_id"]
        for target in targets
        if target["affected"] and target["severity"] in {"warning", "critical"}
    ]
    mild = [target["sensor_id"] for target in targets if target["affected"] and target["severity"] == "advisory"]
    quality_flags = ["data_invalid"] if any(target["class"] == "data_invalid" for target in targets) else []
    explanation = str(model_result.data.get("explanation") or "").strip()
    summary = (
        f"Agent verdict: {network_label.replace('_', ' ')}. "
        + (explanation or "The constrained Agent record was accepted.")
        + " This is monitoring/triage information, not a certified structural-safety assessment."
    )
    geometry_valid = bool((vibration.get("metadata") or {}).get("geometry_valid"))
    localization_status = (
        model_info.get("localization_status") if geometry_valid else "geometry_unavailable"
    )
    inherited_assessment = result.get("network_assessment") or {}
    assessment = {
        key: inherited_assessment.get(key)
        for key in ("schema_version", "analysis_domain", "window_id", "baseline_sensor_id")
        if inherited_assessment.get(key) is not None
    }
    assessment.update(
        {
            "window_id": episode_id,
            "baseline_sensor_id": "baseline",
            "main_agent_label": network_label,
            "network_label": network_label,
            "affected_sensor_ids": affected,
            "significant_sensor_ids": significant,
            "mild_sensor_ids": mild,
            "quality_flags": quality_flags,
            "descriptor_text": "Agent: " + network_label.replace("_", " "),
            "label_uncertain": False,
            "localization_status": localization_status,
            "localization_verified": bool(geometry_valid),
            "geometry_valid": geometry_valid,
            "geometry_source": (vibration.get("metadata") or {}).get("geometry_source"),
            "evidence": [],
            "explanation": summary,
            "model_metadata": model_result.metadata,
        }
    )
    result["network_assessment"] = assessment
    result["main_agent_explanation"] = summary
    result["main_agent_explanation_source"] = "vibrogemma_constrained_record"
    metadata = {}
    metadata.update(
        {
            "agent_mode": "vibrogemma_live",
            "monitor_llm_model_used": True,
            "monitor_llm_model": model,
            "fallbacks_used": [],
            "vibrogemma_monitor": {
                "targets": targets,
                "trained_label_contract": model_info.get("label_contract") or {},
                "decision_mode": model_info.get("decision_mode"),
                "global_class": model_info.get("global_class"),
                "global_severity": model_info.get("global_severity"),
                "global_target_consistent": model_info.get("global_target_consistent"),
                "localization_status": localization_status,
                "model_localization_status": model_info.get("localization_status"),
                "localization_verified": bool(geometry_valid),
                "geometry_valid": geometry_valid,
                "geometry_source": (vibration.get("metadata") or {}).get("geometry_source"),
                "raw_global_choice_probability": model_info.get("raw_global_choice_probability"),
                "raw_target_choice_probabilities": model_info.get("raw_target_choice_probabilities"),
                "structured_target_set_candidate_scores": model_info.get(
                    "structured_target_set_candidate_scores"
                ),
                "gemma_binary_choice_token_ids": model_info.get(
                    "gemma_binary_choice_token_ids"
                ),
                "gemma_binary_choice_logits": model_info.get(
                    "gemma_binary_choice_logits"
                ),
                "gemma_binary_choice_logit_margins": model_info.get(
                    "gemma_binary_choice_logit_margins"
                ),
                "gemma_binary_query_slot_order": model_info.get(
                    "gemma_binary_query_slot_order"
                ),
                "target_set_cardinality_bias": model_info.get(
                    "target_set_cardinality_bias"
                ),
                "embedding_sha256_by_slot": model_info.get(
                    "embedding_sha256_by_slot"
                ),
                "embedding_rms_by_slot": model_info.get(
                    "embedding_rms_by_slot"
                ),
                "encoder_metadata": vibration.get("metadata") or {},
                "model_window": model_result.data.get("window") or {},
                "model_quality": model_result.data.get("quality") or {},
                "model_evidence": model_result.data.get("evidence") or {},
                "capture": capture_metadata,
                "capture_by_slot": capture_by_slot,
                "episode_id": episode_id,
                "episode_arrays_sha256": episode_arrays_sha256,
                "vibration_payload_sha256": vibration_payload_sha256,
                "soft_tokens_valid": sum(bool(value) for value in vibration.get("soft_token_mask") or []),
                "read_duration_s": round(read_duration_s, 3),
                "encoder_duration_s": round(encoder_duration_s, 3),
                "model_duration_s": round(model_duration_s, 3),
                "total_duration_s": round(time.monotonic() - decision_started, 3),
                "fs_native_hz": reference_rate,
                "fs_native_hz_by_slot": native_rates,
            },
        }
    )
    result["model_metadata"] = metadata
    return result
