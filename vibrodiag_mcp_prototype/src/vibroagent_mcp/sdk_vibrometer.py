"""STDATALOG-PYSDK backed vibrometer source.

In the STWIN/STEVAL DATALOG setup, the vibration sensor is exposed as the
IIS3DWB accelerometer component named ``iis3dwb_acc``. These helpers read HSD
acquisition folders through ``stdatalog_core.HSD.HSDatalog`` and return compact
windows suitable for MCP tools.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SENSOR_NAME = "iis3dwb_acc"
DEFAULT_AXIS = "norm"
SDK_SOURCE_NAME = "stdatalog_sdk_hsd"
DEFAULT_MAX_RETURN_SAMPLES = 200_000
LATEST_WINDOW_READ_MARGIN_S = 1.0
LATEST_WINDOW_BACKOFFS_S = (0.0, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0)
LATEST_ZERO_START_MAX_END_S = 600.0
# Hard ceiling for one incremental catch-up decode. A cold seed that lands at
# the file head (all frontier-bounded windows failed) leaves the cursor hours
# behind the duration estimate; decoding that hole in one call materializes
# the whole acquisition as frames (~2.5 GB/h raw, x3-4 in copies) and
# OOM-killed the webchat, its reader workers, and the MCP subprocess on
# 2026-07-06 once acquisitions crossed ~2 h. The live buffer keeps only
# ~20 s, so anything beyond one bounded span is waste: clamp the read to the
# tail of the requested range and accept the gap.
DEFAULT_LIVE_DECODER_MAX_DECODE_SPAN_S = 120.0
# Frontier bisection probes: span of one probe read and the convergence
# resolution. Probes cost one bounded SDK read each (~2 s of data), so a 2 h
# file resolves in ~10 probes.
_FRONTIER_PROBE_SPAN_S = 2.0
_FRONTIER_PROBE_RESOLUTION_S = 15.0
MAX_TIMESTAMP_OVERRUN_S = 120.0
DEFAULT_CURRENT_FILE_MAX_AGE_S = 30.0
DEFAULT_LIVE_DECODER_BUFFER_DURATION_S = 20.0
DEFAULT_LIVE_DECODER_MIN_REFRESH_INTERVAL_S = 0.05
DEFAULT_LIVE_DECODER_SEED_DURATION_S = 10.0
VALID_AXES = {"norm", "magnitude", "vector_norm", "x", "y", "z", "0", "1", "2"}
# STDATALOG-PYSDK/HSDatalog has process-wide mutable decode state.
# Use an RLock so stateless and persistent per-board decoders can safely
# serialize SDK calls, including nested calls made while a caller already
# holds the SDK read lock.
_HSDATALOG_READ_LOCK = threading.RLock()
_LIVE_DECODER_POOL_LOCK = threading.Lock()
_LIVE_DECODER_POOL: dict[tuple[str, str], Any] = {}

# Internal keys that STDATALOG-PYSDK adds/mutates in component status dicts
# while decoding. These must not leak from one independent bounded read to the
# next, especially in live mode where one old failed scan can otherwise make the
# next board read look corrupted.
_SDK_MUTABLE_STATUS_KEYS = {
    "last_index",
    "missing_bytes",
    "saved_bytes",
    "prev_data_byte_counter",
    "is_first_chunk",
    "is_same_dps",
}


@dataclass(frozen=True)
class SdkVibrometerWindow:
    """One vibration window read from an ST Datalog acquisition."""

    signal: np.ndarray
    timestamps_s: np.ndarray | None
    sampling_rate_hz: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _SdkReadResult:
    samples_2d: np.ndarray
    timestamps_s: np.ndarray | None
    axis_columns: list[str]
    read_start_time_s: float
    read_end_time_s: float
    sdk_read_strategy: str
    latest_timestamp_gap_s: float | None
    sdk_read_attempts: list[dict[str, Any]]


@dataclass(frozen=True)
class _TrimmedWindow:
    samples_2d: np.ndarray
    timestamps_s: np.ndarray | None
    start_time_s: float
    end_time_s: float
    timestamp_mode: str
    timestamp_warning: str | None


def _apply_synthetic_target5_test(
    samples_2d: np.ndarray,
    timestamps_s: np.ndarray | None,
    sampling_rate_hz: int,
    *,
    machine_id: str,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Apply an explicit, default-off sine test before axis selection."""
    if os.environ.get("VIBRO_SYNTHETIC_TARGET5_ENABLED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return samples_2d, None
    requested_targets = {
        target.strip().lower()
        for target in os.environ.get("VIBRO_SYNTHETIC_TARGET", "target_5").split(",")
        if target.strip()
    }
    allowed_targets = {f"target_{index}" for index in range(1, 6)}
    targets = allowed_targets if requested_targets == {"all"} else requested_targets
    if not targets <= allowed_targets:
        raise ValueError("VIBRO_SYNTHETIC_TARGET must be all or a comma-separated subset of target_1 through target_5")
    if machine_id not in targets:
        return samples_2d, None

    frequency_hz = _safe_float(os.environ.get("VIBRO_SYNTHETIC_TARGET5_FREQUENCY_HZ", "37"))
    amplitude_g = _safe_float(os.environ.get("VIBRO_SYNTHETIC_TARGET5_AMPLITUDE_G", "0.5"))
    axis_name = os.environ.get("VIBRO_SYNTHETIC_TARGET5_AXIS", "z").strip().lower()
    axis_by_name = {"x": 0, "y": 1, "z": 2}
    if axis_name not in axis_by_name:
        raise ValueError("VIBRO_SYNTHETIC_TARGET5_AXIS must be x, y, or z")
    if frequency_hz is None or frequency_hz <= 0 or frequency_hz >= sampling_rate_hz / 2:
        raise ValueError("synthetic target frequency must be finite and below Nyquist")
    if amplitude_g is None or amplitude_g <= 0 or amplitude_g > 4.0:
        raise ValueError("synthetic target amplitude must be within (0, 4] g")
    if samples_2d.ndim != 2 or samples_2d.shape[1] < 3:
        raise ValueError("synthetic target test requires XYZ samples")

    if timestamps_s is not None and timestamps_s.size == samples_2d.shape[0]:
        time_s = np.asarray(timestamps_s, dtype=np.float64)
    else:
        time_s = np.arange(samples_2d.shape[0], dtype=np.float64) / sampling_rate_hz
    output = np.asarray(samples_2d, dtype=np.float64).copy()
    output[:, axis_by_name[axis_name]] += amplitude_g * np.sin(2 * np.pi * frequency_hz * time_s)
    descriptor = {
        "active": True,
        "test_only": True,
        "kind": "sine",
        "target": machine_id,
        "axis": axis_name,
        "frequency_hz": frequency_hz,
        "amplitude_g": amplitude_g,
        "source": "env_test_hook",
    }
    return output, descriptor


class _PersistentLiveBoardDecoder:
    def __init__(self, folder: Path, sensor: str):
        self.folder = folder
        self.sensor = sensor
        self.data_file = folder / f"{sensor}.dat"
        self.lock = threading.Lock()
        self.hsd: Any | None = None
        self.hsd_instance: Any | None = None
        self.sensor_component: Any | None = None
        self.sensor_status: dict[str, Any] = {}
        self.configured_sampling_rate = 0
        self.axis_columns: list[str] = []
        self.samples_2d = np.empty((0, 0), dtype=np.float64)
        self.timestamps_s = np.empty(0, dtype=np.float64)
        self.latest_timestamp_s: float | None = None
        self.next_start_time_s = 0.0
        self.last_sdk_read_strategy = "persistent_live_uninitialized"
        self.last_sdk_read_attempts: list[dict[str, Any]] = []
        self.last_sdk_read_start_time_s = 0.0
        self.last_sdk_read_end_time_s = 0.0
        self.last_timestamp_mode = "persistent_live_uninitialized"
        self.last_timestamp_warning: str | None = None
        self.last_refresh_monotonic = 0.0
        self.last_file_size: int | None = None
        self.last_file_mtime: float | None = None
        self.latest_gap_calibration_s = 0.0
        self.gap_calibration_updated_monotonic = 0.0
        self.rollover_timestamp_offset_s = 0.0
        self.rollover_events = 0
        # Frontier anchor for the data-duration estimate: (board timestamp,
        # file sample count) captured when the decoder provably consumed
        # everything decodable. See _anchored_data_duration_s_locked.
        self.duration_anchor_ts_s: float | None = None
        self.duration_anchor_samples: int | None = None

    def read_window(
        self,
        *,
        machine_id: str,
        axis: str,
        duration_s: float,
        require_current: bool,
    ) -> SdkVibrometerWindow:
        with self.lock:
            self._ensure_initialized_locked()
            acquisition_duration_s = _acquisition_duration_s(self.hsd_instance)
            initial_data_duration_s = self._estimate_data_duration_s_locked()
            duration_hint_s = initial_data_duration_s or acquisition_duration_s
            freshness = _data_file_freshness_metadata(self.data_file)
            stale_metadata = self._build_stale_metadata_locked(
                machine_id=machine_id,
                axis=axis,
                duration_s=duration_s,
                acquisition_duration_s=acquisition_duration_s,
                data_duration_s=initial_data_duration_s,
                freshness=freshness,
            )
            if require_current and not freshness.get("data_is_current", False):
                raise LiveSdkVibrometerDataRequiredError(stale_metadata)

            self._refresh_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
            self._maybe_update_duration_anchor_locked()
            if self._buffer_span_s_locked() + self._sample_period_s_locked() + 1e-9 < duration_s:
                self._reset_locked()
                reset_data_duration_s = self._estimate_data_duration_s_locked()
                reset_duration_hint_s = reset_data_duration_s or acquisition_duration_s or duration_hint_s
                self._refresh_locked(duration_s=duration_s, duration_hint_s=reset_duration_hint_s)
                self._maybe_update_duration_anchor_locked()

            data_duration_s = self._estimate_data_duration_s_locked()
            duration_hint_s = data_duration_s or acquisition_duration_s
            raw_latest_timestamp_gap_s = self._raw_latest_timestamp_gap_s_locked(duration_hint_s)
            # Absorb estimate wobble into the calibration BEFORE the reset check:
            # a raw-gap jump on a read that provably kept up must not force a full
            # reseed. The absorb rate is bounded (see
            # _maybe_update_gap_calibration_locked), so a genuinely large jump
            # still trips the reset below in the same poll.
            self._maybe_update_gap_calibration_locked(
                raw_gap_s=raw_latest_timestamp_gap_s,
                freshness=freshness,
                # "Never calibrated", NOT "calibrated at zero": with the
                # anchored duration estimate a healthy calibration IS 0.0, and
                # treating that as initial would absorb a real gap wholesale.
                initial=self.gap_calibration_updated_monotonic <= 0.0,
            )
            normalized_gap_for_reset_s = self._normalized_latest_timestamp_gap_s_locked(raw_latest_timestamp_gap_s)
            if (
                normalized_gap_for_reset_s is not None
                and normalized_gap_for_reset_s > self._gap_reset_threshold_s_locked()
            ):
                self._reset_locked()
                reset_data_duration_s = self._estimate_data_duration_s_locked()
                reset_duration_hint_s = reset_data_duration_s or acquisition_duration_s or duration_hint_s
                self._refresh_locked(duration_s=duration_s, duration_hint_s=reset_duration_hint_s)
                self._maybe_update_duration_anchor_locked()
                data_duration_s = self._estimate_data_duration_s_locked()
                duration_hint_s = data_duration_s or acquisition_duration_s
                raw_latest_timestamp_gap_s = self._raw_latest_timestamp_gap_s_locked(duration_hint_s)
                self._maybe_update_gap_calibration_locked(
                    raw_gap_s=raw_latest_timestamp_gap_s,
                    freshness=freshness,
                    initial=self.gap_calibration_updated_monotonic <= 0.0,
                )

            if self.samples_2d.shape[0] < 32 or self.timestamps_s.size < 32:
                raise ValueError(
                    "STDATALOG-PYSDK persistent live decoder could not build a sufficient latest vibrometer window"
                )

            trimmed = _trim_latest_window(
                self.samples_2d,
                self.timestamps_s,
                sampling_rate_hz=self.configured_sampling_rate,
                duration_s=duration_s,
            )
            signal = _select_axis(trimmed.samples_2d, axis=axis)
            if signal.size < 32:
                raise ValueError(
                    f"SDK vibrometer window has only {signal.size} samples. Increase duration_s or adjust start_time_s."
                )
            timestamp_mode = trimmed.timestamp_mode
            timestamp_warning = trimmed.timestamp_warning
            if self.last_timestamp_warning and timestamp_warning is None:
                timestamp_mode = self.last_timestamp_mode
                timestamp_warning = self.last_timestamp_warning

            latest_timestamp_gap_s = self._normalized_latest_timestamp_gap_s_locked(raw_latest_timestamp_gap_s)
            metadata = {
                "machine_id": machine_id,
                "acquisition_folder": str(self.folder),
                "sensor_name": self.sensor,
                "axis": axis,
                "axis_columns": list(self.axis_columns),
                "unit": "g",
                "sample_count": int(signal.size),
                "read_mode": "latest",
                "requested_start_time_s": None,
                "start_time_s": float(trimmed.start_time_s),
                "requested_duration_s": float(duration_s),
                "acquisition_duration_s": _safe_float(acquisition_duration_s),
                "estimated_data_duration_s": _safe_float(data_duration_s),
                "sdk_read_start_time_s": _safe_float(self.last_sdk_read_start_time_s),
                "sdk_read_end_time_s": _safe_float(self.last_sdk_read_end_time_s),
                "sdk_read_strategy": self.last_sdk_read_strategy,
                "sdk_latest_timestamp_raw_gap_s": _safe_float(raw_latest_timestamp_gap_s),
                "sdk_latest_timestamp_gap_calibration_s": _safe_float(self.latest_gap_calibration_s),
                "sdk_latest_timestamp_gap_s": _safe_float(latest_timestamp_gap_s),
                "sdk_duration_estimate_anchored": self.duration_anchor_ts_s is not None,
                "sdk_file_rollovers": int(self.rollover_events),
                "sdk_rollover_timestamp_offset_s": _safe_float(self.rollover_timestamp_offset_s),
                "sdk_read_attempts": list(self.last_sdk_read_attempts),
                "sdk_timestamp_start_s": _safe_float(self.timestamps_s[0]) if self.timestamps_s.size else None,
                "sdk_timestamp_end_s": _safe_float(self.timestamps_s[-1]) if self.timestamps_s.size else None,
                "timestamp_start_s": _safe_float(trimmed.timestamps_s[0]) if trimmed.timestamps_s is not None and trimmed.timestamps_s.size else None,
                "timestamp_end_s": _safe_float(trimmed.timestamps_s[-1]) if trimmed.timestamps_s is not None and trimmed.timestamps_s.size else None,
                "timestamp_mode": timestamp_mode,
                "timestamp_warning": timestamp_warning,
                "configured_odr_hz": _safe_float(self.sensor_status.get("odr")),
                "measured_odr_hz": _safe_float(self.sensor_status.get("measodr")),
                "sampling_rate_hz": int(self.configured_sampling_rate),
                "full_scale_g": _safe_float(self.sensor_status.get("fs")),
                "dimensions": _safe_int(self.sensor_status.get("dim")),
                "sensitivity": _safe_float(self.sensor_status.get("sensitivity")),
                "data_file": str(self.data_file),
                **freshness,
            }
            _apply_decoded_currentness_metadata(metadata)
            if require_current and not metadata.get("data_is_current", False):
                raise LiveSdkVibrometerDataRequiredError(metadata)
            return SdkVibrometerWindow(
                signal=signal,
                timestamps_s=trimmed.timestamps_s,
                sampling_rate_hz=int(self.configured_sampling_rate),
                metadata=metadata,
            )

    def _build_stale_metadata_locked(
        self,
        *,
        machine_id: str,
        axis: str,
        duration_s: float,
        acquisition_duration_s: float | None,
        data_duration_s: float | None,
        freshness: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "machine_id": machine_id,
            "acquisition_folder": str(self.folder),
            "sensor_name": self.sensor,
            "axis": axis,
            "unit": "g",
            "read_mode": "latest",
            "requested_start_time_s": None,
            "requested_duration_s": float(duration_s),
            "acquisition_duration_s": _safe_float(acquisition_duration_s),
            "estimated_data_duration_s": _safe_float(data_duration_s),
            "configured_odr_hz": _safe_float(self.sensor_status.get("odr")),
            "measured_odr_hz": _safe_float(self.sensor_status.get("measodr")),
            "sampling_rate_hz": int(self.configured_sampling_rate),
            "data_file": str(self.data_file),
            **freshness,
        }

    def _ensure_initialized_locked(self) -> None:
        if self.hsd is not None and self.hsd_instance is not None and self.sensor_component:
            return
        self._reset_locked()

    def _reload_sdk_handles_locked(self) -> None:
        self.hsd, self.hsd_instance = _create_hsd(self.folder)
        sensor_component = self.hsd.get_sensor(self.hsd_instance, self.sensor)
        if not sensor_component:
            raise ValueError(f"Sensor {self.sensor!r} is not present in acquisition folder {self.folder}")
        # STDATALOG mutates the component/status dictionaries while decoding
        # batches (for example last_index, missing_bytes, prev counters, and
        # ioffset). Keep an immutable template in the persistent reader and pass
        # fresh deep copies into every SDK call, otherwise one failed/partial read
        # can poison later live reads with false DataCorruptedException errors.
        self.sensor_component = copy.deepcopy(sensor_component)
        self.sensor_status = _sensor_name_and_status(self.sensor_component)[1]
        self.configured_sampling_rate = _sampling_rate_from_sensor_status(self.sensor_status)

    def _reset_locked(self) -> None:
        self._reload_sdk_handles_locked()
        self.axis_columns = []
        self.samples_2d = np.empty((0, 0), dtype=np.float64)
        self.timestamps_s = np.empty(0, dtype=np.float64)
        self.latest_timestamp_s = None
        self.next_start_time_s = 0.0
        self.rollover_timestamp_offset_s = 0.0
        self.last_sdk_read_strategy = "persistent_live_reset"
        self.last_sdk_read_attempts = []
        self.last_sdk_read_start_time_s = 0.0
        self.last_sdk_read_end_time_s = 0.0
        self.last_timestamp_mode = "persistent_live_reset"
        self.last_timestamp_warning = None
        self.last_refresh_monotonic = 0.0
        self.last_file_size = _safe_file_size(self.data_file)
        self.last_file_mtime = _path_mtime(self.data_file)
        self.duration_anchor_ts_s = None
        self.duration_anchor_samples = None

    def _refresh_locked(self, *, duration_s: float, duration_hint_s: float | None) -> None:
        file_size = _safe_file_size(self.data_file)
        file_mtime = _path_mtime(self.data_file)
        if self._file_shrunk_locked(file_size=file_size):
            if self.timestamps_s.size and file_size is not None:
                self._handle_file_rollover_locked(file_size=file_size, file_mtime=file_mtime)
                duration_hint_s = self._estimate_data_duration_s_locked() or duration_hint_s
            else:
                self.latest_gap_calibration_s = 0.0
                self.gap_calibration_updated_monotonic = 0.0
                self._reset_locked()
                file_size = _safe_file_size(self.data_file)
                file_mtime = _path_mtime(self.data_file)
        elif self._should_reset_for_file_change_locked(file_size=file_size, file_mtime=file_mtime):
            self.latest_gap_calibration_s = 0.0
            self.gap_calibration_updated_monotonic = 0.0
            self._reset_locked()
            file_size = _safe_file_size(self.data_file)
            file_mtime = _path_mtime(self.data_file)

        if (
            self.timestamps_s.size
            and file_size == self.last_file_size
            and file_mtime == self.last_file_mtime
            and (time.monotonic() - self.last_refresh_monotonic) < _live_decoder_min_refresh_interval_s()
        ):
            return

        try:
            if self.timestamps_s.size == 0:
                self._seed_buffer_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
                self._catch_up_after_seed_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
            else:
                self._append_new_data_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
        except Exception as exc:
            if not _is_sdk_data_corruption_error(exc) or self.timestamps_s.size == 0:
                raise
            recovery_attempt = {
                "strategy": "persistent_live_incremental_recovery",
                "start_time_s": _safe_float(max(0.0, self.next_start_time_s)),
                "end_time_s": _safe_float(duration_hint_s),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            recovery_file_size = _safe_file_size(self.data_file)
            recovery_file_mtime = _path_mtime(self.data_file)
            if self._file_shrunk_locked(file_size=recovery_file_size) and recovery_file_size is not None:
                rollover_attempt = {**recovery_attempt, "strategy": "persistent_live_rollover_recovery"}
                self._handle_file_rollover_locked(file_size=recovery_file_size, file_mtime=recovery_file_mtime)
                rollover_duration_hint_s = self._estimate_data_duration_s_locked() or duration_hint_s
                try:
                    self._append_new_data_locked(duration_s=duration_s, duration_hint_s=rollover_duration_hint_s)
                except Exception as rollover_exc:
                    if not _is_sdk_data_corruption_error(rollover_exc):
                        raise
                    self.last_sdk_read_strategy = "persistent_live_rollover_pending"
                    self.last_sdk_read_attempts = [
                        rollover_attempt,
                        {
                            "strategy": "persistent_live_rollover_pending",
                            "start_time_s": _safe_float(max(0.0, self.next_start_time_s)),
                            "end_time_s": _safe_float(rollover_duration_hint_s),
                            "error": f"{rollover_exc.__class__.__name__}: {rollover_exc}",
                        },
                    ]
                else:
                    self.last_sdk_read_strategy = f"persistent_live_rollover_recovered:{self.last_sdk_read_strategy}"
                    self.last_sdk_read_attempts = [rollover_attempt, *self.last_sdk_read_attempts]
                file_size = recovery_file_size
                file_mtime = recovery_file_mtime
            else:
                self._reset_locked()
                self._seed_buffer_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
                try:
                    self._catch_up_after_seed_locked(duration_s=duration_s, duration_hint_s=duration_hint_s)
                except Exception:
                    pass  # keep the seeded buffer; the next refresh will catch up
                self.last_sdk_read_strategy = f"persistent_live_recovered:{self.last_sdk_read_strategy}"
                self.last_sdk_read_attempts = [recovery_attempt, *self.last_sdk_read_attempts]

        self.last_refresh_monotonic = time.monotonic()
        self.last_file_size = file_size
        self.last_file_mtime = file_mtime

    def _seed_buffer_locked(self, *, duration_s: float, duration_hint_s: float | None) -> None:
        history_duration_s = min(
            _live_decoder_buffer_duration_s(),
            max(duration_s + LATEST_WINDOW_READ_MARGIN_S, DEFAULT_LIVE_DECODER_SEED_DURATION_S),
        )
        sdk_read = _read_latest_sdk_arrays(
            hsd=self.hsd,
            hsd_instance=self.hsd_instance,
            sensor_component=self.sensor_component,
            duration_s=history_duration_s,
            duration_hint_s=duration_hint_s,
            chunk_size=_dataframe_chunk_size(self.sensor_status, self.configured_sampling_rate, history_duration_s),
            minimum_sample_count=_minimum_latest_sample_count(self.configured_sampling_rate, history_duration_s),
            copy_component=True,
        )
        trimmed = _trim_latest_window(
            sdk_read.samples_2d,
            sdk_read.timestamps_s,
            sampling_rate_hz=self.configured_sampling_rate,
            duration_s=history_duration_s,
        )
        self.samples_2d = trimmed.samples_2d
        self.timestamps_s = np.asarray(trimmed.timestamps_s, dtype=np.float64) if trimmed.timestamps_s is not None else np.empty(0, dtype=np.float64)
        self.axis_columns = list(sdk_read.axis_columns)
        self.latest_timestamp_s = float(self.timestamps_s[-1]) if self.timestamps_s.size else None
        self.next_start_time_s = max(0.0, (self.latest_timestamp_s or 0.0) - self._overlap_duration_s_locked())
        self.last_sdk_read_strategy = f"persistent_live_seed:{sdk_read.sdk_read_strategy}"
        self.last_sdk_read_attempts = list(sdk_read.sdk_read_attempts)
        self.last_sdk_read_start_time_s = sdk_read.read_start_time_s
        self.last_sdk_read_end_time_s = sdk_read.read_end_time_s
        self.last_timestamp_mode = trimmed.timestamp_mode
        self.last_timestamp_warning = trimmed.timestamp_warning
        self._trim_buffer_locked()

    def _catch_up_after_seed_locked(self, *, duration_s: float, duration_hint_s: float | None) -> None:
        # A fresh seed can already be at the decodable tail. Check the caller's
        # duration hint first so normal reads do not repeatedly stat/estimate the
        # file just to prove there is no catch-up work.
        seed_strategy = self.last_sdk_read_strategy
        seed_attempts = self.last_sdk_read_attempts
        caught_up = False
        raw_gap_s = self._raw_latest_timestamp_gap_s_locked(duration_hint_s)
        if raw_gap_s is None or raw_gap_s > self._gap_tolerance_s_locked():
            for _ in range(3):
                prev_tail = self.latest_timestamp_s
                hint_s = self._estimate_data_duration_s_locked()
                try:
                    self._append_new_data_locked(duration_s=duration_s, duration_hint_s=hint_s)
                except Exception as exc:
                    if not _is_sdk_data_corruption_error(exc):
                        raise
                    self._reset_locked()
                    self._seed_buffer_locked(duration_s=duration_s, duration_hint_s=hint_s)
                raw_gap_s = self._raw_latest_timestamp_gap_s_locked(hint_s)
                if raw_gap_s is not None and raw_gap_s <= self._gap_tolerance_s_locked():
                    caught_up = True
                    break
                if self.latest_timestamp_s == prev_tail:
                    break
        else:
            caught_up = True

        # Establish the sample-count/timestamp anchor before calibrating the
        # gap; otherwise the legacy full-file estimate can become a huge
        # calibration that masks all subsequent file growth as decoder lag.
        self._set_initial_duration_anchor_locked()
        anchored_duration_hint_s = self._estimate_data_duration_s_locked()
        anchor_raw_gap_s = self._raw_latest_timestamp_gap_s_locked(
            anchored_duration_hint_s
        )
        if anchor_raw_gap_s is None:
            anchor_raw_gap_s = raw_gap_s
        if anchor_raw_gap_s is not None:
            self._set_gap_calibration_locked(float(anchor_raw_gap_s))
        suffix = "catch_up" if caught_up else "catch_up_partial"
        self.last_sdk_read_strategy = f"{seed_strategy}+{suffix}"
        self.last_sdk_read_attempts = seed_attempts

    def _append_new_data_locked(self, *, duration_s: float, duration_hint_s: float | None) -> None:
        read_start_time_s = max(0.0, self.next_start_time_s)
        aligned_duration_hint_s = self._decode_aligned_duration_hint_s_locked(duration_hint_s)
        if aligned_duration_hint_s is not None:
            read_end_time_s = max(
                read_start_time_s + self._sample_period_s_locked(),
                aligned_duration_hint_s + LATEST_WINDOW_READ_MARGIN_S,
            )
        else:
            latest_for_read_clock_s = self.latest_timestamp_s or read_start_time_s
            if self.rollover_timestamp_offset_s:
                latest_for_read_clock_s = max(0.0, latest_for_read_clock_s - float(self.rollover_timestamp_offset_s))
            read_end_time_s = max(
                read_start_time_s + self._sample_period_s_locked(),
                latest_for_read_clock_s + max(duration_s, self._overlap_duration_s_locked()),
            )
        max_span_s = _max_incremental_decode_span_s()
        if read_end_time_s - read_start_time_s > max_span_s:
            # Decode only the tail of the hole (see DEFAULT_LIVE_DECODER_MAX_DECODE_SPAN_S):
            # the append mask accepts the resulting timestamp jump and the buffer trim
            # keeps ~20 s anyway, so this converges to the frontier in one bounded read
            # instead of materializing hours of history.
            read_start_time_s = max(read_start_time_s, read_end_time_s - max_span_s)
        samples_2d, timestamps_s, axis_columns = self._read_incremental_samples_locked(
            start_time_s=read_start_time_s,
            end_time_s=read_end_time_s,
            duration_s=duration_s,
        )
        if samples_2d.shape[0] == 0 or timestamps_s.size == 0:
            self.last_sdk_read_strategy = "persistent_live_incremental_empty"
            self.last_sdk_read_attempts = []
            self.last_sdk_read_start_time_s = read_start_time_s
            self.last_sdk_read_end_time_s = read_end_time_s
            return

        if not self.axis_columns:
            self.axis_columns = list(axis_columns)
        elif axis_columns:
            samples_2d = _reorder_axis_columns(
                samples_2d=samples_2d,
                axis_columns=axis_columns,
                expected_columns=self.axis_columns,
            )

        epsilon_s = 0.5 * self._sample_period_s_locked()
        if self.timestamps_s.size:
            mask = timestamps_s > (float(self.timestamps_s[-1]) + epsilon_s)
            if not np.any(mask):
                # Advance the cursor only to the end of what the SDK actually decoded,
                # never to read_end_time_s: that end is the file-size duration estimate
                # plus margin, and the estimate runs ahead of the decodable tail by the
                # integrated clock-skew error (~0.2s per hour of file age). Jumping past
                # the tail parks the cursor in not-yet-written data, every later append
                # reads empty, and the gap check then forces a full reset+reseed on every
                # poll — decode time per request degrades from ~20ms to the full seed cost.
                decoded_frontier_s = (
                    float(timestamps_s[-1])
                    - float(self.rollover_timestamp_offset_s)
                    - self._overlap_duration_s_locked()
                )
                self.next_start_time_s = max(self.next_start_time_s, decoded_frontier_s)
                self.last_sdk_read_strategy = "persistent_live_incremental_overlap_only"
                self.last_sdk_read_attempts = []
                self.last_sdk_read_start_time_s = read_start_time_s
                self.last_sdk_read_end_time_s = read_end_time_s
                return
            samples_2d = samples_2d[mask]
            timestamps_s = timestamps_s[mask]

        self._append_samples_locked(samples_2d=samples_2d, timestamps_s=timestamps_s)
        self.latest_timestamp_s = float(self.timestamps_s[-1]) if self.timestamps_s.size else None
        next_latest_s = self.latest_timestamp_s or 0.0
        if self.rollover_timestamp_offset_s:
            next_latest_s = max(0.0, next_latest_s - float(self.rollover_timestamp_offset_s))
        self.next_start_time_s = max(read_start_time_s, next_latest_s - self._overlap_duration_s_locked())
        self.last_sdk_read_strategy = "persistent_live_incremental"
        self.last_sdk_read_attempts = []
        self.last_sdk_read_start_time_s = read_start_time_s
        self.last_sdk_read_end_time_s = read_end_time_s
        self._trim_buffer_locked()

    def _read_incremental_samples_locked(
        self,
        *,
        start_time_s: float,
        end_time_s: float,
        duration_s: float,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        try:
            frames = _read_sdk_dataframe_frames(
                hsd=self.hsd,
                hsd_instance=self.hsd_instance,
                sensor_component=self.sensor_component,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                chunk_size=_dataframe_chunk_size(
                    self.sensor_status,
                    self.configured_sampling_rate,
                    max(duration_s, self._overlap_duration_s_locked()),
                ),
                copy_component=True,
            )
        except ValueError as exc:
            if _is_sdk_no_dataframe_error(exc):
                # A live .dat file can be between SDK-decodable chunks while it is
                # being written, especially when the incremental read is close to
                # the tail.  Treat that as "no new samples yet" and keep serving
                # the previous buffered live window instead of surfacing a false
                # missing-data sensor alert.
                return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), list(self.axis_columns)
            raise
        samples_2d, timestamps_s, axis_columns = _frames_to_arrays(frames)
        samples_2d, timestamps_s = _filter_invalid_timestamp_rows(
            samples_2d,
            timestamps_s,
            max_time_s=end_time_s + MAX_TIMESTAMP_OVERRUN_S,
        )
        if samples_2d.shape[0] == 0:
            return samples_2d, np.empty(0, dtype=np.float64), axis_columns

        timestamp_warning = _timestamp_sequence_warning(timestamps_s)
        if timestamps_s is None or timestamps_s.size != samples_2d.shape[0] or timestamp_warning:
            sample_period_s = self._sample_period_s_locked()
            latest_for_sdk_clock_s = self.latest_timestamp_s or start_time_s
            if self.rollover_timestamp_offset_s:
                latest_for_sdk_clock_s = max(0.0, latest_for_sdk_clock_s - float(self.rollover_timestamp_offset_s))
            base_start_s = max(start_time_s, latest_for_sdk_clock_s)
            timestamps_s = base_start_s + (np.arange(samples_2d.shape[0], dtype=np.float64) * sample_period_s)
        timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
        if self.rollover_timestamp_offset_s:
            timestamps_s = timestamps_s + float(self.rollover_timestamp_offset_s)
        return samples_2d, timestamps_s, axis_columns

    def _append_samples_locked(self, *, samples_2d: np.ndarray, timestamps_s: np.ndarray) -> None:
        if self.samples_2d.shape[0] == 0:
            self.samples_2d = np.asarray(samples_2d, dtype=np.float64)
            self.timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
            return
        self.samples_2d = np.concatenate([self.samples_2d, np.asarray(samples_2d, dtype=np.float64)], axis=0)
        self.timestamps_s = np.concatenate([self.timestamps_s, np.asarray(timestamps_s, dtype=np.float64)], axis=0)

    def _trim_buffer_locked(self) -> None:
        max_samples = max(32, int(round(self.configured_sampling_rate * _live_decoder_buffer_duration_s())))
        if self.samples_2d.shape[0] <= max_samples:
            return
        self.samples_2d = self.samples_2d[-max_samples:]
        self.timestamps_s = self.timestamps_s[-max_samples:]

    def _buffer_span_s_locked(self) -> float:
        if self.timestamps_s.size == 0:
            return 0.0
        return max(0.0, float(self.timestamps_s[-1]) - float(self.timestamps_s[0]) + self._sample_period_s_locked())

    def _sample_period_s_locked(self) -> float:
        if self.configured_sampling_rate <= 0:
            return 0.0
        return 1.0 / float(self.configured_sampling_rate)

    def _packet_duration_s_locked(self) -> float:
        samples_per_ts = _safe_int(self.sensor_status.get("samples_per_ts")) or 0
        if samples_per_ts <= 0 or self.configured_sampling_rate <= 0:
            return 0.0
        return float(samples_per_ts) / float(self.configured_sampling_rate)

    def _overlap_duration_s_locked(self) -> float:
        return max(0.1, self._packet_duration_s_locked() * 2.0)

    def _measured_sample_period_s_locked(self) -> float | None:
        # Per-sample period implied by the SDK-decoded timestamps currently in the
        # buffer. Uses the full buffer span (not adjacent samples) so HSDatalog's
        # 6-decimal timestamp rounding averages out to a negligible error.
        if self.timestamps_s.size >= 2:
            span_s = float(self.timestamps_s[-1]) - float(self.timestamps_s[0])
            intervals = int(self.timestamps_s.size) - 1
            if span_s > 0.0 and intervals > 0:
                return span_s / float(intervals)
        return None

    def _effective_sample_rate_hz_locked(self) -> float | None:
        # The board's true sample rate as seen by the SDK timestamp clock. Used to
        # convert the file-size sample count into a duration on the SAME clock as
        # timestamps_s, so a configured-vs-true ODR mismatch cannot integrate into
        # a steadily growing (false) decode gap.
        period_s = self._measured_sample_period_s_locked()
        if not period_s or period_s <= 0.0:
            return None
        rate_hz = 1.0 / period_s
        configured = float(self.configured_sampling_rate)
        if configured > 0.0 and not (0.8 * configured <= rate_hz <= 1.2 * configured):
            # Reject a degenerate buffer (e.g. one spanning a discontinuity) rather
            # than trust a wild estimate; fall back to the configured rate.
            return None
        return rate_hz

    def _estimate_data_duration_s_locked(self) -> float | None:
        anchored_s = self._anchored_data_duration_s_locked()
        if anchored_s is not None:
            return anchored_s
        return _estimate_data_duration_s(
            folder=self.folder,
            sensor=self.sensor,
            sensor_status=self.sensor_status,
            sampling_rate_hz=self.configured_sampling_rate,
            hsd_instance=self.hsd_instance,
            effective_sampling_rate_hz=self._effective_sample_rate_hz_locked(),
        )

    def _file_sample_count_locked(self) -> int | None:
        return _estimate_file_sample_count(
            folder=self.folder,
            sensor=self.sensor,
            sensor_status=self.sensor_status,
            sampling_rate_hz=self.configured_sampling_rate,
            hsd_instance=self.hsd_instance,
        )

    def _anchored_data_duration_s_locked(self) -> float | None:
        # Extrapolate the duration from the last frontier anchor instead of from
        # the start of the file: the rate error then multiplies only the seconds
        # since last frontier contact, not the acquisition age. A short-window
        # measured rate is accurate to ~0.01-0.06%, which over a multi-hour file
        # integrated into seconds of false "SDK gap" (measured on-device: 5.5s
        # at 2h47m) while staying at milliseconds across one poll interval.
        if self.duration_anchor_ts_s is None or self.duration_anchor_samples is None:
            return None
        if self.rollover_timestamp_offset_s:
            return None  # rollover rebases the timestamp clock; legacy math handles it
        samples_now = self._file_sample_count_locked()
        if samples_now is None:
            return None
        if samples_now < self.duration_anchor_samples:
            # File shrank without a detected rollover: the anchor is stale.
            self.duration_anchor_ts_s = None
            self.duration_anchor_samples = None
            return None
        rate_hz = self._effective_sample_rate_hz_locked() or float(self.configured_sampling_rate)
        if rate_hz <= 0.0:
            return None
        return float(self.duration_anchor_ts_s) + float(samples_now - self.duration_anchor_samples) / rate_hz

    def _set_initial_duration_anchor_locked(self) -> None:
        # Called at the end of seed catch-up: the same moment the initial gap
        # calibration is trusted. Even a catch-up that ends with a residual gap
        # has its tail at the decodable frontier (the loop stops when no more
        # data appears); the residual is estimate error, which anchoring here
        # eliminates going forward. A decoder that later wedges cannot hide:
        # the anchored estimate keeps tracking file growth while the tail
        # stalls, so the gap grows and trips the reset threshold.
        if not self.timestamps_s.size or self.rollover_timestamp_offset_s:
            return
        samples_now = self._file_sample_count_locked()
        if samples_now is None:
            return
        self.duration_anchor_ts_s = float(self.timestamps_s[-1])
        self.duration_anchor_samples = int(samples_now)

    def _maybe_update_duration_anchor_locked(self) -> None:
        # Re-anchor only when the anchored estimate itself confirms the tail is
        # at the frontier, keeping (now - anchor) short so the rate error stays
        # below a packet duration. A lagging decoder fails this check, the
        # anchor stays put, and the estimate keeps tracking file growth so the
        # gap surfaces honestly.
        if self.duration_anchor_ts_s is None:
            return  # the initial anchor is only set at seed catch-up
        if not self.timestamps_s.size or self.rollover_timestamp_offset_s:
            return
        anchored_est_s = self._anchored_data_duration_s_locked()
        if anchored_est_s is None:
            return
        tail_ts_s = float(self.timestamps_s[-1])
        if anchored_est_s - tail_ts_s > max(0.1, self._packet_duration_s_locked() * 2.0):
            return
        samples_now = self._file_sample_count_locked()
        if samples_now is None:
            return
        self.duration_anchor_ts_s = tail_ts_s
        self.duration_anchor_samples = int(samples_now)

    def _raw_latest_timestamp_gap_s_locked(self, duration_hint_s: float | None) -> float | None:
        if duration_hint_s is None or self.timestamps_s.size == 0:
            return None
        latest_timestamp_s = float(self.timestamps_s[-1])
        if self.rollover_timestamp_offset_s:
            latest_timestamp_s = max(0.0, latest_timestamp_s - float(self.rollover_timestamp_offset_s))
        return max(0.0, float(duration_hint_s) - latest_timestamp_s)

    def _decode_aligned_duration_hint_s_locked(self, duration_hint_s: float | None) -> float | None:
        if duration_hint_s is None:
            return None
        latest_timestamp_s = self.latest_timestamp_s
        if latest_timestamp_s is None and self.timestamps_s.size:
            latest_timestamp_s = float(self.timestamps_s[-1])
        if latest_timestamp_s is None:
            return max(0.0, float(duration_hint_s))
        if self.rollover_timestamp_offset_s:
            return max(0.0, float(duration_hint_s))

        raw_gap_s = max(0.0, float(duration_hint_s) - float(latest_timestamp_s))
        residual_s = self._latest_timestamp_gap_residual_s_locked(raw_gap_s) or 0.0
        return float(latest_timestamp_s) + residual_s

    def _normalized_latest_timestamp_gap_s_locked(self, raw_gap_s: float | None) -> float | None:
        # Report decode lag RELATIVE to the calibration anchored at the last
        # (re)seed/catch-up, when the buffer tail provably sat at the decodable
        # frontier. The raw gap compares decoded board-clock timestamps against a
        # file-size duration estimate that falls back to the configured ODR, so a
        # board whose sensor clock deviates from nominal (IIS3DWB ODR tolerance is
        # a few percent) inflates the raw gap by skew x file age — hours into an
        # acquisition that false "gap" reaches tens of seconds. Reporting the raw
        # gap instead of the residual would dump that whole anchor into the value
        # on a small tolerance overshoot, tripping the reset threshold and the
        # data-currentness limit for data that is milliseconds old. Real decode
        # lag cannot hide in the anchor because
        # _maybe_update_gap_calibration_locked rate-limits how fast the anchor
        # may rise; sustained lag therefore surfaces here as residual growth.
        residual_s = self._latest_timestamp_gap_residual_s_locked(raw_gap_s)
        return _normalize_latest_timestamp_gap_s(residual_s, self._gap_tolerance_s_locked())

    def _latest_timestamp_gap_residual_s_locked(self, raw_gap_s: float | None) -> float | None:
        if raw_gap_s is None:
            return None
        return max(0.0, float(raw_gap_s) - float(self.latest_gap_calibration_s))

    def _maybe_update_gap_calibration_locked(
        self,
        *,
        raw_gap_s: float | None,
        freshness: dict[str, Any],
        initial: bool = False,
    ) -> None:
        if raw_gap_s is None or raw_gap_s < 0.0:
            return
        if not freshness.get("data_is_current", False):
            return
        if not self.timestamps_s.size:
            return
        if "persistent_live" not in self.last_sdk_read_strategy:
            return
        # NOTE: "_empty" incremental reads are NOT skipped here. An empty append
        # means nothing more was decodable, i.e. the tail sits at the decodable
        # frontier — exactly when absorbing the raw gap's packet-burst oscillation
        # into the calibration is correct. Skipping them let the residual trip the
        # reset threshold and forced a full reseed (~0.3s) every few reads.

        # A read that appended samples proves the decoder is keeping up with the
        # decodable frontier, so a raw-gap jump in that read is normally wobble in
        # the file-size duration estimate (which scales with file age), not decode
        # lag. Only reads with NO progress keep the threshold guard, so a
        # genuinely stuck decoder still trips the reset.
        made_progress = not (
            self.last_sdk_read_strategy.endswith("_empty")
            or self.last_sdk_read_strategy.endswith("_overlap_only")
        )
        residual_s = self._latest_timestamp_gap_residual_s_locked(raw_gap_s)
        if (
            not initial
            and not made_progress
            and residual_s is not None
            and residual_s > self._gap_reset_threshold_s_locked()
        ):
            return
        if initial:
            self._set_gap_calibration_locked(max(self.latest_gap_calibration_s, float(raw_gap_s)))
            return
        if float(raw_gap_s) <= self.latest_gap_calibration_s:
            return
        # Rate-limit how fast the anchor may rise. Unbounded absorption let the
        # anchor chase a raw gap that was growing because the decoder was
        # genuinely falling behind, hiding real lag forever. Sensor-clock skew
        # (IIS3DWB ODR tolerance is a few percent) accrues slowly, so a rise
        # budget of a few percent of wall time — with a one-packet floor for
        # packet-burst oscillation — absorbs skew while letting sustained real
        # lag accumulate as visible residual.
        max_skew_fraction = 0.05
        now_monotonic = time.monotonic()
        elapsed_s = (
            max(0.0, now_monotonic - self.gap_calibration_updated_monotonic)
            if self.gap_calibration_updated_monotonic > 0.0
            else 0.0
        )
        allowed_rise_s = max(self._packet_duration_s_locked(), max_skew_fraction * elapsed_s)
        self._set_gap_calibration_locked(
            min(float(raw_gap_s), self.latest_gap_calibration_s + allowed_rise_s)
        )

    def _set_gap_calibration_locked(self, value_s: float) -> None:
        self.latest_gap_calibration_s = max(0.0, float(value_s))
        self.gap_calibration_updated_monotonic = time.monotonic()

    def _gap_reset_threshold_s_locked(self) -> float:
        return max(0.18, self._gap_tolerance_s_locked() + self._packet_duration_s_locked())

    def _gap_tolerance_s_locked(self) -> float:
        return self._overlap_duration_s_locked() + self._packet_duration_s_locked()

    def _file_shrunk_locked(self, *, file_size: int | None) -> bool:
        return self.last_file_size is not None and file_size is not None and file_size < self.last_file_size

    def _handle_file_rollover_locked(self, *, file_size: int, file_mtime: float | None) -> None:
        # The anchor pairs a board timestamp with an absolute file sample count;
        # a rollover invalidates both sides of that pair.
        self.duration_anchor_ts_s = None
        self.duration_anchor_samples = None
        previous_latest_timestamp_s = self.latest_timestamp_s
        if previous_latest_timestamp_s is None and self.timestamps_s.size:
            previous_latest_timestamp_s = float(self.timestamps_s[-1])

        self._reload_sdk_handles_locked()
        self.next_start_time_s = 0.0
        self.last_file_size = file_size
        self.last_file_mtime = file_mtime
        self.last_refresh_monotonic = 0.0
        new_file_duration_s = self._estimate_data_duration_s_locked()
        if previous_latest_timestamp_s is not None and new_file_duration_s is not None and new_file_duration_s > 0.0:
            self.rollover_timestamp_offset_s = max(
                0.0,
                float(previous_latest_timestamp_s) - float(new_file_duration_s),
            )
        elif previous_latest_timestamp_s is not None:
            self.rollover_timestamp_offset_s = max(0.0, float(previous_latest_timestamp_s))
        else:
            self.rollover_timestamp_offset_s = 0.0
        self.rollover_events += 1
        self.last_sdk_read_strategy = "persistent_live_rollover_reopen"
        self.last_sdk_read_attempts = []
        self.last_sdk_read_start_time_s = 0.0
        self.last_sdk_read_end_time_s = _safe_float(new_file_duration_s) or 0.0

    def _should_reset_for_file_change_locked(self, *, file_size: int | None, file_mtime: float | None) -> bool:
        if self.last_file_size is None:
            return False
        if file_size is None:
            return True
        if file_size < self.last_file_size:
            return True
        if self.last_file_mtime is not None and file_mtime is not None and file_mtime < self.last_file_mtime and file_size <= self.last_file_size:
            return True
        return False


class LiveSdkVibrometerDataRequiredError(RuntimeError):
    """Raised when the MCP tool is asked to read stale HSD data as live data."""

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        super().__init__(str(metadata.get("data_current_warning") or "Live SDK vibrometer data is required."))


def allowed_hsd_root() -> Path:
    """Return the root from which an LLM-callable tool may read HSD folders."""
    explicit_root = os.environ.get("VIBRO_ALLOWED_HSD_DIR") or os.environ.get("VIBRO_ALLOWED_DATA_DIR")
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    explicit_acquisition = os.environ.get("VIBRO_HSD_ACQUISITION_DIR")
    if explicit_acquisition:
        return Path(explicit_acquisition).expanduser().resolve().parent

    project_root = Path(__file__).resolve().parents[2]
    sdk_root = project_root.parent
    examples_root = sdk_root / "stdatalog_examples"
    if _path_exists(examples_root):
        return examples_root.resolve()
    return Path.cwd().resolve()


def resolve_hsd_acquisition_folder(
    acquisition_folder: str | os.PathLike[str] | None = None,
    sensor_name: str | None = None,
) -> Path:
    """Resolve and validate an HSD acquisition folder for MCP-safe reading."""
    sensor = sensor_name or default_sensor_name()
    if acquisition_folder:
        path = Path(acquisition_folder).expanduser().resolve()
    else:
        env_folder = os.environ.get("VIBRO_HSD_ACQUISITION_DIR")
        path = (
            Path(env_folder).expanduser().resolve()
            if env_folder
            else find_latest_hsd_acquisition_folder(sensor_name=sensor)
        )

    root = allowed_hsd_root()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"acquisition_folder must be inside VIBRO_ALLOWED_HSD_DIR={root}. Got: {path}"
        ) from exc

    if not _path_exists(path):
        raise FileNotFoundError(f"HSD acquisition folder does not exist: {path}")
    if not _path_is_dir(path):
        raise ValueError(f"HSD acquisition path is not a folder: {path}")
    metadata_repair_warnings = _ensure_hsd_metadata_for_live_folder(path, sensor)
    if not _path_exists(path / "device_config.json"):
        raise FileNotFoundError(
            _metadata_error_message(
                f"Missing device_config.json in HSD acquisition folder: {path}",
                metadata_repair_warnings,
            )
        )
    if _path_exists(path / f"{sensor}.dat") and not _path_exists(path / "acquisition_info.json"):
        raise FileNotFoundError(
            _metadata_error_message(
                f"Missing acquisition_info.json in HSD acquisition folder: {path}",
                metadata_repair_warnings,
            )
        )
    return path


def find_latest_hsd_acquisition_folder(sensor_name: str | None = None) -> Path:
    """Find the newest HSD acquisition under the allowed root."""
    root = allowed_hsd_root()
    sensor = sensor_name or default_sensor_name()
    candidates: list[tuple[float, Path]] = []

    if not _path_exists(root):
        raise FileNotFoundError(f"VIBRO_ALLOWED_HSD_DIR does not exist: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in {"__pycache__", "node_modules"}
        ]
        folder = Path(dirpath)
        preferred_dat = folder / f"{sensor}.dat"
        fallback_dats = list(folder.glob("*_acc.dat"))
        newest_input_mtime: float | None = None
        if _path_exists(preferred_dat):
            newest_input_mtime = _path_mtime(preferred_dat)
        elif fallback_dats:
            newest_input_mtime = max(
                (mtime for item in fallback_dats if (mtime := _path_mtime(item)) is not None),
                default=None,
            )
        if newest_input_mtime is None:
            continue

        candidates.append((newest_input_mtime, folder))

    if not candidates:
        raise FileNotFoundError(
            f"No HSD acquisition with {sensor}.dat or *_acc.dat found under VIBRO_ALLOWED_HSD_DIR={root}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].resolve()


def _ensure_hsd_metadata_for_live_folder(folder: Path, sensor: str) -> list[str]:
    """Make a currently written TUI .dat folder readable by HSDatalog.

    The ST TUI creates the sensor .dat immediately, but it can leave
    device_config.json/acquisition_info.json absent until the acquisition is
    stopped. HSDatalog needs those JSON files to decode calibrated samples, so
    reuse the newest compatible metadata from a completed acquisition while the
    live file is still growing.
    """
    warnings: list[str] = []
    data_file = folder / f"{sensor}.dat"
    if not _path_exists(data_file):
        return warnings

    template_folder: Path | None = None
    device_config = folder / "device_config.json"
    if not _path_exists(device_config):
        template_folder = _find_hsd_metadata_template_folder(folder, sensor)
        if template_folder is not None:
            try:
                shutil.copy2(template_folder / "device_config.json", device_config)
            except OSError as exc:
                warnings.append(
                    "metadata_repair_failed: could not copy device_config.json from "
                    f"{template_folder}: {exc.__class__.__name__}: {exc}"
                )
        else:
            warnings.append("metadata_repair_skipped: no compatible device_config.json template was found")

    acquisition_info = folder / "acquisition_info.json"
    if _path_exists(acquisition_info):
        return warnings

    if template_folder is None:
        template_folder = _find_hsd_metadata_template_folder(folder, sensor)
    template_info = (
        _read_json_object(template_folder / "acquisition_info.json")
        if template_folder is not None
        else None
    )
    try:
        _write_live_acquisition_info(folder, template_info)
    except OSError as exc:
        warnings.append(
            "metadata_repair_failed: could not write acquisition_info.json: "
            f"{exc.__class__.__name__}: {exc}"
        )
    return warnings


def _metadata_error_message(message: str, warnings: list[str]) -> str:
    if not warnings:
        return message
    return f"{message}. Metadata repair diagnostics: {'; '.join(warnings)}"


def _find_hsd_metadata_template_folder(current_folder: Path, sensor: str) -> Path | None:
    root = allowed_hsd_root()
    candidates: list[tuple[int, float, Path]] = []

    if not _path_exists(root):
        return None

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in {"__pycache__", "node_modules"}
        ]
        folder = Path(dirpath)
        if folder == current_folder or "device_config.json" not in filenames:
            continue

        preferred_dat = folder / f"{sensor}.dat"
        fallback_dats = list(folder.glob("*_acc.dat"))
        preferred_mtime = _path_mtime(preferred_dat) if _path_exists(preferred_dat) else None
        fallback_mtimes = [mtime for item in fallback_dats if (mtime := _path_mtime(item)) is not None]
        device_config_mtime = _path_mtime(folder / "device_config.json")
        if preferred_mtime is not None:
            candidates.append((0, preferred_mtime, folder))
        elif fallback_mtimes:
            candidates.append((1, max(fallback_mtimes), folder))
        elif device_config_mtime is not None:
            candidates.append((2, device_config_mtime, folder))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][2]


def _write_live_acquisition_info(folder: Path, template_info: dict[str, Any] | None) -> None:
    template = template_info or {}
    start_time = _folder_start_time_utc(folder) or datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)
    end_time = max(start_time + timedelta(days=1), now + timedelta(minutes=10))

    tags = template.get("tags")
    if not isinstance(tags, list):
        tags = []

    payload = {
        "tags": tags,
        "name": str(template.get("name") or "STWIN_Acq"),
        "description": str(template.get("description") or ""),
        "uuid": str(template.get("uuid") or "00000000-0000-0000-0000-000000000000"),
        "start_time": _format_hsd_datetime(start_time),
        "end_time": _format_hsd_datetime(end_time),
        "data_ext": str(template.get("data_ext") or ".dat"),
        "data_fmt": str(template.get("data_fmt") or "HSD_2.0.0"),
        "interface": template.get("interface", 1),
        "schema_version": str(template.get("schema_version") or "2.0.0"),
        "c_type": template.get("c_type", 2),
    }
    (folder / "acquisition_info.json").write_text(
        json.dumps(payload, indent=4) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _folder_start_time_utc(folder: Path) -> datetime | None:
    try:
        return datetime.strptime(folder.name, "%Y%m%d_%H_%M_%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_hsd_datetime(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def default_sensor_name() -> str:
    """Return the configured ST Datalog accelerometer component name."""
    return os.environ.get("VIBRO_HSD_SENSOR_NAME", DEFAULT_SENSOR_NAME)


def list_sdk_vibrometer_sources(
    acquisition_folder: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """List accelerometer/vibration sources available in an HSD folder."""
    folder = resolve_hsd_acquisition_folder(acquisition_folder, sensor_name=default_sensor_name())
    hsd, hsd_instance = _create_hsd(folder)

    sensors = []
    active_sensor_names: list[str] = []
    for sensor_entry in hsd.get_sensor_list(hsd_instance, type_filter="acc"):
        name, status = _sensor_name_and_status(sensor_entry)
        if not name:
            continue
        enabled = bool(status.get("enable", False))
        if enabled:
            active_sensor_names.append(name)
        sensors.append(
            {
                "name": name,
                "enabled": enabled,
                "odr_hz": _safe_float(status.get("odr")),
                "measured_odr_hz": _safe_float(status.get("measodr")),
                "full_scale_g": _safe_float(status.get("fs")),
                "dimensions": _safe_int(status.get("dim")),
                "sensitivity": _safe_float(status.get("sensitivity")),
                "unit": status.get("unit") or "g",
                "data_file_exists": _path_exists(folder / f"{name}.dat"),
            }
        )

    selected = _select_default_sensor_name(sensors)
    return {
        "schema_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SDK_SOURCE_NAME,
        "acquisition_folder": str(folder),
        "allowed_hsd_root": str(allowed_hsd_root()),
        "default_sensor_name": selected,
        "active_sensor_names": active_sensor_names,
        "sensors": sensors,
        "note": "Vibration is read through the IIS3DWB accelerometer stream when iis3dwb_acc is available.",
    }



def _live_decoder_buffer_duration_s() -> float:
    raw = os.environ.get("VIBRO_LIVE_DECODER_BUFFER_DURATION_S")
    value = _safe_float(raw)
    if value is None:
        return DEFAULT_LIVE_DECODER_BUFFER_DURATION_S
    return max(1.0, value)


def _max_incremental_decode_span_s() -> float:
    raw = os.environ.get("VIBRO_LIVE_DECODER_MAX_DECODE_SPAN_S")
    value = _safe_float(raw)
    if value is None:
        return DEFAULT_LIVE_DECODER_MAX_DECODE_SPAN_S
    return max(5.0, value)


def _live_decoder_min_refresh_interval_s() -> float:
    raw = os.environ.get("VIBRO_LIVE_DECODER_MIN_REFRESH_INTERVAL_S")
    value = _safe_float(raw)
    if value is None:
        return DEFAULT_LIVE_DECODER_MIN_REFRESH_INTERVAL_S
    return max(0.0, value)


def _get_live_decoder(folder: Path, sensor: str) -> _PersistentLiveBoardDecoder:
    key = (str(folder.resolve()), str(sensor))
    with _LIVE_DECODER_POOL_LOCK:
        decoder = _LIVE_DECODER_POOL.get(key)
        if decoder is None:
            decoder = _PersistentLiveBoardDecoder(folder=folder.resolve(), sensor=sensor)
            _LIVE_DECODER_POOL[key] = decoder
        return decoder


def _clear_live_decoder_pool() -> None:
    with _LIVE_DECODER_POOL_LOCK:
        _LIVE_DECODER_POOL.clear()


def read_sdk_vibrometer_window(
    *,
    machine_id: str = "stwinbox_iis3dwb",
    acquisition_folder: str | os.PathLike[str] | None = None,
    sensor_name: str | None = None,
    axis: str = DEFAULT_AXIS,
    start_time_s: float | None = None,
    duration_s: float = 5.0,
    require_current: bool = True,
    max_duration_s: float = 60.0,
) -> SdkVibrometerWindow:
    """Read one calibrated vibration window from an HSD acquisition folder.

    If start_time_s is None, read the latest complete-duration window available
    in the acquisition. Pass an explicit start_time_s for repeatable fixed-window
    analysis.
    """
    start_time_s, duration_s = _normalize_window_request(
        start_time_s=start_time_s,
        duration_s=duration_s,
        max_duration_s=max_duration_s,
    )
    axis = _normalize_axis(axis)
    sensor = sensor_name or default_sensor_name()
    folder = resolve_hsd_acquisition_folder(acquisition_folder, sensor_name=sensor)
    if start_time_s is None and duration_s <= _live_decoder_buffer_duration_s():
        decoder = _get_live_decoder(folder, sensor)
        with _HSDATALOG_READ_LOCK:
            return decoder.read_window(
                machine_id=machine_id,
                axis=axis,
                duration_s=duration_s,
                require_current=require_current,
            )
    return _read_sdk_vibrometer_window_stateless(
        machine_id=machine_id,
        folder=folder,
        sensor=sensor,
        axis=axis,
        start_time_s=start_time_s,
        duration_s=duration_s,
        require_current=require_current,
    )


def _read_sdk_vibrometer_window_stateless(
    *,
    machine_id: str,
    folder: Path,
    sensor: str,
    axis: str,
    start_time_s: float | None,
    duration_s: float,
    require_current: bool,
) -> SdkVibrometerWindow:
    with _HSDATALOG_READ_LOCK:
        hsd, hsd_instance = _create_hsd(folder)
        sensor_component = hsd.get_sensor(hsd_instance, sensor)
        if not sensor_component:
            raise ValueError(f"Sensor {sensor!r} is not present in acquisition folder {folder}")

        sensor_status = _sensor_name_and_status(sensor_component)[1]
        configured_sampling_rate = _sampling_rate_from_sensor_status(sensor_status)
        chunk_size = _dataframe_chunk_size(sensor_status, configured_sampling_rate, duration_s)
        acquisition_duration_s = _acquisition_duration_s(hsd_instance)
        data_duration_s = _estimate_data_duration_s(
            folder=folder,
            sensor=sensor,
            sensor_status=sensor_status,
            sampling_rate_hz=configured_sampling_rate,
            hsd_instance=hsd_instance,
        )
        duration_hint_s = data_duration_s or acquisition_duration_s
        data_file = folder / f"{sensor}.dat"
        freshness = _data_file_freshness_metadata(data_file)

        read_latest = start_time_s is None
        if read_latest:
            read_start_time_s = max(0.0, (duration_hint_s or duration_s) - duration_s - LATEST_WINDOW_READ_MARGIN_S)
            read_end_time_s = read_start_time_s + duration_s + LATEST_WINDOW_READ_MARGIN_S
        else:
            read_start_time_s = float(start_time_s)
            end_time_s = read_start_time_s + duration_s
            read_end_time_s = end_time_s + 1.0 if read_start_time_s == 0.0 else end_time_s

        stale_metadata = {
            "machine_id": machine_id,
            "acquisition_folder": str(folder),
            "sensor_name": sensor,
            "axis": axis,
            "unit": "g",
            "read_mode": "latest" if read_latest else "fixed_start_time",
            "requested_start_time_s": _safe_float(start_time_s),
            "requested_duration_s": float(duration_s),
            "acquisition_duration_s": _safe_float(acquisition_duration_s),
            "estimated_data_duration_s": _safe_float(data_duration_s),
            "configured_odr_hz": _safe_float(sensor_status.get("odr")),
            "measured_odr_hz": _safe_float(sensor_status.get("measodr")),
            "sampling_rate_hz": int(configured_sampling_rate),
            "data_file": str(data_file),
            **freshness,
        }
        if require_current and not freshness.get("data_is_current", False):
            raise LiveSdkVibrometerDataRequiredError(stale_metadata)

        if read_latest:
            sdk_read = _read_latest_sdk_arrays(
                hsd=hsd,
                hsd_instance=hsd_instance,
                sensor_component=sensor_component,
                duration_s=duration_s,
                duration_hint_s=duration_hint_s,
                chunk_size=chunk_size,
                minimum_sample_count=_minimum_latest_sample_count(configured_sampling_rate, duration_s),
            )
        else:
            frames = _read_sdk_dataframe_frames(
                hsd=hsd,
                hsd_instance=hsd_instance,
                sensor_component=sensor_component,
                start_time_s=read_start_time_s,
                end_time_s=read_end_time_s,
                chunk_size=chunk_size,
            )
            samples_2d, timestamps_s, axis_columns = _frames_to_arrays(frames)
            sdk_read = _SdkReadResult(
                samples_2d=samples_2d,
                timestamps_s=timestamps_s,
                axis_columns=axis_columns,
                read_start_time_s=read_start_time_s,
                read_end_time_s=read_end_time_s,
                sdk_read_strategy="fixed_start_time_bounded",
                latest_timestamp_gap_s=None,
                sdk_read_attempts=[],
            )
    samples_2d = sdk_read.samples_2d
    timestamps_s = sdk_read.timestamps_s
    axis_columns = sdk_read.axis_columns
    read_start_time_s = sdk_read.read_start_time_s
    read_end_time_s = sdk_read.read_end_time_s
    sdk_timestamp_start_s = _safe_float(timestamps_s[0]) if timestamps_s is not None and timestamps_s.size else None
    sdk_timestamp_end_s = _safe_float(timestamps_s[-1]) if timestamps_s is not None and timestamps_s.size else None
    if read_latest:
        trimmed = _trim_latest_window(
            samples_2d,
            timestamps_s,
            sampling_rate_hz=configured_sampling_rate,
            duration_s=duration_s,
        )
        samples_2d = trimmed.samples_2d
        timestamps_s = trimmed.timestamps_s
        trim_start_time_s = trimmed.start_time_s
        trim_end_time_s = trimmed.end_time_s
        timestamp_mode = trimmed.timestamp_mode
        timestamp_warning = trimmed.timestamp_warning
    else:
        trim_start_time_s = read_start_time_s
        trim_end_time_s = read_start_time_s + duration_s
        timestamp_warning = _timestamp_sequence_warning(timestamps_s)
        if timestamps_s is None or timestamps_s.size != samples_2d.shape[0] or timestamp_warning:
            trimmed = _trim_fixed_window_by_sample_count(
                samples_2d,
                start_time_s=trim_start_time_s,
                sampling_rate_hz=configured_sampling_rate,
                duration_s=duration_s,
                reason=timestamp_warning,
            )
            samples_2d = trimmed.samples_2d
            timestamps_s = trimmed.timestamps_s
            trim_start_time_s = trimmed.start_time_s
            trim_end_time_s = trimmed.end_time_s
            timestamp_mode = trimmed.timestamp_mode
            timestamp_warning = trimmed.timestamp_warning
        else:
            if read_start_time_s == 0.0 and timestamps_s.size:
                # Some HSD acquisitions start their first timestamp slightly after 0.0s.
                # For explicit zero-start reads, keep a full-duration window from the first available sample.
                first_timestamp_s = float(timestamps_s[0])
                if first_timestamp_s > read_start_time_s:
                    trim_start_time_s = first_timestamp_s
                    trim_end_time_s = trim_start_time_s + duration_s
            samples_2d, timestamps_s = _trim_window_by_timestamps(
                samples_2d,
                timestamps_s,
                start_time_s=trim_start_time_s,
                end_time_s=trim_end_time_s,
            )
            timestamp_mode = "sdk_timestamp_window"
            timestamp_warning = None
    signal = _select_axis(samples_2d, axis=axis)
    # HSDatalog_v2 rounds dataframe timestamps to 6 decimals before returning them.
    # Estimating ODR from adjacent rounded timestamps biases high-rate streams, so
    # prefer the measured/configured sensor ODR and keep timestamps only for window trimming.
    sampling_rate_hz = configured_sampling_rate

    if signal.size < 32:
        raise ValueError(
            f"SDK vibrometer window has only {signal.size} samples. Increase duration_s or adjust start_time_s."
        )

    raw_latest_timestamp_gap_s = sdk_read.latest_timestamp_gap_s
    latest_timestamp_gap_s = (
        _normalize_latest_timestamp_gap_s(
            raw_latest_timestamp_gap_s,
            _latest_timestamp_gap_tolerance_s(
                sensor_status=sensor_status,
                sampling_rate_hz=configured_sampling_rate,
            ),
        )
        if read_latest
        else raw_latest_timestamp_gap_s
    )
    metadata = {
        "machine_id": machine_id,
        "acquisition_folder": str(folder),
        "sensor_name": sensor,
        "axis": axis,
        "axis_columns": axis_columns,
        "unit": "g",
        "sample_count": int(signal.size),
        "read_mode": "latest" if read_latest else "fixed_start_time",
        "requested_start_time_s": _safe_float(start_time_s),
        "start_time_s": float(trim_start_time_s),
        "requested_duration_s": float(duration_s),
        "acquisition_duration_s": _safe_float(acquisition_duration_s),
        "estimated_data_duration_s": _safe_float(data_duration_s),
        "sdk_read_start_time_s": _safe_float(read_start_time_s),
        "sdk_read_end_time_s": _safe_float(read_end_time_s),
        "sdk_read_strategy": sdk_read.sdk_read_strategy,
        "sdk_latest_timestamp_raw_gap_s": _safe_float(raw_latest_timestamp_gap_s),
        "sdk_latest_timestamp_gap_s": _safe_float(latest_timestamp_gap_s),
        "sdk_read_attempts": sdk_read.sdk_read_attempts,
        "sdk_timestamp_start_s": sdk_timestamp_start_s,
        "sdk_timestamp_end_s": sdk_timestamp_end_s,
        "timestamp_start_s": _safe_float(timestamps_s[0]) if timestamps_s is not None and timestamps_s.size else None,
        "timestamp_end_s": _safe_float(timestamps_s[-1]) if timestamps_s is not None and timestamps_s.size else None,
        "timestamp_mode": timestamp_mode,
        "timestamp_warning": timestamp_warning,
        "configured_odr_hz": _safe_float(sensor_status.get("odr")),
        "measured_odr_hz": _safe_float(sensor_status.get("measodr")),
        "sampling_rate_hz": int(sampling_rate_hz),
        "full_scale_g": _safe_float(sensor_status.get("fs")),
        "dimensions": _safe_int(sensor_status.get("dim")),
        "sensitivity": _safe_float(sensor_status.get("sensitivity")),
        "data_file": str(data_file),
        **freshness,
    }
    _apply_decoded_currentness_metadata(metadata)
    if require_current and not metadata.get("data_is_current", False):
        raise LiveSdkVibrometerDataRequiredError(metadata)
    return SdkVibrometerWindow(
        signal=signal,
        timestamps_s=timestamps_s,
        sampling_rate_hz=int(sampling_rate_hz),
        metadata=metadata,
    )


def read_sdk_vibrometer_window_payload(
    *,
    machine_id: str = "stwinbox_iis3dwb",
    acquisition_folder: str | os.PathLike[str] | None = None,
    sensor_name: str | None = None,
    axis: str = DEFAULT_AXIS,
    start_time_s: float | None = None,
    duration_s: float = 5.0,
    include_samples: bool = False,
    max_preview_samples: int = 16,
    max_return_samples: int = DEFAULT_MAX_RETURN_SAMPLES,
    require_current: bool = True,
) -> dict[str, Any]:
    """Read one SDK window and return compact JSON-safe metadata/statistics."""
    try:
        window = read_sdk_vibrometer_window(
            machine_id=machine_id,
            acquisition_folder=acquisition_folder,
            sensor_name=sensor_name,
            axis=axis,
            start_time_s=start_time_s,
            duration_s=duration_s,
            require_current=require_current,
        )
    except LiveSdkVibrometerDataRequiredError as exc:
        return _live_data_required_payload(machine_id=machine_id, metadata=exc.metadata)
    except Exception as exc:
        return _sdk_read_failed_payload(
            machine_id=machine_id,
            error="sdk_read_failed",
            message=f"{exc.__class__.__name__}: {exc}",
            request={
                "acquisition_folder": str(acquisition_folder) if acquisition_folder is not None else None,
                "sensor_name": sensor_name,
                "axis": axis,
                "start_time_s": _safe_float(start_time_s),
                "duration_s": _safe_float(duration_s),
                "require_current": bool(require_current),
            },
        )

    signal = np.asarray(window.signal, dtype=np.float64)
    preview_limit = _nonnegative_int(max_preview_samples, default=16)
    return_limit = _nonnegative_int(max_return_samples, default=DEFAULT_MAX_RETURN_SAMPLES)
    payload: dict[str, Any] = {
        "schema_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SDK_SOURCE_NAME,
        "machine_id": machine_id,
        "metadata": window.metadata,
        "sample_count": int(signal.size),
        "sampling_rate_hz": window.sampling_rate_hz,
        "window_duration_s": float(signal.size / float(window.sampling_rate_hz)),
        "preview_samples": _signal_preview(signal, preview_limit),
        "samples_truncated": False,
        "samples_in_payload": 0,
        "preview_note": "Preview only. The full window was read and used for statistics; request include_samples only when raw samples are required.",
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
    }
    if include_samples:
        returned_signal = signal[:return_limit]
        payload["samples"] = [float(value) for value in returned_signal]
        payload["samples_truncated"] = bool(signal.size > returned_signal.size)
        payload["samples_in_payload"] = int(returned_signal.size)
        payload["returned_sample_count"] = int(returned_signal.size)
    return payload


def _live_data_required_payload(*, machine_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    warning = str(metadata.get("data_current_warning") or "Live SDK vibrometer data is required.")
    return {
        "schema_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SDK_SOURCE_NAME,
        "machine_id": machine_id,
        "status": "error",
        "error": "live_data_required",
        "message": warning,
        "metadata": metadata,
        "sample_count": 0,
        "samples_truncated": False,
        "samples_in_payload": 0,
        "preview_samples": [],
        "preview_note": "No samples returned because the configured HSD data is stale; live/current data is required.",
        "recommended_action": (
            "Start a new HSD acquisition or point VIBRO_HSD_ACQUISITION_DIR at a folder whose "
            "IIS3DWB .dat file is actively updating, then retry the read/analyze request."
        ),
    }


def _sdk_read_failed_payload(
    *,
    machine_id: str,
    error: str,
    message: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SDK_SOURCE_NAME,
        "machine_id": machine_id,
        "status": "error",
        "error": error,
        "message": message,
        "request": request,
        "sample_count": 0,
        "samples_truncated": False,
        "samples_in_payload": 0,
        "preview_samples": [],
        "preview_note": "No samples returned because the SDK board read failed before a valid window was decoded.",
        "recommended_action": (
            "Check that the board acquisition folder is inside VIBRO_ALLOWED_HSD_DIR, contains the requested "
            "accelerometer stream, and is readable by STDATALOG-PYSDK."
        ),
    }


def _create_hsd(folder: Path):
    with _HSDATALOG_READ_LOCK:
        try:
            from stdatalog_core.HSD.HSDatalog import HSDatalog
        except ImportError as exc:
            raise RuntimeError(
                "STDATALOG-PYSDK is not importable. Activate the SDK virtual environment "
                "or install stdatalog_core before using SDK-backed vibrometer tools."
            ) from exc

        hsd = HSDatalog()
        try:
            from stdatalog_core.HSD.HSDatalog_v2 import HSDatalog_v2

            hsd_instance = HSDatalog_v2(acquisition_folder=str(folder), update_catalog=False)
        except Exception:
            hsd_instance = hsd.create_hsd(acquisition_folder=str(folder), update_catalog=False)

        if hsd_instance is None:
            raise ValueError(f"Unable to create an HSDatalog instance for acquisition folder: {folder}")
        return hsd, hsd_instance


def _is_sdk_data_corruption_error(exc: BaseException | None) -> bool:
    pending: list[BaseException] = [exc] if exc is not None else []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        name = current.__class__.__name__
        message = str(current)
        if name == "DataCorruptedException":
            return True
        lowered = message.lower()
        if "missing packets revealed" in lowered or "data corrupted" in lowered:
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _read_latest_sdk_arrays(
    *,
    hsd: Any,
    hsd_instance: Any,
    sensor_component: Any,
    duration_s: float,
    duration_hint_s: float | None,
    chunk_size: int,
    minimum_sample_count: int,
    copy_component: bool = True,
) -> _SdkReadResult:
    attempts: list[dict[str, Any]] = []
    best: _SdkReadResult | None = None

    def _try_window(read_start_time_s: float, read_end_time_s: float, strategy: str) -> _SdkReadResult | None:
        nonlocal best
        try:
            frames = _read_sdk_dataframe_frames(
                hsd=hsd,
                hsd_instance=hsd_instance,
                sensor_component=sensor_component,
                start_time_s=read_start_time_s,
                end_time_s=read_end_time_s,
                chunk_size=chunk_size,
                copy_component=copy_component,
            )
            samples_2d, timestamps_s, axis_columns = _frames_to_arrays(frames)
            samples_2d, timestamps_s = _filter_invalid_timestamp_rows(
                samples_2d,
                timestamps_s,
                max_time_s=read_end_time_s + MAX_TIMESTAMP_OVERRUN_S,
            )
            if samples_2d.shape[0] < 32:
                raise ValueError(f"SDK latest-window read returned only {samples_2d.shape[0]} usable samples")

            latest_timestamp_s = _latest_timestamp_or_none(timestamps_s)
            latest_timestamp_gap_s = (
                max(0.0, float(duration_hint_s) - latest_timestamp_s)
                if duration_hint_s is not None and latest_timestamp_s is not None
                else None
            )
            result = _SdkReadResult(
                samples_2d=samples_2d,
                timestamps_s=timestamps_s,
                axis_columns=axis_columns,
                read_start_time_s=read_start_time_s,
                read_end_time_s=read_end_time_s,
                sdk_read_strategy=strategy,
                latest_timestamp_gap_s=latest_timestamp_gap_s,
                sdk_read_attempts=attempts.copy(),
            )
            if best is None or _is_better_latest_candidate(result, best, minimum_sample_count):
                best = result
            return result
        except Exception as exc:
            attempts.append(
                {
                    "start_time_s": _safe_float(read_start_time_s),
                    "end_time_s": _safe_float(read_end_time_s),
                    "strategy": strategy,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            return None

    for read_start_time_s, read_end_time_s, strategy in _estimated_latest_windows(duration_hint_s, duration_s):
        result = _try_window(read_start_time_s, read_end_time_s, strategy)
        if result is not None and result.samples_2d.shape[0] >= minimum_sample_count:
            return result

    # Every estimate-anchored window missed. The cold file-size estimate divides
    # bytes by the configured ODR, and the board's true rate is off by ~1-2%, so
    # the overshoot grows with acquisition age (~150 s at 2 h — beyond every
    # backoff). Seeding from the file head instead would hand catch-up an
    # hours-long hole (the 2026-07-06 OOM cascade) or serve stale data, so
    # bisect the true decodable frontier with cheap 2 s probes and seed there.
    if duration_hint_s is not None and duration_hint_s > 0.0:
        frontier_s = _probe_decodable_frontier(
            hsd=hsd,
            hsd_instance=hsd_instance,
            sensor_component=sensor_component,
            hint_s=float(duration_hint_s),
            chunk_size=chunk_size,
            copy_component=copy_component,
            attempts=attempts,
        )
        if frontier_s is not None:
            probe_start_s = max(0.0, frontier_s - duration_s - LATEST_WINDOW_READ_MARGIN_S)
            probe_end_s = frontier_s + _FRONTIER_PROBE_RESOLUTION_S + LATEST_WINDOW_READ_MARGIN_S
            result = _try_window(probe_start_s, probe_end_s, "frontier_probe_bounded")
            if result is not None:
                return result

    for read_start_time_s, read_end_time_s, strategy in _zero_start_windows(duration_hint_s, duration_s):
        _try_window(read_start_time_s, read_end_time_s, strategy)

    if best is not None:
        return _SdkReadResult(
            samples_2d=best.samples_2d,
            timestamps_s=best.timestamps_s,
            axis_columns=best.axis_columns,
            read_start_time_s=best.read_start_time_s,
            read_end_time_s=best.read_end_time_s,
            sdk_read_strategy=best.sdk_read_strategy,
            latest_timestamp_gap_s=best.latest_timestamp_gap_s,
            sdk_read_attempts=attempts,
        )

    summary = "; ".join(
        f"{item['strategy']}[{item['start_time_s']}, {item['end_time_s']}]: {item['error']}"
        for item in attempts[-4:]
    )
    raise ValueError(
        "STDATALOG-PYSDK could not decode a latest vibrometer window from any bounded SDK read"
        + (f". Attempts: {summary}" if summary else "")
    )


def _is_better_latest_candidate(
    candidate: _SdkReadResult,
    current: _SdkReadResult,
    minimum_sample_count: int,
) -> bool:
    candidate_enough = candidate.samples_2d.shape[0] >= minimum_sample_count
    current_enough = current.samples_2d.shape[0] >= minimum_sample_count
    if candidate_enough != current_enough:
        return candidate_enough

    candidate_ts = _latest_result_timestamp(candidate)
    current_ts = _latest_result_timestamp(current)
    if candidate_ts != current_ts:
        return candidate_ts > current_ts
    return candidate.samples_2d.shape[0] > current.samples_2d.shape[0]


def _latest_result_timestamp(result: _SdkReadResult) -> float:
    latest = _latest_timestamp_or_none(result.timestamps_s)
    return latest if latest is not None else float("-inf")


def _latest_timestamp_or_none(timestamps_s: np.ndarray | None) -> float | None:
    if timestamps_s is None or timestamps_s.size == 0:
        return None
    return float(timestamps_s[-1])


def _dedupe_windows(windows: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    deduped: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float, str]] = set()
    for start, end, strategy in windows:
        key = (round(start, 6), round(end, 6), strategy)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start, end, strategy))
    return deduped


def _estimated_latest_windows(duration_hint_s: float | None, duration_s: float) -> list[tuple[float, float, str]]:
    hint = max(0.0, float(duration_hint_s)) if duration_hint_s is not None else None
    latest_start = max(0.0, (hint or duration_s) - duration_s - LATEST_WINDOW_READ_MARGIN_S)
    read_span_s = duration_s + LATEST_WINDOW_READ_MARGIN_S
    windows = [
        (max(0.0, latest_start - backoff_s), max(0.0, latest_start - backoff_s) + read_span_s, "estimated_latest_bounded")
        for backoff_s in LATEST_WINDOW_BACKOFFS_S
    ]
    return _dedupe_windows(windows)


def _zero_start_windows(duration_hint_s: float | None, duration_s: float) -> list[tuple[float, float, str]]:
    hint = max(0.0, float(duration_hint_s)) if duration_hint_s is not None else None
    minimum_zero_start_end = duration_s + LATEST_WINDOW_READ_MARGIN_S
    zero_start_candidates = [
        minimum_zero_start_end,
        2.0,
        5.0,
        10.0,
        30.0,
        60.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
        min(max(minimum_zero_start_end, (hint or duration_s) + LATEST_WINDOW_READ_MARGIN_S), LATEST_ZERO_START_MAX_END_S),
    ]
    windows = [
        (0.0, end, "zero_start_bounded_scan") for end in zero_start_candidates if end >= minimum_zero_start_end
    ]
    return _dedupe_windows(windows)


def _probe_decodable_frontier(
    *,
    hsd: Any,
    hsd_instance: Any,
    sensor_component: Any,
    hint_s: float,
    chunk_size: int,
    copy_component: bool,
    attempts: list[dict[str, Any]],
) -> float | None:
    """Bisect the newest decodable timestamp using cheap fixed-span probes.

    Returns a time known to hold decodable data within
    _FRONTIER_PROBE_RESOLUTION_S of the true tail, or None when even the file
    head has nothing (empty/rolled-over acquisition — the zero-start scan is
    the right fallback there).
    """

    def _has_data(start_s: float) -> bool:
        try:
            frames = _read_sdk_dataframe_frames(
                hsd=hsd,
                hsd_instance=hsd_instance,
                sensor_component=sensor_component,
                start_time_s=start_s,
                end_time_s=start_s + _FRONTIER_PROBE_SPAN_S,
                chunk_size=chunk_size,
                copy_component=copy_component,
            )
            samples_2d, _, _ = _frames_to_arrays(frames)
            return samples_2d.shape[0] > 0
        except Exception:
            return False

    if _has_data(hint_s):
        return hint_s
    low_s = 0.0
    if not _has_data(low_s):
        return None
    high_s = max(hint_s, low_s + _FRONTIER_PROBE_SPAN_S)
    probes = 2
    while high_s - low_s > _FRONTIER_PROBE_RESOLUTION_S and probes < 24:
        mid_s = (low_s + high_s) / 2.0
        probes += 1
        if _has_data(mid_s):
            low_s = mid_s
        else:
            high_s = mid_s
    attempts.append(
        {
            "start_time_s": _safe_float(low_s),
            "end_time_s": _safe_float(high_s),
            "strategy": "frontier_probe",
            "error": f"bisected decodable frontier in {probes} probes",
        }
    )
    return low_s


def _fresh_sdk_component_for_read(sensor_component: Any) -> Any:
    """Return a clean component copy for one bounded STDATALOG read."""
    component = copy.deepcopy(sensor_component)
    if isinstance(component, dict):
        for status in component.values():
            if isinstance(status, dict):
                for key in _SDK_MUTABLE_STATUS_KEYS:
                    status.pop(key, None)
    return component


def _read_sdk_dataframe_frames(
    *,
    hsd: Any,
    hsd_instance: Any,
    sensor_component: Any,
    start_time_s: float,
    end_time_s: float,
    chunk_size: int,
    copy_component: bool = True,
) -> list[Any]:
    try:
        component = _fresh_sdk_component_for_read(sensor_component) if copy_component else sensor_component
        with _HSDATALOG_READ_LOCK:
            frames = hsd.get_dataframe(
                hsd_instance,
                component,
                start_time=start_time_s,
                end_time=end_time_s,
                raw_data=False,
                chunk_size=chunk_size,
            )
    except TypeError as exc:
        raise ValueError(
            "STDATALOG-PYSDK returned no dataframe for the requested bounded window "
            f"start_time={start_time_s:g}, end_time={end_time_s:g}"
        ) from exc
    if frames is None:
        raise ValueError(
            "STDATALOG-PYSDK returned no dataframe for the requested bounded window "
            f"start_time={start_time_s:g}, end_time={end_time_s:g}"
        )
    return frames


def _is_sdk_no_dataframe_error(exc: BaseException) -> bool:
    return "STDATALOG-PYSDK returned no dataframe for the requested bounded window" in str(exc)


def _frames_to_arrays(frames: list[Any]) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    arrays: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    axis_columns: list[str] | None = None
    axis_column_keys: list[str] | None = None
    timestamps_complete = True

    for frame in frames or []:
        if frame is None:
            continue
        try:
            if len(frame) == 0:
                continue
            columns = list(frame.columns)
        except Exception:
            continue
        time_column = _sdk_time_column(columns)
        value_columns = [column for column in columns if column != time_column]
        if not value_columns:
            continue
        value_column_keys = [_normalize_sdk_column_name(column) for column in value_columns]
        if len(set(value_column_keys)) != len(value_column_keys):
            continue

        if axis_column_keys is None:
            axis_column_keys = value_column_keys
            axis_columns = [str(column) for column in value_columns]
        else:
            if len(value_column_keys) != len(axis_column_keys) or set(value_column_keys) != set(axis_column_keys):
                continue
            column_by_key = dict(zip(value_column_keys, value_columns))
            value_columns = [column_by_key[key] for key in axis_column_keys]

        try:
            values = frame[value_columns].to_numpy(dtype=np.float64)
        except Exception:
            continue
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] == 0:
            continue

        row_mask = np.all(np.isfinite(values), axis=1)
        frame_timestamps = None
        if time_column is not None:
            try:
                frame_timestamps = frame[time_column].to_numpy(dtype=np.float64).reshape(-1)
            except Exception:
                timestamps_complete = False
            else:
                if frame_timestamps.shape[0] == values.shape[0]:
                    row_mask &= np.isfinite(frame_timestamps)
                    row_mask &= frame_timestamps >= 0.0
                else:
                    timestamps_complete = False
                    frame_timestamps = None
        else:
            timestamps_complete = False

        if not np.any(row_mask):
            continue

        values = values[row_mask]
        arrays.append(values)

        if frame_timestamps is not None:
            timestamps.append(frame_timestamps[row_mask])

    if not arrays:
        raise ValueError("No accelerometer samples were returned by STDATALOG-PYSDK for the requested window")

    samples = np.vstack(arrays)
    ts = None
    if timestamps_complete and len(timestamps) == len(arrays):
        ts = np.concatenate(timestamps)
    return samples, ts, axis_columns or []


def _sdk_time_column(columns: list[Any]) -> Any | None:
    for column in columns:
        normalized = _normalize_sdk_column_name(column)
        if normalized in {"t", "time", "times", "timestamp", "timestamps"}:
            return column
    return None


def _normalize_sdk_column_name(column: Any) -> str:
    return "".join(ch for ch in str(column).strip().lower() if ch.isalnum())


def _reorder_axis_columns(
    *,
    samples_2d: np.ndarray,
    axis_columns: list[str],
    expected_columns: list[str],
) -> np.ndarray:
    if not axis_columns or not expected_columns or axis_columns == expected_columns:
        return samples_2d

    axis_by_key = {_normalize_sdk_column_name(column): index for index, column in enumerate(axis_columns)}
    expected_keys = [_normalize_sdk_column_name(column) for column in expected_columns]
    if set(axis_by_key) != set(expected_keys):
        raise ValueError("SDK axis columns changed for a persistent live decoder")
    order = [axis_by_key[key] for key in expected_keys]
    return samples_2d[:, order]


def _select_axis(samples_2d: np.ndarray, axis: str) -> np.ndarray:
    normalized = _normalize_axis(axis)
    if normalized in {"norm", "magnitude", "vector_norm"}:
        return np.linalg.norm(samples_2d, axis=1).astype(np.float64)

    axis_to_index = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
    if normalized not in axis_to_index:
        raise ValueError("axis must be one of: norm, x, y, z, 0, 1, 2")
    index = axis_to_index[normalized]
    if index >= samples_2d.shape[1]:
        raise ValueError(f"axis {axis!r} is not available; sensor returned {samples_2d.shape[1]} channel(s)")
    return samples_2d[:, index].astype(np.float64)


def _sampling_rate_from_sensor_status(sensor_status: dict[str, Any]) -> int:
    for key in ("measodr", "odr"):
        value = _safe_float(sensor_status.get(key))
        if value and value > 0:
            sampling_rate_hz = int(round(value))
            if sampling_rate_hz > 0:
                return sampling_rate_hz
    raise ValueError("Sensor metadata does not contain a positive measured_odr_hz or odr_hz")


def _sampling_rate_from_timestamps(timestamps_s: np.ndarray | None) -> int | None:
    if timestamps_s is None or timestamps_s.size < 2:
        return None
    deltas = np.diff(timestamps_s.astype(np.float64))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return None
    median_delta = float(np.median(positive))
    if median_delta <= 0:
        return None
    return int(round(1.0 / median_delta))


def _latest_timestamp_gap_tolerance_s(
    *,
    sensor_status: dict[str, Any],
    sampling_rate_hz: int,
) -> float:
    samples_per_ts = _safe_int(sensor_status.get("samples_per_ts")) or 0
    packet_duration_s = (
        float(samples_per_ts) / float(sampling_rate_hz)
        if samples_per_ts > 0 and sampling_rate_hz > 0
        else 0.0
    )
    return max(0.1, packet_duration_s * 2.0) + packet_duration_s


def _normalize_latest_timestamp_gap_s(raw_gap_s: float | None, tolerance_s: float) -> float | None:
    if raw_gap_s is None:
        return None
    return max(0.0, float(raw_gap_s) - max(0.0, float(tolerance_s)))


def _minimum_latest_sample_count(sampling_rate_hz: int, duration_s: float) -> int:
    return max(32, int(round(float(sampling_rate_hz) * float(duration_s))))


def _trim_latest_window(
    samples_2d: np.ndarray,
    timestamps_s: np.ndarray | None,
    *,
    sampling_rate_hz: int,
    duration_s: float,
) -> _TrimmedWindow:
    expected_samples = _minimum_latest_sample_count(sampling_rate_hz, duration_s)
    usable_samples = min(expected_samples, int(samples_2d.shape[0]))
    min_timestamp_samples = max(32, int(round(usable_samples * 0.9))) if usable_samples else 32

    if timestamps_s is not None and timestamps_s.size == samples_2d.shape[0] and timestamps_s.size > 0:
        timestamp_warning = _timestamp_sequence_warning(timestamps_s)
        if timestamp_warning:
            return _trim_latest_window_by_sample_count(
                samples_2d,
                None,
                sampling_rate_hz=sampling_rate_hz,
                duration_s=duration_s,
                reason=timestamp_warning,
            )

        sample_period_s = 1.0 / float(sampling_rate_hz)
        latest_timestamp_s = float(timestamps_s[-1])
        trim_end_time_s = latest_timestamp_s + sample_period_s
        trim_start_time_s = max(float(timestamps_s[0]), trim_end_time_s - duration_s)
        trimmed_samples, trimmed_timestamps = _trim_window_by_timestamps(
            samples_2d,
            timestamps_s,
            start_time_s=trim_start_time_s,
            end_time_s=trim_end_time_s,
        )
        if trimmed_samples.shape[0] >= min_timestamp_samples:
            return _TrimmedWindow(
                samples_2d=trimmed_samples,
                timestamps_s=trimmed_timestamps,
                start_time_s=trim_start_time_s,
                end_time_s=trim_end_time_s,
                timestamp_mode="sdk_timestamp_window",
                timestamp_warning=None,
            )

        return _trim_latest_window_by_sample_count(
            samples_2d,
            timestamps_s,
            sampling_rate_hz=sampling_rate_hz,
            duration_s=duration_s,
            reason=(
                "SDK timestamps were discontinuous for the requested latest window; "
                "the returned times were reconstructed from the sample rate while keeping SDK-decoded samples."
            ),
        )

    return _trim_latest_window_by_sample_count(
        samples_2d,
        timestamps_s,
        sampling_rate_hz=sampling_rate_hz,
        duration_s=duration_s,
        reason=(
            "SDK did not return usable sample timestamps for the latest window; "
            "the returned times were reconstructed from the sample rate while keeping SDK-decoded samples."
        ),
    )


def _trim_latest_window_by_sample_count(
    samples_2d: np.ndarray,
    timestamps_s: np.ndarray | None,
    *,
    sampling_rate_hz: int,
    duration_s: float,
    reason: str,
) -> _TrimmedWindow:
    expected_samples = _minimum_latest_sample_count(sampling_rate_hz, duration_s)
    sample_count = min(expected_samples, int(samples_2d.shape[0]))
    if sample_count <= 0:
        return _TrimmedWindow(
            samples_2d=samples_2d[:0],
            timestamps_s=None,
            start_time_s=0.0,
            end_time_s=0.0,
            timestamp_mode="sample_count_tail_from_sdk",
            timestamp_warning=reason,
        )

    trimmed_samples = samples_2d[-sample_count:]
    sample_period_s = 1.0 / float(sampling_rate_hz)
    if timestamps_s is not None and timestamps_s.size:
        latest_timestamp_s = float(timestamps_s[-1])
        trim_end_time_s = latest_timestamp_s + sample_period_s
    else:
        trim_end_time_s = float(sample_count) * sample_period_s
    trim_start_time_s = trim_end_time_s - (float(sample_count) * sample_period_s)
    reconstructed_timestamps = trim_start_time_s + (
        np.arange(sample_count, dtype=np.float64) * sample_period_s
    )

    return _TrimmedWindow(
        samples_2d=trimmed_samples,
        timestamps_s=reconstructed_timestamps,
        start_time_s=float(trim_start_time_s),
        end_time_s=float(trim_end_time_s),
        timestamp_mode="sample_count_tail_from_sdk",
        timestamp_warning=reason,
    )


def _trim_fixed_window_by_sample_count(
    samples_2d: np.ndarray,
    *,
    start_time_s: float,
    sampling_rate_hz: int,
    duration_s: float,
    reason: str | None = None,
) -> _TrimmedWindow:
    expected_samples = _minimum_latest_sample_count(sampling_rate_hz, duration_s)
    sample_count = min(expected_samples, int(samples_2d.shape[0]))
    trimmed_samples = samples_2d[:sample_count]
    sample_period_s = 1.0 / float(sampling_rate_hz)
    reconstructed_timestamps = start_time_s + (
        np.arange(sample_count, dtype=np.float64) * sample_period_s
    )
    end_time_s = start_time_s + (float(sample_count) * sample_period_s)
    return _TrimmedWindow(
        samples_2d=trimmed_samples,
        timestamps_s=reconstructed_timestamps,
        start_time_s=float(start_time_s),
        end_time_s=float(end_time_s),
        timestamp_mode="sample_count_fixed_window_from_sdk",
        timestamp_warning=reason
        or (
            "SDK did not return usable sample timestamps for the fixed window; "
            "the returned times were reconstructed from the requested start time and sample rate."
        ),
    )


def _timestamp_sequence_warning(timestamps_s: np.ndarray | None) -> str | None:
    if timestamps_s is None or timestamps_s.size < 2:
        return None
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    deltas = np.diff(timestamps)
    if not np.all(np.isfinite(deltas)):
        return (
            "SDK timestamps contained non-finite deltas; the returned times were reconstructed "
            "from the sample rate while keeping SDK-decoded samples."
        )
    if np.any(deltas <= 0.0):
        return (
            "SDK timestamps were repeated or non-monotonic; the returned times were reconstructed "
            "from the sample rate while keeping SDK-decoded samples."
        )
    return None


def _trim_window_by_timestamps(
    samples_2d: np.ndarray,
    timestamps_s: np.ndarray | None,
    *,
    start_time_s: float,
    end_time_s: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Trim SDK frame output to the requested sample-time interval.

    STDATALOG-PYSDK returns complete timestamp frames, so the final frame can
    extend past the requested end time. Keeping only timestamps in
    [start_time_s, end_time_s) makes MCP duration/sample counts match the tool
    request while preserving the SDK calibration.
    """
    if timestamps_s is None or timestamps_s.size != samples_2d.shape[0]:
        return samples_2d, timestamps_s

    mask = (timestamps_s >= start_time_s) & (timestamps_s < end_time_s)
    if not np.any(mask):
        return samples_2d[:0], timestamps_s[:0]
    return samples_2d[mask], timestamps_s[mask]


def _filter_invalid_timestamp_rows(
    samples_2d: np.ndarray,
    timestamps_s: np.ndarray | None,
    *,
    max_time_s: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if timestamps_s is None or timestamps_s.size != samples_2d.shape[0]:
        return samples_2d, timestamps_s

    timestamps = timestamps_s.astype(np.float64)
    mask = np.isfinite(timestamps) & (timestamps >= 0.0)
    if max_time_s is not None:
        mask &= timestamps <= max(0.0, float(max_time_s))

    if not np.any(mask):
        return samples_2d[:0], timestamps[:0]
    return samples_2d[mask], timestamps[mask]


def _data_file_freshness_metadata(data_file: Path) -> dict[str, Any]:
    max_age_s = _current_file_max_age_s()
    try:
        modified_at = datetime.fromtimestamp(data_file.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return {
            "data_is_current": False,
            "data_source_mode": "missing_file",
            "data_file_modified_at_utc": None,
            "data_file_age_s": None,
            "data_currentness_max_age_s": max_age_s,
            "data_current_warning": f"Data file is missing: {data_file}",
        }

    age_s = max(0.0, (datetime.now(timezone.utc) - modified_at).total_seconds())
    is_current = age_s <= max_age_s
    warning = None
    if not is_current:
        warning = (
            "This is a saved/stale HSD acquisition, not a current live sensor reading. "
            "Start a new acquisition or point VIBRO_HSD_ACQUISITION_DIR at a file that is actively updating "
            "before interpreting the result as the current machine condition."
        )
    return {
        "data_is_current": bool(is_current),
        "data_source_mode": "current_or_live_file" if is_current else "recorded_file",
        "data_file_modified_at_utc": modified_at.isoformat(),
        "data_file_age_s": float(age_s),
        "data_currentness_max_age_s": max_age_s,
        "data_current_warning": warning,
    }


def _path_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def _safe_file_size(path: Path) -> int | None:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _apply_decoded_currentness_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("read_mode") != "latest":
        return

    timestamp_gap_s = _safe_float(metadata.get("sdk_latest_timestamp_gap_s"))
    max_age_s = _safe_float(metadata.get("data_currentness_max_age_s"))
    if max_age_s is None:
        max_age_s = DEFAULT_CURRENT_FILE_MAX_AGE_S
    if timestamp_gap_s is None or timestamp_gap_s <= max_age_s:
        return

    metadata["data_is_current"] = False
    metadata["data_source_mode"] = "sdk_decoded_stream_lagging"
    metadata["data_current_warning"] = (
        "The HSD .dat file is being updated, but STDATALOG-PYSDK could not decode a current timestamped "
        f"window for this sensor. The newest decoded SDK timestamp is about {timestamp_gap_s:.1f}s behind "
        "the file-size duration estimate. This usually means the live acquisition is not SDK-decodable as "
        "a current stream; with the current STDATALOG-PYSDK limitation, log/read one sensor stream at a time "
        "or restart a single-sensor acquisition for this sensor before using it as live data."
    )


def _current_file_max_age_s() -> float:
    raw = os.environ.get("VIBRO_CURRENT_FILE_MAX_AGE_S")
    if raw is None:
        return DEFAULT_CURRENT_FILE_MAX_AGE_S
    value = _safe_float(raw)
    if value is None:
        return DEFAULT_CURRENT_FILE_MAX_AGE_S
    return max(0.0, value)


def _acquisition_duration_s(hsd_instance: Any) -> float | None:
    get_acquisition_info = getattr(hsd_instance, "get_acquisition_info", None)
    if get_acquisition_info is None:
        return None
    try:
        info = get_acquisition_info() or {}
        start = _parse_iso_datetime(info.get("start_time"))
        end = _parse_iso_datetime(info.get("end_time"))
    except Exception:
        return None
    if start is None or end is None:
        return None
    duration_s = (end - start).total_seconds()
    return duration_s if duration_s > 0 else None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _estimate_data_duration_s(
    *,
    folder: Path,
    sensor: str,
    sensor_status: dict[str, Any],
    sampling_rate_hz: int,
    hsd_instance: Any,
    effective_sampling_rate_hz: float | None = None,
) -> float | None:
    samples = _estimate_file_sample_count(
        folder=folder,
        sensor=sensor,
        sensor_status=sensor_status,
        sampling_rate_hz=sampling_rate_hz,
        hsd_instance=hsd_instance,
    )
    if samples is None:
        return None
    # Converting the sample count to seconds must use the same clock as the
    # SDK-decoded timestamps: using the rounded configured ODR when the board's
    # true rate differs makes the estimate drift from the decoded timestamps,
    # and that constant relative error integrates into a steadily growing
    # (false) decode gap. Prefer the measured rate.
    rate_hz = (
        float(effective_sampling_rate_hz)
        if effective_sampling_rate_hz is not None and effective_sampling_rate_hz > 0.0
        else float(sampling_rate_hz)
    )
    return float(samples / rate_hz)


def _estimate_file_sample_count(
    *,
    folder: Path,
    sensor: str,
    sensor_status: dict[str, Any],
    sampling_rate_hz: int,
    hsd_instance: Any,
) -> int | None:
    """Exact decodable sample count implied by the .dat file's byte layout."""
    data_file = folder / f"{sensor}.dat"
    if sampling_rate_hz <= 0 or not _path_exists(data_file):
        return None

    data_type_len = _data_type_byte_len(sensor_status.get("data_type"))
    dimensions = _safe_int(sensor_status.get("dim")) or 1
    samples_per_ts = _safe_int(sensor_status.get("samples_per_ts")) or 0
    if data_type_len is None or samples_per_ts <= 0:
        return None

    get_protocol_size = getattr(hsd_instance, "get_data_protocol_size", None)
    try:
        protocol_size = int(get_protocol_size()) if get_protocol_size is not None else 0
    except Exception:
        protocol_size = 0

    interface = None
    try:
        interface = (hsd_instance.get_acquisition_info() or {}).get("interface")
    except Exception:
        interface = None
    data_packet_size = _packet_data_size(sensor_status, interface)
    try:
        file_size = data_file.stat().st_size
    except OSError:
        return None
    if data_packet_size and protocol_size >= 0:
        full_packet_size = data_packet_size + protocol_size
        if full_packet_size > 0:
            counter_bytes = (file_size // full_packet_size) * protocol_size
            file_size = max(0, file_size - counter_bytes)

    frame_size = samples_per_ts * dimensions * data_type_len + 8
    if frame_size <= 0:
        return None
    frames = file_size // frame_size
    if frames <= 0:
        return None
    return frames * samples_per_ts


def _data_type_byte_len(data_type: Any) -> int | None:
    text = str(data_type or "").lower()
    if text in {"int8", "int8_t", "uint8", "uint8_t", "byte"}:
        return 1
    if text in {"int16", "int16_t", "uint16", "uint16_t", "float16"}:
        return 2
    if text in {"int24", "int24_t", "uint24", "uint24_t"}:
        return 3
    if text in {"int32", "int32_t", "uint32", "uint32_t", "float", "float32"}:
        return 4
    if text in {"int64", "int64_t", "uint64", "uint64_t", "double", "float64"}:
        return 8
    return None


def _packet_data_size(sensor_status: dict[str, Any], interface: Any) -> int | None:
    key_by_interface = {0: "sd_dps", 1: "usb_dps", 2: "ble_dps", 3: "serial_dps"}
    key = key_by_interface.get(interface)
    if key:
        value = _safe_int(sensor_status.get(key))
        if value:
            return value
    for fallback in ("usb_dps", "sd_dps", "ble_dps", "serial_dps"):
        value = _safe_int(sensor_status.get(fallback))
        if value:
            return value
    return None


def _dataframe_chunk_size(sensor_status: dict[str, Any], sampling_rate_hz: int, duration_s: float) -> int:
    samples_per_ts = _safe_int(sensor_status.get("samples_per_ts")) or 1000
    requested_samples = max(samples_per_ts, int(round(sampling_rate_hz * duration_s)))
    return max(samples_per_ts, min(requested_samples, 1_000_000))


def _sensor_name_and_status(sensor_entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(sensor_entry, dict) and sensor_entry:
        name = next(iter(sensor_entry.keys()))
        status = sensor_entry[name]
        if isinstance(status, dict):
            return str(name), status
    return "", {}


def _select_default_sensor_name(sensors: list[dict[str, Any]]) -> str | None:
    configured = default_sensor_name()
    for sensor in sensors:
        if sensor["name"] == configured and sensor["data_file_exists"]:
            return configured
    for sensor in sensors:
        if sensor["enabled"] and sensor["data_file_exists"]:
            return str(sensor["name"])
    for sensor in sensors:
        if sensor["data_file_exists"]:
            return str(sensor["name"])
    return sensors[0]["name"] if sensors else None


def _normalize_axis(axis: Any) -> str:
    normalized = str(axis or "").strip().lower()
    if normalized not in VALID_AXES:
        raise ValueError("axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2")
    return normalized


def _normalize_window_request(
    *,
    start_time_s: Any,
    duration_s: Any,
    max_duration_s: float = 60.0,
) -> tuple[float | None, float]:
    normalized_duration_s = _safe_float(duration_s)
    if normalized_duration_s is None:
        raise ValueError("duration_s must be a finite number")

    normalized_start_time_s = None
    if start_time_s is not None:
        normalized_start_time_s = _safe_float(start_time_s)
        if normalized_start_time_s is None:
            raise ValueError("start_time_s must be a finite number when provided")

    _validate_window(
        start_time_s=normalized_start_time_s,
        duration_s=normalized_duration_s,
        max_duration_s=max_duration_s,
    )
    return normalized_start_time_s, normalized_duration_s


def _validate_window(*, start_time_s: float | None, duration_s: float, max_duration_s: float = 60.0) -> None:
    if start_time_s is not None and start_time_s < 0:
        raise ValueError("start_time_s must be non-negative")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")
    if duration_s > max_duration_s:
        raise ValueError(f"duration_s is capped at {max_duration_s:g} seconds for edge safety")


def _signal_preview(signal: np.ndarray, max_preview_samples: int) -> list[float]:
    limit = _nonnegative_int(max_preview_samples, default=0)
    return [float(value) for value in signal[:limit]]


def _nonnegative_int(value: Any, *, default: int) -> int:
    parsed = _safe_int(value)
    if parsed is None:
        return max(0, int(default))
    return max(0, parsed)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)
