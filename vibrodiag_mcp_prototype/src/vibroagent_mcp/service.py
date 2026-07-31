"""Building-vibration application services."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .llm_sensor_agent import AutonomousLlmSensorAgent
from .matlab_export import export_validation_dataset
from .model_client import ModelCallResult, ModelClient
from .schemas import SCHEMA_VERSION
from .sdk_vibrometer import (
    LiveSdkVibrometerDataRequiredError,
    read_sdk_vibrometer_window,
    read_sdk_vibrometer_window_payload,
)
from .sensor_registry import SensorRegistry
from .spectral import (
    analyze_impact_ringdown,
    compute_periodogram_psd,
    compute_welch_psd,
    find_distinct_peaks,
    frequency_band_summary,
    integrated_psd_power,
)


DEFAULT_BUILDING_SENSOR_CONFIG = "config/sensors.live.yaml"
ANALYSIS_DOMAIN = "building_vibration_relative_monitoring"


def list_building_sensors_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
) -> dict[str, Any]:
    """List configured building-vibration sensors and their reference sensor."""
    registry = SensorRegistry(_resolve_building_sensor_config(config_path))
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        **registry.as_dict(),
    }


def read_building_sensor_window_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    sensor_id: str = "baseline",
    axis: str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 5.0,
    include_samples: bool = False,
    max_preview_samples: int = 16,
    max_return_samples: int = 200_000,
    require_current: bool = True,
) -> dict[str, Any]:
    """Read one configured building sensor through its registry entry.

    The model never chooses an arbitrary acquisition folder. The selected
    ``sensor_id`` is resolved through the building registry, then the configured
    folder, component name, and axis are used for the read.
    """
    registry = SensorRegistry(_resolve_building_sensor_config(config_path))
    selected_id = registry.resolve_id(sensor_id or registry.baseline_sensor_id)
    if selected_id not in registry.sensors:
        raise ValueError(
            f"Sensor {sensor_id!r} is not present in the building registry. "
            f"Available sensors: {', '.join(registry.sensors)}"
        )
    sensor = registry.sensors[selected_id]
    selected_axis = str(axis or sensor.axis).strip().lower()
    if not sensor.acquisition_folder:
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "status": "error",
            "error": "building_sensor_source_unavailable",
            "message": f"Sensor {selected_id!r} has no configured live acquisition folder.",
            "sensor_id": selected_id,
            "location": sensor.location,
            "role": sensor.role,
        }

    payload = read_sdk_vibrometer_window_payload(
        machine_id=selected_id,
        acquisition_folder=sensor.acquisition_folder,
        sensor_name=sensor.hsd_sensor_name,
        axis=selected_axis,
        start_time_s=start_time_s,
        duration_s=duration_s,
        include_samples=include_samples,
        max_preview_samples=max_preview_samples,
        max_return_samples=max_return_samples,
        require_current=require_current,
    )
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "sensor_id": selected_id,
            "location": sensor.location,
            "role": sensor.role,
            "axis": selected_axis,
            "config_path": str(registry.config_path),
        }
    )
    return payload


def read_building_sensor_psd_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    sensor_id: str = "baseline",
    axis: str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    estimator: str = "welch",
    require_current: bool = True,
) -> dict[str, Any]:
    """Compact spectral summary (peaks, bands, ring-down) for one sensor.

    Measurement only: the numbers are computed deterministically from the
    registered sensor's window; interpreting them is the model's job. The
    payload is deliberately compact (no arrays) so it fits an agent-loop
    observation.
    """
    registry = SensorRegistry(_resolve_building_sensor_config(config_path))
    selected_id = registry.resolve_id(sensor_id or registry.baseline_sensor_id)
    if selected_id not in registry.sensors:
        raise ValueError(
            f"Sensor {sensor_id!r} is not present in the building registry. "
            f"Available sensors: {', '.join(registry.sensors)}"
        )
    sensor = registry.sensors[selected_id]
    selected_axis = str(axis or sensor.axis).strip().lower()
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "sensor_id": selected_id,
        "location": sensor.location,
        "role": sensor.role,
        "axis": selected_axis,
        "config_path": str(registry.config_path),
    }
    if not sensor.acquisition_folder:
        return {
            **base,
            "status": "error",
            "error": "building_sensor_source_unavailable",
            "message": f"Sensor {selected_id!r} has no configured live acquisition folder.",
        }

    try:
        window = read_sdk_vibrometer_window(
            machine_id=selected_id,
            acquisition_folder=sensor.acquisition_folder,
            sensor_name=sensor.hsd_sensor_name,
            axis=selected_axis,
            start_time_s=start_time_s,
            duration_s=duration_s,
            require_current=require_current,
        )
    except LiveSdkVibrometerDataRequiredError as exc:
        return {
            **base,
            "status": "error",
            "error": "live_data_required",
            "message": str(
                exc.metadata.get("data_current_warning")
                or "Live/current sensor data is required for a current spectrum."
            ),
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": "sdk_read_failed",
            "message": f"{exc.__class__.__name__}: {exc}",
        }

    try:
        summary = _psd_summary_from_signal(window.signal, window.sampling_rate_hz, estimator)
    except ValueError as exc:
        return {
            **base,
            "status": "error",
            "error": "invalid_window",
            "message": f"{exc.__class__.__name__}: {exc}",
        }

    return {
        **base,
        "status": "ok",
        **summary,
        "data_is_current": window.metadata.get("data_is_current"),
        "sdk_latest_timestamp_gap_s": window.metadata.get("sdk_latest_timestamp_gap_s"),
    }


def _rounded(value: Any, digits: int) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits)


def _psd_summary_from_signal(signal: Any, sampling_rate_hz: int, estimator: str) -> dict[str, Any]:
    """Compact deterministic PSD summary of one already-selected window.

    Raises ValueError for windows the spectral estimators reject.
    """
    estimator_name = str(estimator or "welch").strip().lower()
    if estimator_name in {"periodogram", "fft", "raw", "raw_fft"}:
        estimate = compute_periodogram_psd(signal, sampling_rate_hz)
        estimator_name = "periodogram"
    else:
        estimate = compute_welch_psd(signal, sampling_rate_hz)
        estimator_name = "welch"
    peaks = find_distinct_peaks(estimate.frequencies_hz, estimate.psd_g2_per_hz, limit=5)
    bands = frequency_band_summary(estimate.frequencies_hz, estimate.psd_g2_per_hz)
    impact = analyze_impact_ringdown(
        signal,
        sampling_rate_hz,
        maximum_frequency_hz=min(200.0, 0.5 * float(sampling_rate_hz)),
    )
    psd_variance_g2 = integrated_psd_power(estimate.frequencies_hz, estimate.psd_g2_per_hz)

    return {
        "estimator": estimator_name,
        "sample_count": int(getattr(signal, "size", 0)),
        "sampling_rate_hz": int(sampling_rate_hz),
        "window_duration_s": _rounded(getattr(signal, "size", 0) / float(sampling_rate_hz), 3),
        "resolution_hz": _rounded(estimate.resolution_hz, 4),
        "psd_rms_g": _rounded(math.sqrt(max(float(psd_variance_g2), 0.0)), 6),
        "top_peaks": [
            {
                "frequency_hz": _rounded(peak.get("frequency_hz"), 2),
                "psd_db_re_1_g2_per_hz": _rounded(peak.get("psd_db_re_1_g2_per_hz"), 1),
                "prominence_db": _rounded(peak.get("prominence_db"), 1),
                "band_rms_g": _rounded(peak.get("band_rms_g"), 6),
            }
            for peak in peaks
        ],
        "frequency_bands": [
            {
                "label": band.get("label"),
                "rms_g": _rounded(band.get("rms_g"), 6),
                "fraction": _rounded(band.get("fraction"), 3),
            }
            for band in bands
        ],
        "impact_ringdown": {
            "detected": bool(impact.get("detected")),
            "status": impact.get("status"),
            "message": impact.get("message"),
            "event_time_s": _rounded(impact.get("event_time_s"), 3),
            "peak_frequency_hz": _rounded(impact.get("peak_frequency_hz"), 2),
            "peak_reliable": bool(impact.get("peak_reliable")),
            "crest_factor": _rounded(impact.get("crest_factor"), 2),
        },
    }


# The boards free-run on their own crystals (no hardware sync), so cross-sensor
# alignment rests on wall-clock anchoring: each board's decoded tail corresponds
# to its file's mtime within roughly one packet (~40 ms) plus decode timing.
# End margin keeps the common end far enough behind every frontier that all
# boards provably have data there; the uncertainty constant is the honest
# cross-board alignment error of this method (packet cadence + mtime/decode
# slop), reported in the payload rather than implied to be zero.
ALIGNED_PSD_END_MARGIN_S = 0.35
ALIGNED_PSD_ALIGNMENT_UNCERTAINTY_S = 0.25
ALIGNED_PSD_WARMUP_DURATION_S = 1.0
ALIGNED_PSD_SLACK_EXTRA_S = 1.2


def compare_building_sensor_psd_aligned_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    sensor_ids: list[str] | None = None,
    axis: str | None = None,
    duration_s: float = 8.0,
    estimator: str = "welch",
    require_current: bool = True,
) -> dict[str, Any]:
    """PSD summaries of the SAME wall-clock window across several sensors.

    Measurement only: every sensor's window is trimmed to end at one common
    wall instant (frontier wall-anchoring, accurate to roughly
    ±ALIGNED_PSD_ALIGNMENT_UNCERTAINTY_S across boards), then summarized with
    the same estimator. Deltas against the reference sensor are arithmetic on
    those numbers; interpreting any of it is the model's job.
    """
    registry = SensorRegistry(_resolve_building_sensor_config(config_path))
    if sensor_ids:
        selected_ids: list[str] = []
        for raw_id in sensor_ids:
            resolved = registry.resolve_id(str(raw_id))
            if resolved not in registry.sensors:
                raise ValueError(
                    f"Sensor {raw_id!r} is not present in the building registry. "
                    f"Available sensors: {', '.join(registry.sensors)}"
                )
            if resolved not in selected_ids:
                selected_ids.append(resolved)
    else:
        selected_ids = list(registry.sensors)
    if len(selected_ids) < 2:
        raise ValueError(
            "compare_building_sensor_psd needs at least two sensors; "
            "use read_building_sensor_psd for a single sensor."
        )
    baseline_id = registry.baseline_sensor_id
    if baseline_id in selected_ids:
        selected_ids.remove(baseline_id)
        selected_ids.insert(0, baseline_id)

    duration_s = max(1.0, min(float(duration_s), 15.0))

    def _read(sensor_id: str, read_duration_s: float) -> Any:
        sensor = registry.sensors[sensor_id]
        if not sensor.acquisition_folder:
            raise ValueError(f"Sensor {sensor_id!r} has no configured live acquisition folder.")
        return read_sdk_vibrometer_window(
            machine_id=sensor_id,
            acquisition_folder=sensor.acquisition_folder,
            sensor_name=sensor.hsd_sensor_name,
            axis=str(axis or sensor.axis).strip().lower(),
            start_time_s=None,
            duration_s=read_duration_s,
            require_current=require_current,
        )

    sensors_payload: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # Pass 1 warms every decoder so the aligned pass reads land close together
    # (a cold seed costs ~1.7s; warm incremental reads cost ~0.1-0.3s).
    for sensor_id in selected_ids:
        try:
            _read(sensor_id, ALIGNED_PSD_WARMUP_DURATION_S)
        except LiveSdkVibrometerDataRequiredError as exc:
            failures.append(
                {
                    "sensor_id": sensor_id,
                    "error": "live_data_required",
                    "message": str(
                        exc.metadata.get("data_current_warning")
                        or "Live/current sensor data is required for an aligned comparison."
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sensor_id": sensor_id,
                    "error": "sdk_read_failed",
                    "message": f"{exc.__class__.__name__}: {exc}",
                }
            )
    failed_ids = {failure["sensor_id"] for failure in failures}

    # Pass 2: one common wall end-instant for every sensor.
    wall_end_common = time.time() - ALIGNED_PSD_END_MARGIN_S
    for sensor_id in selected_ids:
        if sensor_id in failed_ids:
            continue
        sensor = registry.sensors[sensor_id]
        wall_pre = time.time()
        slack_s = max(0.0, wall_pre - wall_end_common) + ALIGNED_PSD_SLACK_EXTRA_S
        try:
            window = _read(sensor_id, duration_s + slack_s)
            segment, trimmed_from_tail_s = _trim_signal_to_wall_end(
                window,
                wall_pre=wall_pre,
                wall_end_common=wall_end_common,
                duration_s=duration_s,
            )
            summary = _psd_summary_from_signal(segment, window.sampling_rate_hz, estimator)
        except LiveSdkVibrometerDataRequiredError as exc:
            failures.append(
                {
                    "sensor_id": sensor_id,
                    "error": "live_data_required",
                    "message": str(
                        exc.metadata.get("data_current_warning")
                        or "Live/current sensor data is required for an aligned comparison."
                    ),
                }
            )
            continue
        except Exception as exc:
            failures.append(
                {
                    "sensor_id": sensor_id,
                    "error": "sdk_read_failed" if not isinstance(exc, ValueError) else "invalid_window",
                    "message": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        sensors_payload.append(
            {
                "sensor_id": sensor_id,
                "location": sensor.location,
                "role": sensor.role,
                "axis": str(axis or sensor.axis).strip().lower(),
                "status": "ok",
                "alignment": {
                    "trimmed_from_tail_s": _rounded(trimmed_from_tail_s, 3),
                    "data_file_age_s": _rounded(window.metadata.get("data_file_age_s"), 3),
                },
                **summary,
                "data_is_current": window.metadata.get("data_is_current"),
            }
        )

    deltas = _psd_deltas_vs_reference(sensors_payload, baseline_id)
    status = "ok" if sensors_payload and not failures else ("partial" if sensors_payload else "error")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "status": status,
        "config_path": str(registry.config_path),
        "baseline_sensor_id": baseline_id,
        "estimator": str(estimator or "welch").strip().lower(),
        "aligned_window": {
            "duration_s": _rounded(duration_s, 3),
            "common_end_unix": _rounded(wall_end_common, 3),
            "alignment_uncertainty_s": ALIGNED_PSD_ALIGNMENT_UNCERTAINTY_S,
            "method": "per-board frontier wall-anchoring (no hardware sync; boards free-run)",
        },
        "sensors": sensors_payload,
        "deltas_vs_baseline": deltas,
        "failures": failures,
    }


def _trim_signal_to_wall_end(
    window: Any,
    *,
    wall_pre: float,
    wall_end_common: float,
    duration_s: float,
) -> tuple[Any, float]:
    """Cut one sensor's window so it ends at the common wall instant.

    The board's decoded tail is anchored to wall time via the data file's age
    (mtime is within about one packet of the newest decodable sample), so the
    amount to drop from the tail is the wall distance between this board's
    tail and the common end.
    """
    signal = np.asarray(window.signal, dtype=np.float64)
    rate_hz = float(window.sampling_rate_hz)
    if signal.size < 32 or rate_hz <= 0:
        raise ValueError("window too small for aligned trimming")
    age_s = window.metadata.get("data_file_age_s")
    age_s = float(age_s) if isinstance(age_s, (int, float)) else 0.05
    wall_tail = wall_pre - age_s
    trim_from_tail_s = max(0.0, wall_tail - wall_end_common)

    timestamps = getattr(window, "timestamps_s", None)
    if timestamps is not None and getattr(timestamps, "size", 0) == signal.size:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        end_ts = float(timestamps[-1]) - trim_from_tail_s
        end_index = int(np.searchsorted(timestamps, end_ts, side="right"))
        start_index = int(np.searchsorted(timestamps, end_ts - duration_s, side="left"))
    else:
        end_index = signal.size - int(round(trim_from_tail_s * rate_hz))
        start_index = end_index - int(round(duration_s * rate_hz))
    end_index = min(signal.size, max(0, end_index))
    start_index = max(0, start_index)

    segment = signal[start_index:end_index]
    achieved_s = segment.size / rate_hz
    if achieved_s < 0.8 * duration_s:
        raise ValueError(
            f"aligned window is only {achieved_s:.2f}s of the requested {duration_s:.2f}s "
            "(buffer too short behind the common end instant)"
        )
    return segment, trim_from_tail_s


def _psd_deltas_vs_reference(
    sensors_payload: list[dict[str, Any]], baseline_id: str
) -> list[dict[str, Any]]:
    reference = next((entry for entry in sensors_payload if entry["sensor_id"] == baseline_id), None)
    if reference is None:
        return []
    reference_bands = {
        band.get("label"): band.get("rms_g") for band in reference.get("frequency_bands", [])
    }
    reference_peak = (reference.get("top_peaks") or [{}])[0].get("frequency_hz")
    deltas: list[dict[str, Any]] = []
    for entry in sensors_payload:
        if entry["sensor_id"] == baseline_id:
            continue
        band_deltas = []
        for band in entry.get("frequency_bands", []):
            label = band.get("label")
            target_rms = band.get("rms_g")
            reference_rms = reference_bands.get(label)
            ratio_db = None
            if isinstance(target_rms, (int, float)) and isinstance(reference_rms, (int, float)):
                if target_rms > 0 and reference_rms > 0:
                    ratio_db = _rounded(20.0 * math.log10(target_rms / reference_rms), 1)
            band_deltas.append({"label": label, "rms_ratio_db": ratio_db})
        entry_peak = (entry.get("top_peaks") or [{}])[0].get("frequency_hz")
        peak_shift_hz = None
        if isinstance(entry_peak, (int, float)) and isinstance(reference_peak, (int, float)):
            peak_shift_hz = _rounded(entry_peak - reference_peak, 2)
        rms_ratio_db = None
        target_total = entry.get("psd_rms_g")
        reference_total = reference.get("psd_rms_g")
        if (
            isinstance(target_total, (int, float))
            and isinstance(reference_total, (int, float))
            and target_total > 0
            and reference_total > 0
        ):
            rms_ratio_db = _rounded(20.0 * math.log10(target_total / reference_total), 1)
        deltas.append(
            {
                "sensor_id": entry["sensor_id"],
                "psd_rms_ratio_db": rms_ratio_db,
                "dominant_peak_shift_hz": peak_shift_hz,
                "band_rms_ratios_db": band_deltas,
            }
        )
    return deltas


def run_building_sensor_agent_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    sensor_id: str = "",
    baseline_sensor_id: str = "baseline",
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
    process_reader: bool = False,
    model_base_url: str | None = None,
    model_api_key: str | None = None,
    model: str | None = None,
    model_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run the autonomous building checker for one selected target sensor."""
    selected_id = str(sensor_id or "").strip()
    if not selected_id:
        # An empty sensor_id used to silently assess EVERY registered sensor and
        # then present the first report as "the" requested sensor. Require an
        # explicit choice instead.
        raise ValueError(
            "run_building_sensor_agent requires sensor_id (for example 'target_1'). "
            "Use run_autonomous_sensor_check to assess every registered sensor."
        )
    selected = [selected_id]
    agent = AutonomousLlmSensorAgent(
        config_path=str(_resolve_building_sensor_config(config_path)),
        baseline_sensor_id=baseline_sensor_id,
        selected_sensor_ids=selected,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
        mode="live" if require_current else "replay",
        small_model_client=_building_model_client(
            role="autonomous_sensor_agent",
            base_url=model_base_url,
            api_key=model_api_key,
            model=model,
            timeout_s=model_timeout_s,
        ),
        main_model_client=_building_model_client(
            role="autonomous_network_agent",
            base_url=model_base_url,
            api_key=model_api_key,
            model=model,
            timeout_s=model_timeout_s,
        ),
        model_timeout_s=model_timeout_s,
    )
    result = agent.run_window()
    result["sensor_agent_report"] = (result.get("sensor_agent_reports") or [None])[0]
    result["process_reader_requested"] = bool(process_reader)
    return result


def run_building_vibration_agent_pipeline_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
    process_reader: bool = False,
    model_base_url: str | None = None,
    model_api_key: str | None = None,
    model: str | None = None,
    model_timeout_s: float | None = None,
    use_model_agents: bool = True,
) -> dict[str, Any]:
    """Run the LLM-assisted building checker across all selected sensors."""
    small_model_client = _building_model_client(
        role="autonomous_sensor_agent",
        base_url=model_base_url,
        api_key=model_api_key,
        model=model,
        timeout_s=model_timeout_s,
    )
    main_model_client = _building_model_client(
        role="autonomous_network_agent",
        base_url=model_base_url,
        api_key=model_api_key,
        model=model,
        timeout_s=model_timeout_s,
    )
    if not use_model_agents:
        small_model_client = _DisabledModelClient("autonomous_sensor_agent")
        main_model_client = _DisabledModelClient("autonomous_network_agent")

    agent = AutonomousLlmSensorAgent(
        config_path=str(_resolve_building_sensor_config(config_path)),
        baseline_sensor_id=baseline_sensor_id,
        selected_sensor_ids=_normalize_sensor_ids(sensor_ids),
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
        mode="live" if require_current else "replay",
        small_model_client=small_model_client,
        main_model_client=main_model_client,
        model_timeout_s=model_timeout_s,
    )
    result = agent.run_window()
    result["process_reader_requested"] = bool(process_reader)
    return result


def run_autonomous_sensor_check_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
    model_base_url: str | None = None,
    model_api_key: str | None = None,
    model: str | None = None,
    model_timeout_s: float | None = None,
    use_model_agents: bool = True,
) -> dict[str, Any]:
    """Run one autonomous building agent that checks every selected sensor."""
    return run_building_vibration_agent_pipeline_service(
        config_path=config_path,
        baseline_sensor_id=baseline_sensor_id,
        sensor_ids=sensor_ids,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
        process_reader=False,
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model=model,
        model_timeout_s=model_timeout_s,
        use_model_agents=use_model_agents,
    )


def _building_model_client(
    *,
    role: str,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    timeout_s: float | None = None,
) -> ModelClient | None:
    if base_url is None and api_key is None and model is None:
        return None
    is_sensor_role = role in {"small_sensor_agent", "autonomous_sensor_agent"}
    return ModelClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        base_url_env="SMALL_AGENT_BASE_URL" if is_sensor_role else "MAIN_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY" if is_sensor_role else "MAIN_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL" if is_sensor_role else "MAIN_AGENT_MODEL",
        role=role,
        timeout_s=timeout_s,
    )


class _DisabledModelClient:
    def __init__(self, role: str):
        self.role = role

    def call_json_model(
        self,
        prompt: str,
        allowed_labels: set[str] | None = None,
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelCallResult:
        return ModelCallResult(
            ok=False,
            used_model=False,
            data=None,
            text=None,
            error="disabled_for_fast_monitor",
            metadata={
                "role": self.role,
                "model_used": False,
                "provider": "deterministic_feature_prepass",
                "model": None,
                "endpoint_configured": False,
                "fallback_reason": "disabled_for_fast_monitor",
            },
        )


def export_building_validation_dataset_service(
    config_path: str = DEFAULT_BUILDING_SENSOR_CONFIG,
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    output_dir: str = "./validation_exports",
    require_current: bool = True,
) -> dict[str, Any]:
    """Run the building checker and export MATLAB validation artifacts."""
    agent = AutonomousLlmSensorAgent(
        config_path=str(_resolve_building_sensor_config(config_path)),
        baseline_sensor_id=baseline_sensor_id,
        selected_sensor_ids=_normalize_sensor_ids(sensor_ids),
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
        mode="live" if require_current else "replay",
    )
    result = agent.run_window()
    try:
        export = export_validation_dataset(
            pipeline_result=result,
            windows=agent.last_windows,
            output_dir=output_dir,
        )
    except OSError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "status": "error",
            "error": "matlab_export_failed",
            "message": f"{exc.__class__.__name__}: {exc}",
            "window_id": result["window_id"],
            "baseline_sensor_id": baseline_sensor_id,
            "network_assessment": result["network_assessment"],
            "model_metadata": result["model_metadata"],
            "requested_output_dir": output_dir,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "status": "ok",
        "window_id": result["window_id"],
        "baseline_sensor_id": baseline_sensor_id,
        "network_assessment": result["network_assessment"],
        "model_metadata": result["model_metadata"],
        "matlab_export": export,
    }


def _resolve_building_sensor_config(config_path: str | None = DEFAULT_BUILDING_SENSOR_CONFIG) -> Path:
    raw = str(config_path or DEFAULT_BUILDING_SENSOR_CONFIG).strip() or DEFAULT_BUILDING_SENSOR_CONFIG
    env_path = os.environ.get("VIBRO_BUILDING_SENSOR_CONFIG")
    candidates: list[Path] = []

    if env_path:
        candidates.append(Path(env_path).expanduser())

    requested = Path(raw).expanduser()
    candidates.append(requested)

    if not requested.is_absolute():
        candidates.append((Path.cwd() / requested).resolve())
        candidates.append((Path(__file__).resolve().parents[2] / requested).resolve())
    elif requested.name == "sensors.live.yaml":
        candidates.append((Path(__file__).resolve().parents[2] / DEFAULT_BUILDING_SENSOR_CONFIG).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def _normalize_sensor_ids(sensor_ids: list[str] | str | None) -> list[str] | None:
    if sensor_ids is None:
        return None
    if isinstance(sensor_ids, str):
        values = [item.strip() for item in sensor_ids.split(",")]
    else:
        values = [str(item).strip() for item in sensor_ids]
    return [item for item in values if item] or None
