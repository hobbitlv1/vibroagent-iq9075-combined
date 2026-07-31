"""Autonomous LLM-assisted building sensor checker.

This module owns one concrete building-monitoring loop:

1. load the configured sensor registry,
2. read the baseline and every selected target sensor,
3. compute compact building-vibration metrics,
4. ask one OpenAI-compatible model endpoint for strict JSON labels covering
   every target sensor plus the fused network assessment.

Raw vibration samples stay in process and are never placed in model prompts.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import compute_features, preprocess_signal
from .model_client import ModelCallResult, ModelClient
from .schemas import (
    ANALYSIS_DOMAIN,
    NETWORK_LABELS,
    SCHEMA_VERSION,
    SENSOR_LABELS,
    SensorConfig,
    ensure_building_domain_only,
    validate_network_label,
    validate_sensor_label,
)
from .sdk_vibrometer import (
    LiveSdkVibrometerDataRequiredError,
    read_sdk_vibrometer_window,
)
from .sensor_registry import SensorRegistry

VALID_AXES = {"norm", "magnitude", "vector_norm", "x", "y", "z", "0", "1", "2"}
QUALITY_SENSOR_LABELS = {"possible_sensor_issue", "insufficient_data"}
QUALITY_NETWORK_LABELS = {"sensor_network_quality_issue", "insufficient_data"}

# Reasons that mean "no model exists in this deployment" (tests, offline export,
# the fast monitor prepass). Only these may fall back to deterministic rule
# labels, clearly marked via decision_source. A CONFIGURED model that fails or
# returns unusable output never falls back: the decision is the model's or it is
# an explicit error.
NO_MODEL_CONFIGURED_REASONS = {
    "endpoint_not_configured",
    "openai_client_not_installed",
    "disabled_for_fast_monitor",
}


def _parallel_read_workers() -> int:
    raw = os.environ.get("VIBRO_BOARD_READER_PARALLELISM")
    try:
        value = int(raw) if raw is not None else 16
    except ValueError:
        value = 16
    return max(1, value)
EPS = 1e-12


@dataclass(frozen=True)
class SensorWindow:
    config: SensorConfig
    signal: np.ndarray
    timestamps_s: np.ndarray | None
    sampling_rate_hz: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ReadFailure:
    config: SensorConfig
    axis: str
    duration_s: float
    quality_flags: list[str]
    message: str
    metadata: dict[str, Any]


class AutonomousLlmSensorAgent:
    """Autonomously read and assess all configured building sensors."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        baseline_sensor_id: str = "baseline",
        selected_sensor_ids: list[str] | tuple[str, ...] | None = None,
        start_time_s: float | None = None,
        duration_s: float = 10.0,
        axis: str | None = None,
        require_current: bool = True,
        mode: str | None = None,
        small_model_client: Any | None = None,
        main_model_client: Any | None = None,
        model_base_url: str | None = None,
        model_api_key: str | None = None,
        model: str | None = None,
        model_timeout_s: float | None = None,
    ):
        self.registry = SensorRegistry(config_path)
        self.baseline_sensor_id = baseline_sensor_id or self.registry.baseline_sensor_id
        self.selected_sensor_ids = list(selected_sensor_ids) if selected_sensor_ids else None
        self.start_time_s = start_time_s
        self.duration_s = float(duration_s)
        self.axis = _normalize_optional_axis(axis)
        self.require_current = bool(require_current)
        self.mode = (mode or ("live" if self.require_current else "replay")).strip().lower()
        self._small_model_client_explicit = small_model_client is not None
        self._main_model_client_explicit = main_model_client is not None
        self.small_model_client = small_model_client or _default_model_client(
            role="autonomous_sensor_agent",
            base_url=model_base_url,
            api_key=model_api_key,
            model=model,
            timeout_s=model_timeout_s,
            base_url_env="SMALL_AGENT_BASE_URL",
            api_key_env="SMALL_AGENT_API_KEY",
            model_env="SMALL_AGENT_MODEL",
        )
        self.main_model_client = main_model_client or _default_model_client(
            role="autonomous_network_agent",
            base_url=model_base_url,
            api_key=model_api_key,
            model=model,
            timeout_s=model_timeout_s,
            base_url_env="MAIN_AGENT_BASE_URL",
            api_key_env="MAIN_AGENT_API_KEY",
            model_env="MAIN_AGENT_MODEL",
        )
        self.last_windows: dict[str, SensorWindow] = {}

    def run_window(
        self,
        *,
        start_time_s: float | None = None,
        duration_s: float | None = None,
        axis: str | None = None,
        require_current: bool | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        start = self.start_time_s if start_time_s is None else start_time_s
        duration = self.duration_s if duration_s is None else float(duration_s)
        selected_axis = self.axis if axis is None else _normalize_optional_axis(axis)
        current_required = self.require_current if require_current is None else bool(require_current)
        selected_mode = (mode or self.mode or ("live" if current_required else "replay")).strip().lower()
        _validate_window(start_time_s=start, duration_s=duration, axis=selected_axis, mode=selected_mode)

        window_id = _new_window_id()
        baseline_config = self.registry.baseline(self.baseline_sensor_id)
        target_configs = self.registry.targets(self.selected_sensor_ids, self.baseline_sensor_id)
        sensor_configs = _dedupe_sensor_configs([baseline_config, *target_configs])
        self.last_windows = {}

        # Decode the boards concurrently (same bounded worker count as
        # BuildingAgentPipeline): each read is independent and _read_sensor
        # already converts every failure into a _ReadFailure.
        reads: dict[str, SensorWindow | _ReadFailure] = {}
        with ThreadPoolExecutor(
            max_workers=min(len(sensor_configs), _parallel_read_workers()),
            thread_name_prefix="vibro-agent-read",
        ) as executor:
            futures = {
                config.sensor_id: executor.submit(
                    self._read_sensor,
                    config=config,
                    axis=selected_axis or config.axis,
                    start_time_s=start,
                    duration_s=duration,
                    mode=selected_mode,
                    require_current=current_required,
                )
                for config in sensor_configs
            }
            for sensor_id, future in futures.items():
                reads[sensor_id] = future.result()

        baseline_read = reads[baseline_config.sensor_id]
        if isinstance(baseline_read, SensorWindow):
            baseline_metrics = build_building_metrics(
                window_id=window_id,
                window=baseline_read,
                duration_s=duration,
            )
            self.last_windows[baseline_config.sensor_id] = baseline_read
        else:
            baseline_metrics = _failure_metrics(window_id=window_id, failure=baseline_read)

        sensor_metrics: list[dict[str, Any]] = []
        all_sensor_metrics: list[dict[str, Any]] = [baseline_metrics]
        sensor_prechecks: list[dict[str, Any]] = []
        read_failures: list[str] = []

        for target in target_configs:
            read = reads[target.sensor_id]
            if isinstance(read, SensorWindow):
                metrics = build_building_metrics(window_id=window_id, window=read, duration_s=duration)
                self.last_windows[target.sensor_id] = read
            else:
                metrics = _failure_metrics(window_id=window_id, failure=read)
                read_failures.append(target.sensor_id)
            comparison = compare_building_metrics(metrics, baseline_metrics)
            rule_label, rule_confidence, rule_evidence = deterministic_sensor_label(
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                comparison=comparison,
            )
            sensor_prechecks.append(
                {
                    "target": target,
                    "axis": selected_axis or target.axis,
                    "duration_s": duration,
                    "metrics": metrics,
                    "comparison": comparison,
                    "rule_label": rule_label,
                    "rule_confidence": rule_confidence,
                    "rule_evidence": rule_evidence,
                }
            )
            sensor_metrics.append(metrics)
            all_sensor_metrics.append(metrics)

        provisional_reports = [
            _precheck_report_for_network(precheck, baseline_metrics=baseline_metrics)
            for precheck in sensor_prechecks
        ]
        network_rule_label, network_rule_confidence, affected, network_rule_evidence, network_rule_explanation = deterministic_network_label(
            reports=provisional_reports,
            baseline_metrics=baseline_metrics,
        )
        attempted_sensor_ids = [config.sensor_id for config in sensor_configs]
        expected_sensor_ids = list(self.registry.sensors)
        checked_sensor_ids = [sensor_id for sensor_id, read in reads.items() if isinstance(read, SensorWindow)]
        failed_sensor_ids = read_failures + [
            baseline_config.sensor_id
            for read in [baseline_read]
            if isinstance(read, _ReadFailure)
        ]
        window_payload = {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "window_id": window_id,
            "start_time_s": _json_float(start),
            "duration_s": float(duration),
            "mode": selected_mode,
            "require_current": bool(current_required),
        }

        batch_client = self.main_model_client
        if self._small_model_client_explicit and not self._main_model_client_explicit:
            batch_client = self.small_model_client
        try:
            batch_decision = _model_all_sensor_decision(
                client=batch_client,
                baseline_metrics=baseline_metrics,
                sensor_prechecks=sensor_prechecks,
                network_rule_label=network_rule_label,
                network_rule_confidence=network_rule_confidence,
                affected_sensor_ids=affected,
                network_rule_evidence=network_rule_evidence,
                network_rule_explanation=network_rule_explanation,
            )
        except ModelDecisionUnavailableError as exc:
            # LLM-only policy: a configured model that produced no usable
            # decision is an explicit error; measurements are preserved but no
            # deterministic labels or explanation impersonate the agent.
            result = {
                "schema_version": SCHEMA_VERSION,
                "analysis_domain": ANALYSIS_DOMAIN,
                "status": "error",
                "error": "model_decision_unavailable",
                "message": (
                    f"The building-monitoring model did not return a usable decision ({exc.reason}). "
                    "No deterministic fallback answer was generated."
                ),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "agent": "AutonomousLlmSensorAgent",
                "window_id": window_id,
                "baseline_sensor_id": baseline_config.sensor_id,
                "mode": selected_mode,
                "window": window_payload,
                "expected_sensor_ids": expected_sensor_ids,
                "attempted_sensor_ids": attempted_sensor_ids,
                "checked_sensor_ids": checked_sensor_ids,
                "failed_sensor_ids": failed_sensor_ids,
                "all_sensors_checked": self.selected_sensor_ids is None and set(attempted_sensor_ids) == set(expected_sensor_ids),
                "baseline_metrics": baseline_metrics,
                "sensor_metrics": sensor_metrics,
                "all_sensor_metrics": all_sensor_metrics,
                "sensor_agent_reports": [],
                "network_assessment": None,
                "main_agent_explanation": None,
                "main_agent_explanation_source": None,
                "model_metadata": {
                    **exc.metadata,
                    "agent_mode": "single_llm_batch",
                    "raw_samples_sent_to_models": False,
                    "fallbacks_used": [],
                },
            }
            ensure_building_domain_only(result)
            return result
        sensor_reports = [
            self._build_sensor_report(
                window_id=window_id,
                target=precheck["target"],
                axis=precheck["axis"],
                duration_s=precheck["duration_s"],
                metrics=precheck["metrics"],
                baseline_metrics=baseline_metrics,
                comparison=precheck["comparison"],
                rule_label=precheck["rule_label"],
                model_report=batch_decision["sensor_decisions"][precheck["target"].sensor_id],
            )
            for precheck in sensor_prechecks
        ]
        network = self._build_network_assessment(
            window_id=window_id,
            baseline_sensor_id=baseline_config.sensor_id,
            baseline_metrics=baseline_metrics,
            reports=sensor_reports,
            rule_label=network_rule_label,
            model_report=batch_decision["network_decision"],
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "status": "ok",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "agent": "AutonomousLlmSensorAgent",
            "window_id": window_id,
            "baseline_sensor_id": baseline_config.sensor_id,
            "mode": selected_mode,
            "window": window_payload,
            "expected_sensor_ids": expected_sensor_ids,
            "attempted_sensor_ids": attempted_sensor_ids,
            "checked_sensor_ids": checked_sensor_ids,
            "failed_sensor_ids": failed_sensor_ids,
            "all_sensors_checked": self.selected_sensor_ids is None and set(attempted_sensor_ids) == set(expected_sensor_ids),
            "baseline_metrics": baseline_metrics,
            "sensor_metrics": sensor_metrics,
            "all_sensor_metrics": all_sensor_metrics,
            "sensor_agent_reports": sensor_reports,
            "network_assessment": network,
            "main_agent_explanation": network["explanation"],
            "main_agent_explanation_source": network["explanation_source"],
            "model_metadata": _aggregate_model_metadata(sensor_reports, network),
        }
        ensure_building_domain_only(result)
        return result

    def _read_sensor(
        self,
        *,
        config: SensorConfig,
        axis: str,
        start_time_s: float | None,
        duration_s: float,
        mode: str,
        require_current: bool,
    ) -> SensorWindow | _ReadFailure:
        try:
            if mode == "replay" and config.replay_signal_path:
                signal, timestamps_s, sampling_rate_hz, metadata = _load_replay_window(
                    path=Path(config.replay_signal_path),
                    axis=axis,
                    start_time_s=start_time_s,
                    duration_s=duration_s,
                    configured_sampling_rate_hz=config.sampling_rate_hz,
                )
            else:
                if not config.acquisition_folder:
                    raise ValueError(
                        "Sensor has no acquisition_folder for live/HSD reading and no replay_signal_path for replay mode"
                    )
                window = read_sdk_vibrometer_window(
                    machine_id=config.sensor_id,
                    acquisition_folder=config.acquisition_folder,
                    sensor_name=config.hsd_sensor_name,
                    axis=axis,
                    start_time_s=start_time_s,
                    duration_s=duration_s,
                    require_current=require_current if mode == "live" else False,
                )
                signal = window.signal
                timestamps_s = window.timestamps_s
                sampling_rate_hz = window.sampling_rate_hz
                metadata = {"source": "stdatalog_sdk_hsd", **window.metadata}
            quality_flags = _quality_flags_for_signal(signal, sampling_rate_hz, duration_s)
            metadata = {
                **metadata,
                "quality_flags": quality_flags,
                "mode": mode,
                "role": config.role,
                "location": config.location,
            }
            return SensorWindow(
                config=config,
                signal=np.asarray(signal, dtype=np.float64).reshape(-1),
                timestamps_s=timestamps_s,
                sampling_rate_hz=int(sampling_rate_hz),
                metadata=metadata,
            )
        except LiveSdkVibrometerDataRequiredError as exc:
            metadata = dict(exc.metadata)
            warning = str(metadata.get("data_current_warning") or "Live/current sensor data is required.")
            return _ReadFailure(
                config=config,
                axis=axis,
                duration_s=duration_s,
                quality_flags=["missing_data", "stale_data"],
                message=warning,
                metadata={"source": "stdatalog_sdk_hsd", "read_error": warning, **metadata},
            )
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}"
            return _ReadFailure(
                config=config,
                axis=axis,
                duration_s=duration_s,
                quality_flags=["missing_data", "read_error"],
                message=message,
                metadata={"read_error": message, "mode": mode},
            )

    def _build_sensor_report(
        self,
        *,
        window_id: str,
        target: SensorConfig,
        axis: str,
        duration_s: float,
        metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        comparison: dict[str, Any],
        rule_label: str,
        model_report: dict[str, Any],
    ) -> dict[str, Any]:
        report = {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "window_id": window_id,
            "sensor_id": target.sensor_id,
            "baseline_sensor_id": baseline_metrics["sensor_id"],
            "location": target.location,
            "role": target.role,
            "axis": axis,
            "sampling_rate_hz": int(metrics.get("sampling_rate_hz") or 0),
            "duration_s": float(duration_s),
            "quality_flags": _dedupe_strings([*metrics.get("quality_flags", []), *baseline_metrics.get("quality_flags", [])]),
            "python_rule_label": validate_sensor_label(rule_label),
            "small_agent_label": validate_sensor_label(model_report["label"]),
            "small_agent_confidence": _clamp01(model_report["confidence"]),
            "evidence": _dedupe_strings(model_report["evidence"])[:8],
            "model_metadata": model_report["model_metadata"],
            "metrics": metrics,
            "baseline_comparison": comparison,
        }
        ensure_building_domain_only(report)
        return report

    def _build_network_assessment(
        self,
        *,
        window_id: str,
        baseline_sensor_id: str,
        baseline_metrics: dict[str, Any],
        reports: list[dict[str, Any]],
        rule_label: str,
        model_report: dict[str, Any],
    ) -> dict[str, Any]:
        network = {
            "schema_version": SCHEMA_VERSION,
            "analysis_domain": ANALYSIS_DOMAIN,
            "window_id": window_id,
            "baseline_sensor_id": baseline_sensor_id,
            "python_rule_label": validate_network_label(rule_label),
            "network_label": validate_network_label(model_report["label"]),
            "main_agent_label": validate_network_label(model_report["label"]),
            "main_agent_confidence": _clamp01(model_report["confidence"]),
            "quality_flags": _dedupe_strings(
                [*baseline_metrics.get("quality_flags", []), *(flag for report in reports for flag in report.get("quality_flags", []))]
            ),
            "affected_sensor_ids": model_report["affected_sensor_ids"],
            "evidence": _dedupe_strings(model_report["evidence"])[:10],
            "explanation": model_report["explanation"],
            "explanation_source": model_report["explanation_source"],
            "model_metadata": model_report["model_metadata"],
        }
        ensure_building_domain_only(network)
        return network


def build_building_metrics(*, window_id: str, window: SensorWindow, duration_s: float) -> dict[str, Any]:
    signal = np.asarray(window.signal, dtype=np.float64).reshape(-1)
    processed = preprocess_signal(signal, sampling_rate_hz=window.sampling_rate_hz)
    features = compute_features(processed)
    td = features["time_domain"]
    fd = features["frequency_domain"]
    quality_flags = _dedupe_strings([*window.metadata.get("quality_flags", []), *_quality_flags_for_signal(signal, window.sampling_rate_hz, duration_s)])
    dominant_frequency_hz = _json_float(fd.get("dominant_frequency_hz"))
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "window_id": window_id,
        "sensor_id": window.config.sensor_id,
        "location": window.config.location,
        "role": window.config.role,
        "axis": str(window.metadata.get("axis") or window.config.axis),
        "sampling_rate_hz": int(window.sampling_rate_hz),
        "duration_s": float(td["duration_s"]),
        "sample_count": int(td["num_samples"]),
        "acceleration_rms_g": _json_float(td["rms"]),
        "acceleration_peak_g": _json_float(td["abs_peak"]),
        "acceleration_peak_to_peak_g": _json_float(td["peak_to_peak"]),
        "crest_factor": _json_float(td["crest_factor"]),
        "kurtosis": _json_float(td["kurtosis"]),
        "estimated_ppv_mm_s": _estimated_ppv_mm_s(td["rms"], dominant_frequency_hz),
        "dominant_frequency_hz": dominant_frequency_hz,
        "band_powers": _relative_band_powers(processed.centered, window.sampling_rate_hz),
        "quality_flags": quality_flags,
        "source_metadata": _compact_metadata(window.metadata),
    }


def compare_building_metrics(metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    band_powers = metrics.get("band_powers") or {}
    baseline_bands = baseline_metrics.get("band_powers") or {}
    band_keys = sorted(set(band_powers) | set(baseline_bands))
    spectral_distance = math.sqrt(
        sum((float(band_powers.get(key, 0.0)) - float(baseline_bands.get(key, 0.0))) ** 2 for key in band_keys)
    )
    comparison = {
        "baseline_sensor_id": baseline_metrics.get("sensor_id"),
        "sensor_id": metrics.get("sensor_id"),
        "rms_ratio": _bounded_ratio(metrics.get("acceleration_rms_g"), baseline_metrics.get("acceleration_rms_g")),
        "peak_ratio": _bounded_ratio(metrics.get("acceleration_peak_g"), baseline_metrics.get("acceleration_peak_g")),
        "peak_to_peak_ratio": _bounded_ratio(metrics.get("acceleration_peak_to_peak_g"), baseline_metrics.get("acceleration_peak_to_peak_g")),
        "ppv_ratio": _bounded_ratio(metrics.get("estimated_ppv_mm_s"), baseline_metrics.get("estimated_ppv_mm_s")),
        "rms_delta_g": _delta(metrics.get("acceleration_rms_g"), baseline_metrics.get("acceleration_rms_g")),
        "peak_delta_g": _delta(metrics.get("acceleration_peak_g"), baseline_metrics.get("acceleration_peak_g")),
        "dominant_frequency_delta_hz": _delta(metrics.get("dominant_frequency_hz"), baseline_metrics.get("dominant_frequency_hz")),
        "kurtosis_delta": _delta(metrics.get("kurtosis"), baseline_metrics.get("kurtosis")),
        "crest_factor_delta": _delta(metrics.get("crest_factor"), baseline_metrics.get("crest_factor")),
        "spectral_distance": _json_float(spectral_distance),
        "baseline_quality_flags": list(baseline_metrics.get("quality_flags") or []),
        "sensor_quality_flags": list(metrics.get("quality_flags") or []),
    }
    return comparison


def deterministic_sensor_label(
    *,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[str, float, list[str]]:
    sensor_flags = set(metrics.get("quality_flags") or [])
    baseline_flags = set(baseline_metrics.get("quality_flags") or [])
    evidence: list[str] = []

    if "missing_data" in sensor_flags or "insufficient_samples" in sensor_flags:
        return "insufficient_data", 0.96, _dedupe_strings(["Sensor window could not be read with enough samples.", *sensor_flags])
    if "missing_data" in baseline_flags or "insufficient_samples" in baseline_flags:
        return "insufficient_data", 0.96, _dedupe_strings(["Reference sensor window is not usable.", *baseline_flags])
    if sensor_flags or baseline_flags:
        return "possible_sensor_issue", 0.9, _dedupe_strings(["Quality flags are present in the sensor or reference window.", *sensor_flags, *baseline_flags])

    rms_ratio = _num(comparison.get("rms_ratio"), default=1.0)
    peak_ratio = _num(comparison.get("peak_ratio"), default=1.0)
    ppv_ratio = _num(comparison.get("ppv_ratio"), default=1.0)
    spectral_distance = _num(comparison.get("spectral_distance"), default=0.0)
    kurtosis = _num(metrics.get("kurtosis"), default=0.0)
    crest = _num(metrics.get("crest_factor"), default=0.0)

    evidence.append(f"RMS ratio to reference is {rms_ratio:.2f}.")
    evidence.append(f"Peak ratio to reference is {peak_ratio:.2f}.")
    evidence.append(f"Spectral distance to reference is {spectral_distance:.2f}.")

    if kurtosis >= 8.0 or crest >= 7.0 or peak_ratio >= 5.0:
        evidence.append("The local window has a high impulsive-shape indicator.")
        return "impulsive_or_shock_event", 0.88, evidence
    if rms_ratio >= 3.0 or peak_ratio >= 3.0 or ppv_ratio >= 3.0 or spectral_distance >= 0.65:
        evidence.append("The local vibration metrics are far above the reference window.")
        return "significant_local_deviation", 0.86, evidence
    if rms_ratio >= 1.5 or peak_ratio >= 1.5 or ppv_ratio >= 1.5 or spectral_distance >= 0.35:
        evidence.append("The local vibration metrics are elevated relative to the reference window.")
        return "mild_deviation", 0.74, evidence
    evidence.append("The compact metrics remain close to the reference sensor.")
    return "normal_relative_to_baseline", 0.82, evidence


def deterministic_network_label(
    *,
    reports: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> tuple[str, float, list[str], list[str], str]:
    if not reports:
        return (
            "insufficient_data",
            0.98,
            [],
            ["No target sensor reports were produced."],
            "No target sensors were available for this building-vibration check.",
        )

    quality_reports = [
        report
        for report in reports
        if report.get("small_agent_label") in QUALITY_SENSOR_LABELS or report.get("quality_flags")
    ]
    if baseline_metrics.get("quality_flags") or quality_reports:
        affected = [str(report.get("sensor_id")) for report in quality_reports]
        return (
            "sensor_network_quality_issue",
            0.9,
            affected,
            ["One or more sensor windows have quality or read flags."],
            "The run completed, but at least one sensor or the reference window has a data-quality issue. Treat this as a sensor-network triage result before interpreting vibration deviations.",
        )

    affected_reports = [
        report
        for report in reports
        if report.get("small_agent_label")
        in {"mild_deviation", "significant_local_deviation", "impulsive_or_shock_event"}
    ]
    affected = [str(report.get("sensor_id")) for report in affected_reports]
    significant = [
        report
        for report in affected_reports
        if report.get("small_agent_label") in {"significant_local_deviation", "impulsive_or_shock_event"}
    ]
    mild = [
        report
        for report in affected_reports
        if report.get("small_agent_label") == "mild_deviation"
    ]

    if not affected_reports:
        return (
            "normal_relative_to_reference_sensor",
            0.84,
            [],
            ["All target sensors are close to the reference sensor."],
            (
                "All checked target sensors remain close to the reference sensor for this synchronized window, "
                "and no affected target sensors were identified. Continue monitoring comparable windows and "
                "watch for sustained trend changes before escalating this relative triage result."
            ),
        )
    if len(significant) >= 2 or len(affected_reports) >= max(2, math.ceil(len(reports) * 0.6)):
        return (
            "multi_sensor_structure_wide_vibration_event",
            0.84,
            affected,
            ["Multiple target sensors deviated from the reference in the same window."],
            "Multiple target sensors deviated from the reference sensor in the same window. This is a monitoring triage result, not a certified structural safety assessment.",
        )
    if len(affected_reports) > 1 and not significant:
        return (
            "mild_multi_sensor_deviation",
            0.76,
            affected,
            ["More than one target sensor has a mild relative deviation."],
            "Several target sensors show mild relative deviation. Continue monitoring and repeat the measurement under comparable conditions.",
        )
    if significant:
        return (
            "localized_vibration_deviation",
            0.82,
            affected,
            ["One target sensor has a significant or impulsive relative deviation."],
            "One checked target sensor stands out from the reference sensor. Inspect that location and repeat the measurement before taking action.",
        )
    if mild:
        return (
            "mild_local_deviation",
            0.72,
            affected,
            ["One target sensor has a mild relative deviation."],
            "One checked target sensor is mildly elevated relative to the reference sensor. Continue monitoring and repeat the window for confirmation.",
        )
    return (
        "normal_relative_to_reference_sensor",
        0.8,
        [],
        ["No actionable relative deviation was detected."],
        "No actionable relative deviation was detected in this window.",
    )


class ModelDecisionUnavailableError(RuntimeError):
    """A configured model endpoint produced no usable decision.

    Raised instead of substituting deterministic rule labels: the caller gets an
    explicit error result and no fallback answer is generated (LLM-only policy).
    """

    def __init__(self, reason: str, metadata: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata


class _ModelDecisionRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Fix 1 (FIXES_TODO_20260709), OPT-IN and default OFF: re-derive network_status
# from the model's own sensor_reports. The psdtext production model's network
# head is unreliable (never answered "normal" on real ambient data, 0/151,
# while its own sensor_reports were all-normal in 89/151); derivation was
# validated at +15/+4/+59/+14 pts network accuracy on the four stage-1 sets.
# Default OFF because the LLM-only decision policy keeps every runtime judgment
# the model's; the aggregation thresholds below are a deterministic rule. The
# root fix is training-side (ordered schema — where derivation measurably adds
# nothing); set VIBRO_DERIVE_NETWORK_STATUS=1 only as a deliberate interim
# crutch for the frozen psdtext deployment.
_DERIVE_NETWORK_FROM_REPORTS = os.environ.get(
    "VIBRO_DERIVE_NETWORK_STATUS", "0"
).strip().lower() in {"1", "true", "yes"}


def _derive_network_from_sensor_reports(
    data: dict[str, Any], *, baseline_metrics: dict[str, Any]
) -> str | None:
    """Network label from the model's own sensor_reports via the deterministic
    rule; None (fail open to the model's raw network_status) when the reports
    are missing or carry an unknown label."""
    raw_reports = data.get("sensor_reports")
    if not isinstance(raw_reports, list) or not raw_reports:
        return None
    reports: list[dict[str, Any]] = []
    for item in raw_reports:
        if not isinstance(item, dict):
            return None
        try:
            label = validate_sensor_label(
                str(item.get("local_status") or item.get("small_agent_label") or item.get("label") or "")
            )
        except ValueError:
            return None
        reports.append({"sensor_id": str(item.get("sensor_id") or ""), "small_agent_label": label, "quality_flags": []})
    label, _, _, _, _ = deterministic_network_label(reports=reports, baseline_metrics=baseline_metrics)
    return label


def _precheck_report_for_network(precheck: dict[str, Any], *, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    target = precheck["target"]
    metrics = precheck["metrics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "window_id": metrics.get("window_id"),
        "sensor_id": target.sensor_id,
        "baseline_sensor_id": baseline_metrics.get("sensor_id"),
        "location": target.location,
        "role": target.role,
        "axis": precheck["axis"],
        "sampling_rate_hz": int(metrics.get("sampling_rate_hz") or 0),
        "duration_s": float(precheck["duration_s"]),
        "quality_flags": _dedupe_strings([*metrics.get("quality_flags", []), *baseline_metrics.get("quality_flags", [])]),
        "python_rule_label": validate_sensor_label(precheck["rule_label"]),
        "small_agent_label": validate_sensor_label(precheck["rule_label"]),
        "small_agent_confidence": _clamp01(precheck["rule_confidence"]),
        "evidence": _dedupe_strings(precheck["rule_evidence"])[:8],
        "model_metadata": {
            "role": "autonomous_all_sensor_agent",
            "model_used": False,
            "provider": "deterministic_precheck",
            "model": None,
            "endpoint_configured": False,
            "fallback_reason": "deterministic_precheck_only",
        },
        "metrics": metrics,
        "baseline_comparison": precheck["comparison"],
    }


def _model_all_sensor_decision(
    *,
    client: Any,
    baseline_metrics: dict[str, Any],
    sensor_prechecks: list[dict[str, Any]],
    network_rule_label: str,
    network_rule_confidence: float,
    affected_sensor_ids: list[str],
    network_rule_evidence: list[str],
    network_rule_explanation: str,
) -> dict[str, Any]:
    prompt = _build_all_sensor_prompt(
        baseline_metrics=baseline_metrics,
        sensor_prechecks=sensor_prechecks,
        network_rule_label=network_rule_label,
        network_rule_confidence=network_rule_confidence,
        affected_sensor_ids=affected_sensor_ids,
        network_rule_evidence=network_rule_evidence,
        network_rule_explanation=network_rule_explanation,
    )
    call = _call_model(
        client,
        prompt,
        allowed_labels=set(),
        extra_body=_all_sensor_response_format(
            [str(precheck["target"].sensor_id) for precheck in sensor_prechecks]
        ),
    )
    batch_metadata = _batch_metadata(call.metadata)
    if not call.ok or not call.data:
        reason = str(call.error or call.metadata.get("fallback_reason") or "model_call_failed")
        if reason not in NO_MODEL_CONFIGURED_REASONS:
            # A configured model failed (timeout, HTTP error, unparseable
            # response): surface it, never answer with rule labels.
            raise ModelDecisionUnavailableError(reason, _rejected_metadata(batch_metadata, reason))
        return _fallback_all_sensor_decisions(
            sensor_prechecks=sensor_prechecks,
            network_rule_label=network_rule_label,
            network_rule_confidence=network_rule_confidence,
            affected_sensor_ids=affected_sensor_ids,
            network_rule_evidence=network_rule_evidence,
            network_rule_explanation=network_rule_explanation,
            metadata=batch_metadata,
        )

    try:
        data = call.data
        try:
            ensure_building_domain_only(data)
        except ValueError as exc:
            raise _ModelDecisionRejected("model_output_rejected_for_out_of_domain_language") from exc
        network_data = _network_data_from_batch(data)
        try:
            network_label = validate_network_label(
                str(network_data.get("network_status") or network_data.get("network_label") or network_data.get("main_agent_label") or "")
            )
        except ValueError as exc:
            raise _ModelDecisionRejected("model_output_invalid_network_label") from exc
        network_affected_value = network_data.get("affected_sensor_ids")
        if network_affected_value is None:
            network_affected_value = network_data.get("affected_sensors")
        network_affected_ids = _string_list(network_affected_value)
        model_network_label_raw = network_label
        if _DERIVE_NETWORK_FROM_REPORTS:
            derived = _derive_network_from_sensor_reports(data, baseline_metrics=baseline_metrics)
            if derived is not None:
                network_label = derived
        allow_network_normal_expansion = (
            network_label == "normal_relative_to_reference_sensor"
            and network_rule_label == "normal_relative_to_reference_sensor"
            and not network_affected_ids
            and not baseline_metrics.get("quality_flags")
        )
        sensor_decisions: dict[str, dict[str, Any]] = {}
        for precheck in sensor_prechecks:
            sensor_id = str(precheck["target"].sensor_id)
            try:
                item = _find_sensor_decision_data(data, sensor_id)
            except ValueError as exc:
                if not allow_network_normal_expansion:
                    raise _ModelDecisionRejected("model_output_missing_sensor_decision") from exc
                item = None
            if item is None:
                label = "normal_relative_to_baseline"
                evidence = _string_list(network_data.get("evidence") or [])
                confidence = _clamp01(network_data.get("confidence"))
            else:
                try:
                    label = validate_sensor_label(str(item.get("local_status") or item.get("small_agent_label") or item.get("label") or ""))
                except ValueError as exc:
                    raise _ModelDecisionRejected("model_output_invalid_sensor_label") from exc
                evidence = _string_list(item.get("evidence") or item.get("main_reason") or [])
                confidence = _clamp01(item.get("confidence"))
            decision_metadata = _accepted_metadata(
                _batch_metadata(call.metadata, decision_type="sensor", sensor_id=sensor_id)
            )
            if _has_quality_conflict(
                label,
                precheck["metrics"].get("quality_flags"),
                baseline_metrics.get("quality_flags"),
                rule_label=precheck["rule_label"],
            ):
                # LLM-only policy: the model's label stands even when it
                # disagrees with the deterministic quality heuristic; the
                # disagreement is recorded instead of vetoing the decision.
                decision_metadata["quality_gate_disagreement"] = True
            sensor_decisions[sensor_id] = {
                "label": label,
                # LLM-only policy: the confidence is the model's own number,
                # reported as-is, never substituted from the rule precheck. The
                # prompt carries no numeric anchor and the response grammar
                # forces a generated JSON number in this slot. The per-sensor
                # evidence stays measurement text because the prompt
                # deliberately does not request it.
                "confidence": confidence,
                "evidence": evidence or precheck["rule_evidence"],
                "model_metadata": decision_metadata,
            }
        final_reports = [
            {
                "sensor_id": sensor_id,
                "small_agent_label": decision["label"],
                "quality_flags": _dedupe_strings(
                    [*precheck["metrics"].get("quality_flags", []), *baseline_metrics.get("quality_flags", [])]
                ),
            }
            for precheck in sensor_prechecks
            for sensor_id, decision in [(str(precheck["target"].sensor_id), sensor_decisions[str(precheck["target"].sensor_id)])]
        ]
        network_decision_metadata = _accepted_metadata(_batch_metadata(call.metadata, decision_type="network"))
        if network_label != model_network_label_raw:
            # Fix 1 (FIXES_TODO_20260709): network verdict re-derived from the
            # model's own sensor_reports; the raw head output is kept for audit.
            network_decision_metadata["network_status_derived_from_sensor_reports"] = True
            network_decision_metadata["model_network_status_raw"] = model_network_label_raw
        if _network_quality_conflict(network_label, final_reports, baseline_metrics, rule_label=network_rule_label):
            # LLM-only policy: record the disagreement, keep the model's label.
            network_decision_metadata["quality_gate_disagreement"] = True
        network_evidence = _string_list(network_data.get("evidence") or [])
        model_explanation = _safe_text(network_data.get("explanation"))
        if not model_explanation:
            raise _ModelDecisionRejected("model_output_rejected_for_missing_explanation")
        network_decision = {
            "label": network_label,
            # LLM-only policy: the model's own confidence, reported as-is (see
            # the sensor-decision note above).
            "confidence": _clamp01(network_data.get("confidence")),
            # The model's affected list stands (filtered to real sensor ids so a
            # hallucinated id cannot enter the report); an empty list is the
            # model's statement, not an omission to be backfilled from rules.
            "affected_sensor_ids": _sanitize_affected_ids(network_affected_value, final_reports, []),
            "evidence": network_evidence or network_rule_evidence,
            "explanation": model_explanation,
            "explanation_source": "model",
            "model_metadata": network_decision_metadata,
        }
        return {"sensor_decisions": sensor_decisions, "network_decision": network_decision}
    except _ModelDecisionRejected as exc:
        _log_rejected_model_output(exc.reason, call)
        raise ModelDecisionUnavailableError(exc.reason, _rejected_metadata(batch_metadata, exc.reason)) from exc
    except ModelDecisionUnavailableError:
        raise
    except Exception as exc:
        reason = "model_output_rejected_for_invalid_or_unsafe_json"
        _log_rejected_model_output(reason, call)
        raise ModelDecisionUnavailableError(reason, _rejected_metadata(batch_metadata, reason)) from exc


def _log_rejected_model_output(reason: str, call: ModelCallResult) -> None:
    """Log the rejected raw model text to the server log for diagnosis.

    The text goes to stderr only (webchat/service log) — rejected model output
    is never placed in the chat-facing result or error message.
    """
    text = " ".join(str(call.text or "").split())
    if len(text) > 2000:
        text = text[:2000] + "…"
    print(f"[vibroagent] model output rejected ({reason}): {text or '<no text>'}", file=sys.stderr)


def _fallback_all_sensor_decisions(
    *,
    sensor_prechecks: list[dict[str, Any]],
    network_rule_label: str,
    network_rule_confidence: float,
    affected_sensor_ids: list[str],
    network_rule_evidence: list[str],
    network_rule_explanation: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sensor_decisions": {
            str(precheck["target"].sensor_id): _fallback_sensor_decision(
                precheck["rule_label"],
                precheck["rule_confidence"],
                precheck["rule_evidence"],
                metadata,
            )
            for precheck in sensor_prechecks
        },
        "network_decision": _fallback_network_decision(
            network_rule_label,
            network_rule_confidence,
            affected_sensor_ids,
            network_rule_evidence,
            network_rule_explanation,
            metadata,
        ),
    }


def _all_sensor_response_format(target_ids: list[str]) -> dict[str, Any]:
    """response_format for the batch decision, grammar-enforced on the GenieX shim.

    Forces every confidence to be a model-generated JSON number (no rule value can leak
    in) and pins sensor ids/labels to the real sets. Adapters without json_schema
    support ignore the field and fall back to prompt-guided JSON.
    """
    sensor_id_schema: dict[str, Any] = {"enum": sorted(target_ids)} if target_ids else {"type": "string"}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "all_sensor_decision",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sensor_reports": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sensor_id": sensor_id_schema,
                                    "local_status": {"enum": sorted(SENSOR_LABELS)},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["sensor_id", "local_status", "confidence"],
                            },
                        },
                        "network_status": {"enum": sorted(NETWORK_LABELS)},
                        "confidence": {"type": "number"},
                        "affected_sensor_ids": {"type": "array", "items": sensor_id_schema},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "sensor_reports",
                        "network_status",
                        "confidence",
                        "affected_sensor_ids",
                        "evidence",
                        "explanation",
                    ],
                },
            },
        }
    }


def _build_all_sensor_prompt(
    *,
    baseline_metrics: dict[str, Any],
    sensor_prechecks: list[dict[str, Any]],
    network_rule_label: str,
    network_rule_confidence: float,
    affected_sensor_ids: list[str],
    network_rule_evidence: list[str],
    network_rule_explanation: str,
) -> str:
    target_ids = [str(precheck["target"].sensor_id) for precheck in sensor_prechecks]
    requested_duration_s = _round_prompt(sensor_prechecks[0]["duration_s"]) if sensor_prechecks else None
    observed_durations = [
        _optional_float(precheck["metrics"].get("duration_s"))
        for precheck in sensor_prechecks
    ]
    observed_durations = [value for value in observed_durations if value is not None]
    payload = {
        "ref": _compact_metric_row(baseline_metrics),
        "targets": [_compact_precheck_for_prompt(precheck) for precheck in sensor_prechecks],
        "duration_check": {
            "requested_s": requested_duration_s,
            "ref_s": _round_prompt(baseline_metrics.get("duration_s")),
            "target_min_s": _round_prompt(min(observed_durations)) if observed_durations else None,
            "target_max_s": _round_prompt(max(observed_durations)) if observed_durations else None,
        },
        "net_rule": {
            "label": network_rule_label,
            "conf": _round_prompt(network_rule_confidence),
            "affected": affected_sensor_ids,
        },
    }
    # The confidence values must be the model's own numbers. The example deliberately
    # carries NO numeric anchor (greedy decoding echoes any literal number verbatim);
    # on the GenieX shim the grammar from _all_sensor_response_format forces a JSON
    # number in each slot, so the model has to generate its own value there.
    schema = {
        "sensor_reports": [
            {"sensor_id": sensor_id, "local_status": "<sensor_label>", "confidence": "<your own 0-1 estimate>"}
            for sensor_id in target_ids
        ],
        "network_status": "<network_label>",
        "confidence": "<your own 0-1 estimate>",
        "affected_sensor_ids": [],
        "evidence": ["short reason"],
        "explanation": "two complete operator sentences that mention duration, finding, affected sensors, and next monitoring action",
    }
    return (
        "Autonomous building-vibration check. Raw vibration sample arrays are not provided. "
        "Use only this compact relative metrics JSON. Discuss only building response, sensor quality, and relative deviations. No certified structural safety claims. "
        "Return one compact strict JSON object only, matching this shape and every target id exactly. "
        "For each sensor_reports item, output only sensor_id, local_status, and confidence; do not include per-sensor evidence arrays. "
        "Set every confidence to your own 0-to-1 estimate of that decision judged from the relative metrics; vary it with evidence strength and never copy a number from this prompt. "
        "Use at most two network evidence strings. Do not copy every ratio into the response. "
        "Write explanation as exactly two complete sentences, 30-75 words total. Sentence 1 must mention the observed duration from duration_check and the building-level finding relative to the reference sensor. "
        "Sentence 2 must mention affected target sensors and the next monitoring step. For normal windows, include the exact phrase no affected target sensors were identified. "
        "Use reference sensor in prose, not baseline. Stop immediately after the closing brace: "
        f"{json.dumps(schema, separators=(',', ':'), sort_keys=True)}\n"
        f"sensor_labels={','.join(sorted(SENSOR_LABELS))}\n"
        f"network_labels={','.join(sorted(NETWORK_LABELS))}\n"
        f"data={json.dumps(payload, separators=(',', ':'), sort_keys=True)}"
    )


def _compact_precheck_for_prompt(precheck: dict[str, Any]) -> dict[str, Any]:
    target = precheck["target"]
    metrics = precheck["metrics"]
    comparison = precheck["comparison"]
    return {
        "id": target.sensor_id,
        "loc": target.location,
        "q": _dedupe_strings(metrics.get("quality_flags")),
        "m": _compact_metric_row(metrics),
        "rel": {
            "rms": _round_prompt(comparison.get("rms_ratio")),
            "peak": _round_prompt(comparison.get("peak_ratio")),
            "ppv": _round_prompt(comparison.get("ppv_ratio")),
            "spec": _round_prompt(comparison.get("spectral_distance")),
            "df": _round_prompt(comparison.get("dominant_frequency_delta_hz")),
        },
        "rule": precheck["rule_label"],
    }


def _compact_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": metrics.get("sensor_id"),
        "n": int(metrics.get("sample_count") or 0),
        "rms": _round_prompt(metrics.get("acceleration_rms_g")),
        "peak": _round_prompt(metrics.get("acceleration_peak_g")),
        "ppv": _round_prompt(metrics.get("estimated_ppv_mm_s")),
        "freq": _round_prompt(metrics.get("dominant_frequency_hz")),
        "crest": _round_prompt(metrics.get("crest_factor")),
        "kurt": _round_prompt(metrics.get("kurtosis")),
        "q": _dedupe_strings(metrics.get("quality_flags")),
    }


def _round_prompt(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    return round(float(number), 4)

def _find_sensor_decision_data(data: dict[str, Any], sensor_id: str) -> dict[str, Any]:
    for item in _sensor_decision_items(data):
        if str(item.get("sensor_id") or "") == sensor_id:
            return item
    raise ValueError(f"model response missing decision for sensor_id {sensor_id!r}")


def _sensor_decision_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("sensor_reports") or data.get("sensor_decisions") or data.get("sensors")
    if isinstance(value, dict):
        items: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, dict):
                items.append({"sensor_id": str(item.get("sensor_id") or key), **item})
        return items
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, dict)]
    return []


def _network_data_from_batch(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    nested = data.get("network_assessment") or data.get("network")
    if isinstance(nested, dict):
        result.update(nested)
    return result


def _batch_metadata(metadata: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        **metadata,
        "role": metadata.get("role") or "autonomous_all_sensor_agent",
        "decision_scope": "all_sensor_batch",
        **extra,
    }


def _load_replay_window(
    *,
    path: Path,
    axis: str,
    start_time_s: float | None,
    duration_s: float,
    configured_sampling_rate_hz: int | None,
) -> tuple[np.ndarray, np.ndarray | None, int, dict[str, Any]]:
    signal, timestamps_s, source_columns = _load_replay_signal(path, axis)
    sampling_rate_hz = configured_sampling_rate_hz or _sampling_rate_from_timestamps(timestamps_s)
    if sampling_rate_hz is None:
        raise ValueError(f"Sensor replay file {path} needs sampling_rate_hz in the registry or a usable time column")
    signal, timestamps_s, read_start = _trim_replay_signal(
        signal=signal,
        timestamps_s=timestamps_s,
        sampling_rate_hz=sampling_rate_hz,
        start_time_s=start_time_s,
        duration_s=duration_s,
    )
    if signal.size < 32:
        raise ValueError(f"Replay window has only {signal.size} samples")
    metadata = {
        "source": "replay_signal_file",
        "replay_signal_path": str(path),
        "axis": axis,
        "source_columns": source_columns,
        "read_mode": "fixed_start_time" if start_time_s is not None else "latest",
        "start_time_s": _json_float(read_start),
        "requested_start_time_s": _json_float(start_time_s),
        "requested_duration_s": float(duration_s),
        "sample_count": int(signal.size),
        "sampling_rate_hz": int(sampling_rate_hz),
    }
    return signal, timestamps_s, int(sampling_rate_hz), metadata


def _load_replay_signal(path: Path, axis: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Replay signal file does not exist: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64)
        return _select_from_array(arr, axis=axis, names=None, time_values=None)

    delimiter = "," if path.suffix.lower() == ".csv" else None
    first_line = _first_data_line(path)
    has_header = bool(first_line and re.search(r"[A-Za-z_]", first_line))
    if has_header:
        data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float64, encoding="utf-8")
        if data.shape == ():
            data = np.asarray([data], dtype=data.dtype)
        names = [str(name) for name in (data.dtype.names or [])]
        columns = {name: np.asarray(data[name], dtype=np.float64).reshape(-1) for name in names}
        time_name = _time_column_name(names)
        time_values = columns.pop(time_name) if time_name else None
        arr = np.column_stack([columns[name] for name in columns]) if columns else np.empty((0, 0))
        return _select_from_array(arr, axis=axis, names=list(columns), time_values=time_values)

    arr = np.asarray(np.loadtxt(path, delimiter=delimiter), dtype=np.float64)
    return _select_from_array(arr, axis=axis, names=None, time_values=None)


def _select_from_array(
    arr: np.ndarray,
    *,
    axis: str,
    names: list[str] | None,
    time_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        signal = data
        columns = names or ["value"]
    elif data.ndim == 2:
        if data.shape[1] == 0:
            raise ValueError("Replay signal has no data columns")
        if time_values is None and data.shape[1] >= 2 and np.all(np.diff(data[:, 0]) >= 0):
            time_values = data[:, 0]
            data = data[:, 1:]
            if names:
                names = names[1:]
        columns = names or [str(index) for index in range(data.shape[1])]
        signal = _select_axis_from_columns(data, columns, axis)
    else:
        raise ValueError("Replay signal must be a 1D or 2D array")
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(signal)
    if time_values is not None:
        time_values = np.asarray(time_values, dtype=np.float64).reshape(-1)
        finite = finite & np.isfinite(time_values)
        time_values = time_values[finite]
    return signal[finite], time_values, list(columns)


def _select_axis_from_columns(data: np.ndarray, columns: list[str], axis: str) -> np.ndarray:
    normalized = {name.lower(): index for index, name in enumerate(columns)}
    axis = axis.lower()
    if axis in {"norm", "magnitude", "vector_norm"}:
        xyz = [normalized.get(name) for name in ("x", "y", "z")]
        if all(index is not None for index in xyz):
            return np.linalg.norm(data[:, [int(index) for index in xyz]], axis=1)
        if data.shape[1] > 1:
            return np.linalg.norm(data, axis=1)
        return data[:, 0]
    if axis in normalized:
        return data[:, normalized[axis]]
    if axis in {"x", "y", "z"}:
        index = {"x": 0, "y": 1, "z": 2}[axis]
        if index < data.shape[1]:
            return data[:, index]
    if axis in {"0", "1", "2"}:
        index = int(axis)
        if index < data.shape[1]:
            return data[:, index]
    if data.shape[1] == 1:
        return data[:, 0]
    raise ValueError(f"Axis {axis!r} is not available in replay columns {columns}")


def _trim_replay_signal(
    *,
    signal: np.ndarray,
    timestamps_s: np.ndarray | None,
    sampling_rate_hz: int,
    start_time_s: float | None,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    requested_samples = max(1, int(round(float(duration_s) * float(sampling_rate_hz))))
    if timestamps_s is not None and timestamps_s.size == signal.size:
        if start_time_s is None:
            start = max(float(timestamps_s[0]), float(timestamps_s[-1]) - float(duration_s))
        else:
            start = float(start_time_s)
        end = start + float(duration_s)
        mask = (timestamps_s >= start) & (timestamps_s < end)
        return signal[mask], timestamps_s[mask], start

    if start_time_s is None:
        start_index = max(0, signal.size - requested_samples)
    else:
        start_index = max(0, int(round(float(start_time_s) * float(sampling_rate_hz))))
    end_index = min(signal.size, start_index + requested_samples)
    timestamps = np.arange(start_index, end_index, dtype=np.float64) / float(sampling_rate_hz)
    return signal[start_index:end_index], timestamps, float(start_index / float(sampling_rate_hz))


def _relative_band_powers(signal_centered: np.ndarray, sampling_rate_hz: int) -> dict[str, float]:
    x = np.asarray(signal_centered, dtype=np.float64).reshape(-1)
    if x.size < 32 or sampling_rate_hz <= 0:
        return {}
    spectrum = np.fft.rfft(x * np.hanning(x.size))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sampling_rate_hz))
    power = np.abs(spectrum) ** 2
    if power.size:
        power[0] = 0.0
    total = float(np.sum(power) + EPS)
    nyquist = float(sampling_rate_hz) / 2.0
    bands = [
        ("0_5_hz", 0.0, min(5.0, nyquist)),
        ("5_20_hz", 5.0, min(20.0, nyquist)),
        ("20_50_hz", 20.0, min(50.0, nyquist)),
        ("50_100_hz", 50.0, min(100.0, nyquist)),
        ("100_200_hz", 100.0, min(200.0, nyquist)),
        ("200_plus_hz", 200.0, nyquist + EPS),
    ]
    result: dict[str, float] = {}
    for key, lo, hi in bands:
        if hi <= lo:
            continue
        mask = (freqs >= lo) & (freqs < hi)
        result[key] = float(np.sum(power[mask]) / total)
    return result


def _failure_metrics(*, window_id: str, failure: _ReadFailure) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_domain": ANALYSIS_DOMAIN,
        "window_id": window_id,
        "sensor_id": failure.config.sensor_id,
        "location": failure.config.location,
        "role": failure.config.role,
        "axis": failure.axis,
        "sampling_rate_hz": int(failure.config.sampling_rate_hz or 0),
        "duration_s": float(failure.duration_s),
        "sample_count": 0,
        "acceleration_rms_g": None,
        "acceleration_peak_g": None,
        "acceleration_peak_to_peak_g": None,
        "crest_factor": None,
        "kurtosis": None,
        "estimated_ppv_mm_s": None,
        "dominant_frequency_hz": None,
        "band_powers": {},
        "quality_flags": _dedupe_strings(failure.quality_flags),
        "source_metadata": _compact_metadata(failure.metadata),
        "read_error": failure.message,
    }


def _quality_flags_for_signal(signal: np.ndarray, sampling_rate_hz: int, duration_s: float) -> list[str]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    flags: list[str] = []
    if x.size < 32:
        flags.append("insufficient_samples")
    expected = int(round(float(duration_s) * max(1, int(sampling_rate_hz))))
    if expected >= 64 and x.size < int(expected * 0.75):
        flags.append("short_window")
    if x.size and float(np.std(x)) < 1e-10:
        flags.append("flatline")
    if x.size and not np.all(np.isfinite(x)):
        flags.append("non_finite_values")
    return _dedupe_strings(flags)


def _estimated_ppv_mm_s(acceleration_rms_g: Any, dominant_frequency_hz: Any) -> float | None:
    acceleration = _optional_float(acceleration_rms_g)
    frequency = _optional_float(dominant_frequency_hz)
    if acceleration is None or frequency is None or frequency <= 0.2:
        return None
    return _json_float((acceleration * 9.80665 / (2.0 * math.pi * frequency)) * 1000.0)


def _call_model(
    client: Any,
    prompt: str,
    allowed_labels: set[str],
    extra_body: dict[str, Any] | None = None,
) -> ModelCallResult:
    if client is None:
        return ModelCallResult(
            ok=False,
            used_model=False,
            data=None,
            text=None,
            error="endpoint_not_configured",
            metadata={
                "role": "autonomous_llm_agent",
                "model_used": False,
                "provider": "deterministic_fallback",
                "model": None,
                "endpoint_configured": False,
                "fallback_reason": "endpoint_not_configured",
            },
        )
    return client.call_json_model(prompt, allowed_labels=allowed_labels, extra_body=extra_body)


def _fallback_sensor_decision(
    label: str,
    confidence: float,
    evidence: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "confidence": confidence,
        "evidence": evidence,
        "model_metadata": _fallback_metadata(metadata),
    }


def _fallback_network_decision(
    label: str,
    confidence: float,
    affected_sensor_ids: list[str],
    evidence: list[str],
    explanation: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "confidence": confidence,
        "affected_sensor_ids": affected_sensor_ids,
        "evidence": evidence,
        "explanation": explanation,
        "explanation_source": "deterministic_fallback",
        "model_metadata": _fallback_metadata(metadata),
    }


def _default_model_client(
    *,
    role: str,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    timeout_s: float | None,
    base_url_env: str,
    api_key_env: str,
    model_env: str,
) -> ModelClient:
    return ModelClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        model_env=model_env,
        role=role,
        timeout_s=timeout_s,
    )


def _accepted_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        **metadata,
        "model_used": bool(metadata.get("model_used")),
        "fallback_reason": None,
        "decision_source": "model",
    }


def _fallback_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    reason = metadata.get("fallback_reason") or metadata.get("error") or "deterministic_rule_fallback"
    return {
        **metadata,
        "model_used": False,
        "fallback_reason": reason,
        "decision_source": "deterministic_rules",
    }


def _rejected_metadata(metadata: dict[str, Any], reason: str) -> dict[str, Any]:
    return {**metadata, "model_used": False, "fallback_reason": reason, "decision_source": "none"}


def _aggregate_model_metadata(reports: list[dict[str, Any]], network: dict[str, Any]) -> dict[str, Any]:
    metadatas = [report.get("model_metadata", {}) for report in reports]
    metadatas.append(network.get("model_metadata", {}))
    fallbacks = _dedupe_strings(
        [
            str(metadata.get("fallback_reason"))
            for metadata in metadatas
            if metadata.get("fallback_reason")
        ]
    )
    return {
        "raw_samples_sent_to_models": False,
        "model_used": any(bool(metadata.get("model_used")) for metadata in metadatas),
        "fallbacks_used": fallbacks,
        "sensor_model_metadata": [report.get("model_metadata", {}) for report in reports],
        "network_model_metadata": network.get("model_metadata", {}),
    }


def _has_quality_conflict(label: str, sensor_flags: Any, baseline_flags: Any, *, rule_label: str | None = None) -> bool:
    flags = set(sensor_flags or []) | set(baseline_flags or [])
    if flags and label not in QUALITY_SENSOR_LABELS:
        return True
    return bool(rule_label and rule_label not in QUALITY_SENSOR_LABELS and label in QUALITY_SENSOR_LABELS)


def _network_quality_conflict(
    label: str,
    reports: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    *,
    rule_label: str | None = None,
) -> bool:
    has_quality = bool(baseline_metrics.get("quality_flags")) or any(report.get("quality_flags") for report in reports)
    if has_quality and label not in QUALITY_NETWORK_LABELS:
        return True
    return bool(rule_label and rule_label not in QUALITY_NETWORK_LABELS and label in QUALITY_NETWORK_LABELS)


def _sanitize_affected_ids(value: Any, reports: list[dict[str, Any]], fallback: list[str]) -> list[str]:
    available = {str(report.get("sensor_id")) for report in reports}
    selected = [item for item in _string_list(value) if item in available]
    return _dedupe_strings(selected or fallback)


def _first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def _time_column_name(names: list[str]) -> str | None:
    for name in names:
        if name.lower() in {"time", "time_s", "timestamp", "timestamp_s", "t"}:
            return name
    return None


def _sampling_rate_from_timestamps(timestamps_s: np.ndarray | None) -> int | None:
    if timestamps_s is None or timestamps_s.size < 2:
        return None
    diffs = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    sr = round(1.0 / float(np.median(diffs)))
    return int(sr) if sr > 0 else None


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "source",
        "mode",
        "read_mode",
        "replay_signal_path",
        "acquisition_folder",
        "sensor_name",
        "axis",
        "sample_count",
        "sampling_rate_hz",
        "start_time_s",
        "requested_start_time_s",
        "requested_duration_s",
        "data_is_current",
        "data_current_warning",
        "read_error",
        "quality_flags",
    }
    return {key: value for key, value in metadata.items() if key in keep}


def _dedupe_sensor_configs(configs: list[SensorConfig]) -> list[SensorConfig]:
    seen: set[str] = set()
    result: list[SensorConfig] = []
    for config in configs:
        if config.sensor_id in seen:
            continue
        seen.add(config.sensor_id)
        result.append(config)
    return result


def _dedupe_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_optional_axis(axis: str | None) -> str | None:
    if axis is None:
        return None
    value = str(axis).strip().lower()
    if value not in VALID_AXES:
        raise ValueError("axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2")
    return value


def _validate_window(*, start_time_s: float | None, duration_s: float, axis: str | None, mode: str) -> None:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if duration_s > 300:
        raise ValueError("duration_s is capped at 300 seconds")
    if start_time_s is not None and start_time_s < 0:
        raise ValueError("start_time_s must be non-negative")
    if mode not in {"live", "replay"}:
        raise ValueError("mode must be live or replay")
    if axis is not None and axis not in VALID_AXES:
        raise ValueError("axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2")


def _new_window_id() -> str:
    return "building_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _bounded_ratio(value: Any, baseline: Any) -> float | None:
    numerator = _optional_float(value)
    denominator = _optional_float(baseline)
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < EPS:
        return 999.0 if abs(numerator) >= EPS else 1.0
    return _json_float(min(999.0, max(0.0, numerator / denominator)))


def _delta(value: Any, baseline: Any) -> float | None:
    left = _optional_float(value)
    right = _optional_float(baseline)
    if left is None or right is None:
        return None
    return _json_float(left - right)


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _json_float(value: Any) -> float | None:
    result = _optional_float(value)
    return None if result is None else float(result)


def _num(value: Any, *, default: float) -> float:
    result = _optional_float(value)
    return float(default if result is None else result)


def _clamp01(value: Any, default: float = 0.0) -> float:
    result = _optional_float(value)
    if result is None:
        result = float(default)
    return max(0.0, min(1.0, float(result)))


