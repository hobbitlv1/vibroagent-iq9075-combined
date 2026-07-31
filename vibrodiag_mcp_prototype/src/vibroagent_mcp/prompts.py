"""Prompt builders for building-vibration logical agents."""

from __future__ import annotations

import json
from typing import Any

from .schemas import NETWORK_LABELS, SENSOR_LABELS


def build_small_sensor_agent_prompt(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    payload = {
        "sensor_metrics": _compact_metrics(metrics),
        "baseline_reference_metrics": _compact_metrics(baseline_metrics),
        "baseline_comparison": comparison,
    }
    return (
        "You are a small logical agent for building vibration relative monitoring.\n"
        "Discuss only relative building vibration, sensor quality, transient events, and monitoring actions.\n"
        "Use only the compact numeric metrics below. raw vibration sample arrays are not provided and must not be requested.\n"
        "Use confidence > 0 for every chosen label; only report insufficient_data when quality_flags show missing, stale, or too few samples.\n"
        "Return strict JSON only, with one local_status from this fixed set: "
        f"{sorted(SENSOR_LABELS)}.\n"
        "Output schema:\n"
        "{\n"
        '  "sensor_id": "...",\n'
        '  "local_status": "normal_relative_to_baseline | mild_deviation | significant_local_deviation | impulsive_or_shock_event | possible_sensor_issue | insufficient_data",\n'
        '  "confidence": 0.85,\n'
        '  "main_reason": "...",\n'
        '  "evidence": ["..."],\n'
        '  "quality_flags": [],\n'
        '  "send_to_main_agent": true\n'
        "}\n"
        "Metrics:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_main_building_agent_prompt(
    sensor_reports: list[dict[str, Any]],
    network_assessment: dict[str, Any],
) -> str:
    payload = {
        "sensor_reports": sensor_reports,
        "python_network_assessment": network_assessment,
    }
    return (
        "You are the main fusion agent for building vibration relative monitoring.\n"
        "Compare target sensor reports against the calm/reference baseline sensor.\n"
        "Classify whether the result is normal, local deviation, multi-sensor deviation, structure-wide event across the monitored sensor network, sensor quality issue, or insufficient data.\n"
        "Do not make certified structural-safety claims. State that this is monitoring/triage, not a certified structural safety assessment.\n"
        "Keep every finding within the supplied building-sensor label vocabulary.\n"
        "Use confidence > 0 for every chosen label; only report insufficient_data or sensor_network_quality_issue when the provided reports include quality flags.\n"
        "Return strict JSON only, with network_status from this fixed set: "
        f"{sorted(NETWORK_LABELS)}.\n"
        "Output schema:\n"
        "{\n"
        '  "network_status": "normal_relative_to_reference_sensor | mild_local_deviation | mild_multi_sensor_deviation | localized_vibration_deviation | multi_sensor_structure_wide_vibration_event | sensor_network_quality_issue | insufficient_data",\n'
        '  "confidence": 0.85,\n'
        '  "summary": "...",\n'
        '  "evidence": ["..."],\n'
        '  "recommended_action": "..."\n'
        "}\n"
        "Reports:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "schema_version",
        "analysis_domain",
        "window_id",
        "sensor_id",
        "location",
        "role",
        "axis",
        "sampling_rate_hz",
        "duration_s",
        "sample_count",
        "acceleration_rms_g",
        "acceleration_peak_g",
        "acceleration_peak_to_peak_g",
        "crest_factor",
        "kurtosis",
        "estimated_ppv_mm_s",
        "dominant_frequency_hz",
        "band_powers",
        "quality_flags",
    }
    return {key: metrics.get(key) for key in sorted(allowed_keys) if key in metrics}


def build_autonomous_sensor_agent_messages(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison: dict[str, Any],
    python_label: str,
    evidence: list[str],
) -> tuple[str, str]:
    """Build a compact tool-loop prompt for one target sensor.

    The local Qualcomm Genie adapter does not consume OpenAI tool schemas, so
    the system instruction also defines the project's text tool-call protocol.
    """

    system = (
        "You are one bounded building-vibration sensor agent. This is relative monitoring against a reference "
        "sensor. The task is relative building monitoring and not a structural-safety certification. Use only supplied "
        "compact metrics; never request or invent raw samples. Call exactly one tool per turn. Usually call "
        "finalize_sensor immediately. Use compare_axis only when the evidence is genuinely ambiguous. The final "
        f"local_status must be one of {sorted(SENSOR_LABELS)}. Quality-gated deterministic labels "
        "(insufficient_data or possible_sensor_issue) may not be overridden without valid fresh evidence. "
        "For endpoints without native tool calls, output only this syntax and no prose: "
        "<tool_call><function=TOOL><parameter=name>value</parameter></function></tool_call>."
    )
    payload = {
        "deterministic_label": python_label,
        "deterministic_evidence": [str(item) for item in evidence[:8]],
        "sensor": _compact_metrics(metrics),
        "reference": _compact_metrics(baseline_metrics),
        "comparison": _compact_json(comparison),
    }
    user = (
        "Decide this sensor now. If finalizing, include local_status, confidence, main_reason, and a short evidence "
        "array in finalize_sensor. Compact input:"
        + json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    )
    return system, user


def build_autonomous_orchestrator_messages(
    sensor_reports: list[dict[str, Any]],
    python_label: str,
    network_seed: dict[str, Any],
) -> tuple[str, str]:
    """Build a compact tool-loop prompt for network-level fusion."""

    system = (
        "You are the bounded fusion agent for a building vibration sensor network. Fuse target reports relative "
        "to the baseline. Keep conclusions within building-monitoring evidence and do not make certified structural-safety claims. Call exactly "
        "one tool per turn. Usually call finalize_building immediately; request_sensor_recheck is only for one "
        "borderline sensor when the supplied report is insufficient. The final network_status must be one of "
        f"{sorted(NETWORK_LABELS)}. Do not override insufficient_data or sensor_network_quality_issue unless a "
        "successful recheck resolves the quality evidence. For text-only endpoints, output only: "
        "<tool_call><function=TOOL><parameter=name>value</parameter></function></tool_call>."
    )
    payload = {
        "deterministic_network_label": python_label,
        "network_seed": _compact_json(network_seed),
        "sensor_reports": [_compact_sensor_report(item) for item in sensor_reports[:12]],
    }
    user = (
        "Fuse the network now. finalize_building should include network_status, confidence, summary, evidence, "
        "and recommended_action. Compact input:"
        + json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    )
    return system, user


def _compact_sensor_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        try:
            report = vars(report)
        except TypeError:
            return {"value": str(report)[:300]}
    allowed = (
        "sensor_id",
        "location",
        "axis",
        "small_agent_label",
        "python_rule_label",
        "small_agent_confidence",
        "main_reason",
        "evidence",
        "quality_flags",
        "baseline_comparison",
        "metrics",
    )
    compact: dict[str, Any] = {}
    for key in allowed:
        if key not in report:
            continue
        value = report[key]
        if key == "metrics" and isinstance(value, dict):
            compact[key] = _compact_metrics(value)
        else:
            compact[key] = _compact_json(value)
    return compact


def _compact_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "<nested omitted>"
    if isinstance(value, dict):
        return {
            str(key): _compact_json(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
            if str(key).lower() not in {"samples", "samples_g", "raw_signal", "waveform"}
        }
    if isinstance(value, (list, tuple)):
        compact = [_compact_json(item, depth=depth + 1) for item in value[:12]]
        if len(value) > 12:
            compact.append({"omitted_items": len(value) - 12})
        return compact
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value
