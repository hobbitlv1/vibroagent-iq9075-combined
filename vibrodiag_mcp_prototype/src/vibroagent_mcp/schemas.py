"""Strict schemas and label guards for building vibration monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

SCHEMA_VERSION = "0.5.0"
ANALYSIS_DOMAIN = "building_vibration_relative_monitoring"

SENSOR_LABELS = {
    "normal_relative_to_baseline",
    "mild_deviation",
    "significant_local_deviation",
    "impulsive_or_shock_event",
    "possible_sensor_issue",
    "insufficient_data",
}

NETWORK_LABELS = {
    "normal_relative_to_reference_sensor",
    "mild_local_deviation",
    "mild_multi_sensor_deviation",
    "localized_vibration_deviation",
    "multi_sensor_structure_wide_vibration_event",
    "sensor_network_quality_issue",
    "insufficient_data",
}

OUT_OF_DOMAIN_COMPONENT_TERMS = {
    "bearing fault",
    "bearing defect",
    "bearing damage",
    "bearing failure",
    "bearing wear",
    "bearing diagnosis",
    "bearing condition",
    "normal bearing",
    "inner race",
    "outer race",
    "rolling element",
    "ball fault",
    "cage fault",
    "bpfo",
    "bpfi",
    "ball spin frequency",
    "fundamental train frequency",
    "gearbox fault",
    "shaft fault",
    "rotor fault",
    "motor fault",
}

SensorRole = Literal["baseline", "target"]


@dataclass(frozen=True)
class WindowSpec:
    schema_version: str
    analysis_domain: str
    window_id: str
    start_time_s: float | None
    duration_s: float
    mode: str
    require_current: bool


@dataclass(frozen=True)
class SensorConfig:
    sensor_id: str
    location: str
    role: SensorRole
    acquisition_folder: str | None = None
    hsd_sensor_name: str = "iis3dwb_acc"
    axis: str = "z"
    replay_signal_path: str | None = None
    sampling_rate_hz: int | None = None


@dataclass(frozen=True)
class ModelMetadata:
    model_used: bool
    provider: str
    model: str | None
    endpoint_configured: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class BuildingMetrics:
    schema_version: str
    analysis_domain: str
    window_id: str
    sensor_id: str
    location: str
    role: str
    axis: str
    sampling_rate_hz: int
    duration_s: float
    sample_count: int
    acceleration_rms_g: float | None
    acceleration_peak_g: float | None
    acceleration_peak_to_peak_g: float | None
    crest_factor: float | None
    kurtosis: float | None
    estimated_ppv_mm_s: float | None
    dominant_frequency_hz: float | None
    band_powers: dict[str, float]
    quality_flags: list[str]


@dataclass(frozen=True)
class SensorAgentReport:
    schema_version: str
    analysis_domain: str
    window_id: str
    sensor_id: str
    baseline_sensor_id: str
    location: str
    role: str
    axis: str
    sampling_rate_hz: int
    duration_s: float
    quality_flags: list[str]
    python_rule_label: str
    small_agent_label: str
    small_agent_confidence: float
    evidence: list[str]
    model_metadata: dict[str, Any]
    metrics: dict[str, Any]
    baseline_comparison: dict[str, Any]


@dataclass(frozen=True)
class NetworkAssessment:
    schema_version: str
    analysis_domain: str
    window_id: str
    baseline_sensor_id: str
    python_rule_label: str
    main_agent_label: str
    main_agent_confidence: float
    quality_flags: list[str]
    affected_sensor_ids: list[str]
    evidence: list[str]
    explanation: str
    model_metadata: dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    schema_version: str
    analysis_domain: str
    window_id: str
    baseline_sensor_id: str
    mode: str
    window: dict[str, Any]
    baseline_metrics: dict[str, Any]
    sensor_metrics: list[dict[str, Any]]
    sensor_agent_reports: list[dict[str, Any]]
    network_assessment: dict[str, Any]
    main_agent_explanation: str
    model_metadata: dict[str, Any]


def validate_sensor_label(label: str) -> str:
    if label not in SENSOR_LABELS:
        raise ValueError(f"Unsupported building sensor label: {label!r}")
    return label


def validate_network_label(label: str) -> str:
    if label not in NETWORK_LABELS:
        raise ValueError(f"Unsupported building network label: {label!r}")
    return label


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_dict(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain_dict(item) for item in value]
    return value


def find_out_of_domain_component_terms(obj: Any) -> list[str]:
    """Return component-specific terms that do not belong in building reports.

    This guard is intentionally separate from prompts and tool descriptions. It
    exists only to reject out-of-domain model output before it reaches the webchat.
    """
    offenders: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            # Also match snake_case / hyphenated variants ("bearing_diagnosis",
            # "inner-race"): models trained on legacy machine-readable labels
            # emit them as single tokens, which the spaced phrases alone miss.
            lower = value.casefold().replace("_", " ").replace("-", " ")
            for phrase in OUT_OF_DOMAIN_COMPONENT_TERMS:
                if phrase in lower:
                    offenders.append(phrase)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(str(key))
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
        elif is_dataclass(value):
            walk(asdict(value))

    walk(obj)
    return sorted(set(offenders))


def redact_out_of_domain_component_text(
    obj: Any,
    *,
    replacement: str = "redacted_for_building_monitoring",
) -> Any:
    """Return a copy with out-of-domain component text replaced.

    The matched vocabulary is never returned to callers. This is used on
    response envelopes so a rejected model phrase cannot leak through raw
    tool results or error metadata even when the human-facing answer is safe.
    """
    if isinstance(obj, str):
        return replacement if find_out_of_domain_component_terms(obj) else obj
    if isinstance(obj, dict):
        return {
            str(redact_out_of_domain_component_text(key, replacement=replacement)):
                redact_out_of_domain_component_text(value, replacement=replacement)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact_out_of_domain_component_text(item, replacement=replacement) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact_out_of_domain_component_text(item, replacement=replacement) for item in obj)
    if isinstance(obj, set):
        return {redact_out_of_domain_component_text(item, replacement=replacement) for item in obj}
    if is_dataclass(obj):
        return redact_out_of_domain_component_text(asdict(obj), replacement=replacement)
    return obj


def ensure_building_domain_only(obj: Any) -> None:
    """Reject component-specific diagnoses in building-monitoring output."""
    offenders = find_out_of_domain_component_terms(obj)
    if offenders:
        raise ValueError(
            "Building-monitoring payload contains out-of-domain component terminology."
        )
