from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import atomic_write_json, utc_now_iso


def _recall(metrics: dict[str, Any], label: str) -> tuple[float | None, int]:
    labels = list(metrics.get("class_labels") or [])
    if label not in labels:
        return None, 0
    index = labels.index(label)
    matrix = list(metrics.get("class_confusion_matrix") or [])
    if index >= len(matrix):
        return None, 0
    row = list(matrix[index])
    support = int(sum(row))
    return (float(row[index]) / support if support else None), support


def _minimum_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    value: float | None,
    minimum: float,
    required: bool = True,
    note: str | None = None,
) -> None:
    observed_passed = value is not None and float(value) >= float(minimum)
    checks.append(
        {
            "name": name,
            "value": value,
            "minimum": float(minimum),
            "required": bool(required),
            "passed": bool(observed_passed) if required else True,
            "observed_passed": bool(observed_passed),
            "note": note,
        }
    )


def _maximum_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    value: float | None,
    maximum: float,
    required: bool = True,
    note: str | None = None,
) -> None:
    observed_passed = value is not None and float(value) <= float(maximum)
    checks.append(
        {
            "name": name,
            "value": value,
            "maximum": float(maximum),
            "required": bool(required),
            "passed": bool(observed_passed) if required else True,
            "observed_passed": bool(observed_passed),
            "note": note,
        }
    )


def _dataset_state_required(
    thresholds: dict[str, Any], dataset_id: str, state: str
) -> bool:
    """Return whether one dataset/state cell is estimable and therefore gated.

    Some publisher corpora have an unavoidable complete-group coverage limitation.
    The state remains reported, but a missing held-out cell is not converted into a
    fabricated example and is not made a hard failure when explicitly preregistered.
    """

    limited = dict(thresholds.get("coverage_limited_states") or {})
    values = {str(value) for value in limited.get(str(dataset_id), ())}
    return str(state) not in values


def _presence_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    present: bool,
    required: bool = True,
    note: str | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "value": bool(present),
            "expected": True,
            "required": bool(required),
            "passed": bool(present) if required else True,
            "observed_passed": bool(present),
            "note": note,
        }
    )


def _localization_checks(
    checks: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    prefix: str = "",
    required: bool = True,
) -> None:
    """Gate target decisions only where genuine target labels exist.

    A schema with five rows is not evidence that the model localizes. These gates
    explicitly detect the all-five/all-zero collapse observed in controlled target
    perturbations. Missing target labels are never interpreted as five negatives.
    """

    localization = dict(metrics.get("localization") or {})
    supervised = int(localization.get("target_supervised_episode_count") or 0)
    single_count = int(localization.get("single_target_episode_count") or 0)
    _presence_check(
        checks,
        name=f"{prefix}target_supervision_support",
        present=supervised > 0,
        required=required,
    )
    minimum_single = int(thresholds.get("minimum_single_target_episode_count", 1))
    _presence_check(
        checks,
        name=f"{prefix}single_target_support",
        present=single_count >= minimum_single,
        required=required,
        note=f"requires at least {minimum_single} genuinely labelled single-target episodes",
    )
    _minimum_check(
        checks,
        name=f"{prefix}affected_f1",
        value=(
            float(metrics["affected_f1"])
            if metrics.get("affected_f1") is not None
            else None
        ),
        minimum=float(thresholds["overall_affected_f1"]),
        required=required,
    )
    _minimum_check(
        checks,
        name=f"{prefix}single_target_exact_set_match",
        value=localization.get("single_target_exact_set_match"),
        minimum=float(thresholds.get("minimum_single_target_exact_set_match", 0.75)),
        required=required,
    )
    per_target = dict(localization.get("per_target") or {})
    configured_targets = thresholds.get("required_targets")
    if configured_targets is not None:
        required_targets = tuple(str(value) for value in configured_targets)
        invalid_targets = sorted(
            set(required_targets).difference(
                {f"target_{index}" for index in range(1, 6)}
            )
        )
        if invalid_targets or not required_targets:
            raise ValueError(
                "required_targets must be a non-empty subset of target_1..target_5; "
                f"invalid={invalid_targets}"
            )
        for target_name in required_targets:
            values = dict(per_target.get(target_name) or {})
            support = int(values.get("support", 0) or 0)
            _presence_check(
                checks,
                name=f"{prefix}{target_name}_support",
                present=support > 0,
                required=required,
                note="dataset-specific preregistered source-zone support",
            )
            if "minimum_per_target_f1" in thresholds:
                _minimum_check(
                    checks,
                    name=f"{prefix}{target_name}_f1",
                    value=(
                        float(values.get("f1", 0.0) or 0.0)
                        if support > 0
                        else None
                    ),
                    minimum=float(thresholds["minimum_per_target_f1"]),
                    required=required,
                )
        scored_targets = {
            target_name: dict(per_target.get(target_name) or {})
            for target_name in required_targets
        }
    else:
        scored_targets = per_target
    supported_target_f1 = [
        float(values.get("f1", 0.0) or 0.0)
        for values in scored_targets.values()
        if int(values.get("support", 0) or 0) > 0
    ]
    if "minimum_supported_target_count" in thresholds:
        minimum_supported = int(thresholds["minimum_supported_target_count"])
        _presence_check(
            checks,
            name=f"{prefix}supported_target_count",
            present=len(supported_target_f1) >= minimum_supported,
            required=required,
            note=f"requires positive localization support for at least {minimum_supported} targets",
        )
    if "minimum_per_target_f1" in thresholds:
        _minimum_check(
            checks,
            name=f"{prefix}minimum_per_target_f1",
            value=min(supported_target_f1) if supported_target_f1 else None,
            minimum=float(thresholds["minimum_per_target_f1"]),
            required=required,
        )
    _maximum_check(
        checks,
        name=f"{prefix}single_target_all_five_rate",
        value=localization.get("single_target_all_five_rate"),
        maximum=float(thresholds.get("maximum_single_target_all_five_rate", 0.10)),
        required=required,
    )
    _maximum_check(
        checks,
        name=f"{prefix}single_target_all_zero_rate",
        value=localization.get("single_target_all_zero_rate"),
        maximum=float(thresholds.get("maximum_single_target_all_zero_rate", 0.10)),
        required=required,
    )
    _minimum_check(
        checks,
        name=f"{prefix}global_target_consistency_rate",
        value=metrics.get("global_target_consistency_rate"),
        minimum=float(thresholds.get("minimum_global_target_consistency_rate", 0.95)),
        required=required,
    )


def validate_operational_summary(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Gate a balanced validation probe before complete validation.

    The probe includes localization-collapse components. Passing authorizes only
    complete validation; it never authorizes the frozen test.
    """

    components = dict(summary.get("components") or {})
    per_dataset = dict(summary.get("by_dataset") or {})
    required_datasets = tuple(str(value) for value in thresholds.get("required_datasets", ()))
    if not required_datasets:
        required_datasets = tuple(sorted(per_dataset))

    checks: list[dict[str, Any]] = []
    _minimum_check(
        checks,
        name="probe_overall_macro_f1",
        value=components.get("overall_macro_f1"),
        minimum=float(thresholds["overall_macro_f1"]),
    )
    _minimum_check(
        checks,
        name="probe_overall_normal_recall",
        value=components.get("overall_normal_recall"),
        minimum=float(thresholds["overall_normal_recall"]),
    )
    _minimum_check(
        checks,
        name="probe_overall_anomaly_recall",
        value=components.get("overall_anomaly_recall"),
        minimum=float(thresholds["overall_anomaly_recall"]),
    )
    _minimum_check(
        checks,
        name="probe_overall_affected_f1",
        value=components.get("affected_f1"),
        minimum=float(thresholds["overall_affected_f1"]),
    )
    if bool(thresholds.get("require_localization_gate", False)):
        _minimum_check(
            checks,
            name="probe_single_target_exact_set_match",
            value=components.get("single_target_exact_set_match"),
            minimum=float(thresholds.get("minimum_single_target_exact_set_match", 0.60)),
        )
        _minimum_check(
            checks,
            name="probe_single_target_noncollapse",
            value=components.get("single_target_noncollapse"),
            minimum=float(thresholds.get("minimum_single_target_noncollapse", 0.80)),
        )
        if "minimum_supported_target_count" in thresholds:
            _minimum_check(
                checks,
                name="probe_supported_target_count",
                value=components.get("supported_target_count"),
                minimum=float(thresholds["minimum_supported_target_count"]),
            )
        if "minimum_per_target_f1" in thresholds:
            _minimum_check(
                checks,
                name="probe_minimum_per_target_f1",
                value=components.get("minimum_per_target_f1"),
                minimum=float(thresholds["minimum_per_target_f1"]),
            )

    for dataset_id in required_datasets:
        values = per_dataset.get(dataset_id)
        _presence_check(
            checks,
            name=f"probe_dataset:{dataset_id}:present",
            present=values is not None,
        )
        if values is None:
            continue
        _minimum_check(
            checks,
            name=f"probe_dataset:{dataset_id}:macro_f1",
            value=values.get("class_macro_f1"),
            minimum=float(thresholds["minimum_dataset_macro_f1"]),
        )
        normal_required = _dataset_state_required(thresholds, dataset_id, "normal")
        anomaly_required = _dataset_state_required(thresholds, dataset_id, "unknown_anomaly")
        _minimum_check(
            checks,
            name=f"probe_dataset:{dataset_id}:normal_recall",
            value=values.get("normal_recall"),
            minimum=float(thresholds["minimum_dataset_state_recall"]),
            required=normal_required,
            note=(
                None if normal_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )
        _minimum_check(
            checks,
            name=f"probe_dataset:{dataset_id}:anomaly_recall",
            value=values.get("anomaly_recall"),
            minimum=float(thresholds["minimum_dataset_state_recall"]),
            required=anomaly_required,
            note=(
                None if anomaly_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )

    failures = [item for item in checks if item["required"] and not item["passed"]]
    return {
        "created_utc": utc_now_iso(),
        "gate": "balanced_checkpoint_selection_preflight_v2_factorized_localization",
        "passed": not failures,
        "full_validation_authorized": not failures,
        "test_authorized": False,
        "required_datasets": list(required_datasets),
        "checks": checks,
        "required_failures": failures,
        "note": "Passing authorizes complete validation only; the frozen test still requires the complete validation and intervention gates.",
    }


def validate_selection_report_file(
    report_path: str | Path,
    *,
    output: str | Path,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    source = Path(report_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected_checkpoint = str(payload.get("selected_checkpoint", ""))
    selected = next(
        (
            candidate
            for candidate in payload.get("candidates", [])
            if str(candidate.get("checkpoint", "")) == selected_checkpoint
        ),
        None,
    )
    if selected is None:
        raise ValueError("selection report does not contain its selected checkpoint candidate")
    report = validate_operational_summary(selected, thresholds=thresholds)
    report.update(
        {
            "selection_report": str(source),
            "selected_checkpoint": selected_checkpoint,
            "selected_epoch": selected.get("epoch"),
            "selected_operational_score": selected.get("operational_score"),
        }
    )
    atomic_write_json(output, report)
    return report


def validate_metrics_gates(
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply the complete global and target-localization validation gates."""

    checks: list[dict[str, Any]] = []
    overall_normal, normal_support = _recall(metrics, "normal")
    overall_anomaly, anomaly_support = _recall(metrics, "unknown_anomaly")
    _presence_check(checks, name="overall_normal_support", present=normal_support > 0)
    _presence_check(checks, name="overall_anomaly_support", present=anomaly_support > 0)
    _minimum_check(
        checks,
        name="overall_class_macro_f1",
        value=float(metrics["class_macro_f1"]),
        minimum=float(thresholds["overall_macro_f1"]),
    )
    _minimum_check(
        checks,
        name="overall_normal_recall",
        value=overall_normal,
        minimum=float(thresholds["overall_normal_recall"]),
    )
    _minimum_check(
        checks,
        name="overall_anomaly_recall",
        value=overall_anomaly,
        minimum=float(thresholds["overall_anomaly_recall"]),
    )
    require_localization = bool(thresholds.get("require_localization_gate", False))
    if require_localization:
        _localization_checks(checks, metrics, thresholds=thresholds, required=True)
    else:
        _minimum_check(
            checks,
            name="overall_affected_f1",
            value=(float(metrics["affected_f1"]) if metrics.get("affected_f1") is not None else None),
            minimum=float(thresholds["overall_affected_f1"]),
        )

    by_dataset = dict(metrics.get("by_dataset") or {})
    required_datasets = tuple(str(value) for value in thresholds.get("required_datasets", ()))
    if not required_datasets:
        required_datasets = tuple(sorted(by_dataset))
    required_target_datasets = tuple(
        str(value) for value in thresholds.get("required_target_datasets", ())
    )
    target_dataset_thresholds = {
        str(dataset_id): dict(values or {})
        for dataset_id, values in dict(
            thresholds.get("target_dataset_localization") or {}
        ).items()
    }
    required_target_datasets = tuple(
        dict.fromkeys((*required_target_datasets, *target_dataset_thresholds))
    )
    datasets_to_check = tuple(
        dict.fromkeys((*required_datasets, *required_target_datasets))
    )

    for dataset_id in datasets_to_check:
        values = by_dataset.get(dataset_id)
        _presence_check(
            checks,
            name=f"dataset:{dataset_id}:present",
            present=values is not None,
        )
        if values is None:
            continue
        _minimum_check(
            checks,
            name=f"dataset:{dataset_id}:macro_f1",
            value=float(values["class_macro_f1"]),
            minimum=float(thresholds["minimum_dataset_macro_f1"]),
        )
        dataset_normal, dataset_normal_support = _recall(values, "normal")
        dataset_anomaly, dataset_anomaly_support = _recall(values, "unknown_anomaly")
        normal_required = _dataset_state_required(thresholds, dataset_id, "normal")
        anomaly_required = _dataset_state_required(thresholds, dataset_id, "unknown_anomaly")
        _presence_check(
            checks,
            name=f"dataset:{dataset_id}:normal_support",
            present=dataset_normal_support > 0,
            required=normal_required,
            note=(
                None if normal_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )
        _presence_check(
            checks,
            name=f"dataset:{dataset_id}:anomaly_support",
            present=dataset_anomaly_support > 0,
            required=anomaly_required,
            note=(
                None if anomaly_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )
        _minimum_check(
            checks,
            name=f"dataset:{dataset_id}:normal_recall",
            value=dataset_normal,
            minimum=float(thresholds["minimum_dataset_state_recall"]),
            required=normal_required,
            note=(
                None if normal_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )
        _minimum_check(
            checks,
            name=f"dataset:{dataset_id}:anomaly_recall",
            value=dataset_anomaly,
            minimum=float(thresholds["minimum_dataset_state_recall"]),
            required=anomaly_required,
            note=(
                None if anomaly_required
                else "publisher complete-group coverage limitation preregistered"
            ),
        )
        if dataset_id in required_target_datasets:
            dataset_localization_thresholds = dict(thresholds)
            dataset_localization_thresholds.update(
                target_dataset_thresholds.get(dataset_id, {})
            )
            _localization_checks(
                checks,
                values,
                thresholds=dataset_localization_thresholds,
                prefix=f"dataset:{dataset_id}:",
                required=True,
            )

    for dataset_id in required_target_datasets:
        _presence_check(
            checks,
            name=f"target_dataset:{dataset_id}:present",
            present=dataset_id in by_dataset,
        )

    ece = metrics.get("global_ece")
    if "maximum_global_ece" in thresholds:
        _maximum_check(
            checks,
            name="global_ece",
            value=(float(ece) if ece is not None else None),
            maximum=float(thresholds["maximum_global_ece"]),
            required=bool(thresholds.get("require_global_ece", True)),
        )

    persistence = dict(metrics.get("persistence") or {})
    healthy_hours = float(persistence.get("healthy_evaluated_chronological_hours") or 0.0)
    false_rate = persistence.get("false_alert_events_per_building_day")
    minimum_hours = float(thresholds.get("minimum_healthy_hours_for_false_alert_gate", 24.0))
    false_required = healthy_hours >= minimum_hours
    maximum_false_rate = float(thresholds.get("maximum_false_alert_events_per_building_day", 0.5))
    false_passed = false_rate is not None and float(false_rate) <= maximum_false_rate
    checks.append(
        {
            "name": "false_alert_events_per_building_day",
            "value": false_rate,
            "maximum": maximum_false_rate,
            "healthy_chronological_hours": healthy_hours,
            "minimum_hours_required": minimum_hours,
            "required": false_required,
            "passed": bool(false_passed) if false_required else True,
            "observed_passed": bool(false_passed),
            "note": None if false_required else "reported but not gated because duration is too short",
        }
    )

    required_failures = [item for item in checks if item["required"] and not item["passed"]]
    protocol_test_authorized = bool(thresholds.get("frozen_test_authorized", True))
    return {
        "created_utc": utc_now_iso(),
        "gate": "complete_validation_authorization_v2_factorized_localization",
        "passed": not required_failures,
        "checks": checks,
        "required_failures": required_failures,
        "required_datasets": list(required_datasets),
        "required_target_datasets": list(required_target_datasets),
        "test_authorized": not required_failures and protocol_test_authorized,
        "protocol_test_authorized": protocol_test_authorized,
        "confidence_operational": bool(metrics.get("confidence_operational", False)),
        "confidence_note": (
            "Confidence may remain disabled; this does not authorize confidence-driven abstention."
            if not metrics.get("confidence_operational", False)
            else "Validation-fitted confidence passed its own gate."
        ),
        "note": (
            "Validation passed, but the protocol explicitly keeps the frozen test unauthorized."
            if not required_failures and not protocol_test_authorized
            else "The frozen test also requires the preregistered localization intervention audit and an explicit user authorization flag."
        ),
    }


def validate_metrics_file(
    metrics_path: str | Path,
    *,
    output: str | Path,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    source = Path(metrics_path)
    metrics = json.loads(source.read_text(encoding="utf-8"))
    report = validate_metrics_gates(metrics, thresholds=thresholds)
    report["metrics"] = str(source)
    atomic_write_json(output, report)
    return report
