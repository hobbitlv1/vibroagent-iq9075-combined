from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..data.lumo_geometry import LUMO_PANEL_LEVELS, LUMO_TARGET_DAMAGE_LEVELS
from .conditioning_v017 import MEASURED_TRANSFER_CALIBRATION_SCHEMA
from .lumo_fe_v019 import LUMO_ABAQUS_SOURCE_SHA256
from .lumo_inverse_v019 import (
    LUMO_HEALTHY_MODE_BANK_POLICY,
    LUMO_INVERSE_CALIBRATION_SCHEMA,
    LUMO_INVERSE_METHOD,
    healthy_train_mode_bank,
)
from .modal_synthesis_v019 import SensorModalState

SYNTHETIC_MANIFEST_ENV = "VIBROGEMMA_SYNTHETIC_AUGMENTATION_MANIFEST"
REQUIRE_CALIBRATED_SYNTHETIC_ENV = "VIBROGEMMA_REQUIRE_CALIBRATED_SYNTHETIC"
REQUIRE_INVERSE_FE_SYNTHETIC_ENV = "VIBROGEMMA_REQUIRE_INVERSE_FE_SYNTHETIC"
LEGACY_SYNTHETIC_SOURCE_KIND = "opensees_measured_conditioned_v017"
CALIBRATED_SYNTHETIC_SOURCE_KIND = "opensees_measured_calibrated_v018"
INVERSE_FE_SYNTHETIC_SOURCE_KIND = "opensees_lumo_full_inverse_fe_v019"
LUMO_INVERSE_FE_SCHEMA = "vibrogemma-lumo-full-inverse-fe-calibration-v0.19.0"
SYNTHETIC_AUDIT_SCHEMA = "vibrogemma-opensees-augmentation-audit-v0.19.0"
SYNTHETIC_MANIFEST_TEMPLATE_INDEX_NAMESPACE = "expanded_train_view_v1"
SYNTHETIC_RUNTIME_PROJECTION_NAMESPACE = "joint_physical_event_v1"
LUMO_INVERSE_MODAL_STATES = {"healthy"} | {
    f"{damage}_{severity}"
    for levels in LUMO_TARGET_DAMAGE_LEVELS.values()
    for damage in levels
    for severity in ("010", "111")
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _inverse_fe_artifact_issues(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    modal_library = dict(payload.get("modal_library") or {})
    synthesis = dict(payload.get("modal_synthesis_contract") or {})
    acceptance = dict(payload.get("acceptance") or {})
    runtime = dict(payload.get("runtime_verification") or {})
    expected_level_basis = {
        "DAM1": "publisher_geometry_unmeasured",
        "DAM2": "publisher_geometry_unmeasured",
        "DAM3": "measured_train_group",
        "DAM4": "measured_train_group",
        "DAM5": "publisher_geometry_unmeasured",
        "DAM6": "measured_train_group",
    }
    expected_pattern_basis = {
        "010": "measured_train_groups",
        "111": "publisher_geometry_extrapolation_from_010_residual",
    }
    if (
        payload.get("schema") != LUMO_INVERSE_FE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("dataset_id") != "lumo"
        or payload.get("source_split") != "train"
        or payload.get("validation_access") is not False
        or payload.get("test_access") is not False
        or payload.get("target_damage_levels") != LUMO_TARGET_DAMAGE_LEVELS
        or set(modal_library) != LUMO_INVERSE_MODAL_STATES
        or not _is_sha256(payload.get("episode_store_index_sha256"))
        or not _is_sha256(payload.get("request_sha256"))
        or not _is_sha256(dict(payload.get("source_model") or {}).get("sha256"))
        or dict(payload.get("source_model") or {}).get("sha256")
        != LUMO_ABAQUS_SOURCE_SHA256
        or payload.get("official_fe_sha256") != LUMO_ABAQUS_SOURCE_SHA256
        or payload.get("inverse_modal_calibration_schema")
        != LUMO_INVERSE_CALIBRATION_SCHEMA
        or synthesis.get("shared_counterfactual_excitation") is not True
        or synthesis.get("measured_magnitude_replacement") is not False
        or synthesis.get("panel_levels") != list(LUMO_PANEL_LEVELS)
        or synthesis.get("panel_axes") != ["x", "y"]
        or synthesis.get("modal_residue_source")
        != "OpenSees_mass_normalized_modes_and_modal_participation_factors"
        or synthesis.get("shared_generalized_excitation_components")
        != ["MX", "MY", "MZ", "RMX", "RMY", "RMZ"]
        or synthesis.get("healthy_train_mode_bank_policy")
        != LUMO_HEALTHY_MODE_BANK_POLICY
        or synthesis.get("mode_alignment")
        != "frequency_MAC_Hungarian_sign_plus_exact_degenerate_Procrustes"
        or synthesis.get("forward_mode_count") != 60
        or synthesis.get("forward_mode_search_counts") != [10, 30, 60]
        or synthesis.get("forward_mode_policy")
        != "adaptive_panel_observable_10_30_60_then_group_consensus_v1"
        or synthesis.get("retained_forward_mode_count") != 10
        or synthesis.get("resolved_forward_mode_count") not in {10, 30, 60}
        or synthesis.get("artifact_mode_count") != 10
        or float(acceptance.get("maximum_fit_cost", float("nan"))) != 0.08
        or acceptance.get("reject_parameter_bound_hits") is not True
        or runtime.get("full_2535_node_opensees_eigen_completed") is not True
        or runtime.get("healthy_and_all_damage_modal_library_completed") is not True
        or runtime.get("modal_properties_participation_factors_completed") is not True
        or int(runtime.get("checksummed_candidate_count", 0)) < 4
        or payload.get("damage_level_basis") != expected_level_basis
        or payload.get("damage_pattern_basis") != expected_pattern_basis
    ):
        issues.append("provenance or modal coverage")
        return issues
    receipt = dict(payload.get("inverse_modal_calibration") or {})
    receipt_value = dict(receipt)
    receipt_hash = str(receipt_value.pop("receipt_sha256", ""))
    try:
        expected_receipt_hash = hashlib.sha256(
            json.dumps(
                receipt_value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    except (TypeError, ValueError):
        issues.append("inverse modal fit receipt is not canonical JSON")
        return issues
    optimizer = dict(receipt.get("optimizer") or {})
    groups = dict(receipt.get("groups") or {})
    damage_levels = {summary.get("damage_level") for summary in groups.values()}
    group_matches_valid = all(
        len(summary.get("matched_forward_mode_indices") or [])
        == len(summary.get("frequencies_hz") or [])
        and len(set(summary.get("matched_forward_mode_indices") or []))
        == len(summary.get("matched_forward_mode_indices") or [])
        for summary in groups.values()
    )
    healthy_assignments = [
        tuple(summary.get("matched_forward_mode_indices") or [])
        for summary in groups.values()
        if summary.get("damage_level") is None
    ]
    try:
        expected_healthy_bank = list(
            healthy_train_mode_bank(healthy_assignments, maximum_modes=10)
        )
    except ValueError:
        expected_healthy_bank = []
    expected_support = {
        str(index): sum(index in assignment for assignment in healthy_assignments)
        for index in expected_healthy_bank
    }
    if (
        receipt.get("schema") != LUMO_INVERSE_CALIBRATION_SCHEMA
        or receipt.get("dataset_id") != "lumo"
        or receipt.get("source_split") != "train"
        or receipt.get("method") != LUMO_INVERSE_METHOD
        or receipt_hash != expected_receipt_hash
        or optimizer.get("success") is not True
        or not np.isfinite(float(optimizer.get("cost", float("nan"))))
        or float(optimizer["cost"]) > float(acceptance["maximum_fit_cost"])
        or int(optimizer.get("jacobian_rank", 0)) != 3
        or not np.isfinite(float(optimizer.get("jacobian_condition", float("nan"))))
        or float(optimizer["jacobian_condition"]) > 1.0e10
        or list(optimizer.get("bound_hits") or [])
        or not groups
        or any(summary.get("source_split") != "train" for summary in groups.values())
        or not {None, "DAM3", "DAM4", "DAM6"}.issubset(damage_levels)
        or not group_matches_valid
        or expected_healthy_bank
        != synthesis.get("selected_healthy_forward_mode_indices")
        or int(synthesis.get("healthy_train_group_count", 0))
        != len(healthy_assignments)
        or synthesis.get("healthy_train_mode_support_counts") != expected_support
    ):
        issues.append("inverse modal fit receipt is invalid")
    mode_counts: set[int] = set()
    for key, value in modal_library.items():
        try:
            state = SensorModalState.from_payload(value)
            mode_counts.add(len(state.frequencies_hz))
        except Exception as exc:
            issues.append(f"modal state {key}: {exc}")
    if len(mode_counts) != 1 or not mode_counts or not 1 <= next(iter(mode_counts)) <= 10:
        issues.append("modal states do not share one-to-ten modes")
    selected_indices = list(synthesis.get("selected_healthy_forward_mode_indices") or [])
    selected_count = int(synthesis.get("selected_artifact_mode_count", 0))
    if (
        not mode_counts
        or selected_count != next(iter(mode_counts))
        or len(selected_indices) != selected_count
        or len(set(selected_indices)) != selected_count
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 10
            for index in selected_indices
        )
    ):
        issues.append("selected forward-mode indices are invalid")
    return issues


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            rows.append(value)
    return rows


def audit_synthetic_manifest(
    path: Path, *, require_calibrated: bool = False, require_inverse_fe: bool = False
) -> dict[str, Any]:
    path = path.resolve()
    rows = _read_jsonl(path)
    issues: list[str] = []
    if not rows:
        issues.append("manifest is empty")
    physical: dict[tuple[str, str], set[str]] = {}
    target_positive_counts = {target: 0 for target in range(1, 6)}
    parent_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    source_groups: set[str] = set()
    calibration_payloads: dict[Path, dict[str, Any]] = {}
    calibration_hashes: dict[Path, str] = {}
    inverse_payloads: dict[Path, dict[str, Any]] = {}
    inverse_hashes: dict[Path, str] = {}
    inverse_validation_issues: dict[Path, list[str]] = {}
    record_ids: set[str] = set()
    calibrated_groups: set[tuple[str, str]] = set()
    tasks_by_state: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    inverse_groups: set[tuple[str, str]] = set()
    inverse_damage_coverage: dict[int, set[tuple[str, str]]] = {
        target: set() for target in range(1, 6)
    }
    joint_projection_values: set[bool] = set()

    for row in rows:
        if row.get("split") != "train":
            issues.append(f"non-train synthetic row: {row.get('record_id')}")
        source_kind = str(row.get("source_kind") or "")
        if source_kind not in {
            LEGACY_SYNTHETIC_SOURCE_KIND,
            CALIBRATED_SYNTHETIC_SOURCE_KIND,
            INVERSE_FE_SYNTHETIC_SOURCE_KIND,
        }:
            issues.append(f"invalid source_kind: {row.get('record_id')}")
        if require_calibrated and source_kind not in {
            CALIBRATED_SYNTHETIC_SOURCE_KIND,
            INVERSE_FE_SYNTHETIC_SOURCE_KIND,
        }:
            issues.append(f"calibrated source kind required: {row.get('record_id')}")
        if require_inverse_fe and source_kind != INVERSE_FE_SYNTHETIC_SOURCE_KIND:
            issues.append(f"inverse-FE source kind required: {row.get('record_id')}")
        joint_projection = row.get("joint_event_projection_only", False)
        if not isinstance(joint_projection, bool):
            issues.append(
                "joint_event_projection_only must be boolean: "
                f"{row.get('record_id')}"
            )
        else:
            joint_projection_values.add(joint_projection)
        record_id = str(row.get("record_id") or "")
        if not record_id or record_id in record_ids:
            issues.append(f"missing or duplicate record_id: {record_id}")
        record_ids.add(record_id)
        if row.get("target_permutation_augmentation") is not False:
            issues.append(f"target permutation not false: {row.get('record_id')}")
        parent = str(row.get("parent_dataset") or "")
        group = str(row.get("counterfactual_group") or "")
        physical_state = str(row.get("physical_state") or "")
        if not parent or not group or not physical_state:
            issues.append(f"missing parent/group/state: {row.get('record_id')}")
            continue
        physical.setdefault((parent, group), set()).add(physical_state)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        task_family = str(row.get("task_record", {}).get("_task_family") or "")
        task_counts[task_family] = task_counts.get(task_family, 0) + 1
        tasks_by_state.setdefault((parent, group, physical_state), []).append(row)
        if source_kind in {
            CALIBRATED_SYNTHETIC_SOURCE_KIND,
            INVERSE_FE_SYNTHETIC_SOURCE_KIND,
        }:
            calibrated_groups.add((parent, group))
        source_group = str(row.get("source_real_group") or "")
        if not source_group:
            issues.append(f"missing source_real_group: {row.get('record_id')}")
        else:
            source_groups.add(source_group)
        waveform = Path(str(row.get("waveform_path") or ""))
        if not waveform.is_absolute():
            waveform = (path.parent / waveform).resolve()
        if not waveform.is_file():
            issues.append(f"missing waveform archive: {waveform}")
        elif row.get("waveform_sha256") != _sha256_file(waveform):
            issues.append(f"waveform hash mismatch: {waveform}")
        affected = row.get("affected_target")
        if affected is not None and task_family == "global":
            try:
                target_positive_counts[int(affected)] += 1
            except Exception:
                issues.append(f"invalid affected target: {row.get('record_id')}")
        if source_kind in {
            CALIBRATED_SYNTHETIC_SOURCE_KIND,
            INVERSE_FE_SYNTHETIC_SOURCE_KIND,
        }:
            expected_row_schema = (
                "vibrogemma-opensees-synthetic-task-v0.19.0"
                if source_kind == INVERSE_FE_SYNTHETIC_SOURCE_KIND
                else "vibrogemma-opensees-synthetic-task-v0.18.1"
            )
            if row.get("schema") != expected_row_schema:
                issues.append(f"invalid calibrated row schema: {row.get('record_id')}")
            calibration = Path(str(row.get("measured_calibration_path") or ""))
            if not calibration.is_absolute():
                calibration = (path.parent / calibration).resolve()
            if not calibration.is_file():
                issues.append(f"missing measured calibration: {calibration}")
                continue
            if calibration not in calibration_hashes:
                calibration_hashes[calibration] = _sha256_file(calibration)
            observed_calibration_hash = calibration_hashes[calibration]
            if row.get("measured_calibration_sha256") != observed_calibration_hash:
                issues.append(f"measured calibration hash mismatch: {calibration}")
                continue
            try:
                if calibration not in calibration_payloads:
                    calibration_payloads[calibration] = json.loads(
                        calibration.read_text(encoding="utf-8")
                    )
                payload = calibration_payloads[calibration]
            except Exception as exc:
                issues.append(f"invalid measured calibration {calibration}: {exc}")
                continue
            parent_payload = dict((payload.get("calibrations") or {}).get(parent) or {})
            if (
                payload.get("schema") != MEASURED_TRANSFER_CALIBRATION_SCHEMA
                or payload.get("source_split") != "train"
                or parent_payload.get("dataset_id") != parent
                or parent_payload.get("source_splits") != ["train"]
                or row.get("measured_calibration_schema")
                != MEASURED_TRANSFER_CALIBRATION_SCHEMA
                or row.get("measured_calibration_source_split") != "train"
            ):
                issues.append(f"invalid measured calibration provenance: {row.get('record_id')}")
            if affected is not None and f"target_{affected}" not in dict(
                parent_payload.get("gains_by_target") or {}
            ):
                issues.append(f"missing target calibration: {row.get('record_id')}")
            if affected is not None:
                target_key = f"target_{affected}"
                expected_basis = dict(parent_payload.get("basis_by_target") or {}).get(
                    target_key
                )
                expected_sources = dict(
                    parent_payload.get("source_targets_by_target") or {}
                ).get(target_key)
                if (
                    row.get("measured_calibration_basis") != expected_basis
                    or row.get("measured_calibration_source_targets") != expected_sources
                ):
                    issues.append(
                        f"invalid target calibration provenance: {row.get('record_id')}"
                    )
            error = row.get("measured_calibration_mean_log_error")
            maximum_error = row.get("measured_calibration_max_mean_log_error")
            if affected is not None and (
                error is None
                or maximum_error is None
                or not np.isfinite(float(error))
                or not np.isfinite(float(maximum_error))
                or float(error) < 0.0
                or float(maximum_error) <= 0.0
                or float(error) > float(maximum_error)
            ):
                issues.append(f"invalid measured calibration error: {row.get('record_id')}")
        if source_kind == INVERSE_FE_SYNTHETIC_SOURCE_KIND:
            inverse_groups.add((parent, group))
            inverse_path = Path(str(row.get("inverse_fe_calibration_path") or ""))
            if not inverse_path.is_absolute():
                inverse_path = (path.parent / inverse_path).resolve()
            if not inverse_path.is_file():
                issues.append(f"missing inverse-FE calibration: {inverse_path}")
                continue
            if inverse_path not in inverse_hashes:
                inverse_hashes[inverse_path] = _sha256_file(inverse_path)
            if row.get("inverse_fe_calibration_sha256") != inverse_hashes[inverse_path]:
                issues.append(f"inverse-FE calibration hash mismatch: {inverse_path}")
                continue
            try:
                if inverse_path not in inverse_payloads:
                    inverse_payloads[inverse_path] = json.loads(
                        inverse_path.read_text(encoding="utf-8")
                    )
                    inverse_validation_issues[inverse_path] = _inverse_fe_artifact_issues(
                        inverse_payloads[inverse_path]
                    )
                inverse = inverse_payloads[inverse_path]
            except Exception as exc:
                issues.append(f"invalid inverse-FE calibration {inverse_path}: {exc}")
                continue
            modal_key = row.get("inverse_fe_modal_state_key")
            modal_library = dict(inverse.get("modal_library") or {})
            artifact_issues = inverse_validation_issues.get(inverse_path, [])
            if artifact_issues:
                issues.append(
                    f"invalid inverse-FE artifact {inverse_path}: {'; '.join(artifact_issues)}"
                )
            if (
                inverse.get("schema") != LUMO_INVERSE_FE_SCHEMA
                or inverse.get("status") != "complete"
                or inverse.get("dataset_id") != "lumo"
                or inverse.get("source_split") != "train"
                or inverse.get("validation_access") is not False
                or inverse.get("test_access") is not False
                or parent != "lumo"
                or row.get("inverse_fe_calibration_schema") != LUMO_INVERSE_FE_SCHEMA
                or row.get("inverse_fe_source_split") != "train"
                or row.get("inverse_fe_request_sha256") != inverse.get("request_sha256")
                or row.get("inverse_fe_source_model_sha256")
                != dict(inverse.get("source_model") or {}).get("sha256")
                or modal_key not in modal_library
            ):
                issues.append(f"invalid inverse-FE provenance: {row.get('record_id')}")
            if affected is None and modal_key != "healthy":
                issues.append(f"healthy row uses damaged inverse-FE state: {row.get('record_id')}")
            if affected is None and (
                row.get("inverse_fe_damage_level") is not None
                or row.get("inverse_fe_damage_pattern") is not None
                or row.get("inverse_fe_damage_level_basis") is not None
                or row.get("inverse_fe_damage_pattern_basis") is not None
            ):
                issues.append(f"healthy row carries inverse-FE damage metadata: {row.get('record_id')}")
            if affected is not None:
                damage_level = str(row.get("inverse_fe_damage_level") or "")
                damage_pattern = str(row.get("inverse_fe_damage_pattern") or "")
                expected_levels = dict(inverse.get("target_damage_levels") or {}).get(
                    f"target_{affected}", []
                )
                if (
                    damage_level not in expected_levels
                    or damage_pattern not in {"010", "111"}
                    or modal_key != f"{damage_level}_{damage_pattern}"
                    or row.get("inverse_fe_damage_level_basis")
                    != dict(inverse.get("damage_level_basis") or {}).get(damage_level)
                    or row.get("inverse_fe_damage_pattern_basis")
                    != dict(inverse.get("damage_pattern_basis") or {}).get(damage_pattern)
                ):
                    issues.append(f"inverse-FE damage mapping mismatch: {row.get('record_id')}")
                elif task_family == "global":
                    inverse_damage_coverage[int(affected)].add(
                        (damage_level, damage_pattern)
                    )

    expected_states = {"healthy", "target_1", "target_2", "target_3", "target_4", "target_5"}
    if len(joint_projection_values) > 1:
        issues.append("manifest mixes joint-only and reusable label templates")
    for key, states in sorted(physical.items()):
        if states != expected_states:
            issues.append(f"counterfactual group {key} states={sorted(states)}")
    for parent, group in sorted(calibrated_groups):
        for physical_state in sorted(expected_states):
            state_rows = tasks_by_state.get((parent, group, physical_state), [])
            expected_affected = (
                None if physical_state == "healthy" else int(physical_state.rsplit("_", 1)[-1])
            )
            if any(row.get("affected_target") != expected_affected for row in state_rows):
                issues.append(f"calibrated state/affected mismatch: {(parent, group, physical_state)}")
            global_rows = [
                row for row in state_rows if row.get("task_record", {}).get("_task_family") == "global"
            ]
            target_rows = [
                row for row in state_rows if row.get("task_record", {}).get("_task_family") == "target"
            ]
            target_indices = [
                int(row.get("task_record", {}).get("_task_target_index", 0))
                for row in target_rows
            ]
            if len(global_rows) != 1 or sorted(target_indices) != [1, 2, 3, 4, 5]:
                issues.append(f"incomplete calibrated tasks: {(parent, group, physical_state)}")
                continue
            expected_global_state = "normal" if expected_affected is None else "unknown_anomaly"
            if global_rows[0].get("task_record", {}).get("_task_state") != expected_global_state:
                issues.append(f"invalid calibrated global task: {(parent, group, physical_state)}")
            for row in target_rows:
                focus = int(row["task_record"]["_task_target_index"])
                expected_state = (
                    "unknown_anomaly" if focus == expected_affected else "normal"
                )
                if row["task_record"].get("_task_state") != expected_state:
                    issues.append(
                        f"invalid calibrated target task: {(parent, group, physical_state, focus)}"
                    )
    nonzero = [value for value in target_positive_counts.values() if value > 0]
    if nonzero and len(set(nonzero)) != 1:
        issues.append(f"target positives are imbalanced: {target_positive_counts}")
    if len(inverse_groups) >= 4:
        for target in range(1, 6):
            levels = LUMO_TARGET_DAMAGE_LEVELS[f"target_{target}"]
            expected = {
                (damage_level, pattern)
                for damage_level in levels
                for pattern in ("010", "111")
            }
            if inverse_damage_coverage[target] != expected:
                issues.append(
                    f"inverse-FE target_{target} damage coverage is incomplete: "
                    f"{sorted(inverse_damage_coverage[target])}"
                )

    return {
        "schema": SYNTHETIC_AUDIT_SCHEMA,
        "require_calibrated": bool(require_calibrated),
        "require_inverse_fe": bool(require_inverse_fe),
        "passed": not issues,
        "manifest": str(path),
        "manifest_sha256": _sha256_file(path),
        "row_count": len(rows),
        "counterfactual_group_count": len(physical),
        "source_real_group_count": len(source_groups),
        "parent_row_counts": dict(sorted(parent_counts.items())),
        "task_family_counts": dict(sorted(task_counts.items())),
        "target_positive_counts": target_positive_counts,
        "joint_event_projection_only": (
            next(iter(joint_projection_values))
            if len(joint_projection_values) == 1
            else None
        ),
        "measured_calibration_files": [str(value) for value in sorted(calibration_payloads)],
        "inverse_fe_calibration_files": [str(value) for value in sorted(inverse_payloads)],
        "inverse_fe_damage_coverage": {
            f"target_{target}": [list(value) for value in sorted(values)]
            for target, values in inverse_damage_coverage.items()
        },
        "issues": issues,
    }


def _find_waveform_field(board: Any) -> str:
    if not dataclasses.is_dataclass(board):
        raise TypeError(f"BoardWindow is not a dataclass: {type(board)!r}")
    candidates: list[tuple[int, str]] = []
    for field in dataclasses.fields(board):
        value = getattr(board, field.name)
        if isinstance(value, np.ndarray) and value.ndim in {1, 2}:
            candidates.append((int(value.size), field.name))
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor) and value.ndim in {1, 2}:
                    candidates.append((int(value.numel()), field.name))
            except Exception:
                pass
    if not candidates:
        for name in ("samples", "values", "waveform", "data"):
            if hasattr(board, name):
                return name
        raise AttributeError(f"cannot identify waveform field in {type(board)!r}")
    return max(candidates)[1]


def _convert_like(template: Any, array: np.ndarray) -> Any:
    if isinstance(template, np.ndarray):
        return np.asarray(array, dtype=template.dtype)
    try:
        import torch

        if isinstance(template, torch.Tensor):
            return torch.as_tensor(array, dtype=template.dtype, device=template.device)
    except Exception:
        pass
    return array


def _replace_dataclass_if_present(value: Any, updates: Mapping[str, Any]) -> Any:
    if not dataclasses.is_dataclass(value):
        return value
    field_names = {field.name for field in dataclasses.fields(value)}
    selected = {key: item for key, item in updates.items() if key in field_names}
    return dataclasses.replace(value, **selected) if selected else value


def _strip_supervision_task_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one joint-event record from an expanded supervision-task record."""

    output = dict(record)
    for key in (
        "_supervision_task",
        "_task_family",
        "_task_target_index",
        "_task_original_target_index",
        "_task_state",
        "_task_group",
        "_target_permutation",
    ):
        output.pop(key, None)
    return output


def _synthetic_event_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int | None]:
    affected = row.get("affected_target")
    return (
        str(row.get("counterfactual_group") or ""),
        str(row.get("physical_state") or ""),
        str(row.get("parent_dataset") or ""),
        str(row.get("waveform_path") or ""),
        (None if affected is None else int(affected)),
    )


def _event_record_id(row: Mapping[str, Any]) -> str:
    record_id = str(row.get("record_id") or "")
    return record_id.rsplit("-task-", 1)[0] if "-task-" in record_id else record_id


def _joint_synthetic_label(affected_target: int | None) -> Any:
    """Build the complete five-target label for one synthetic physical event."""

    from vibrogemma.contracts import EpisodeLabel, TargetLabel

    if affected_target is not None and affected_target not in {1, 2, 3, 4, 5}:
        raise RuntimeError(f"invalid synthetic affected target: {affected_target}")
    anomalous = affected_target is not None
    return EpisodeLabel(
        targets=[
            TargetLabel(
                sensor_id=f"target_{target_index}",
                affected=(target_index == affected_target),
                class_name=(
                    "unknown_anomaly" if target_index == affected_target else "normal"
                ),
                severity=("advisory" if target_index == affected_target else "none"),
                source="opensees_counterfactual_localisation",
            )
            for target_index in range(1, 6)
        ],
        explanation_target="",
        global_class_name=("unknown_anomaly" if anomalous else "normal"),
        global_severity=("advisory" if anomalous else "none"),
        target_supervision_available=True,
        supervision_scope="joint",
        focus_target_index=None,
    )


def wrap_episode_dataset_class(base_class: type) -> type:
    """Return an environment-gated EpisodeDataset subclass.

    When the manifest variable is absent, behaviour is identical to the pinned
    base class. Synthetic records are appended only for the train split.

    Legacy OpenSees manifests store six *factorized task* rows for each physical
    counterfactual event and their template indices refer to an expanded
    EpisodeDataset. Structured-set training intentionally uses unexpanded whole
    events. In that mode the wrapper therefore projects each six-row family back
    to one canonical joint event and materializes its five-target label directly.
    This avoids duplicated synthetic events, index-space mismatches, and label
    leakage from unrelated factorized templates.
    """

    class OpenSeesAugmentedEpisodeDataset(base_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._opensees_real_record_count = len(self.records)
            self._opensees_manifest_path: Path | None = None
            self._opensees_template_dataset: Any | None = None
            self._opensees_joint_event_projection = False
            split = getattr(self, "split", None)
            if split is None:
                split = args[1] if len(args) > 1 else kwargs.get("split")
            manifest_value = os.environ.get(SYNTHETIC_MANIFEST_ENV)
            if str(split) != "train" or not manifest_value:
                return
            manifest_path = Path(manifest_value).expanduser().resolve()
            require_calibrated = os.environ.get(REQUIRE_CALIBRATED_SYNTHETIC_ENV) == "1"
            require_inverse_fe = os.environ.get(REQUIRE_INVERSE_FE_SYNTHETIC_ENV) == "1"
            audit = audit_synthetic_manifest(
                manifest_path,
                require_calibrated=require_calibrated,
                require_inverse_fe=require_inverse_fe,
            )
            if not audit["passed"]:
                raise RuntimeError(f"synthetic augmentation admission failed: {audit}")
            rows = _read_jsonl(manifest_path)
            fallback_index = args[0] if args else kwargs.get("index_path")
            current_index = Path(getattr(self, "index_path", fallback_index)).resolve()
            for calibration_value in {
                str(row.get("measured_calibration_path"))
                for row in rows
                if row.get("source_kind")
                in {CALIBRATED_SYNTHETIC_SOURCE_KIND, INVERSE_FE_SYNTHETIC_SOURCE_KIND}
            }:
                calibration_path = Path(calibration_value)
                if not calibration_path.is_absolute():
                    calibration_path = (manifest_path.parent / calibration_path).resolve()
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                if calibration.get("episode_store_index_sha256") != _sha256_file(
                    current_index
                ):
                    raise RuntimeError(
                        "measured calibration was fitted from a different episode index"
                    )
            for inverse_value in {
                str(row.get("inverse_fe_calibration_path"))
                for row in rows
                if row.get("source_kind") == INVERSE_FE_SYNTHETIC_SOURCE_KIND
            }:
                inverse_path = Path(inverse_value)
                if not inverse_path.is_absolute():
                    inverse_path = (manifest_path.parent / inverse_path).resolve()
                inverse = json.loads(inverse_path.read_text(encoding="utf-8"))
                if inverse.get("episode_store_index_sha256") != _sha256_file(current_index):
                    raise RuntimeError(
                        "inverse-FE calibration was fitted from a different episode index"
                    )

            # Manifest template indices were generated against the task-expanded
            # train view. Recreate that exact index space independently of the
            # caller's factorized/structured training mode.
            template_kwargs = dict(kwargs)
            for filter_key in (
                "include_dataset_ids",
                "include_training_roles",
                "train_only_dataset_ids",
            ):
                template_kwargs.pop(filter_key, None)
            template_kwargs["require_labels"] = True
            template_kwargs["expand_supervision_tasks"] = True
            template_kwargs["target_permutation_augmentation"] = False
            template_kwargs["episode_ids"] = None
            self._opensees_template_dataset = base_class(*args, **template_kwargs)

            base_is_factorized = any(
                bool(record.get("_supervision_task"))
                or str(record.get("_task_family") or "") in {"global", "target"}
                for record in self.records
            )
            projection_only_values = {
                bool(row.get("joint_event_projection_only")) for row in rows
            }
            if len(projection_only_values) != 1:
                raise RuntimeError(
                    "synthetic manifest mixes joint-only and reusable label templates"
                )
            joint_event_projection_only = next(iter(projection_only_values))
            if base_is_factorized and joint_event_projection_only:
                raise RuntimeError(
                    "joint-event-only synthetic labels cannot enter factorized training"
                )
            appended: list[dict[str, Any]] = []
            if base_is_factorized:
                for row in rows:
                    task_record = dict(row["task_record"])
                    task_record["_opensees_synthetic_row"] = row
                    appended.append(task_record)
            else:
                self._opensees_joint_event_projection = True
                grouped: dict[
                    tuple[str, str, str, str, int | None], list[dict[str, Any]]
                ] = {}
                for row in rows:
                    grouped.setdefault(_synthetic_event_key(row), []).append(row)
                for event_rows in grouped.values():
                    global_rows = [
                        row
                        for row in event_rows
                        if str(dict(row.get("task_record") or {}).get("_task_family"))
                        == "global"
                    ]
                    canonical = global_rows[0] if global_rows else event_rows[0]
                    waveform_template_index = int(canonical["waveform_template_index"])
                    if not 0 <= waveform_template_index < len(
                        self._opensees_template_dataset.records
                    ):
                        raise RuntimeError(
                            "synthetic waveform template index is outside the expanded "
                            f"train view: {waveform_template_index}"
                        )
                    template_record = _strip_supervision_task_fields(
                        self._opensees_template_dataset.records[
                            waveform_template_index
                        ]
                    )
                    affected_value = canonical.get("affected_target")
                    affected_target = (
                        None if affected_value is None else int(affected_value)
                    )
                    synthetic_group = str(canonical["counterfactual_group"])
                    healthy = affected_target is None
                    template_record.update(
                        {
                            "dataset_id": str(canonical["parent_dataset"]),
                            "split": "train",
                            "episode_id": _event_record_id(canonical),
                            "source_kind": str(canonical["source_kind"]),
                            "sampling_group": synthetic_group,
                            "split_group": synthetic_group,
                            "session_id": synthetic_group,
                            "has_label": True,
                            "healthy": healthy,
                            "global_class_name": (
                                "normal" if healthy else "unknown_anomaly"
                            ),
                            "class_name": (
                                "normal" if healthy else "unknown_anomaly"
                            ),
                            "severity": "none" if healthy else "advisory",
                            "target_supervision_available": True,
                            "affected_target_indices": (
                                [] if healthy else [affected_target]
                            ),
                            "affected_targets": (
                                [] if healthy else [affected_target]
                            ),
                            "_opensees_synthetic_row": {
                                **canonical,
                                "_structured_set_event_projection": True,
                                "_factorized_task_row_count": len(event_rows),
                                "manifest_template_index_namespace": (
                                    SYNTHETIC_MANIFEST_TEMPLATE_INDEX_NAMESPACE
                                ),
                                "runtime_projection_namespace": (
                                    SYNTHETIC_RUNTIME_PROJECTION_NAMESPACE
                                ),
                            },
                        }
                    )
                    appended.append(template_record)
            include_datasets = kwargs.get("include_dataset_ids")
            if include_datasets is not None:
                allowed_datasets = {str(value) for value in include_datasets}
                appended = [
                    record
                    for record in appended
                    if str(record.get("dataset_id", "")) in allowed_datasets
                ]
            include_roles = kwargs.get("include_training_roles")
            if include_roles is not None:
                allowed_roles = {str(value) for value in include_roles}
                missing_roles = [
                    str(record.get("episode_id", ""))
                    for record in appended
                    if not str(record.get("training_role", "")).strip()
                ]
                if missing_roles:
                    raise RuntimeError(
                        "configured training-role filtering requires synthetic "
                        f"training_role metadata; missing examples={missing_roles[:5]}"
                    )
                appended = [
                    record
                    for record in appended
                    if str(record["training_role"]) in allowed_roles
                ]
            self.records = list(self.records) + appended
            self._opensees_manifest_path = manifest_path

        def __getitem__(self, index: int) -> Any:
            if index < self._opensees_real_record_count:
                return super().__getitem__(index)
            record = self.records[index]
            descriptor = dict(record.get("_opensees_synthetic_row") or {})
            if not descriptor:
                return super().__getitem__(index)
            if self._opensees_template_dataset is None:
                raise RuntimeError("synthetic template dataset was not initialized")
            waveform_template_index = int(descriptor["waveform_template_index"])
            if not 0 <= waveform_template_index < len(
                self._opensees_template_dataset.records
            ):
                raise RuntimeError(
                    "synthetic waveform template index is outside the expanded "
                    f"train view: {waveform_template_index}"
                )
            base_episode = self._opensees_template_dataset[
                waveform_template_index
            ]
            structured_projection = bool(
                descriptor.get("_structured_set_event_projection")
            )
            if structured_projection:
                affected_value = descriptor.get("affected_target")
                affected_target = (
                    None if affected_value is None else int(affected_value)
                )
                label = _joint_synthetic_label(affected_target)
            else:
                label_template_index = int(descriptor["label_template_index"])
                if not 0 <= label_template_index < len(
                    self._opensees_template_dataset.records
                ):
                    raise RuntimeError(
                        "synthetic label template index is outside the expanded "
                        f"train view: {label_template_index}"
                    )
                label = self._opensees_template_dataset[
                    label_template_index
                ].label
            waveform_path = Path(descriptor["waveform_path"])
            if not waveform_path.is_absolute() and self._opensees_manifest_path is not None:
                waveform_path = (self._opensees_manifest_path.parent / waveform_path).resolve()
            with np.load(waveform_path, allow_pickle=False) as archive:
                arrays = [archive[f"board_{board_index}"] for board_index in range(6)]
            if len(base_episode.boards) != 6:
                raise RuntimeError("synthetic augmentation requires exactly six boards")
            boards = []
            for board_index, (board, array) in enumerate(
                zip(base_episode.boards, arrays, strict=True)
            ):
                waveform_field = _find_waveform_field(board)
                template = getattr(board, waveform_field)
                if tuple(np.asarray(template).shape) != tuple(array.shape):
                    raise RuntimeError(
                        f"synthetic board_{board_index} shape {array.shape} != "
                        f"template {np.asarray(template).shape}"
                    )
                boards.append(
                    dataclasses.replace(
                        board,
                        **{waveform_field: _convert_like(template, array)},
                    )
                )
            metadata = dict(getattr(base_episode, "metadata", {}) or {})
            metadata.update(
                {
                    "source_kind": descriptor.get("source_kind"),
                    "synthetic_counterfactual_group": descriptor.get(
                        "counterfactual_group"
                    ),
                    "synthetic_physical_state": descriptor.get("physical_state"),
                    "synthetic_affected_target": descriptor.get("affected_target"),
                    "structured_set_event_projection": structured_projection,
                    "synthetic_manifest_template_index_namespace": descriptor.get(
                        "manifest_template_index_namespace"
                    ),
                    "synthetic_runtime_projection_namespace": descriptor.get(
                        "runtime_projection_namespace"
                    ),
                }
            )
            board_collection = (
                tuple(boards)
                if isinstance(base_episode.boards, tuple)
                else list(boards)
            )
            provenance = getattr(base_episode, "provenance", None)
            if dataclasses.is_dataclass(provenance):
                operations = list(getattr(provenance, "derived_operations", []) or [])
                operation = (
                    "opensees_factorized_manifest_to_joint_event_projection"
                    if structured_projection
                    else "opensees_factorized_task_augmentation"
                )
                if operation not in operations:
                    operations.append(operation)
                provenance = _replace_dataclass_if_present(
                    provenance,
                    {
                        "dataset_id": descriptor.get("parent_dataset"),
                        "provenance_type": "synthetic_counterfactual",
                        "structure_id": descriptor.get("synthetic_structure_id"),
                        "session_id": descriptor.get("counterfactual_group"),
                        "experiment_id": (
                            _event_record_id(descriptor)
                            if structured_projection
                            else descriptor.get("record_id")
                        ),
                        "split_group": descriptor.get("counterfactual_group"),
                        "derived_operations": operations,
                        "local_training_data": False,
                    },
                )
            episode = _replace_dataclass_if_present(
                base_episode,
                {
                    "boards": board_collection,
                    "label": label,
                    "provenance": provenance,
                    "episode_id": (
                        _event_record_id(descriptor)
                        if structured_projection
                        else descriptor.get("record_id")
                    ),
                    "dataset_id": descriptor.get("parent_dataset"),
                    "structure_id": descriptor.get("synthetic_structure_id"),
                    "session_id": descriptor.get("counterfactual_group"),
                    "split": "train",
                    "metadata": metadata,
                },
            )
            validate = getattr(episode, "validate", None)
            if callable(validate):
                validate(require_label=True)
            return episode

    OpenSeesAugmentedEpisodeDataset.__name__ = base_class.__name__
    OpenSeesAugmentedEpisodeDataset.__qualname__ = base_class.__qualname__
    OpenSeesAugmentedEpisodeDataset.__module__ = base_class.__module__
    return OpenSeesAugmentedEpisodeDataset
