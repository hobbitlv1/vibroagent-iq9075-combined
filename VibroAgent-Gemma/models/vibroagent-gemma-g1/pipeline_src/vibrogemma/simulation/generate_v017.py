from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .conditioning_v017 import (
    MEASURED_TRANSFER_CALIBRATION_SCHEMA,
    MeasuredTransferCalibration,
    apply_simulated_transfer,
    estimate_dataset_calibration,
    fit_measured_transfer_calibration,
    measured_transfer_calibration_error,
)
from .lumo_fe_v019 import LUMO_ABAQUS_SOURCE_SHA256
from .modal_synthesis_v019 import SensorModalState, synthesize_modal_family
from .opensees_v017 import TwinParameters, run_counterfactual_family, sha256_file
from .synthetic_dataset_v017 import (
    CALIBRATED_SYNTHETIC_SOURCE_KIND,
    INVERSE_FE_SYNTHETIC_SOURCE_KIND,
    LUMO_INVERSE_FE_SCHEMA,
    LUMO_TARGET_DAMAGE_LEVELS,
    _inverse_fe_artifact_issues,
    audit_synthetic_manifest,
)


def _record_group(record: Mapping[str, Any]) -> str:
    return str(
        record.get("_task_group")
        or record.get("sampling_group")
        or record.get("split_group")
        or record.get("session_id")
        or ""
    )


def _task_key(record: Mapping[str, Any]) -> tuple[str, int | None, str]:
    family = str(record.get("_task_family") or "global")
    target = record.get("_task_target_index")
    return family, (int(target) if target is not None else None), str(record.get("_task_state") or "")


def _extract_array(board: Any) -> np.ndarray:
    import dataclasses as dc

    if not dc.is_dataclass(board):
        raise TypeError(type(board))
    candidates: list[tuple[int, np.ndarray]] = []
    for field in dc.fields(board):
        value = getattr(board, field.name)
        if isinstance(value, np.ndarray) and value.ndim in {1, 2}:
            candidates.append((int(value.size), value))
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor) and value.ndim in {1, 2}:
                    array = value.detach().cpu().numpy()
                    candidates.append((int(array.size), array))
            except Exception:
                pass
    if not candidates:
        raise AttributeError(f"no waveform array found in {type(board)!r}")
    return np.asarray(max(candidates, key=lambda item: item[0])[1])


def _sample_rate(episode: Any) -> float:
    for owner in (episode, *getattr(episode, "boards", ())):
        for name in ("sample_rate_hz", "sampling_rate_hz", "fs_hz", "sample_rate"):
            value = getattr(owner, name, None)
            if value is not None and float(value) > 0:
                return float(value)
    raise AttributeError("cannot determine episode sample rate")


def _affected_targets_from_record(record: Mapping[str, Any]) -> set[int]:
    direct = record.get("affected_target_indices")
    if direct is None:
        direct = record.get("affected_targets")
    if isinstance(direct, (list, tuple, set)):
        return {int(value) for value in direct}
    label = record.get("label")
    if isinstance(label, Mapping):
        direct = label.get("affected_targets")
        if isinstance(direct, (list, tuple, set)):
            return {int(value) for value in direct}
        targets = label.get("targets")
        if isinstance(targets, Sequence):
            output: set[int] = set()
            for item in targets:
                if not isinstance(item, Mapping):
                    continue
                state = str(item.get("state") or item.get("label") or item.get("class") or "")
                if state in {"unknown_anomaly", "anomaly", "affected"}:
                    index = item.get("target_index") or item.get("index") or item.get("sensor_id")
                    if index is not None:
                        text = str(index).replace("target_", "").replace("T", "")
                        if text.isdigit():
                            output.add(int(text))
            return output
    for key in ("localization", "supervision", "metadata"):
        child = record.get(key)
        if isinstance(child, Mapping):
            nested = _affected_targets_from_record(child)
            if nested:
                return nested
    return set()


def _affected_targets_from_episode(episode: Any) -> set[int]:
    label = getattr(episode, "label", None)
    if label is None:
        return set()
    direct = getattr(label, "affected_targets", None)
    if direct is not None:
        try:
            return {int(value) for value in direct}
        except Exception:
            pass
    output: set[int] = set()
    for collection_name in ("targets", "target_records", "sensors", "sensor_records"):
        collection = getattr(label, collection_name, None)
        if collection is None:
            continue
        try:
            iterator = list(collection)
        except Exception:
            continue
        for ordinal, item in enumerate(iterator, start=1):
            state = str(
                getattr(item, "state", None)
                or getattr(item, "label", None)
                or getattr(item, "class_label", None)
                or ""
            )
            if state not in {"unknown_anomaly", "anomaly", "affected"}:
                continue
            index = (
                getattr(item, "target_index", None)
                or getattr(item, "index", None)
                or getattr(item, "sensor_id", None)
                or getattr(item, "board_id", None)
                or ordinal
            )
            text = str(index).replace("target_", "").replace("T", "")
            if text.isdigit():
                output.add(int(text))
    return output


def _choose_template_maps(
    dataset: Any,
    parent_datasets: Sequence[str],
    *,
    allow_joint_event_label_placeholders: bool = False,
) -> tuple[
    dict[tuple[str, str, int | None], int],
    dict[tuple[str, int | None, str], int],
]:
    waveform_templates: dict[tuple[str, str, int | None], int] = {}
    label_templates: dict[tuple[str, int | None, str], int] = {}
    for index, record in enumerate(dataset.records):
        family, target, state = _task_key(record)
        dataset_id = str(record.get("dataset_id") or "")
        if state == "normal" and dataset_id in parent_datasets:
            waveform_templates.setdefault((dataset_id, family, target), index)
        if dataset_id == "qugs":
            if family == "target":
                label_templates.setdefault((family, target, state), index)
            elif family == "global" and state == "normal":
                label_templates.setdefault((family, None, state), index)
            elif family == "global" and state == "unknown_anomaly":
                affected = _affected_targets_from_record(record)
                if len(affected) == 1:
                    label_templates.setdefault((family, int(next(iter(affected))), state), index)
    # Some installations use a longer QUGS identifier. Re-scan by suffix.
    if not any(key[0] == "global" and key[2] == "unknown_anomaly" for key in label_templates):
        for index, record in enumerate(dataset.records):
            if "qugs" not in str(record.get("dataset_id") or "").lower():
                continue
            family, target, state = _task_key(record)
            if family == "target":
                label_templates.setdefault((family, target, state), index)
            elif family == "global" and state == "normal":
                label_templates.setdefault((family, None, state), index)
            elif family == "global" and state == "unknown_anomaly":
                affected = _affected_targets_from_record(record)
                if len(affected) == 1:
                    label_templates.setdefault((family, int(next(iter(affected))), state), index)
    # Final fallback: inspect materialized global labels.  This remains train-only
    # and is needed when the raw task record does not repeat affected_targets.
    missing_global_targets = [
        target for target in range(1, 6)
        if ("global", target, "unknown_anomaly") not in label_templates
    ]
    if missing_global_targets:
        for index, record in enumerate(dataset.records):
            if not missing_global_targets:
                break
            if "qugs" not in str(record.get("dataset_id") or "").lower():
                continue
            family, _target, state = _task_key(record)
            if family != "global" or state != "unknown_anomaly":
                continue
            affected = _affected_targets_from_episode(dataset[index])
            if len(affected) == 1:
                target = int(next(iter(affected)))
                label_templates.setdefault(("global", target, state), index)
                if target in missing_global_targets:
                    missing_global_targets.remove(target)

    missing_waveform = [
        (parent, family, target)
        for parent in parent_datasets
        for family, target in [("global", None), *[("target", i) for i in range(1, 6)]]
        if (parent, family, target) not in waveform_templates
    ]
    missing_labels = []
    for family, target, state in [
        ("global", None, "normal"),
        *[("global", i, "unknown_anomaly") for i in range(1, 6)],
        *[("target", i, "normal") for i in range(1, 6)],
        *[("target", i, "unknown_anomaly") for i in range(1, 6)],
    ]:
        if (family, target, state) not in label_templates:
            missing_labels.append((family, target, state))
    if missing_labels and allow_joint_event_label_placeholders:
        # Structured-set training reconstructs a complete synthetic label from
        # ``affected_target`` at runtime.  Its legacy six-row manifest still
        # carries label-template indices, but those templates are never used by
        # the joint-event projection.  Permit a same-family/state train-only
        # placeholder when a LUMO-only store lacks QUGS templates (notably the
        # publisher-unmeasured T1/T4 positives).  The dataset wrapper rejects
        # this manifest if factorized training is requested.
        placeholders: dict[tuple[str, str], int] = {}
        for index, record in enumerate(dataset.records):
            dataset_id = str(record.get("dataset_id") or "")
            if dataset_id not in parent_datasets:
                continue
            family, _target, state = _task_key(record)
            placeholders.setdefault((family, state), index)
        still_missing: list[tuple[str, int | None, str]] = []
        for key in missing_labels:
            placeholder = placeholders.get((key[0], key[2]))
            if placeholder is None:
                still_missing.append(key)
            else:
                label_templates[key] = placeholder
        missing_labels = still_missing
    if missing_waveform or missing_labels:
        raise RuntimeError({"missing_waveform_templates": missing_waveform, "missing_label_templates": missing_labels})
    return waveform_templates, label_templates


def _band_limited_force(rng: np.random.Generator, steps: int, dt: float, max_hz: float) -> np.ndarray:
    white = rng.normal(size=steps)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(steps, d=dt)
    envelope = np.exp(-np.square(frequencies / max(max_hz, 0.5)))
    envelope[frequencies < 0.1] *= frequencies[frequencies < 0.1] / 0.1
    force = np.fft.irfft(spectrum * envelope, n=steps)
    force -= np.mean(force)
    force /= np.std(force) + 1e-12
    return force.astype(np.float64)


def _make_parameters(calibration: Any, rng: np.random.Generator, sim_rate_hz: float, duration_seconds: float, damage_range: tuple[float, float]) -> TwinParameters:
    masses = rng.lognormal(mean=0.0, sigma=0.15, size=6)
    base = float(calibration.stiffness_scale)
    stiffnesses = base * rng.lognormal(mean=0.0, sigma=0.22, size=6)
    return TwinParameters(
        masses=tuple(float(v) for v in masses),
        stiffnesses=tuple(float(v) for v in stiffnesses),
        damping_ratio=float(rng.uniform(0.01, 0.05)),
        damage_fraction=float(rng.uniform(*damage_range)),
        dt=1.0 / float(sim_rate_hz),
        steps=int(round(duration_seconds * sim_rate_hz)),
    )


def _clone_task_record(
    template: Mapping[str, Any],
    *,
    parent_dataset: str,
    group: str,
    record_id: str,
    source_kind: str,
) -> dict[str, Any]:
    record = copy.deepcopy(dict(template))
    record["dataset_id"] = parent_dataset
    record["split"] = "train"
    for key in ("_task_group", "sampling_group", "split_group", "session_id"):
        if key in record or key == "_task_group":
            record[key] = group
    record["episode_id"] = record_id
    record["source_kind"] = source_kind
    return record


def _load_lumo_inverse_fe_artifact(
    path: Path, *, episode_store_index: Path
) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_targets = LUMO_TARGET_DAMAGE_LEVELS
    modal_library = dict(payload.get("modal_library") or {})
    required_states = {"healthy"} | {
        f"{damage}_{severity}"
        for levels in expected_targets.values()
        for damage in levels
        for severity in ("010", "111")
    }
    request_sha = str(payload.get("request_sha256") or "")
    source_model_sha = str(dict(payload.get("source_model") or {}).get("sha256") or "")
    def valid_sha(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )
    if (
        payload.get("schema") != LUMO_INVERSE_FE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("dataset_id") != "lumo"
        or payload.get("source_split") != "train"
        or payload.get("validation_access") is not False
        or payload.get("test_access") is not False
        or payload.get("episode_store_index_sha256")
        != sha256_file(Path(episode_store_index).resolve())
        or payload.get("target_damage_levels") != expected_targets
        or set(modal_library) != required_states
        or not valid_sha(request_sha)
        or not valid_sha(source_model_sha)
        or source_model_sha != LUMO_ABAQUS_SOURCE_SHA256
        or payload.get("official_fe_sha256") != LUMO_ABAQUS_SOURCE_SHA256
    ):
        raise ValueError("LUMO inverse-FE artifact provenance or modal coverage is invalid")
    artifact_issues = _inverse_fe_artifact_issues(payload)
    if artifact_issues:
        raise ValueError(f"LUMO inverse-FE artifact is invalid: {artifact_issues}")
    return payload, sha256_file(path)


def _inverse_modal_selection(
    *, target: int, group_index: int, damage_levels: Sequence[str]
) -> tuple[str, str]:
    if target not in range(1, 6) or group_index < 0 or not damage_levels:
        raise ValueError("invalid inverse-FE modal selection inputs")
    damage_level = str(damage_levels[(group_index // 2) % len(damage_levels)])
    # Severity is shared across target counterfactuals in one family so it cannot
    # become a target-identity shortcut.
    pattern = ("010", "111")[group_index % 2]
    return damage_level, pattern


def _fit_measured_calibrations(
    dataset: Any,
    parent_datasets: Sequence[str],
    *,
    max_transfer_hz: float,
    minimum_target_groups: Mapping[str, int],
    missing_target_policy: str,
    maximum_geometry_consistency_error: float,
) -> dict[str, MeasuredTransferCalibration]:
    examples: dict[
        str, list[tuple[int | None, str, Sequence[np.ndarray], float]]
    ] = {parent: [] for parent in parent_datasets}
    for index, record in enumerate(dataset.records):
        if str(record.get("split") or "") != "train":
            raise RuntimeError("measured calibration dataset contains a non-train row")
        parent = str(record.get("dataset_id") or "")
        if parent not in examples or not bool(record.get("target_supervision_available", True)):
            continue
        affected = _affected_targets_from_record(record)
        global_state = str(record.get("global_class_name") or record.get("class_name") or "")
        if not affected and global_state == "normal":
            target = None
        elif len(affected) == 1:
            target = int(next(iter(affected)))
        else:
            raise RuntimeError(
                f"{parent} calibration row {record.get('episode_id')} has ambiguous targets: "
                f"{sorted(affected)}"
            )
        group = str(record.get("split_group") or "")
        if not group:
            raise RuntimeError(f"{parent} calibration row lacks a physical split_group")
        episode = dataset[index]
        examples[parent].append(
            (
                target,
                group,
                [_extract_array(board) for board in episode.boards],
                _sample_rate(episode),
            )
        )
    return {
        parent: fit_measured_transfer_calibration(
            dataset_id=parent,
            examples=examples[parent],
            max_frequency_hz=max_transfer_hz,
            minimum_target_groups=int(minimum_target_groups[parent]),
            missing_target_policy=missing_target_policy,
            maximum_geometry_consistency_error=maximum_geometry_consistency_error,
            expected_measured_targets=(
                (1, 2, 3, 4, 5)
                if parent == "qugs"
                else (2, 3, 5) if parent == "lumo" else None
            ),
        )
        for parent in parent_datasets
    }


def generate_balanced_augmentation(
    *,
    episode_store_index: Path,
    binary: Path,
    output_root: Path,
    parent_datasets: Sequence[str] = ("qugs", "lumo"),
    synthetic_fraction: float = 0.30,
    seed: int = 20260824,
    max_transfer_hz: float = 200.0,
    simulation_rate_hz: float = 512.0,
    damage_range: tuple[float, float] = (0.08, 0.40),
    synthetic_groups_per_parent: int | None = None,
    calibration_strength: float = 1.0,
    calibration_min_groups_per_target: int = 2,
    lumo_calibration_min_groups_per_target: int = 1,
    calibration_max_mean_log_error: float = 0.35,
    calibration_missing_target_policy: str = "lumo_ordinal_derived",
    calibration_max_geometry_consistency_error: float = 0.70,
    lumo_inverse_fe_calibration: Path | None = None,
    inverse_fe_max_mean_log_error: float = 0.70,
    joint_event_projection_only: bool = False,
) -> dict[str, Any]:
    """Generate train-only, measured-calibrated target-balanced counterfactual augmentation."""

    if not 0.0 < synthetic_fraction < 0.8:
        raise ValueError("synthetic_fraction must be in (0, 0.8)")
    if synthetic_groups_per_parent is not None and int(synthetic_groups_per_parent) < 1:
        raise ValueError("synthetic_groups_per_parent must be >= 1 when provided")
    if not 0.0 <= calibration_strength <= 1.0:
        raise ValueError("calibration_strength must be in [0, 1]")
    if calibration_min_groups_per_target < 1:
        raise ValueError("calibration_min_groups_per_target must be positive")
    if lumo_calibration_min_groups_per_target < 1:
        raise ValueError("lumo_calibration_min_groups_per_target must be positive")
    if calibration_max_mean_log_error <= 0:
        raise ValueError("calibration_max_mean_log_error must be positive")
    if calibration_missing_target_policy not in {"fail", "lumo_ordinal_derived"}:
        raise ValueError("invalid calibration_missing_target_policy")
    if calibration_max_geometry_consistency_error <= 0.0:
        raise ValueError("calibration_max_geometry_consistency_error must be positive")
    if inverse_fe_max_mean_log_error <= 0.0:
        raise ValueError("inverse_fe_max_mean_log_error must be positive")
    os.environ.pop("VIBROGEMMA_SYNTHETIC_AUGMENTATION_MANIFEST", None)
    from vibrogemma.data.episodes import EpisodeDataset

    parent_datasets = tuple(str(value) for value in parent_datasets)
    inverse_payload: dict[str, Any] | None = None
    inverse_sha: str | None = None
    inverse_path: Path | None = None
    if lumo_inverse_fe_calibration is not None:
        if parent_datasets != ("lumo",):
            raise ValueError("the publisher-geometry inverse-FE challenger is LUMO-only")
        inverse_path = Path(lumo_inverse_fe_calibration).expanduser().resolve()
        inverse_payload, inverse_sha = _load_lumo_inverse_fe_artifact(
            inverse_path, episode_store_index=episode_store_index
        )
    if joint_event_projection_only and inverse_payload is None:
        raise ValueError(
            "joint_event_projection_only requires the strict LUMO inverse-FE path"
        )
    calibration_dataset = EpisodeDataset(
        episode_store_index,
        "train",
        require_labels=True,
        expand_supervision_tasks=False,
        target_permutation_augmentation=False,
    )
    calibrations = _fit_measured_calibrations(
        calibration_dataset,
        parent_datasets,
        max_transfer_hz=max_transfer_hz,
        minimum_target_groups={
            parent: (
                lumo_calibration_min_groups_per_target
                if parent == "lumo"
                else calibration_min_groups_per_target
            )
            for parent in parent_datasets
        },
        missing_target_policy=calibration_missing_target_policy,
        maximum_geometry_consistency_error=calibration_max_geometry_consistency_error,
    )
    dataset = EpisodeDataset(
        episode_store_index,
        "train",
        require_labels=True,
        expand_supervision_tasks=True,
        target_permutation_augmentation=False,
        target_permutation_seed=seed + 2,
    )
    waveform_templates, label_templates = _choose_template_maps(
        dataset,
        parent_datasets,
        allow_joint_event_label_placeholders=joint_event_projection_only,
    )

    real_groups_by_parent: dict[str, list[str]] = {parent: [] for parent in parent_datasets}
    healthy_global_indices: dict[tuple[str, str], int] = {}
    for index, record in enumerate(dataset.records):
        parent = str(record.get("dataset_id") or "")
        if parent not in parent_datasets:
            continue
        family, target, state = _task_key(record)
        if family == "global" and state == "normal":
            group = _record_group(record)
            if group:
                healthy_global_indices.setdefault((parent, group), index)
                if group not in real_groups_by_parent[parent]:
                    real_groups_by_parent[parent].append(group)
    for parent, groups in real_groups_by_parent.items():
        if not groups:
            raise RuntimeError(f"no healthy train groups for parent dataset {parent}")

    output_root = output_root.resolve()
    wave_root = output_root / "waveforms"
    sim_root = output_root / "simulations"
    wave_root.mkdir(parents=True, exist_ok=True)
    sim_root.mkdir(parents=True, exist_ok=True)
    calibration_path = output_root / "measured_transfer_calibration.json"
    calibration_payload = {
        "schema": MEASURED_TRANSFER_CALIBRATION_SCHEMA,
        "source_split": "train",
        "episode_store_index": str(Path(episode_store_index).resolve()),
        "episode_store_index_sha256": sha256_file(Path(episode_store_index).resolve()),
        "calibrations": {
            parent: calibrations[parent].to_payload() for parent in parent_datasets
        },
    }
    calibration_path.write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calibration_sha = sha256_file(calibration_path)
    manifest_path = output_root / "synthetic_tasks.jsonl"
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {}

    for parent in parent_datasets:
        measured_calibration = calibrations[parent]
        groups = sorted(real_groups_by_parent[parent])
        synthetic_groups = (
            int(synthetic_groups_per_parent)
            if synthetic_groups_per_parent is not None
            else max(1, int(round(len(groups) * synthetic_fraction / (1.0 - synthetic_fraction))))
        )
        parent_summary = {
            "real_healthy_groups": len(groups),
            "synthetic_groups": synthetic_groups,
            "synthetic_group_policy": (
                "fixed_per_parent" if synthetic_groups_per_parent is not None else "fraction_derived"
            ),
            "calibration_healthy_groups": measured_calibration.healthy_group_count,
            "calibration_target_groups": measured_calibration.target_group_counts,
        }
        group_summary[parent] = parent_summary
        for group_index in range(synthetic_groups):
            source_group = groups[group_index % len(groups)]
            source_global_index = healthy_global_indices[(parent, source_group)]
            source_episode = dataset[source_global_index]
            board_values = [_extract_array(board) for board in source_episode.boards]
            sample_rate = _sample_rate(source_episode)
            sample_count = min(max(np.asarray(value).shape) for value in board_values)
            duration = sample_count / sample_rate
            synthetic_group = f"opensees::{parent}::{seed}::{group_index:05d}"
            state_sim_dir = sim_root / parent / f"group-{group_index:05d}"
            selected_modal_states: dict[str, dict[str, Any]] = {}
            if inverse_payload is not None:
                modal_library = dict(inverse_payload["modal_library"])
                target_damage_levels = dict(inverse_payload["target_damage_levels"])
                states = {"healthy": SensorModalState.from_payload(modal_library["healthy"])}
                selected_modal_states["healthy"] = {
                    "modal_state_key": "healthy",
                    "damage_level": None,
                    "damage_pattern": None,
                }
                for target in range(1, 6):
                    levels = target_damage_levels[f"target_{target}"]
                    damage_level, severity = _inverse_modal_selection(
                        target=target,
                        group_index=group_index,
                        damage_levels=levels,
                    )
                    key = f"{damage_level}_{severity}"
                    states[f"target_{target}"] = SensorModalState.from_payload(
                        modal_library[key]
                    )
                    selected_modal_states[f"target_{target}"] = {
                        "modal_state_key": key,
                        "damage_level": damage_level,
                        "damage_pattern": severity,
                    }
                simulation = synthesize_modal_family(
                    states=states,
                    rng=rng,
                    sample_rate_hz=simulation_rate_hz,
                    duration_seconds=min(duration, 10.0),
                    max_frequency_hz=min(max_transfer_hz, simulation_rate_hz / 3.0),
                )
                state_sim_dir.mkdir(parents=True, exist_ok=True)
                (state_sim_dir / "counterfactual_receipt.json").write_text(
                    json.dumps(
                        {
                            "schema": "vibrogemma-lumo-modal-counterfactual-v0.19.0",
                            "inverse_fe_calibration": str(inverse_path),
                            "inverse_fe_calibration_sha256": inverse_sha,
                            "request_sha256": inverse_payload["request_sha256"],
                            "selected_modal_states": selected_modal_states,
                            "shared_modal_excitation": True,
                            "simulation_rate_hz": simulation_rate_hz,
                            "duration_seconds": min(duration, 10.0),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                source_kind = INVERSE_FE_SYNTHETIC_SOURCE_KIND
            else:
                carrier_calibration = estimate_dataset_calibration(
                    board_values=board_values,
                    sample_rate_hz=sample_rate,
                    duration_seconds=duration,
                    max_frequency_hz=max_transfer_hz,
                )
                parameters = _make_parameters(
                    carrier_calibration,
                    rng,
                    simulation_rate_hz,
                    min(duration, 10.0),
                    damage_range,
                )
                force = _band_limited_force(
                    rng,
                    parameters.steps,
                    parameters.dt,
                    max_hz=min(max_transfer_hz, simulation_rate_hz / 3.0),
                )
                simulation = run_counterfactual_family(
                    binary=binary,
                    work_dir=state_sim_dir,
                    force=force,
                    parameters=parameters,
                )
                source_kind = CALIBRATED_SYNTHETIC_SOURCE_KIND
            carriers: list[np.ndarray] = []
            for board, real_board in zip(source_episode.boards, board_values, strict=True):
                gain = float(rng.uniform(0.97, 1.03))
                noise_scale = float(np.std(real_board) * rng.uniform(0.001, 0.005))
                noise = rng.normal(scale=noise_scale, size=np.asarray(real_board).shape)
                axis_mask = np.asarray(getattr(board, "axis_mask", ()), dtype=bool)
                if axis_mask.shape == (3,) and np.asarray(real_board).ndim == 2:
                    if np.asarray(real_board).shape[0] == 3:
                        noise *= axis_mask[:, None]
                    elif np.asarray(real_board).shape[1] == 3:
                        noise *= axis_mask[None, :]
                carriers.append(
                    np.asarray(
                        np.asarray(real_board) * gain + noise,
                        dtype=np.asarray(real_board).dtype,
                    )
                )
            physical_states: list[tuple[str, int | None]] = [("healthy", None)] + [
                (f"target_{target}", target) for target in range(1, 6)
            ]
            for physical_state, affected_target in physical_states:
                state_waveforms: list[np.ndarray] = []
                for board_index, carrier in enumerate(carriers):
                    if affected_target is None:
                        synthetic_board = carrier
                    else:
                        synthetic_board = apply_simulated_transfer(
                            real_values=np.asarray(carrier),
                            simulated_healthy=simulation["healthy"][board_index],
                            simulated_damaged=simulation[physical_state][board_index],
                            real_sample_rate_hz=sample_rate,
                            simulated_sample_rate_hz=simulation_rate_hz,
                            max_transfer_hz=max_transfer_hz,
                            measured_band_gains=(
                                None
                                if inverse_payload is not None
                                else measured_calibration.gains_by_target[affected_target][
                                    board_index
                                ]
                            ),
                            calibration_band_edges_hz=(
                                None
                                if inverse_payload is not None
                                else measured_calibration.band_edges_hz
                            ),
                            calibration_strength=(
                                0.0 if inverse_payload is not None else calibration_strength
                            ),
                        )
                    state_waveforms.append(
                        np.asarray(synthetic_board, dtype=np.asarray(carrier).dtype)
                    )
                calibration_error = None
                if affected_target is not None:
                    calibration_error = measured_transfer_calibration_error(
                        calibration=measured_calibration,
                        target=affected_target,
                        healthy_board_values=carriers,
                        damaged_board_values=state_waveforms,
                        sample_rate_hz=sample_rate,
                    )
                    allowed_calibration_error = (
                        inverse_fe_max_mean_log_error
                        if inverse_payload is not None
                        else calibration_max_mean_log_error
                    )
                    if calibration_error > allowed_calibration_error:
                        raise RuntimeError(
                            f"{synthetic_group} target_{affected_target} measured calibration "
                            f"error {calibration_error:.6f} exceeds "
                            f"{allowed_calibration_error:.6f}"
                        )
                state_id = f"{parent}-{seed}-{group_index:05d}-{physical_state}"
                waveform_path = wave_root / parent / f"{state_id}.npz"
                waveform_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    waveform_path,
                    **{f"board_{index}": value for index, value in enumerate(state_waveforms)},
                )
                waveform_sha = sha256_file(waveform_path)

                tasks: list[tuple[str, int | None, str, int]] = []
                if affected_target is None:
                    tasks.append(("global", None, "normal", label_templates[("global", None, "normal")]))
                    for focus in range(1, 6):
                        tasks.append(("target", focus, "normal", label_templates[("target", focus, "normal")]))
                else:
                    tasks.append(("global", None, "unknown_anomaly", label_templates[("global", affected_target, "unknown_anomaly")]))
                    for focus in range(1, 6):
                        state = "unknown_anomaly" if focus == affected_target else "normal"
                        tasks.append(("target", focus, state, label_templates[("target", focus, state)]))

                for task_ordinal, (family, focus, state, label_template_index) in enumerate(tasks):
                    waveform_template_index = waveform_templates[(parent, family, focus)]
                    record_id = f"{state_id}-task-{task_ordinal}"
                    task_record = _clone_task_record(
                        dataset.records[label_template_index],
                        parent_dataset=parent,
                        group=synthetic_group,
                        record_id=record_id,
                        source_kind=source_kind,
                    )
                    task_record["_task_family"] = family
                    task_record["_task_state"] = state
                    if focus is None:
                        task_record.pop("_task_target_index", None)
                    else:
                        task_record["_task_target_index"] = int(focus)
                    rows.append(
                        {
                            "schema": (
                                "vibrogemma-opensees-synthetic-task-v0.19.0"
                                if inverse_payload is not None
                                else "vibrogemma-opensees-synthetic-task-v0.18.1"
                            ),
                            "record_id": record_id,
                            "split": "train",
                            "source_kind": source_kind,
                            "measured_data": False,
                            "parent_dataset": parent,
                            "counterfactual_group": synthetic_group,
                            "source_real_group": source_group,
                            "physical_state": physical_state,
                            "affected_target": affected_target,
                            "target_permutation_augmentation": False,
                            "joint_event_projection_only": bool(
                                joint_event_projection_only
                            ),
                            "synthetic_structure_id": f"opensees::{parent}::{seed}::{group_index:05d}",
                            "waveform_path": str(waveform_path),
                            "waveform_sha256": waveform_sha,
                            "waveform_template_index": waveform_template_index,
                            "label_template_index": label_template_index,
                            "simulation_receipt": str(state_sim_dir / "counterfactual_receipt.json"),
                            "measured_calibration_path": str(calibration_path),
                            "measured_calibration_sha256": calibration_sha,
                            "measured_calibration_schema": MEASURED_TRANSFER_CALIBRATION_SCHEMA,
                            "measured_calibration_source_split": "train",
                            "measured_calibration_basis": measured_calibration.basis_by_target.get(
                                affected_target
                            ) if affected_target is not None else "healthy_measured_train_groups",
                            "measured_calibration_source_targets": list(
                                measured_calibration.source_targets_by_target.get(
                                    affected_target, ()
                                )
                            ),
                            "measured_calibration_mean_log_error": calibration_error,
                            "measured_calibration_max_mean_log_error": (
                                inverse_fe_max_mean_log_error
                                if inverse_payload is not None
                                else calibration_max_mean_log_error
                            ),
                            **(
                                {
                                    "inverse_fe_calibration_path": str(inverse_path),
                                    "inverse_fe_calibration_sha256": inverse_sha,
                                    "inverse_fe_calibration_schema": LUMO_INVERSE_FE_SCHEMA,
                                    "inverse_fe_source_split": "train",
                                    "inverse_fe_request_sha256": inverse_payload["request_sha256"],
                                    "inverse_fe_source_model_sha256": inverse_payload[
                                        "source_model"
                                    ]["sha256"],
                                    "inverse_fe_modal_state_key": selected_modal_states[
                                        physical_state
                                    ]["modal_state_key"],
                                    "inverse_fe_damage_level": selected_modal_states[
                                        physical_state
                                    ]["damage_level"],
                                    "inverse_fe_damage_pattern": selected_modal_states[
                                        physical_state
                                    ]["damage_pattern"],
                                    "inverse_fe_damage_level_basis": (
                                        None
                                        if affected_target is None
                                        else inverse_payload["damage_level_basis"][
                                            selected_modal_states[physical_state]["damage_level"]
                                        ]
                                    ),
                                    "inverse_fe_damage_pattern_basis": (
                                        None
                                        if affected_target is None
                                        else inverse_payload["damage_pattern_basis"][
                                            selected_modal_states[physical_state]["damage_pattern"]
                                        ]
                                    ),
                                }
                                if inverse_payload is not None
                                else {}
                            ),
                            "task_record": task_record,
                        }
                    )

    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    audit = audit_synthetic_manifest(
        manifest_path,
        require_calibrated=True,
        require_inverse_fe=inverse_payload is not None,
    )
    audit["generation"] = {
        "seed": seed,
        "parent_datasets": list(parent_datasets),
        "synthetic_fraction": synthetic_fraction,
        "max_transfer_hz": max_transfer_hz,
        "simulation_rate_hz": simulation_rate_hz,
        "damage_range": list(damage_range),
        "synthetic_groups_per_parent": synthetic_groups_per_parent,
        "calibration_strength": calibration_strength,
        "calibration_min_groups_per_target": calibration_min_groups_per_target,
        "lumo_calibration_min_groups_per_target": lumo_calibration_min_groups_per_target,
        "calibration_max_mean_log_error": calibration_max_mean_log_error,
        "calibration_missing_target_policy": calibration_missing_target_policy,
        "calibration_max_geometry_consistency_error": calibration_max_geometry_consistency_error,
        "inverse_fe_max_mean_log_error": inverse_fe_max_mean_log_error,
        "joint_event_projection_only": bool(joint_event_projection_only),
        "measured_calibration": {
            "schema": MEASURED_TRANSFER_CALIBRATION_SCHEMA,
            "path": str(calibration_path),
            "sha256": calibration_sha,
            "source_split": "train",
        },
        "inverse_fe_calibration": (
            {
                "schema": LUMO_INVERSE_FE_SCHEMA,
                "path": str(inverse_path),
                "sha256": inverse_sha,
                "request_sha256": inverse_payload["request_sha256"],
                "source_model_sha256": inverse_payload["source_model"]["sha256"],
            }
            if inverse_payload is not None
            else None
        ),
        "group_summary": group_summary,
        "episode_store_index": str(episode_store_index),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
    }
    audit_path = output_root / "augmentation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"generated augmentation failed audit: {audit}")
    return {"manifest": str(manifest_path), "audit": str(audit_path), **audit}
