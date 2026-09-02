from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import torch
from safetensors.torch import load_file

from ..config import resolve_from_config, resolve_static_asset
from ..contracts import (
    ClassificationResult,
    EpisodeLabel,
    TargetLabel,
    TargetPrediction,
)
from ..data.split_audit import is_supported_assignment_strategy
from ..data.splits import assert_disjoint_groups
from ..inference.policy import GeneratedOutputAlertPolicy
from ..model.pipeline import VibroGemmaForClassification
from ..model.records import GlobalPrediction
from ..training.data import build_episode_loader
from ..utils import append_jsonl, atomic_write_json, atomic_write_text, utc_now_iso
from .metrics import (
    bootstrap_intervals,
    compute_metrics,
    confidence_correctness_auroc,
    expected_calibration_error,
    global_expected_calibration_error,
)


def _load_checkpoint(model: VibroGemmaForClassification, checkpoint: Path) -> None:
    model.load_components(checkpoint, strict=False)
    adapter = checkpoint / "adapter"
    if adapter.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to evaluate a LoRA adapter checkpoint") from exc
        model.gemma = PeftModel.from_pretrained(model.gemma, adapter)
        model.structured_generator.model = model.gemma
    elif (checkpoint / "gemma_trainable.safetensors").exists():
        model.gemma.load_state_dict(
            load_file(str(checkpoint / "gemma_trainable.safetensors")), strict=False
        )


def _read_index_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _label_from_row(row: dict[str, Any]) -> EpisodeLabel:
    payload = row["ground_truth"]
    if isinstance(payload, list):
        # Backward-compatible reader for historical prediction files.
        targets_payload = payload
        target_available = True
        global_class = (
            "unknown_anomaly"
            if any(str(item["class"]) != "normal" for item in targets_payload)
            else "normal"
        )
        global_severity = (
            "advisory" if global_class == "unknown_anomaly" else "none"
        )
    else:
        targets_payload = list(payload.get("targets") or [])
        target_available = bool(payload.get("target_supervision_available", False))
        global_payload = dict(payload.get("global") or {})
        global_class = str(global_payload["class"])
        global_severity = str(global_payload["severity"])
    return EpisodeLabel(
        targets=[
            TargetLabel(
                sensor_id=str(item["sensor_id"]),
                affected=bool(item["affected"]),
                class_name=str(item["class"]),
                severity=str(item["severity"]),
                source=str(item.get("source", "evaluation")),
            )
            for item in targets_payload
        ],
        global_class_name=global_class,
        global_severity=global_severity,
        target_supervision_available=target_available,
        supervision_scope=("joint" if target_available else "global_only"),
    )


def _predictions_from_row(row: dict[str, Any]) -> list[TargetPrediction]:
    return [
        TargetPrediction(
            sensor_id=str(item["sensor_id"]),
            affected=bool(item["affected"]),
            class_name=str(item["class"]),
            severity=str(item["severity"]),
            choice_probability=(
                float(item["choice_probability"])
                if item.get("choice_probability") is not None
                else None
            ),
            raw_choice_probability=(
                float(item["raw_choice_probability"])
                if item.get("raw_choice_probability") is not None
                else None
            ),
            confidence_operational=bool(item.get("confidence_operational", False)),
        )
        for item in row["prediction"]["targets"]
    ]


def _global_prediction_from_row(
    row: dict[str, Any], *, raw: bool = False
) -> GlobalPrediction:
    model_payload = dict((row.get("prediction") or {}).get("model") or {})
    class_name = model_payload.get("global_class")
    severity = model_payload.get("global_severity")
    probability = model_payload.get(
        "raw_global_choice_probability" if raw else "global_choice_probability"
    )
    if class_name is None:
        targets = _predictions_from_row(row)
        anomalous = any(
            target.class_name not in {"normal", "data_invalid"} for target in targets
        )
        class_name = "unknown_anomaly" if anomalous else "normal"
        severity = "advisory" if anomalous else "none"
    return GlobalPrediction(
        str(class_name),
        str(severity),
        float(probability) if probability is not None else None,
    )


def _classification_result(row: dict[str, Any]) -> ClassificationResult:
    payload = row["prediction"]
    return ClassificationResult(
        schema_version=str(payload["schema_version"]),
        window=dict(payload["window"]),
        system_state=str(payload["system_state"]),
        targets=_predictions_from_row(row),
        explanation=str(payload["explanation"]),
        quality=dict(payload.get("quality") or {}),
        model=dict(payload.get("model") or {}),
        evidence=dict(payload.get("evidence") or {}),
    )


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid predictions JSONL at line {line_number}: {exc}") from exc
    return rows


def _persistence_metrics(rows: list[dict[str, Any]], evaluation_cfg: dict[str, Any]) -> dict[str, Any]:
    persistence = dict(evaluation_cfg.get("persistence") or {})
    trigger_count = int(persistence.get("trigger_count", 3))
    trigger_window = int(persistence.get("trigger_window", 5))
    clear_after = int(persistence.get("clear_after_consecutive_normal", 3))
    cooldown_seconds = float(persistence.get("cooldown_seconds", 30.0))

    sessions: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[(str(row["dataset_id"]), str(row["structure_id"]), str(row["session_id"]))].append(row)

    false_events = 0
    healthy_seconds = 0.0
    changed_sessions = 0
    detected_changed_sessions = 0
    detection_delays: list[float] = []
    by_dataset: dict[str, dict[str, float | int | None]] = defaultdict(
        lambda: {
            "healthy_seconds": 0.0,
            "false_alert_events": 0,
            "changed_sessions": 0,
            "detected_changed_sessions": 0,
        }
    )

    for (dataset_id, _structure_id, _session_id), session_rows in sessions.items():
        session_rows.sort(key=lambda item: float(item["prediction"]["window"].get("start_time_s", 0.0)))
        policy = GeneratedOutputAlertPolicy(
            trigger_count=trigger_count,
            trigger_window=trigger_window,
            clear_after_consecutive_normal=clear_after,
            cooldown_seconds=cooldown_seconds,
            alert_on_data_invalid=True,
        )
        first_truth = session_rows[0]["ground_truth"]
        if isinstance(first_truth, dict):
            healthy = str((first_truth.get("global") or {}).get("class")) == "normal"
        else:
            healthy = all(target["class"] == "normal" for target in first_truth)
        starts = [float(item["prediction"]["window"].get("start_time_s", 0.0)) for item in session_rows]
        durations = [
            float(item["prediction"]["window"].get("effective_duration_seconds", item["prediction"]["window"]["duration_seconds"]))
            for item in session_rows
        ]
        if healthy:
            session_seconds = max(start + duration for start, duration in zip(starts, durations, strict=True)) - min(starts)
            session_seconds = max(session_seconds, max(durations, default=0.0))
            healthy_seconds += session_seconds
            by_dataset[dataset_id]["healthy_seconds"] = float(by_dataset[dataset_id]["healthy_seconds"]) + session_seconds
        else:
            changed_sessions += 1
            by_dataset[dataset_id]["changed_sessions"] = int(by_dataset[dataset_id]["changed_sessions"]) + 1

        emitted_at: float | None = None
        for row, start in zip(session_rows, starts, strict=True):
            decision = policy.evaluate(_classification_result(row), now_monotonic=start)
            if decision.emit:
                if healthy:
                    false_events += 1
                    by_dataset[dataset_id]["false_alert_events"] = int(by_dataset[dataset_id]["false_alert_events"]) + 1
                elif emitted_at is None:
                    emitted_at = start
        if not healthy and emitted_at is not None:
            detected_changed_sessions += 1
            by_dataset[dataset_id]["detected_changed_sessions"] = int(by_dataset[dataset_id]["detected_changed_sessions"]) + 1
            detection_delays.append(max(0.0, emitted_at - min(starts)))

    healthy_days = healthy_seconds / 86400.0
    for values in by_dataset.values():
        dataset_days = float(values["healthy_seconds"]) / 86400.0
        values["false_alert_events_per_building_day"] = (
            float(values["false_alert_events"]) / dataset_days if dataset_days > 0 else None
        )
        changed = int(values["changed_sessions"])
        values["known_event_detection_recall"] = (
            int(values["detected_changed_sessions"]) / changed if changed > 0 else None
        )

    return {
        "policy": {
            "trigger_count": trigger_count,
            "trigger_window": trigger_window,
            "clear_after_consecutive_normal": clear_after,
            "cooldown_seconds": cooldown_seconds,
        },
        "healthy_evaluated_chronological_hours": healthy_seconds / 3600.0,
        "false_alert_events": false_events,
        "false_alert_events_per_building_day": false_events / healthy_days if healthy_days > 0 else None,
        "changed_sessions": changed_sessions,
        "detected_changed_sessions": detected_changed_sessions,
        "known_event_detection_recall": (
            detected_changed_sessions / changed_sessions if changed_sessions > 0 else None
        ),
        "median_detection_delay_seconds": (
            float(sorted(detection_delays)[len(detection_delays) // 2]) if detection_delays else None
        ),
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def _explanation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    explanations = [str(row["prediction"].get("explanation", "")) for row in rows]
    sources = [str((row["prediction"].get("model") or {}).get("explanation_source", "unknown")) for row in rows]
    counts: dict[str, int] = defaultdict(int)
    for source in sources:
        counts[source] += 1
    unique = len(set(explanations))
    return {
        "episode_count": len(rows),
        "unique_explanations": unique,
        "unique_fraction": unique / len(rows) if rows else 0.0,
        "source_counts": dict(sorted(counts.items())),
        "gemma_grounded_fraction": counts.get("gemma_grounded", 0) / len(rows) if rows else 0.0,
        "deterministic_fallback_fraction": (
            sum(value for key, value in counts.items() if key.startswith("deterministic_")) / len(rows)
            if rows
            else 0.0
        ),
    }


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_manifest_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if root.is_file():
        records.append(
            {
                "path": root.name,
                "size": root.stat().st_size,
                "sha256": _content_sha256(root),
            }
        )
    elif root.is_dir():
        for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
            records.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": _content_sha256(candidate),
                }
            )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), records


def _episode_store_fingerprint(
    config: dict[str, Any],
    *,
    split: str,
    episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Fingerprint the index and serialized metadata used by one evaluation.

    Episode IDs are intentionally stable across a label-only rebuild. Hashing only
    those IDs therefore permits stale predictions (and their embedded ground truth)
    to be reused after relabelling. The index hash catches index-level changes, while
    the two aggregate metadata hashes catch label, board-topology, provenance, and
    preprocessing-metadata changes in the serialized episodes selected for this run.
    Waveform arrays are content-addressed by the episode build and are deliberately
    not reread here.
    """

    episode_root_value = config.get("data", {}).get("episode_root")
    if not episode_root_value:
        return {
            "schema": "vibrogemma-episode-store-fingerprint-v1",
            "available": False,
            "reason": "episode_root_not_configured",
        }

    episode_root = resolve_from_config(config, episode_root_value)
    index_path = episode_root / "index.jsonl"
    if not index_path.is_file():
        return {
            "schema": "vibrogemma-episode-store-fingerprint-v1",
            "available": False,
            "reason": "episode_index_missing",
            "episode_root": str(episode_root),
            "index_path": str(index_path),
        }

    index_bytes = index_path.read_bytes()
    index_rows = [
        json.loads(line)
        for line in index_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    selected_rows = []
    for row in index_rows:
        if str(split) != "all" and str(row.get("split")) != str(split):
            continue
        episode_id = str(row.get("episode_id") or "")
        if episode_ids is not None and episode_id not in episode_ids:
            continue
        selected_rows.append(row)
    selected_rows.sort(
        key=lambda row: (
            str(row.get("episode_id") or ""),
            str(row.get("metadata_path") or ""),
        )
    )

    metadata_records: list[dict[str, Any]] = []
    ground_truth_records: list[dict[str, Any]] = []
    for row in selected_rows:
        episode_id = str(row.get("episode_id") or "")
        metadata_value = row.get("metadata_path")
        if not metadata_value:
            metadata_records.append(
                {
                    "episode_id": episode_id,
                    "metadata_path": None,
                    "sha256": None,
                    "missing": True,
                }
            )
            ground_truth_records.append(
                {
                    "episode_id": episode_id,
                    "index_ground_truth": {
                        "has_label": row.get("has_label"),
                        "global_class_name": row.get("global_class_name"),
                        "target_supervision_available": row.get(
                            "target_supervision_available"
                        ),
                        "affected_target_indices": row.get(
                            "affected_target_indices"
                        ),
                    },
                    "serialized_label": None,
                }
            )
            continue

        metadata_path = Path(str(metadata_value))
        if not metadata_path.is_absolute():
            metadata_path = episode_root / metadata_path
        metadata_path = metadata_path.resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Episode {episode_id!r} references missing metadata: {metadata_path}"
            )
        metadata_bytes = metadata_path.read_bytes()
        payload = json.loads(metadata_bytes)
        metadata_records.append(
            {
                "episode_id": episode_id,
                "metadata_path": str(metadata_path),
                "size": len(metadata_bytes),
                "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "missing": False,
            }
        )
        ground_truth_records.append(
            {
                "episode_id": episode_id,
                "index_ground_truth": {
                    "has_label": row.get("has_label"),
                    "global_class_name": row.get("global_class_name"),
                    "target_supervision_available": row.get(
                        "target_supervision_available"
                    ),
                    "affected_target_indices": row.get(
                        "affected_target_indices"
                    ),
                },
                "serialized_label": payload.get("label"),
            }
        )

    canonical_metadata = json.dumps(
        metadata_records, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    canonical_ground_truth = json.dumps(
        ground_truth_records, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return {
        "schema": "vibrogemma-episode-store-fingerprint-v1",
        "available": True,
        "episode_root": str(episode_root),
        "index_path": str(index_path),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "split": str(split),
        "selected_episode_count": len(selected_rows),
        "selected_episode_ids_sha256": hashlib.sha256(
            "\n".join(str(row.get("episode_id") or "") for row in selected_rows).encode(
                "utf-8"
            )
        ).hexdigest(),
        "serialized_metadata_sha256": hashlib.sha256(canonical_metadata).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(canonical_ground_truth).hexdigest(),
        "metadata_file_count": sum(
            not bool(record["missing"]) for record in metadata_records
        ),
        "missing_metadata_count": sum(
            bool(record["missing"]) for record in metadata_records
        ),
    }


def _evaluation_signature(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    episode_ids: set[str] | None = None,
    include_explanations: bool = True,
    *,
    episode_store_fingerprint: dict[str, Any] | None = None,
) -> str:
    calibration = config.get("generation", {}).get("confidence_calibration_path")
    calibration_record: dict[str, Any] | None = None
    if calibration:
        calibration_path = resolve_from_config(config, str(calibration))
        calibration_record = {
            "path": str(calibration_path),
            "sha256": (
                hashlib.sha256(calibration_path.read_bytes()).hexdigest()
                if calibration_path.exists()
                else None
            ),
        }

    profile_path_value = (
        config.get("model", {})
        .get("multiresolution", {})
        .get("physics", {})
        .get("healthy_profile_path")
    )
    profile_record: dict[str, Any] | None = None
    if profile_path_value:
        profile_path = resolve_from_config(config, str(profile_path_value))
        profile_record = {
            "path": str(profile_path),
            "sha256": (
                hashlib.sha256(profile_path.read_bytes()).hexdigest()
                if profile_path.exists()
                else None
            ),
        }

    checkpoint_manifest_sha256, checkpoint_manifest = _file_manifest_hash(checkpoint_path)
    episode_store = (
        episode_store_fingerprint
        if episode_store_fingerprint is not None
        else _episode_store_fingerprint(
            config,
            split=split,
            episode_ids=episode_ids,
        )
    )
    schema_path = resolve_static_asset(config, "schemas/alert.schema.json")
    schema_record = {
        "path": str(schema_path),
        "sha256": (
            hashlib.sha256(schema_path.read_bytes()).hexdigest()
            if schema_path.exists()
            else None
        ),
    }
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "split": split,
        "episode_store": episode_store,
        "calibration": calibration_record,
        "healthy_profile": profile_record,
        "alert_schema": schema_record,
        "config_sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "episode_ids_sha256": (
            hashlib.sha256("\n".join(sorted(episode_ids)).encode("utf-8")).hexdigest()
            if episode_ids is not None
            else None
        ),
        "include_explanations": bool(include_explanations),
        "model": config.get("model", {}),
        "labels": config.get("labels", {}),
        "generation": config.get("generation", {}),
        "data_window_seconds": config.get("data", {}).get("window_seconds"),
        "schema": "factorized_partial_label_evaluation_signature_v4",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _common_mode_residual_metadata(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the non-classifying residual adapter diagnostics.

    These values are descriptive only. They never become a class score or an
    anomaly threshold. The gate is part of the vibration-token correction path.
    """

    records: list[dict[str, Any]] = []
    for row in prediction_rows:
        model_record = dict(row.get("prediction", {}).get("model") or {})
        residual = dict(model_record.get("common_mode_residual") or {})
        if residual:
            records.append(residual)
    if not records:
        return {
            "enabled": False,
            "gate_values": [],
            "maximum_absolute_gate": 0.0,
            "mean_correction_rms": 0.0,
            "maximum_correction_rms": 0.0,
            "maximum_correction_abs": 0.0,
            "episode_count": 0,
        }

    gate_values = [float(value) for value in records[0].get("gate_values", [])]
    correction_rms = [float(record.get("correction_rms", 0.0)) for record in records]
    correction_max = [float(record.get("correction_max_abs", 0.0)) for record in records]
    return {
        "enabled": any(bool(record.get("enabled", False)) for record in records),
        "gate_values": gate_values,
        "maximum_absolute_gate": max((abs(value) for value in gate_values), default=0.0),
        "mean_correction_rms": sum(correction_rms) / max(len(correction_rms), 1),
        "maximum_correction_rms": max(correction_rms, default=0.0),
        "maximum_correction_abs": max(correction_max, default=0.0),
        "episode_count": len(records),
    }


def evaluate_checkpoint(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    split: str = "test",
    overwrite: bool = False,
    episode_ids: set[str] | None = None,
    output_name: str | None = None,
    bootstrap_samples: int | None = None,
    include_persistence: bool = True,
    include_explanations: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    model = VibroGemmaForClassification.from_config(config, training=False)
    _load_checkpoint(model, checkpoint_path)
    model.move_components_to_gemma_device()
    model.eval()

    from ..model.gemma_bridge import resolve_text_model

    text_model = resolve_text_model(model.gemma)
    evaluation_device = text_model.embed_tokens.weight.device
    require_cuda = bool(config.get("evaluation", {}).get("require_cuda", torch.cuda.is_available()))
    if require_cuda and evaluation_device.type != "cuda":
        raise RuntimeError(
            "CUDA is available and evaluation.require_cuda is enabled, but Gemma was loaded on "
            f"{evaluation_device}. Set model.device=auto and rerun with --overwrite."
        )
    print(
        json.dumps(
            {
                "event": "evaluation_runtime",
                "device": str(evaluation_device),
                "cuda_available": torch.cuda.is_available(),
                "checkpoint": str(checkpoint_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    index_path = resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"
    all_rows = _read_index_rows(index_path)
    assert_disjoint_groups(all_rows)
    expected_rows = [
        row
        for row in all_rows
        if row["split"] == split
        and bool(row.get("has_label", True))
        and (episode_ids is None or str(row["episode_id"]) in episode_ids)
    ]
    expected_ids = {str(row["episode_id"]) for row in expected_rows}
    if episode_ids is not None:
        missing_requested = set(episode_ids) - expected_ids
        if missing_requested:
            raise RuntimeError(
                f"Requested evaluation episode IDs are absent from split {split!r}: "
                f"{sorted(missing_requested)[:10]}"
            )

    output_directory_name = output_name or f"evaluation-{split}"
    output_root = resolve_from_config(config, config["project"]["output_dir"]) / output_directory_name
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    state_path = output_root / "evaluation_state.json"
    episode_store_fingerprint = _episode_store_fingerprint(
        config,
        split=split,
        episode_ids=episode_ids,
    )
    evaluation_signature = _evaluation_signature(
        config,
        checkpoint_path,
        split,
        episode_ids,
        include_explanations,
        episode_store_fingerprint=episode_store_fingerprint,
    )
    if overwrite:
        predictions_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    existing_rows = _load_prediction_rows(predictions_path)
    completed_ids = {str(row["episode_id"]) for row in existing_rows}
    unexpected = completed_ids - expected_ids
    if unexpected:
        raise RuntimeError(f"Existing evaluation contains episodes outside the requested split: {sorted(unexpected)[:5]}")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            str(state.get("checkpoint")) != str(checkpoint_path)
            or str(state.get("split")) != split
            or str(state.get("evaluation_signature", "")) != evaluation_signature
        ):
            raise RuntimeError(
                "Existing evaluation output belongs to another checkpoint, split, config, "
                "episode-store fingerprint, or confidence-calibration configuration; "
                "pass --overwrite"
            )
    elif existing_rows:
        raise RuntimeError(
            "predictions.jsonl exists without evaluation_state.json; pass --overwrite to avoid mixing runs"
        )

    evaluation_cfg = config.get("evaluation", {})
    progress_every = max(1, int(evaluation_cfg.get("progress_every_episodes", 100)))
    schema_path = resolve_static_asset(config, "schemas/alert.schema.json")
    schema_validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    loader = build_episode_loader(
        config,
        split,
        shuffle=False,
        batch_size=1,
        require_labels=True,
        stage="evaluation",
        prepare_multiresolution=True,
        retain_prepared=True,
        episode_ids=episode_ids,
    )
    started = time.perf_counter()
    newly_completed = 0
    for prepared_batch in loader:
        prepared = prepared_batch.require_full_prepared()[0]
        episode = prepared.episode
        if episode.episode_id in completed_ids:
            continue
        result = model.classify(
            prepared,
            checkpoint_name=str(checkpoint_path),
            generate_explanation=include_explanations,
        )
        schema_validator.validate(result.to_dict())
        assert episode.label is not None
        append_jsonl(
            predictions_path,
            {
                "episode_id": episode.episode_id,
                "dataset_id": episode.provenance.dataset_id,
                "structure_id": episode.provenance.structure_id,
                "session_id": episode.provenance.session_id,
                "experiment_id": episode.provenance.experiment_id,
                "ground_truth": {
                    "global": {
                        "class": episode.label.global_class_name,
                        "severity": episode.label.global_severity,
                    },
                    "target_supervision_available": (
                        episode.label.target_supervision_available
                    ),
                    "targets": [
                        {
                            "sensor_id": target.sensor_id,
                            "affected": target.affected,
                            "class": target.class_name,
                            "severity": target.severity,
                            "source": target.source,
                        }
                        for target in episode.label.targets
                    ],
                },
                "audit_metadata": dict(episode.metadata.get("audit_metadata") or {}),
                "prediction": result.to_dict(),
            },
        )
        completed_ids.add(episode.episode_id)
        newly_completed += 1
        if newly_completed % progress_every == 0 or len(completed_ids) == len(expected_ids):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "split": split,
                        "completed": len(completed_ids),
                        "total": len(expected_ids),
                        "newly_completed": newly_completed,
                        "episodes_per_hour": newly_completed / elapsed * 3600.0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            atomic_write_json(
                state_path,
                {
                    "checkpoint": str(checkpoint_path),
                    "split": split,
                    "evaluation_signature": evaluation_signature,
                    "episode_store_fingerprint": episode_store_fingerprint,
                    "completed": len(completed_ids),
                    "total": len(expected_ids),
                    "updated_utc": utc_now_iso(),
                    "complete": len(completed_ids) == len(expected_ids),
                },
            )

    prediction_rows = _load_prediction_rows(predictions_path)
    prediction_rows.sort(key=lambda row: str(row["episode_id"]))
    if {str(row["episode_id"]) for row in prediction_rows} != expected_ids:
        missing = sorted(expected_ids - {str(row["episode_id"]) for row in prediction_rows})
        raise RuntimeError(f"Evaluation ended with missing predictions: {missing[:10]}")

    truth = [_label_from_row(row) for row in prediction_rows]
    predictions = [_predictions_from_row(row) for row in prediction_rows]
    global_predictions = [
        _global_prediction_from_row(row, raw=False) for row in prediction_rows
    ]
    raw_global_predictions = [
        _global_prediction_from_row(row, raw=True) for row in prediction_rows
    ]
    metrics = compute_metrics(
        truth,
        predictions,
        window_seconds=float(config["data"]["window_seconds"]),
        global_predictions=global_predictions,
    )
    metrics["raw_global_choice_probability_ece"] = global_expected_calibration_error(
        truth, raw_global_predictions
    )
    metrics["global_choice_probability_ece"] = global_expected_calibration_error(
        truth, global_predictions
    )
    metrics["global_ece"] = metrics["global_choice_probability_ece"]
    metrics["global_confidence_operational"] = any(
        bool((row.get("prediction", {}).get("model") or {}).get("global_confidence_operational", False))
        for row in prediction_rows
    )
    metrics["raw_choice_probability_ece"] = expected_calibration_error(truth, predictions, raw=True)
    metrics["raw_choice_probability_auroc"] = confidence_correctness_auroc(truth, predictions, raw=True)
    metrics["calibrated_choice_probability_ece"] = expected_calibration_error(truth, predictions, raw=False)
    metrics["calibrated_choice_probability_auroc"] = confidence_correctness_auroc(truth, predictions, raw=False)
    metrics["target_confidence_operational"] = any(
        target.confidence_operational
        for episode_predictions in predictions
        for target in episode_predictions
    )
    metrics["confidence_operational"] = bool(
        metrics["target_confidence_operational"]
        or metrics["global_confidence_operational"]
    )
    metrics["choice_probability_ece"] = metrics["calibrated_choice_probability_ece"]
    metrics["bootstrap_intervals"] = bootstrap_intervals(
        truth,
        predictions,
        window_seconds=float(config["data"]["window_seconds"]),
        samples=(
            int(bootstrap_samples)
            if bootstrap_samples is not None
            else int(evaluation_cfg.get("bootstrap_samples", 1000))
        ),
        confidence_level=float(evaluation_cfg.get("confidence_level", 0.95)),
        seed=int(config["project"].get("seed", 0)),
        global_predictions=global_predictions,
    )
    metrics["by_dataset"] = {}
    per_dataset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(prediction_rows):
        per_dataset[str(row["dataset_id"])].append(index)
    for dataset_id, indices in per_dataset.items():
        dataset_metrics = compute_metrics(
            [truth[index] for index in indices],
            [predictions[index] for index in indices],
            window_seconds=float(config["data"]["window_seconds"]),
            global_predictions=[global_predictions[index] for index in indices],
        )
        dataset_metrics["global_ece"] = global_expected_calibration_error(
            [truth[index] for index in indices],
            [global_predictions[index] for index in indices],
        )
        dataset_metrics["raw_global_choice_probability_ece"] = (
            global_expected_calibration_error(
                [truth[index] for index in indices],
                [raw_global_predictions[index] for index in indices],
            )
        )
        metrics["by_dataset"][dataset_id] = dataset_metrics
    metrics["persistence"] = (
        _persistence_metrics(prediction_rows, evaluation_cfg)
        if include_persistence
        else {"not_computed": True, "reason": "balanced_checkpoint_selection_subset"}
    )
    metrics["explanations"] = (
        _explanation_metrics(prediction_rows)
        if include_explanations
        else {"not_computed": True, "reason": "record_only_evaluation"}
    )

    build_report_path = index_path.parent / "build_report.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8")) if build_report_path.exists() else None
    metrics["metadata"] = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "evaluated_utc": utc_now_iso(),
        "group_disjointness_checked": True,
        "dataset_stratified_split_checked": bool(
            build_report
            and is_supported_assignment_strategy(build_report.get("split_strategy"))
        ),
        "split_strategy": (build_report.get("split_strategy") if build_report else None),
        "materialize_splits": (build_report.get("materialize_splits") if build_report else None),
        "datasets_in_split": sorted(per_dataset),
        "local_training_data_used": False,
        "include_explanations": bool(include_explanations),
        "resumed_prediction_count": len(existing_rows),
        "new_prediction_count": newly_completed,
        "evaluation_signature": evaluation_signature,
        "episode_store_fingerprint": episode_store_fingerprint,
        "subset_evaluation": episode_ids is not None,
        "requested_episode_count": len(episode_ids) if episode_ids is not None else None,
        "evaluation_device": str(evaluation_device),
        "common_mode_residual": _common_mode_residual_metadata(prediction_rows),
    }
    atomic_write_json(output_root / "metrics.json", metrics)
    _write_confusion_csv(
        output_root / "class_confusion_matrix.csv",
        metrics["class_labels"],
        metrics["class_confusion_matrix"],
    )
    _write_confusion_csv(
        output_root / "severity_confusion_matrix.csv",
        metrics["severity_labels"],
        metrics["severity_confusion_matrix"],
    )
    atomic_write_text(output_root / "report.md", _render_report(metrics))
    atomic_write_json(
        state_path,
        {
            "checkpoint": str(checkpoint_path),
            "split": split,
            "evaluation_signature": evaluation_signature,
            "episode_store_fingerprint": episode_store_fingerprint,
            "completed": len(prediction_rows),
            "total": len(expected_ids),
            "updated_utc": utc_now_iso(),
            "complete": True,
        },
    )
    return metrics


def _write_confusion_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _render_report(metrics: dict[str, Any]) -> str:
    persistence = metrics.get("persistence", {})
    false_event_rate = persistence.get("false_alert_events_per_building_day")
    false_event_text = "not available" if false_event_rate is None else f"{false_event_rate:.6f}"
    lines = [
        "# VibroGemma binary-state v3 evaluation report",
        "",
        f"- Checkpoint: `{metrics['metadata']['checkpoint']}`",
        f"- Split: `{metrics['metadata']['split']}`",
        f"- Datasets represented: {', '.join(metrics['metadata']['datasets_in_split'])}",
        f"- Episodes: {metrics['episode_count']}",
        f"- Target decisions: {metrics['target_count']}",
        f"- Episode exact match: {metrics['episode_exact_match']:.6f}",
        f"- Class macro F1: {metrics['class_macro_f1']:.6f}",
        (
            f"- Affected-sensor F1 (target-labelled episodes only): {metrics['affected_f1']:.6f}"
            if metrics.get("affected_f1") is not None
            else "- Affected-sensor F1: not applicable (no target labels)"
        ),
        f"- Global/target consistency: {metrics.get('global_target_consistency_rate')}",
        f"- Single-target exact-set match: {metrics.get('localization', {}).get('single_target_exact_set_match')}",
        f"- Single-target all-five collapse rate: {metrics.get('localization', {}).get('single_target_all_five_rate')}",
        f"- Single-target all-zero collapse rate: {metrics.get('localization', {}).get('single_target_all_zero_rate')}",
        f"- Supported-severity macro F1: {metrics['severity_macro_f1']:.6f}",
        f"- Persistence-filtered false-alert events/building-day: {false_event_text}",
        f"- Known changed-session detection recall: {persistence.get('known_event_detection_recall')}",
        f"- Gemma grounded-explanation fraction: {metrics.get('explanations', {}).get('gemma_grounded_fraction')}",
        f"- Confidence operational: {metrics.get('confidence_operational', False)}",
        "",
        "## Validity boundary",
        "",
        "These values are computed only from the executed dataset-stratified public test split. They do not establish accuracy on an unseen building or mounting configuration. No local STWIN recording was used as labelled training data.",
        "",
        "## Bootstrap intervals",
        "",
    ]
    for name, interval in metrics["bootstrap_intervals"].items():
        lines.append(f"- {name}: [{interval['lower']:.6f}, {interval['upper']:.6f}]")
    return "\n".join(lines) + "\n"
