from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import resolve_from_config
from ..utils import atomic_write_json, iter_jsonl, utc_now_iso
from .evaluator import (
    _episode_store_fingerprint,
    _file_manifest_hash,
    evaluate_checkpoint,
)
from .gates import validate_operational_summary


def _selection_ids(
    index_path: Path,
    *,
    split: str,
    episodes_per_stratum: int,
    seed: int,
    include_dataset_ids: set[str] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    if episodes_per_stratum < 1:
        raise ValueError("episodes_per_stratum must be >= 1")
    strata: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in iter_jsonl(index_path):
        if str(row.get("split")) != split or not bool(row.get("has_label", True)):
            continue
        if include_dataset_ids is not None and str(row.get("dataset_id")) not in include_dataset_ids:
            continue
        stratum = str(row.get("split_stratum", ""))
        if stratum == "representation_only":
            continue
        strata[(str(row["dataset_id"]), stratum)][str(row["split_group"])].append(row)
    if not strata:
        raise RuntimeError(f"No labelled strata are available in split {split!r}")

    selected: set[str] = set()
    report_rows: list[dict[str, Any]] = []
    for (dataset_id, stratum), grouped in sorted(strata.items()):
        group_names = sorted(grouped)
        digest = hashlib.sha256(
            f"{seed}\0{split}\0{dataset_id}\0{stratum}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        queues: dict[str, list[dict[str, Any]]] = {}
        for group_name in group_names:
            values = list(grouped[group_name])
            rng.shuffle(values)
            queues[group_name] = values
        rng.shuffle(group_names)
        cursor = 0
        picked = 0
        exhausted: set[str] = set()
        while picked < episodes_per_stratum and len(exhausted) < len(group_names):
            group_name = group_names[cursor % len(group_names)]
            cursor += 1
            queue = queues[group_name]
            if queue:
                selected.add(str(queue.pop()["episode_id"]))
                picked += 1
            else:
                exhausted.add(group_name)
        report_rows.append(
            {
                "dataset_id": dataset_id,
                "stratum": stratum,
                "available_groups": len(grouped),
                "available_episodes": sum(len(values) for values in grouped.values()),
                "selected_episodes": picked,
            }
        )
    return selected, {
        "split": split,
        "seed": int(seed),
        "episodes_per_stratum": episodes_per_stratum,
        "include_dataset_ids": (
            sorted(include_dataset_ids) if include_dataset_ids is not None else None
        ),
        "selected_episode_count": len(selected),
        "strata": report_rows,
    }


def _recall(metrics: dict[str, Any], label: str) -> float | None:
    labels = list(metrics.get("class_labels") or [])
    if label not in labels:
        return None
    index = labels.index(label)
    matrix = metrics["class_confusion_matrix"]
    support = float(sum(matrix[index]))
    return float(matrix[index][index]) / support if support > 0 else None


def operational_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset_id, values in sorted((metrics.get("by_dataset") or {}).items()):
        per_dataset[dataset_id] = {
            "class_macro_f1": float(values["class_macro_f1"]),
            "normal_recall": _recall(values, "normal"),
            "anomaly_recall": _recall(values, "unknown_anomaly"),
            "affected_f1": (
                float(values["affected_f1"])
                if values.get("affected_f1") is not None
                else None
            ),
        }

    normal_recall = _recall(metrics, "normal")
    anomaly_recall = _recall(metrics, "unknown_anomaly")
    dataset_macro = [item["class_macro_f1"] for item in per_dataset.values()]
    dataset_normal = [
        item["normal_recall"]
        for item in per_dataset.values()
        if item["normal_recall"] is not None
    ]
    dataset_anomaly = [
        item["anomaly_recall"]
        for item in per_dataset.values()
        if item["anomaly_recall"] is not None
    ]
    localization = dict(metrics.get("localization") or {})
    raw_per_target = dict(localization.get("per_target") or {})
    per_target: dict[str, dict[str, float | int]] = {}
    for target_name, values in sorted(raw_per_target.items()):
        per_target[target_name] = {
            "support": int(values.get("support", 0) or 0),
            "precision": float(values.get("precision", 0.0) or 0.0),
            "recall": float(values.get("recall", 0.0) or 0.0),
            "f1": float(values.get("f1", 0.0) or 0.0),
        }
    supported_targets = {
        target_name: values
        for target_name, values in per_target.items()
        if int(values["support"]) > 0
    }
    supported_target_f1 = [
        float(values["f1"])
        for values in supported_targets.values()
    ]
    worst_target = (
        min(
            supported_targets,
            key=lambda target_name: (
                float(supported_targets[target_name]["f1"]),
                target_name,
            ),
        )
        if supported_targets
        else None
    )
    target_exact = localization.get("single_target_exact_set_match")
    all_five = localization.get("single_target_all_five_rate")
    all_zero = localization.get("single_target_all_zero_rate")
    noncollapse = (
        1.0 - max(float(all_five or 0.0), float(all_zero or 0.0))
        if target_exact is not None
        else 0.0
    )
    components = {
        "overall_macro_f1": float(metrics["class_macro_f1"]),
        "overall_normal_recall": float(normal_recall or 0.0),
        "overall_anomaly_recall": float(anomaly_recall or 0.0),
        "affected_f1": float(metrics.get("affected_f1") or 0.0),
        "single_target_exact_set_match": float(target_exact or 0.0),
        "single_target_noncollapse": float(noncollapse),
        "supported_target_count": len(supported_targets),
        "minimum_per_target_f1": min(supported_target_f1) if supported_target_f1 else 0.0,
        "minimum_dataset_macro_f1": min(dataset_macro) if dataset_macro else 0.0,
        "minimum_dataset_normal_recall": min(dataset_normal) if dataset_normal else 0.0,
        "minimum_dataset_anomaly_recall": min(dataset_anomaly) if dataset_anomaly else 0.0,
    }
    weights = {
        "overall_macro_f1": 0.12,
        "overall_normal_recall": 0.08,
        "overall_anomaly_recall": 0.06,
        "affected_f1": 0.18,
        "single_target_exact_set_match": 0.18,
        "single_target_noncollapse": 0.08,
        "minimum_per_target_f1": 0.18,
        "minimum_dataset_macro_f1": 0.06,
        "minimum_dataset_normal_recall": 0.03,
        "minimum_dataset_anomaly_recall": 0.03,
    }
    # A weighted geometric mean makes a single collapsed dataset/state costly,
    # unlike language-model loss or overall accuracy on imbalanced windows.
    score = math.exp(
        sum(
            weight * math.log(max(1e-6, float(components[name])))
            for name, weight in weights.items()
        )
    )
    return {
        "operational_score": score,
        "components": components,
        "by_dataset": per_dataset,
        "localization_diagnostics": {
            "per_target": per_target,
            "supported_target_count": len(supported_targets),
            "worst_target": worst_target,
            "worst_target_f1": (
                float(supported_targets[worst_target]["f1"])
                if worst_target is not None
                else None
            ),
            "zero_f1_targets": [
                target_name
                for target_name, values in supported_targets.items()
                if float(values["f1"]) == 0.0
            ],
            "single_target_episode_count": int(
                localization.get("single_target_episode_count", 0) or 0
            ),
            "single_target_all_five_rate": (
                float(all_five) if all_five is not None else None
            ),
            "single_target_all_zero_rate": (
                float(all_zero) if all_zero is not None else None
            ),
        },
    }


def _selection_signature(
    config: dict[str, Any],
    *,
    stage: str,
    split: str,
    episodes_per_stratum: int,
    selected_ids: set[str],
    checkpoints: list[Path],
) -> tuple[str, dict[str, Any]]:
    canonical_config = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    candidates = []
    for checkpoint in checkpoints:
        manifest_sha256, manifest = _file_manifest_hash(checkpoint)
        candidates.append(
            {
                "path": str(checkpoint.resolve()),
                "manifest_sha256": manifest_sha256,
                "manifest": manifest,
            }
        )
    candidate_payload = json.dumps(
        candidates, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    payload = {
        "schema": "vibrogemma-checkpoint-selection-signature-v1",
        "stage": str(stage),
        "split": str(split),
        "episodes_per_stratum": int(episodes_per_stratum),
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "selected_episode_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode("utf-8")
        ).hexdigest(),
        "episode_store": _episode_store_fingerprint(
            config,
            split=split,
            episode_ids=selected_ids,
        ),
        "candidate_inventory_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "candidates": candidates,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return signature, payload


def select_operational_checkpoint(
    config: dict[str, Any],
    *,
    stage: str,
    split: str = "validation",
    episodes_per_stratum: int = 32,
    overwrite: bool = False,
) -> dict[str, Any]:
    if stage not in {"align", "lora"}:
        raise ValueError("stage must be 'align' or 'lora'")
    checkpoint_stage_root = (
        resolve_from_config(config, config["project"]["checkpoint_dir"]) / stage
    )
    checkpoints = sorted(
        path
        for path in checkpoint_stage_root.glob("epoch-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    )
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_stage_root}")

    selection_cfg = dict(config.get("evaluation", {}).get("checkpoint_selection", {}) or {})
    configured_probe_offset = selection_cfg.get("probe_seed_offset")
    if isinstance(configured_probe_offset, dict):
        probe_seed_offset = int(
            configured_probe_offset.get(stage, 301 if stage == "align" else 302)
        )
    elif configured_probe_offset is None:
        probe_seed_offset = 301 if stage == "align" else 302
    else:
        probe_seed_offset = int(configured_probe_offset)
    index_path = resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"
    stage_corpus = dict(
        config.get("training", {}).get(stage, {}).get("corpus") or {}
    )
    configured_datasets = stage_corpus.get("include_dataset_ids")
    include_dataset_ids = (
        {str(value) for value in configured_datasets}
        if configured_datasets is not None
        else None
    )
    ids, selection_manifest = _selection_ids(
        index_path,
        split=split,
        episodes_per_stratum=episodes_per_stratum,
        seed=int(config["project"].get("seed", 0)) + probe_seed_offset,
        include_dataset_ids=include_dataset_ids,
    )
    selection_root = (
        resolve_from_config(config, config["project"]["output_dir"])
        / "checkpoint-selection"
        / stage
    )
    selection_root.mkdir(parents=True, exist_ok=True)
    report_path = selection_root / "selection_report.json"
    selection_signature, selection_signature_payload = _selection_signature(
        config,
        stage=stage,
        split=split,
        episodes_per_stratum=episodes_per_stratum,
        selected_ids=ids,
        checkpoints=checkpoints,
    )
    stale_cached_selection = False
    if report_path.exists() and not overwrite:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        selected_path = Path(str(existing.get("selected_checkpoint", "")))
        current_candidates = {path.resolve() for path in checkpoints}
        if (
            existing.get("selection_signature") == selection_signature
            and selected_path.exists()
            and selected_path.resolve() in current_candidates
        ):
            return existing
        stale_cached_selection = True
    atomic_write_json(selection_root / "selection_manifest.json", selection_manifest)

    candidates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        output_name = f"checkpoint-selection/{stage}/{checkpoint.name}"
        metrics = evaluate_checkpoint(
            config,
            checkpoint=checkpoint,
            split=split,
            overwrite=bool(overwrite or stale_cached_selection),
            episode_ids=ids,
            output_name=output_name,
            bootstrap_samples=100,
            include_persistence=False,
            include_explanations=False,
        )
        summary = operational_summary(metrics)
        preflight_thresholds = dict(
            config.get("evaluation", {})
            .get("checkpoint_selection", {})
            .get("preflight_gates", {})
        )
        gate = (
            validate_operational_summary(summary, thresholds=preflight_thresholds)
            if preflight_thresholds
            else {"passed": True, "checks": []}
        )
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": int(state.get("epoch", 0)),
                **summary,
                "preflight_gate": gate,
            }
        )

    candidates.sort(key=lambda item: (float(item["operational_score"]), -int(item["epoch"])), reverse=True)
    per_stage_requirement = dict(selection_cfg.get("require_passing_candidate_by_stage") or {})
    require_passing = bool(
        per_stage_requirement.get(stage, selection_cfg.get("require_passing_candidate", False))
    )
    eligible = [item for item in candidates if bool(item.get("preflight_gate", {}).get("passed", True))]
    if require_passing and not eligible:
        report = {
            "created_utc": utc_now_iso(),
            "stage": stage,
            "split": split,
            "selection_manifest": str(selection_root / "selection_manifest.json"),
            "selected_checkpoint": None,
            "selected_epoch": None,
            "selected_operational_score": None,
            "passed": False,
            "reason": "no checkpoint satisfied the preregistered operational preflight gates",
            "selection_signature": selection_signature,
            "selection_signature_payload": selection_signature_payload,
            "candidates": candidates,
        }
        atomic_write_json(report_path, report)
        raise RuntimeError(report["reason"] + f"; see {report_path}")
    selected = (eligible if require_passing else candidates)[0]
    report = {
        "created_utc": utc_now_iso(),
        "stage": stage,
        "split": split,
        "selection_manifest": str(selection_root / "selection_manifest.json"),
        "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": selected["epoch"],
        "selected_operational_score": selected["operational_score"],
        "passed": True,
        "require_passing_candidate": require_passing,
        "selection_signature": selection_signature,
        "selection_signature_payload": selection_signature_payload,
        "candidates": candidates,
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        checkpoint_stage_root / "best-operational.json",
        {
            "path": selected["checkpoint"],
            "epoch": selected["epoch"],
            "operational_score": selected["operational_score"],
            "report": str(selection_root / "selection_report.json"),
        },
    )
    return report
