from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

from ..config import resolve_from_config
from ..utils import atomic_write_json, iter_jsonl, utc_now_iso
from .evaluator import evaluate_checkpoint
from .selection import _selection_ids, operational_summary

_LABELS = ("normal", "unknown_anomaly")


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    if not normalized:
        raise ValueError("candidate name cannot be empty")
    return normalized


def _load_candidates(path: str | Path) -> tuple[str, list[dict[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    champion = str(payload.get("champion", ""))
    raw = list(payload.get("candidates") or [])
    if not champion or not raw:
        raise ValueError("candidate manifest requires champion and candidates")
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name", ""))
        checkpoint = Path(str(item.get("checkpoint", ""))).expanduser()
        if not name or name in seen:
            raise ValueError(f"duplicate or empty candidate name: {name!r}")
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        seen.add(name)
        candidates.append({"name": name, "checkpoint": str(checkpoint.resolve())})
    if champion not in seen:
        raise ValueError(f"champion {champion!r} is not in candidates")
    return champion, candidates


def _macro_f1_from_rows(rows: Iterable[dict[str, Any]]) -> float:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in _LABELS}
    for row in rows:
        truth = list(row["ground_truth"])
        prediction = list(row["prediction"]["targets"])
        for expected, actual in zip(truth, prediction, strict=True):
            true_label = str(expected["class"])
            predicted_label = str(actual["class"])
            for label in _LABELS:
                if true_label == label and predicted_label == label:
                    counts[label]["tp"] += 1
                elif true_label != label and predicted_label == label:
                    counts[label]["fp"] += 1
                elif true_label == label and predicted_label != label:
                    counts[label]["fn"] += 1
    f1_values: list[float] = []
    for label in _LABELS:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return sum(f1_values) / len(f1_values)


def _paired_macro_f1_interval(
    champion_predictions: Path,
    challenger_predictions: Path,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    champion_rows = {
        str(row["episode_id"]): row for row in iter_jsonl(champion_predictions)
    }
    challenger_rows = {
        str(row["episode_id"]): row for row in iter_jsonl(challenger_predictions)
    }
    if champion_rows.keys() != challenger_rows.keys():
        raise RuntimeError("paired checkpoint comparison received different episode sets")
    episode_ids = sorted(champion_rows)
    if not episode_ids:
        raise RuntimeError("paired checkpoint comparison has no predictions")
    point = _macro_f1_from_rows(challenger_rows.values()) - _macro_f1_from_rows(
        champion_rows.values()
    )
    rng = random.Random(int(seed))
    differences: list[float] = []
    for _ in range(max(1, int(samples))):
        sampled = [rng.choice(episode_ids) for _ in episode_ids]
        differences.append(
            _macro_f1_from_rows(challenger_rows[episode_id] for episode_id in sampled)
            - _macro_f1_from_rows(champion_rows[episode_id] for episode_id in sampled)
        )
    differences.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(confidence_level)))
    lower_index = min(
        len(differences) - 1,
        int(math.floor(alpha * 0.5 * len(differences))),
    )
    upper_index = min(
        len(differences) - 1,
        int(math.ceil((1.0 - alpha * 0.5) * len(differences))) - 1,
    )
    return {
        "point": float(point),
        "lower": float(differences[lower_index]),
        "upper": float(differences[upper_index]),
    }


def _recall(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    return float(value) if value is not None else 0.0


def _regression_guard(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    paired_interval: dict[str, float],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    champion_summary = champion["summary"]
    challenger_summary = challenger["summary"]
    champion_components = champion_summary["components"]
    challenger_components = challenger_summary["components"]
    checks: list[dict[str, Any]] = []

    def minimum(name: str, value: float, required: float) -> None:
        checks.append(
            {
                "name": name,
                "value": float(value),
                "minimum": float(required),
                "passed": float(value) >= float(required),
            }
        )

    def maximum_drop(
        name: str,
        champion_value: float,
        challenger_value: float,
        allowed: float,
    ) -> None:
        drop = float(champion_value) - float(challenger_value)
        checks.append(
            {
                "name": name,
                "champion": float(champion_value),
                "challenger": float(challenger_value),
                "drop": float(drop),
                "maximum_drop": float(allowed),
                "passed": drop <= float(allowed),
            }
        )

    minimum(
        "operational_score_gain",
        float(challenger_summary["operational_score"])
        - float(champion_summary["operational_score"]),
        float(thresholds.get("minimum_operational_score_gain", 0.01)),
    )
    minimum(
        "overall_macro_f1_gain",
        float(challenger_components["overall_macro_f1"])
        - float(champion_components["overall_macro_f1"]),
        float(thresholds.get("minimum_macro_f1_gain", 0.005)),
    )
    minimum(
        "paired_macro_f1_lower_bound",
        float(paired_interval["lower"]),
        float(thresholds.get("minimum_paired_macro_f1_lower_bound", 0.0)),
    )
    maximum_drop(
        "affected_f1_non_regression",
        float(champion_components["affected_f1"]),
        float(challenger_components["affected_f1"]),
        float(thresholds.get("maximum_affected_f1_drop", 0.01)),
    )
    maximum_drop(
        "single_target_exact_set_match_non_regression",
        float(champion_components["single_target_exact_set_match"]),
        float(challenger_components["single_target_exact_set_match"]),
        float(thresholds.get("maximum_single_target_exact_set_match_drop", 0.0)),
    )
    maximum_drop(
        "single_target_noncollapse_non_regression",
        float(champion_components["single_target_noncollapse"]),
        float(challenger_components["single_target_noncollapse"]),
        float(thresholds.get("maximum_single_target_noncollapse_drop", 0.0)),
    )
    minimum(
        "minimum_per_target_f1_gain",
        float(challenger_components["minimum_per_target_f1"])
        - float(champion_components["minimum_per_target_f1"]),
        float(thresholds.get("minimum_per_target_f1_gain", 0.0)),
    )

    champion_by_dataset = champion_summary["by_dataset"]
    challenger_by_dataset = challenger_summary["by_dataset"]
    maximum_dataset_macro_drop = float(
        thresholds.get("maximum_dataset_macro_f1_drop", 0.025)
    )
    maximum_state_recall_drop = float(
        thresholds.get("maximum_dataset_state_recall_drop", 0.05)
    )
    for dataset_id, champion_values in sorted(champion_by_dataset.items()):
        challenger_values = challenger_by_dataset.get(dataset_id)
        if challenger_values is None:
            checks.append({"name": f"dataset:{dataset_id}:present", "passed": False})
            continue
        maximum_drop(
            f"dataset:{dataset_id}:macro_f1_non_regression",
            float(champion_values["class_macro_f1"]),
            float(challenger_values["class_macro_f1"]),
            maximum_dataset_macro_drop,
        )
        for state_key in ("normal_recall", "anomaly_recall"):
            maximum_drop(
                f"dataset:{dataset_id}:{state_key}_non_regression",
                _recall(champion_values, state_key),
                _recall(challenger_values, state_key),
                maximum_state_recall_drop,
            )

    return {
        "passed": all(bool(item.get("passed")) for item in checks),
        "checks": checks,
        "paired_macro_f1_difference": paired_interval,
    }


def compare_candidate_checkpoints(
    config: dict[str, Any],
    *,
    candidate_manifest: str | Path,
    split: str = "validation",
    stage1_episodes_per_stratum: int = 32,
    stage2_episodes_per_stratum: int = 128,
    shortlist_challengers: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    champion_name, candidates = _load_candidates(candidate_manifest)
    output_root = resolve_from_config(config, config["project"]["output_dir"])
    comparison_root = output_root / "regression-controlled-selection"
    comparison_root.mkdir(parents=True, exist_ok=True)
    report_path = comparison_root / "selection_report.json"
    if report_path.exists() and not overwrite:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        selected = Path(str(existing.get("selected_checkpoint", "")))
        if selected.is_dir():
            return existing

    index_path = (
        resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"
    )
    base_seed = int(config["project"].get("seed", 0))
    stage1_ids, stage1_manifest = _selection_ids(
        index_path,
        split=split,
        episodes_per_stratum=int(stage1_episodes_per_stratum),
        seed=base_seed + 901,
    )
    stage2_ids, stage2_manifest = _selection_ids(
        index_path,
        split=split,
        episodes_per_stratum=int(stage2_episodes_per_stratum),
        seed=base_seed + 902,
    )
    atomic_write_json(comparison_root / "stage1_manifest.json", stage1_manifest)
    atomic_write_json(comparison_root / "stage2_manifest.json", stage2_manifest)

    stage1: list[dict[str, Any]] = []
    for item in candidates:
        safe = _safe_name(item["name"])
        output_name = f"regression-controlled-selection/stage1/{safe}"
        metrics = evaluate_checkpoint(
            config,
            checkpoint=item["checkpoint"],
            split=split,
            overwrite=overwrite,
            episode_ids=stage1_ids,
            output_name=output_name,
            bootstrap_samples=100,
            include_persistence=False,
            include_explanations=False,
        )
        stage1.append(
            {
                **item,
                "summary": operational_summary(metrics),
                "metrics_path": str(output_root / output_name / "metrics.json"),
                "predictions_path": str(
                    output_root / output_name / "predictions.jsonl"
                ),
            }
        )

    challengers = sorted(
        (item for item in stage1 if item["name"] != champion_name),
        key=lambda item: float(item["summary"]["operational_score"]),
        reverse=True,
    )
    stage2_names = {champion_name}
    stage2_names.update(
        item["name"]
        for item in challengers[: max(1, int(shortlist_challengers))]
    )

    stage2: list[dict[str, Any]] = []
    for item in candidates:
        if item["name"] not in stage2_names:
            continue
        safe = _safe_name(item["name"])
        output_name = f"regression-controlled-selection/stage2/{safe}"
        metrics = evaluate_checkpoint(
            config,
            checkpoint=item["checkpoint"],
            split=split,
            overwrite=overwrite,
            episode_ids=stage2_ids,
            output_name=output_name,
            bootstrap_samples=300,
            include_persistence=False,
            include_explanations=False,
        )
        stage2.append(
            {
                **item,
                "summary": operational_summary(metrics),
                "metrics_path": str(output_root / output_name / "metrics.json"),
                "predictions_path": str(
                    output_root / output_name / "predictions.jsonl"
                ),
            }
        )

    champion = next(item for item in stage2 if item["name"] == champion_name)
    comparison_cfg = dict(
        config.get("evaluation", {}).get("regression_control") or {}
    )
    challenger_reports: list[dict[str, Any]] = []
    for challenger in stage2:
        if challenger["name"] == champion_name:
            continue
        paired = _paired_macro_f1_interval(
            Path(champion["predictions_path"]),
            Path(challenger["predictions_path"]),
            samples=int(comparison_cfg.get("paired_bootstrap_samples", 500)),
            confidence_level=float(comparison_cfg.get("confidence_level", 0.95)),
            seed=base_seed + 903,
        )
        guard = _regression_guard(champion, challenger, paired, comparison_cfg)
        challenger_reports.append({**challenger, "regression_guard": guard})

    passing = [
        item for item in challenger_reports if item["regression_guard"]["passed"]
    ]
    if passing:
        passing.sort(
            key=lambda item: float(item["summary"]["operational_score"]),
            reverse=True,
        )
        selected = passing[0]
        selected_reason = "challenger_passed_all_non_regression_checks"
    else:
        selected = champion
        selected_reason = "champion_retained_no_challenger_passed"

    report = {
        "created_utc": utc_now_iso(),
        "protocol": "regression_controlled_champion_challenger_v1",
        "split": split,
        "champion": champion_name,
        "selected_name": selected["name"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_reason": selected_reason,
        "stage1": stage1,
        "stage2_champion": champion,
        "stage2_challengers": challenger_reports,
        "test_authorized": False,
        "note": (
            "This comparison selects a validation champion only. Full validation "
            "gates must pass before the frozen test can run."
        ),
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        comparison_root / "selected_checkpoint.json",
        {
            "name": selected["name"],
            "path": selected["checkpoint"],
            "reason": selected_reason,
            "report": str(report_path),
        },
    )
    return report
