from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def localization_collapse_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total_positive = 0
    exact = 0
    all_five = 0
    all_zero = 0
    partial = 0
    per_target = {index: {"tp": 0, "fp": 0, "fn": 0} for index in range(1, 6)}
    for row in records:
        truth = {int(v) for v in row.get("true_affected_targets", [])}
        pred = {int(v) for v in row.get("predicted_affected_targets", [])}
        if truth:
            total_positive += 1
            exact += int(pred == truth)
            all_five += int(pred == {1,2,3,4,5})
            all_zero += int(not pred)
            partial += int(0 < len(pred) < 5)
        for target in range(1, 6):
            t, p = target in truth, target in pred
            per_target[target]["tp"] += int(t and p)
            per_target[target]["fp"] += int((not t) and p)
            per_target[target]["fn"] += int(t and (not p))
    target_f1 = {}
    for target, c in per_target.items():
        denominator = 2*c["tp"] + c["fp"] + c["fn"]
        target_f1[f"target_{target}"] = (2*c["tp"] / denominator) if denominator else None
    valid = [v for v in target_f1.values() if v is not None]
    return {
        "positive_episode_count": total_positive,
        "exact_affected_set_match": exact / total_positive if total_positive else None,
        "positive_all_five_rate": all_five / total_positive if total_positive else None,
        "positive_all_zero_rate": all_zero / total_positive if total_positive else None,
        "positive_partial_target_rate": partial / total_positive if total_positive else None,
        "per_target_f1": target_f1,
        "target_macro_f1": sum(valid) / len(valid) if valid else None,
    }
