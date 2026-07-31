#!/usr/bin/env python3
"""Score Phase 0.5 benchmark predictions against the frozen eval truth.

Scores any number of prediction files (one arm each, produced by
run_phase05_benchmark.py) plus the built-in "rules" arm (the deterministic
rule labels embedded in the eval set). Reports, per arm:

  * sensor-level: accuracy, per-class precision/recall/F1, macro-F1,
    truth x prediction confusion matrix
  * binary deviation screen: false-positive rate on healthy windows,
    recall on damaged windows (mild / significant separately)
  * network-level: accuracy, confusion; affected-sensor-id set P/R/F1
  * robustness: failed calls, unparseable JSON, missing sensor ids
  * latency: mean / p50 / p95
  * slices: single vs multi examples

With exactly two arms, prints the Phase 0.5 disagreement buckets
(both right / A-only right / B-only right / both wrong) that feed the plan's
decision gate (big-right/small-wrong = capability gap; both-wrong =
representation ceiling).

Usage:
  python3 scripts/score_phase05_benchmark.py \
      --eval-jsonl data/phase05_benchmark/eval.jsonl \
      --predictions data/phase05_benchmark/predictions_qwen3_4b_geniex_npu.jsonl \
      [--predictions predictions_qwen3_235b.jsonl ...] \
      [--report data/phase05_benchmark/report.json]

Stdlib only; runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

SENSOR_LABELS = [
    "normal_relative_to_baseline",
    "mild_deviation",
    "significant_local_deviation",
    "impulsive_or_shock_event",
    "possible_sensor_issue",
    "insufficient_data",
]
DEVIATION_SENSOR_LABELS = {"mild_deviation", "significant_local_deviation", "impulsive_or_shock_event"}
MISSING = "__missing__"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_json_payload(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def sensor_label_map(payload: dict[str, Any]) -> dict[str, str]:
    items = payload.get("sensor_reports") or payload.get("sensor_decisions") or payload.get("sensors")
    reports: list[dict[str, Any]] = []
    if isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, dict):
                reports.append({"sensor_id": str(value.get("sensor_id") or key), **value})
    elif isinstance(items, (list, tuple)):
        reports = [item for item in items if isinstance(item, dict)]
    out: dict[str, str] = {}
    for report in reports:
        sensor_id = str(report.get("sensor_id") or "")
        label = str(report.get("local_status") or report.get("small_agent_label") or report.get("label") or "")
        if sensor_id and label:
            out[sensor_id] = label
    return out


def network_label_of(payload: dict[str, Any]) -> str:
    return str(payload.get("network_status") or payload.get("network_label") or payload.get("main_agent_label") or "")


def affected_ids_of(payload: dict[str, Any]) -> set[str]:
    value = payload.get("affected_sensor_ids")
    if value is None:
        value = payload.get("affected_sensors")
    if not isinstance(value, (list, tuple)):
        return set()
    return {str(x) for x in value}


def extract_arm_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """example_id -> {sensor_labels, network_label, affected, latency_s, ok, parse_ok}."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id"))
        if not row.get("ok"):
            out[example_id] = {"ok": False, "parse_ok": False, "sensor_labels": {}, "network_label": "", "affected": set(), "latency_s": None}
            continue
        payload = parse_json_payload(str(row.get("text") or ""))
        out[example_id] = {
            "ok": True,
            "parse_ok": bool(payload),
            "sensor_labels": sensor_label_map(payload),
            "network_label": network_label_of(payload),
            "affected": affected_ids_of(payload),
            "latency_s": row.get("latency_s"),
            "usage": row.get("usage"),
        }
    return out


def rules_arm_predictions(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for example in examples:
        rule = example.get("rule_predictions") or {}
        out[example["example_id"]] = {
            "ok": True,
            "parse_ok": True,
            "sensor_labels": dict(rule.get("sensor_labels") or {}),
            "network_label": str(rule.get("network_label") or ""),
            "affected": {str(x) for x in rule.get("affected_sensor_ids") or []},
            "latency_s": None,
        }
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_arm(
    arm: str,
    examples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, int]] = {label: {"tp": 0, "fp": 0, "fn": 0} for label in SENSOR_LABELS}
    sensor_total = sensor_correct = 0
    healthy_total = healthy_fp = 0
    damaged_stats = {"mild": {"total": 0, "flagged": 0, "exact": 0}, "significant": {"total": 0, "flagged": 0, "exact": 0}}
    network_total = network_correct = 0
    network_confusion: dict[str, dict[str, int]] = {}
    affected_tp = affected_fp = affected_fn = 0
    failed_calls = parse_failures = missing_sensor_slots = 0
    latencies: list[float] = []
    by_kind: dict[str, dict[str, int]] = {}

    per_decision: dict[tuple[str, str], bool] = {}
    per_network: dict[str, bool] = {}

    for example in examples:
        example_id = example["example_id"]
        truth = example["truth"]
        kind = example["strata"]["kind"]
        prediction = predictions.get(example_id)
        if prediction is None or not prediction.get("ok"):
            failed_calls += 1
            prediction = {"parse_ok": False, "sensor_labels": {}, "network_label": "", "affected": set()}
        elif not prediction.get("parse_ok"):
            parse_failures += 1
        if prediction.get("latency_s") is not None:
            latencies.append(float(prediction["latency_s"]))

        window_state = {w["sensor_id"]: w["state"] for w in example["windows"]}
        for sensor_id, truth_label in truth["sensor_labels"].items():
            predicted = prediction["sensor_labels"].get(sensor_id, MISSING)
            if predicted == MISSING:
                missing_sensor_slots += 1
            sensor_total += 1
            correct = predicted == truth_label
            sensor_correct += int(correct)
            per_decision[(example_id, sensor_id)] = correct
            confusion.setdefault(truth_label, {})
            confusion[truth_label][predicted] = confusion[truth_label].get(predicted, 0) + 1
            if truth_label in per_class:
                if correct:
                    per_class[truth_label]["tp"] += 1
                else:
                    per_class[truth_label]["fn"] += 1
                    if predicted in per_class:
                        per_class[predicted]["fp"] += 1
            slot = by_kind.setdefault(kind, {"total": 0, "correct": 0})
            slot["total"] += 1
            slot["correct"] += int(correct)

            state = window_state.get(sensor_id)
            flagged = predicted in DEVIATION_SENSOR_LABELS
            if state == "healthy":
                healthy_total += 1
                healthy_fp += int(flagged)
            elif state in damaged_stats:
                damaged_stats[state]["total"] += 1
                damaged_stats[state]["flagged"] += int(flagged)
                damaged_stats[state]["exact"] += int(correct)

        truth_network = truth["network_label"]
        predicted_network = prediction["network_label"] or MISSING
        network_total += 1
        network_correct += int(predicted_network == truth_network)
        per_network[example_id] = predicted_network == truth_network
        network_confusion.setdefault(truth_network, {})
        network_confusion[truth_network][predicted_network] = network_confusion[truth_network].get(predicted_network, 0) + 1

        truth_affected = set(truth["affected_sensor_ids"])
        predicted_affected = set(prediction["affected"])
        affected_tp += len(truth_affected & predicted_affected)
        affected_fp += len(predicted_affected - truth_affected)
        affected_fn += len(truth_affected - predicted_affected)

    per_class_report = {}
    macro_f1_values = []
    for label in SENSOR_LABELS:
        counts = per_class[label]
        support = counts["tp"] + counts["fn"]
        if support == 0 and counts["fp"] == 0:
            continue
        precision, recall, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
        per_class_report[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "support": support}
        if support > 0:
            macro_f1_values.append(f1)

    ap, ar, af1 = prf(affected_tp, affected_fp, affected_fn)
    result = {
        "arm": arm,
        "sensor": {
            "total": sensor_total,
            "accuracy": round(sensor_correct / sensor_total, 3) if sensor_total else None,
            "macro_f1": round(sum(macro_f1_values) / len(macro_f1_values), 3) if macro_f1_values else None,
            "per_class": per_class_report,
            "confusion": confusion,
        },
        "binary_screen": {
            "healthy_windows": healthy_total,
            "false_positive_rate": round(healthy_fp / healthy_total, 3) if healthy_total else None,
            "mild_recall_any_deviation": round(damaged_stats["mild"]["flagged"] / damaged_stats["mild"]["total"], 3)
            if damaged_stats["mild"]["total"]
            else None,
            "significant_recall_any_deviation": round(
                damaged_stats["significant"]["flagged"] / damaged_stats["significant"]["total"], 3
            )
            if damaged_stats["significant"]["total"]
            else None,
        },
        "network": {
            "total": network_total,
            "accuracy": round(network_correct / network_total, 3) if network_total else None,
            "confusion": network_confusion,
            "affected_ids": {"precision": round(ap, 3), "recall": round(ar, 3), "f1": round(af1, 3)},
        },
        "robustness": {
            "failed_calls": failed_calls,
            "parse_failures": parse_failures,
            "missing_sensor_slots": missing_sensor_slots,
        },
        "latency_s": {
            "n": len(latencies),
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 2) if latencies else None,
        },
        "by_kind": {
            kind: {"total": v["total"], "accuracy": round(v["correct"] / v["total"], 3)} for kind, v in sorted(by_kind.items())
        },
        "_per_decision": per_decision,
        "_per_network": per_network,
    }
    return result


def disagreement_buckets(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a["_per_decision"]) & set(b["_per_decision"]))
    buckets = {"both_right": 0, f"{a['arm']}_only": 0, f"{b['arm']}_only": 0, "both_wrong": 0}
    samples: dict[str, list[str]] = {k: [] for k in buckets}
    for key in keys:
        ra, rb = a["_per_decision"][key], b["_per_decision"][key]
        if ra and rb:
            bucket = "both_right"
        elif ra:
            bucket = f"{a['arm']}_only"
        elif rb:
            bucket = f"{b['arm']}_only"
        else:
            bucket = "both_wrong"
        buckets[bucket] += 1
        if len(samples[bucket]) < 5:
            samples[bucket].append(f"{key[0]}/{key[1]}")
    network_keys = sorted(set(a["_per_network"]) & set(b["_per_network"]))
    network_buckets = {"both_right": 0, f"{a['arm']}_only": 0, f"{b['arm']}_only": 0, "both_wrong": 0}
    for key in network_keys:
        ra, rb = a["_per_network"][key], b["_per_network"][key]
        bucket = "both_right" if ra and rb else (f"{a['arm']}_only" if ra else (f"{b['arm']}_only" if rb else "both_wrong"))
        network_buckets[bucket] += 1
    return {"sensor_decisions": buckets, "sensor_examples": samples, "network": network_buckets}


def print_arm(result: dict[str, Any]) -> None:
    print(f"\n=== arm: {result['arm']} ===")
    sensor = result["sensor"]
    print(f"sensor: acc={sensor['accuracy']} macro_f1={sensor['macro_f1']} (n={sensor['total']})")
    for label, row in sensor["per_class"].items():
        print(f"  {label:32s} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} (n={row['support']})")
    screen = result["binary_screen"]
    print(
        f"screen: FP(healthy)={screen['false_positive_rate']} "
        f"recall(mild)={screen['mild_recall_any_deviation']} recall(significant)={screen['significant_recall_any_deviation']}"
    )
    network = result["network"]
    print(f"network: acc={network['accuracy']} affected_f1={network['affected_ids']['f1']} (n={network['total']})")
    robustness = result["robustness"]
    print(
        f"robustness: failed_calls={robustness['failed_calls']} parse_failures={robustness['parse_failures']} "
        f"missing_slots={robustness['missing_sensor_slots']}"
    )
    latency = result["latency_s"]
    if latency["n"]:
        print(f"latency_s: mean={latency['mean']} p50={latency['p50']} p95={latency['p95']} (n={latency['n']})")
    print(f"by_kind: {result['by_kind']}")
    print("confusion (truth -> prediction):")
    for truth_label, row in sorted(sensor["confusion"].items()):
        total = sum(row.values())
        top = ", ".join(f"{k}={v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {truth_label} (n={total}): {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--predictions", action="append", default=[], help="prediction JSONL (repeatable)")
    parser.add_argument("--skip-rules-arm", action="store_true")
    parser.add_argument("--report", default=None, help="write full JSON report here")
    args = parser.parse_args()

    examples = read_jsonl(Path(args.eval_jsonl))
    print(f"eval examples: {len(examples)}, sensor decisions: {sum(len(e['truth']['sensor_labels']) for e in examples)}")

    results: list[dict[str, Any]] = []
    if not args.skip_rules_arm:
        results.append(score_arm("rules", examples, rules_arm_predictions(examples)))
    for prediction_path in args.predictions:
        rows = read_jsonl(Path(prediction_path))
        arm = str(rows[0].get("arm") or Path(prediction_path).stem) if rows else Path(prediction_path).stem
        # Keep the last row per example (reruns append).
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[str(row.get("example_id"))] = row
        results.append(score_arm(arm, examples, extract_arm_predictions(list(latest.values()))))

    for result in results:
        print_arm(result)

    comparison = None
    model_arms = [r for r in results if r["arm"] != "rules"]
    if len(model_arms) == 2:
        comparison = disagreement_buckets(model_arms[0], model_arms[1])
        print("\n=== disagreement buckets (sensor decisions) ===")
        for bucket, count in comparison["sensor_decisions"].items():
            print(f"  {bucket}: {count}")
        print("network:", comparison["network"])

    if args.report:
        payload = {
            "arms": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
            "comparison": comparison and {k: v for k, v in comparison.items() if k != "sensor_examples"},
        }
        Path(args.report).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
