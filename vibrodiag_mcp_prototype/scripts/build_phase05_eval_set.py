#!/usr/bin/env python3
"""Build the Phase 0.5 model-capability benchmark eval set from LUMO.

Produces a frozen, portable eval JSONL: every line carries the exact production
prompt (messages), the exact grammar response_format, the ground truth from the
LUMO damage states, the deterministic rule-arm predictions, and raw-window
pointers so later arms (PSD-as-text, native ChatTS probes) can replay the very
same windows. Any OpenAI-compatible endpoint can then be benchmarked with
run_phase05_benchmark.py under byte-identical conditions.

Design (probe-informed, 2026-07-07):
  * Same-channel cross-time comparison. Cross-position pairing is dominated by
    mast transfer-function differences; cross-time raw levels are dominated by
    ambient (wind) excitation. The committed dataset truth (damage state per
    session) matches the same-channel cross-time frame.
  * The reference row is a per-channel field-wise MEDIAN over healthy windows
    drawn from all healthy sessions. A single healthy window is not
    "representative normal" on LUMO (10x day-to-day level swings); the median
    composite is the honest analog of the production assumption that the
    baseline sensor behaves normally.
  * Two example kinds: "single" (1 target; raw per-window discrimination) and
    "multi" (5 targets of the same channel from different sessions; tests
    cross-target relative reasoning inside one prompt).
  * Truth labels: session state -> normal_relative_to_baseline / mild_deviation
    (DAM*_010) / significant_local_deviation (DAM*_111), per the committed
    label_rules in config/vibration_sft_datasets.example.yaml. Network truth =
    deterministic_network_label applied to the truth sensor labels (offline
    referee construction only).

Usage:
  PYTHONPATH=src ../vibroagent-venv/bin/python scripts/build_phase05_eval_set.py \
      --lumo-root data/external/lumo --out data/phase05_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_building_vibration_sft import (  # noqa: E402
    PromptTarget,
    compute_pipeline_metric_payload,
)
from vibroagent_mcp.llm_sensor_agent import (  # noqa: E402
    _all_sensor_response_format,
    _build_all_sensor_prompt,
    compare_building_metrics,
    deterministic_network_label,
    deterministic_sensor_label,
)

# The exact system message the production ModelClient sends (model_client.py).
PRODUCTION_SYSTEM_MESSAGE = "Return strict JSON only. Do not include markdown."

SEED = 20260707
WINDOW_DURATION_S = 10.0
# Channels: 6 accelerometer channels spread over mast height and both axes.
CHANNELS = ["accel01x", "accel02y", "accel04x", "accel05y", "accel07x", "accel08y"]
# Windows per (file, channel) candidate grid: start seconds, away from file edges.
CANDIDATE_STARTS_S = [45.0, 105.0, 165.0, 225.0, 285.0, 345.0, 405.0, 465.0, 525.0]
REF_WINDOWS_PER_HEALTHY_FILE = 2  # per channel, drawn from the candidate grid

SINGLE_PER_CHANNEL = {"healthy": 20, "mild": 10, "significant": 10}
MULTI_EXAMPLES = 60
MULTI_TARGETS = 5
# Composition cycle for multi examples: number of (mild, significant) targets;
# the rest are healthy. Chosen to exercise all reachable network truth labels.
MULTI_COMPOSITIONS = [
    (0, 0),  # all healthy            -> normal_relative_to_reference_sensor
    (1, 0),  # one mild               -> mild_local_deviation
    (0, 1),  # one significant        -> localized_vibration_deviation
    (2, 0),  # two mild               -> mild_multi_sensor_deviation
    (0, 2),  # two significant        -> multi_sensor_building_wide_vibration_event
    (2, 1),  # mixed, 3 affected      -> multi_sensor_building_wide_vibration_event
]

STATE_TO_LABEL = {
    "healthy": "normal_relative_to_baseline",
    "mild": "mild_deviation",
    "significant": "significant_local_deviation",
}


@dataclass(frozen=True)
class WindowKey:
    session: str
    file: str
    channel: str
    start_s: float


def session_state(session: str) -> str:
    if "Healthy" in session:
        return "healthy"
    if session.endswith("_010"):
        return "mild"
    if session.endswith("_111"):
        return "significant"
    raise ValueError(f"Unrecognized LUMO session name: {session}")


def damage_type(session: str) -> str | None:
    match = re.search(r"(DAM\d)", session)
    return match.group(1) if match else None


def load_lumo_file(path: Path) -> tuple[np.ndarray, list[str], float]:
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    dat = mat["Dat"]
    data = np.asarray(dat.Data, dtype=np.float64)
    if data.shape[0] < data.shape[1]:
        data = data.T
    names = [str(x) for x in np.asarray(dat.ChannelNames, dtype=object).reshape(-1)]
    fs = float(np.asarray(dat.Fs, dtype=np.float64).reshape(-1)[0])
    return data, names, fs


def iter_session_files(lumo_root: Path) -> Iterator[tuple[str, Path]]:
    for session_dir in sorted(p for p in lumo_root.iterdir() if p.is_dir() and re.match(r"\d\d_", p.name)):
        for file_path in sorted(session_dir.glob("*.mat")):
            yield session_dir.name, file_path


def window_metrics(
    data: np.ndarray,
    names: list[str],
    fs: float,
    channel: str,
    start_s: float,
    *,
    sensor_id: str,
) -> dict[str, Any]:
    ci = names.index(channel)
    start = int(round(start_s * fs))
    length = int(round(WINDOW_DURATION_S * fs))
    signal = data[start : start + length, ci]
    return compute_pipeline_metric_payload(
        signal,
        sampling_rate_hz=int(round(fs)),
        sensor_id=sensor_id,
        location=channel,
        role="target",
        axis=channel[-1],
        source_quality_flags=[],
    )


def median_reference(channel: str, healthy_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Field-wise median composite of healthy windows for one channel."""
    scalar_keys = [
        "acceleration_rms_g",
        "acceleration_peak_g",
        "acceleration_peak_to_peak_g",
        "crest_factor",
        "kurtosis",
        "estimated_ppv_mm_s",
        "dominant_frequency_hz",
        "duration_s",
    ]
    ref = dict(healthy_metrics[0])
    for key in scalar_keys:
        values = [m.get(key) for m in healthy_metrics]
        values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        ref[key] = float(np.median(values)) if values else None
    counts = [int(m.get("sample_count") or 0) for m in healthy_metrics]
    ref["sample_count"] = int(np.median(counts)) if counts else 0
    band_keys = sorted({b for m in healthy_metrics for b in (m.get("band_powers") or {})})
    bands = {b: float(np.median([float((m.get("band_powers") or {}).get(b, 0.0)) for m in healthy_metrics])) for b in band_keys}
    total = sum(bands.values())
    if total > 0:
        bands = {b: v / total for b, v in bands.items()}
    ref["band_powers"] = bands
    ref["sensor_id"] = "baseline"
    ref["location"] = channel
    ref["role"] = "baseline"
    ref["axis"] = channel[-1]
    ref["quality_flags"] = []
    ref["window_id"] = "phase05_reference"
    return ref


def build_example(
    *,
    example_id: str,
    kind: str,
    channel: str,
    reference: dict[str, Any],
    reference_members: list[WindowKey],
    targets: list[tuple[WindowKey, dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    order = list(range(len(targets)))
    rng.shuffle(order)
    targets = [targets[i] for i in order]

    prechecks: list[dict[str, Any]] = []
    truth_sensor_labels: dict[str, str] = {}
    truth_reports: list[dict[str, Any]] = []
    rule_sensor_labels: dict[str, str] = {}
    rule_reports: list[dict[str, Any]] = []
    windows_meta: list[dict[str, Any]] = []

    for index, (key, metrics) in enumerate(targets, start=1):
        sensor_id = f"target_{index}" if len(targets) > 1 else "target_1"
        metrics = dict(metrics)
        metrics["sensor_id"] = sensor_id
        comparison = compare_building_metrics(metrics, reference)
        rule_label, rule_confidence, rule_evidence = deterministic_sensor_label(
            metrics=metrics, baseline_metrics=reference, comparison=comparison
        )
        prechecks.append(
            {
                "target": PromptTarget(sensor_id=sensor_id, location=channel),
                "axis": channel[-1],
                "duration_s": WINDOW_DURATION_S,
                "metrics": metrics,
                "comparison": comparison,
                "rule_label": rule_label,
                "rule_confidence": rule_confidence,
                "rule_evidence": rule_evidence,
            }
        )
        state = session_state(key.session)
        truth_label = STATE_TO_LABEL[state]
        truth_sensor_labels[sensor_id] = truth_label
        truth_reports.append({"sensor_id": sensor_id, "small_agent_label": truth_label, "quality_flags": []})
        rule_sensor_labels[sensor_id] = rule_label
        rule_reports.append({"sensor_id": sensor_id, "small_agent_label": rule_label, "quality_flags": []})
        windows_meta.append(
            {
                "sensor_id": sensor_id,
                "session": key.session,
                "state": state,
                "damage_type": damage_type(key.session),
                "file": key.file,
                "channel": key.channel,
                "start_s": key.start_s,
                "duration_s": WINDOW_DURATION_S,
            }
        )

    clean_reference = dict(reference)
    truth_network_label, _, truth_affected, _, _ = deterministic_network_label(
        reports=truth_reports, baseline_metrics=clean_reference
    )
    rule_network_label, rule_network_confidence, rule_affected, rule_evidence_net, rule_explanation = (
        deterministic_network_label(reports=rule_reports, baseline_metrics=clean_reference)
    )

    prompt = _build_all_sensor_prompt(
        baseline_metrics=reference,
        sensor_prechecks=prechecks,
        network_rule_label=rule_network_label,
        network_rule_confidence=rule_network_confidence,
        affected_sensor_ids=rule_affected,
        network_rule_evidence=rule_evidence_net,
        network_rule_explanation=rule_explanation,
    )
    response_format = _all_sensor_response_format([p["target"].sensor_id for p in prechecks])["response_format"]

    n_damaged = sum(1 for m in windows_meta if m["state"] != "healthy")
    return {
        "example_id": example_id,
        "task": "all_sensor_agent",
        "messages": [
            {"role": "system", "content": PRODUCTION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "response_format": response_format,
        "truth": {
            "sensor_labels": truth_sensor_labels,
            "network_label": truth_network_label,
            "affected_sensor_ids": sorted(truth_affected),
        },
        "rule_predictions": {
            "sensor_labels": rule_sensor_labels,
            "network_label": rule_network_label,
            "affected_sensor_ids": sorted(rule_affected),
        },
        "strata": {
            "kind": kind,
            "channel": channel,
            "n_targets": len(targets),
            "n_damaged": n_damaged,
            "sessions": sorted({m["session"] for m in windows_meta}),
            "damage_types": sorted({m["damage_type"] for m in windows_meta if m["damage_type"]}),
        },
        "windows": windows_meta,
        "reference": {
            "channel": channel,
            "kind": "median_composite_of_healthy_windows",
            "member_windows": [
                {"session": k.session, "file": k.file, "channel": k.channel, "start_s": k.start_s}
                for k in reference_members
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lumo-root", default="data/external/lumo")
    parser.add_argument("--out", default="data/phase05_benchmark")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    lumo_root = Path(args.lumo_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    session_files: dict[str, list[Path]] = {}
    for session, file_path in iter_session_files(lumo_root):
        session_files.setdefault(session, []).append(file_path)
    sessions = sorted(session_files)
    by_state: dict[str, list[str]] = {"healthy": [], "mild": [], "significant": []}
    for session in sessions:
        by_state[session_state(session)].append(session)
    print(f"sessions: {len(sessions)} {by_state}")

    # Pass 1: compute metrics for the full candidate window grid, one file load each.
    print("computing candidate-window metrics ...")
    metrics_by_key: dict[WindowKey, dict[str, Any]] = {}
    sampling_rate_hz: float | None = None
    for session in sessions:
        for file_path in session_files[session]:
            data, names, fs = load_lumo_file(file_path)
            sampling_rate_hz = fs
            for channel in CHANNELS:
                for start_s in CANDIDATE_STARTS_S:
                    key = WindowKey(session=session, file=file_path.name, channel=channel, start_s=start_s)
                    metrics_by_key[key] = window_metrics(data, names, fs, channel, start_s, sensor_id="candidate")
            print(f"  {session}/{file_path.name}: ok")

    # Reference pool: per channel, REF_WINDOWS_PER_HEALTHY_FILE windows from each
    # healthy file (deterministic picks from the candidate grid), median-composited.
    used_keys: set[WindowKey] = set()
    references: dict[str, dict[str, Any]] = {}
    reference_members: dict[str, list[WindowKey]] = {}
    for channel in CHANNELS:
        members: list[WindowKey] = []
        for session in by_state["healthy"]:
            for file_index, file_path in enumerate(session_files[session]):
                starts = CANDIDATE_STARTS_S[file_index % 2 :: 3][:REF_WINDOWS_PER_HEALTHY_FILE]
                for start_s in starts:
                    members.append(WindowKey(session=session, file=file_path.name, channel=channel, start_s=start_s))
        references[channel] = median_reference(channel, [metrics_by_key[k] for k in members])
        reference_members[channel] = members
        used_keys.update(members)

    # Target pools per (channel, state), excluding reference members.
    pools: dict[tuple[str, str], list[WindowKey]] = {}
    for key in metrics_by_key:
        if key in used_keys:
            continue
        pools.setdefault((key.channel, session_state(key.session)), []).append(key)
    for pool in pools.values():
        pool.sort(key=lambda k: (k.session, k.file, k.start_s))
        rng.shuffle(pool)

    def draw(channel: str, state: str, count: int) -> list[WindowKey]:
        pool = pools[(channel, state)]
        if len(pool) < count:
            raise SystemExit(f"pool exhausted: {channel}/{state} needs {count}, has {len(pool)}")
        picked, pools[(channel, state)] = pool[:count], pool[count:]
        return picked

    examples: list[dict[str, Any]] = []

    # Kind "single": per channel, SINGLE_PER_CHANNEL windows of each state.
    for channel in CHANNELS:
        for state, count in SINGLE_PER_CHANNEL.items():
            for key in draw(channel, state, count):
                example_id = f"single_{channel}_{state}_{hashlib.sha256(repr(key).encode()).hexdigest()[:10]}"
                examples.append(
                    build_example(
                        example_id=example_id,
                        kind="single",
                        channel=channel,
                        reference=references[channel],
                        reference_members=reference_members[channel],
                        targets=[(key, metrics_by_key[key])],
                        rng=rng,
                    )
                )

    # Kind "multi": MULTI_EXAMPLES examples, 5 same-channel targets across sessions.
    for index in range(MULTI_EXAMPLES):
        channel = CHANNELS[index % len(CHANNELS)]
        # index//len(CHANNELS) so composition is not confounded with channel.
        n_mild, n_significant = MULTI_COMPOSITIONS[(index // len(CHANNELS)) % len(MULTI_COMPOSITIONS)]
        n_healthy = MULTI_TARGETS - n_mild - n_significant
        keys = draw(channel, "healthy", n_healthy) + draw(channel, "mild", n_mild) + draw(channel, "significant", n_significant)
        example_id = f"multi_{channel}_{index:03d}"
        examples.append(
            build_example(
                example_id=example_id,
                kind="multi",
                channel=channel,
                reference=references[channel],
                reference_members=reference_members[channel],
                targets=[(k, metrics_by_key[k]) for k in keys],
                rng=rng,
            )
        )

    rng.shuffle(examples)
    eval_path = out_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")

    n_sensor_decisions = sum(len(e["truth"]["sensor_labels"]) for e in examples)
    truth_counts: dict[str, int] = {}
    for example in examples:
        for label in example["truth"]["sensor_labels"].values():
            truth_counts[label] = truth_counts.get(label, 0) + 1
    network_counts: dict[str, int] = {}
    for example in examples:
        label = example["truth"]["network_label"]
        network_counts[label] = network_counts.get(label, 0) + 1
    prompt_chars = [len(e["messages"][1]["content"]) for e in examples]
    manifest = {
        "generated_with": Path(__file__).name,
        "seed": args.seed,
        "lumo_root": str(lumo_root),
        "sampling_rate_hz": sampling_rate_hz,
        "window_duration_s": WINDOW_DURATION_S,
        "channels": CHANNELS,
        "examples_total": len(examples),
        "examples_by_kind": {
            kind: sum(1 for e in examples if e["strata"]["kind"] == kind) for kind in ("single", "multi")
        },
        "sensor_decisions_total": n_sensor_decisions,
        "truth_sensor_label_counts": truth_counts,
        "truth_network_label_counts": network_counts,
        "prompt_chars": {
            "mean": int(np.mean(prompt_chars)),
            "max": int(np.max(prompt_chars)),
        },
        "production_conditions": {
            "system_message": PRODUCTION_SYSTEM_MESSAGE,
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": "json_schema (frozen per example)",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {eval_path} ({len(examples)} examples, {n_sensor_decisions} sensor decisions)")


if __name__ == "__main__":
    main()
