"""Memory-bounded receipts for reusing an already evaluated checkpoint probe.

The checkpoint selector already writes one JSON object per evaluated episode to
``predictions.jsonl``.  Reusing those IDs is both more exact and much cheaper
than rebuilding the selection from a potentially very large episode index.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROBE_MANIFEST_SCHEMA = "vibrogemma-evaluated-probe-manifest-v0.19.6.7"


def _truth_route(row: dict[str, Any]) -> str:
    ground_truth = dict(row.get("ground_truth") or {})
    if not bool(ground_truth.get("target_supervision_available", False)):
        global_class = str((ground_truth.get("global") or {}).get("class", "unknown"))
        return f"GLOBAL_ONLY_{global_class}"
    affected = [
        str(target.get("sensor_id"))
        for target in (ground_truth.get("targets") or [])
        if bool(target.get("affected", False))
    ]
    if not affected:
        return "NONE"
    if len(affected) == 1:
        return affected[0]
    return "MULTI"


def build_evaluated_probe_manifest(
    predictions_path: Path,
    *,
    evaluation_state_path: Path,
    expected_checkpoint: Path,
    expected_split: str,
    expected_count: int,
    episodes_per_stratum: int,
    selection_stage: str,
) -> dict[str, Any]:
    """Stream an existing prediction file and retain only compact probe receipts."""

    predictions_path = predictions_path.expanduser().resolve()
    evaluation_state_path = evaluation_state_path.expanduser().resolve()
    expected_checkpoint = expected_checkpoint.expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if not evaluation_state_path.is_file():
        raise FileNotFoundError(evaluation_state_path)

    state = json.loads(evaluation_state_path.read_text(encoding="utf-8"))
    if not bool(state.get("complete", False)):
        raise RuntimeError("checkpoint-selection evaluation_state is not complete")
    if str(state.get("split")) != str(expected_split):
        raise RuntimeError(
            f"evaluation_state split mismatch: {state.get('split')!r} != {expected_split!r}"
        )
    state_checkpoint = Path(str(state.get("checkpoint") or "")).expanduser().resolve()
    if state_checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"evaluation_state checkpoint mismatch: {state_checkpoint} != {expected_checkpoint}"
        )

    digest = hashlib.sha256()
    episode_ids: list[str] = []
    seen: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    truth_route_counts: Counter[str] = Counter()
    with predictions_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid prediction JSON at {predictions_path}:{line_number}: {exc}"
                ) from exc
            episode_id = str(row.get("episode_id") or "")
            if not episode_id:
                raise RuntimeError(
                    f"prediction row {line_number} has no non-empty episode_id"
                )
            if episode_id in seen:
                raise RuntimeError(f"duplicate evaluated episode_id: {episode_id}")
            seen.add(episode_id)
            episode_ids.append(episode_id)
            dataset_counts[str(row.get("dataset_id") or "unknown")] += 1
            truth_route_counts[_truth_route(row)] += 1

    if len(episode_ids) != int(expected_count):
        raise RuntimeError(
            f"expected {expected_count} evaluated episodes, observed {len(episode_ids)}"
        )
    for key in ("completed", "total"):
        if int(state.get(key, -1)) != int(expected_count):
            raise RuntimeError(
                f"evaluation_state {key}={state.get(key)!r}, expected {expected_count}"
            )

    sorted_ids = sorted(episode_ids)
    ids_sha256 = hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()
    return {
        "schema": PROBE_MANIFEST_SCHEMA,
        "source": "completed_checkpoint_selection_predictions",
        "selection_stage": str(selection_stage),
        "split": str(expected_split),
        "episodes_per_stratum": int(episodes_per_stratum),
        "checkpoint": str(expected_checkpoint),
        "predictions_path": str(predictions_path),
        "predictions_sha256": digest.hexdigest(),
        "evaluation_state_path": str(evaluation_state_path),
        "evaluation_signature": state.get("evaluation_signature"),
        "selected_episode_count": len(sorted_ids),
        "selected_episode_ids_sha256": ids_sha256,
        "episode_ids": sorted_ids,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "truth_route_counts": dict(sorted(truth_route_counts.items())),
        "memory_policy": {
            "full_episode_index_scanned": False,
            "full_prediction_rows_retained": False,
            "retained_payload": "episode_ids_and_aggregate_counters_only",
        },
    }


def load_evaluated_probe_manifest(
    path: Path,
    *,
    expected_checkpoint: Path | None = None,
    expected_split: str | None = None,
    expected_count: int | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Load and validate a compact evaluated-probe manifest."""

    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROBE_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported probe manifest schema: {payload.get('schema')!r}")
    ids = [str(value) for value in (payload.get("episode_ids") or [])]
    if len(ids) != len(set(ids)):
        raise RuntimeError("probe manifest contains duplicate episode IDs")
    observed_hash = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    if observed_hash != str(payload.get("selected_episode_ids_sha256") or ""):
        raise RuntimeError("probe manifest episode-ID digest mismatch")
    if expected_count is not None and len(ids) != int(expected_count):
        raise RuntimeError(f"probe manifest has {len(ids)} IDs, expected {expected_count}")
    if expected_split is not None and str(payload.get("split")) != str(expected_split):
        raise RuntimeError("probe manifest split mismatch")
    if expected_checkpoint is not None:
        observed = Path(str(payload.get("checkpoint") or "")).expanduser().resolve()
        if observed != expected_checkpoint.expanduser().resolve():
            raise RuntimeError("probe manifest checkpoint mismatch")
    return set(ids), payload
