from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import resolve_from_config
from ..exceptions import DataContractError
from .splits import assert_disjoint_groups


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataContractError(f"invalid episode index JSON at line {line_number}: {exc}") from exc
    return rows


def supervised_coverage_policy(
    audit: Mapping[str, Any],
    supervised_dataset_ids: Iterable[str],
    *,
    required_strata: Iterable[str] = ("healthy", "changed::unknown_anomaly"),
    active_splits: Iterable[str] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Classify supervised split coverage as valid, limited, or erroneous.

    Complete session groups may never be divided merely to populate every split.
    A stratum with fewer independent groups than active splits is therefore an
    explicit coverage limitation, not a split-construction error. Dataset-level
    labelled coverage is still required in every active split, and any missing
    stratum coverage that was avoidable remains fatal.

    The distinction matters for datasets such as QUGS, whose single independent
    healthy group is correctly retained in training while changed episodes still
    provide target-level normal and anomaly decisions in held-out evaluation.
    """

    split_names = tuple(str(value) for value in active_splits)
    strata = tuple(str(value) for value in required_strata)
    datasets = tuple(str(value) for value in supervised_dataset_ids)
    labelled = dict(audit.get("labelled_by_dataset") or {})
    buckets = {
        (str(item.get("dataset_id")), str(item.get("stratum"))): dict(item)
        for item in audit.get("by_dataset_stratum", [])
    }

    errors: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []

    for dataset_id in datasets:
        counts = dict(labelled.get(dataset_id) or {})
        missing_dataset_splits = [
            split for split in split_names if int(counts.get(split, 0)) <= 0
        ]
        if not counts:
            errors.append(
                {
                    "type": "supervised_dataset_missing",
                    "dataset_id": dataset_id,
                }
            )
        elif missing_dataset_splits:
            errors.append(
                {
                    "type": "supervised_dataset_missing_labelled_split",
                    "dataset_id": dataset_id,
                    "missing_splits": missing_dataset_splits,
                    "episode_counts": {split: int(counts.get(split, 0)) for split in split_names},
                }
            )

        stratum_reports: list[dict[str, Any]] = []
        for stratum in strata:
            item = buckets.get((dataset_id, stratum))
            if item is None:
                error = {
                    "type": "supervised_stratum_missing",
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                }
                errors.append(error)
                stratum_reports.append({**error, "status": "error"})
                continue

            episode_counts = dict(item.get("episode_counts") or {})
            missing_splits = [
                split for split in split_names if int(episode_counts.get(split, 0)) <= 0
            ]
            coverage_limited = bool(item.get("coverage_limited", False))
            report = {
                "dataset_id": dataset_id,
                "stratum": stratum,
                "group_count": int(item.get("group_count", 0)),
                "coverage_limited": coverage_limited,
                "episode_counts": {
                    split: int(episode_counts.get(split, 0)) for split in split_names
                },
                "present_splits": [split for split in split_names if split not in missing_splits],
                "missing_splits": missing_splits,
            }

            if not missing_splits:
                report["status"] = "complete"
            elif coverage_limited:
                # A one-group stratum is intentionally train-only; with two groups
                # the splitter prioritises train and test. Never manufacture
                # validation/test windows by dividing a complete session.
                if "train" in split_names and int(episode_counts.get("train", 0)) <= 0:
                    error = {
                        **report,
                        "type": "coverage_limited_stratum_missing_training",
                        "status": "error",
                    }
                    errors.append(error)
                    report = error
                else:
                    limitation = {
                        **report,
                        "type": "insufficient_independent_groups",
                        "status": "coverage_limited",
                        "note": (
                            "Complete-group disjointness makes the missing split coverage "
                            "unavoidable; held-out session-level claims for this stratum are not available."
                        ),
                    }
                    limitations.append(limitation)
                    report = limitation
            else:
                error = {
                    **report,
                    "type": "avoidable_supervised_stratum_gap",
                    "status": "error",
                }
                errors.append(error)
                report = error
            stratum_reports.append(report)

        dataset_reports.append(
            {
                "dataset_id": dataset_id,
                "labelled_episode_counts": {
                    split: int(counts.get(split, 0)) for split in split_names
                },
                "strata": stratum_reports,
            }
        )

    return {
        "policy": "complete_group_coverage_with_explicit_limitations_v1",
        "active_splits": list(split_names),
        "required_strata": list(strata),
        "supervised_datasets": list(datasets),
        "passed": not errors,
        "errors": errors,
        "coverage_limitations": limitations,
        "datasets": dataset_reports,
    }

SUPPORTED_ASSIGNMENT_STRATEGIES = frozenset(
    {
        "dataset_stratified_complete_groups_v2",
        "publisher_explicit_plus_dataset_stratified_complete_groups_v1",
        "explicit_locked_complete_groups_v1",
    }
)
_SPLIT_NAMES = ("train", "validation", "test")


def is_supported_assignment_strategy(value: Any) -> bool:
    """Return whether ``value`` names an accepted complete-group strategy."""

    return str(value) in SUPPORTED_ASSIGNMENT_STRATEGIES


def _normalise_materialized_splits(value: Any, *, source: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    requested = {str(item) for item in values}
    unknown = requested - set(_SPLIT_NAMES)
    if unknown:
        raise DataContractError(
            f"{source} materialize_splits contains invalid values: {sorted(unknown)}"
        )
    ordered = tuple(split for split in _SPLIT_NAMES if split in requested)
    if not ordered:
        raise DataContractError(f"{source} materialize_splits cannot be empty")
    return ordered


def _resolve_materialized_splits(
    config: Mapping[str, Any],
    assignment: Mapping[str, Any],
    build: Mapping[str, Any],
) -> tuple[str, ...]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    data_cfg = dict(config.get("data") or {})
    for source, value in (
        ("config", data_cfg.get("materialize_splits")),
        ("assignment", assignment.get("materialize_splits")),
        ("build", build.get("materialize_splits")),
    ):
        parsed = _normalise_materialized_splits(value, source=source)
        if parsed is not None:
            candidates.append((source, parsed))
    if not candidates:
        split_cfg = dict(data_cfg.get("split") or {})
        active = tuple(
            split for split in _SPLIT_NAMES if float(split_cfg.get(split, 1.0)) > 0
        )
        return active or _SPLIT_NAMES
    expected = candidates[0][1]
    if any(value != expected for _, value in candidates[1:]):
        raise DataContractError(
            "materialize_splits drift between config, assignment, and build reports: "
            + json.dumps({name: list(value) for name, value in candidates}, sort_keys=True)
        )
    return expected


def _generated_assignment_report(assignment: Mapping[str, Any]) -> Mapping[str, Any]:
    strategy = str(assignment.get("strategy", ""))
    if strategy != "publisher_explicit_plus_dataset_stratified_complete_groups_v1":
        return assignment
    generated = assignment.get("generated_assignment_report")
    if not isinstance(generated, Mapping):
        raise DataContractError("hybrid split report is missing generated_assignment_report")
    generated_strategy = str(generated.get("strategy", ""))
    generated_group_count = int(assignment.get("generated_group_count", 0) or 0)
    if generated_group_count > 0 and generated_strategy != "dataset_stratified_complete_groups_v2":
        raise DataContractError(
            "hybrid split report has generated groups but its nested strategy is not "
            "dataset_stratified_complete_groups_v2"
        )
    if generated_group_count == 0 and generated_strategy not in {
        "",
        "explicit_locked_complete_groups_v1",
        "dataset_stratified_complete_groups_v2",
    }:
        raise DataContractError(
            f"unsupported generated split strategy inside hybrid report: {generated_strategy!r}"
        )
    return generated


def _assignment_map(assignment: Mapping[str, Any]) -> dict[str, str]:
    raw = assignment.get("assignments")
    if isinstance(raw, Mapping) and raw:
        result = {str(group): str(split) for group, split in raw.items()}
    else:
        result: dict[str, str] = {}
        for bucket in assignment.get("by_dataset_stratum", []):
            for split, groups in dict(bucket.get("groups") or {}).items():
                for group in groups:
                    previous = result.setdefault(str(group), str(split))
                    if previous != str(split):
                        raise DataContractError(
                            f"split group {group!r} has conflicting report assignments"
                        )
    if not result:
        raise DataContractError("split assignment report contains no complete-group assignments")
    invalid = sorted({split for split in result.values() if split not in _SPLIT_NAMES})
    if invalid:
        raise DataContractError(f"split assignment report contains invalid splits: {invalid}")
    return result


def audit_episode_split(config: dict[str, Any]) -> dict[str, Any]:
    """Audit the splits intentionally materialized by an episode store.

    Development stores materialize only train and validation while the immutable
    assignment lock already reserves test groups.  An authorized test-only store
    later reuses the same lock and materializes only test.  The audit accepts both
    modes and rejects assignment drift, accidental test access, missing assigned
    materialized strata, or inconsistent build reports.
    """

    root = resolve_from_config(config, config["data"]["episode_root"])
    index_path = root / "index.jsonl"
    assignment_path = root / "split_assignment_report.json"
    build_path = root / "build_report.json"
    for path in (index_path, assignment_path, build_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = _read_jsonl(index_path)
    if not rows:
        raise DataContractError("episode index is empty")
    assert_disjoint_groups(rows)

    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    build = json.loads(build_path.read_text(encoding="utf-8"))
    strategy = str(assignment.get("strategy", ""))
    if not is_supported_assignment_strategy(strategy):
        raise DataContractError(
            "episode store was not built with a supported complete-group strategy; "
            f"observed={strategy!r}, supported={sorted(SUPPORTED_ASSIGNMENT_STRATEGIES)}"
        )
    if build.get("split_strategy") not in (None, strategy):
        raise DataContractError(
            "build report split_strategy does not match split_assignment_report: "
            f"{build.get('split_strategy')!r} versus {strategy!r}"
        )

    generated_report = _generated_assignment_report(assignment)
    assignments = _assignment_map(assignment)
    materialized_splits = _resolve_materialized_splits(config, assignment, build)
    materialized_set = set(materialized_splits)
    nonmaterialized_splits = tuple(
        split for split in _SPLIT_NAMES if split not in materialized_set
    )
    gaps: list[dict[str, Any]] = []

    assignment_lock_verified: bool | None = None
    lock_value = assignment.get("assignment_lock_path")
    if lock_value:
        lock_path = Path(str(lock_value))
        if not lock_path.is_absolute():
            lock_path = (assignment_path.parent / lock_path).resolve()
        if not lock_path.is_file():
            assignment_lock_verified = False
            gaps.append({"type": "assignment_lock_missing", "path": str(lock_path)})
        else:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_assignments = {
                str(group): str(split)
                for group, split in dict(lock_payload.get("assignments") or {}).items()
            }
            assignment_lock_verified = lock_assignments == assignments
            if not assignment_lock_verified:
                gaps.append(
                    {
                        "type": "assignment_lock_mismatch",
                        "path": str(lock_path),
                        "assignment_group_count": len(assignments),
                        "lock_group_count": len(lock_assignments),
                    }
                )

    episode_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    dataset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in _SPLIT_NAMES}
    )
    labelled_dataset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in _SPLIT_NAMES}
    )
    observed_split_counts = {split: 0 for split in _SPLIT_NAMES}

    for row in rows:
        dataset_id = str(row["dataset_id"])
        split = str(row["split"])
        stratum = str(row.get("split_stratum", "unknown"))
        group = str(row.get("split_group", ""))
        if split not in _SPLIT_NAMES:
            gaps.append(
                {"type": "episode_has_invalid_split", "episode_id": row.get("episode_id"), "split": split}
            )
            continue
        if split not in materialized_set:
            gaps.append(
                {
                    "type": "episode_from_nonmaterialized_split",
                    "episode_id": row.get("episode_id"),
                    "split": split,
                }
            )
        expected_split = assignments.get(group)
        if expected_split is None:
            gaps.append(
                {
                    "type": "episode_group_missing_from_assignment",
                    "episode_id": row.get("episode_id"),
                    "split_group": group,
                }
            )
        elif expected_split != split:
            gaps.append(
                {
                    "type": "episode_assignment_mismatch",
                    "episode_id": row.get("episode_id"),
                    "split_group": group,
                    "observed_split": split,
                    "assigned_split": expected_split,
                }
            )
        episode_counts[(dataset_id, stratum, split)] += 1
        dataset_counts[dataset_id][split] += 1
        observed_split_counts[split] += 1
        if bool(row.get("has_label", True)):
            labelled_dataset_counts[dataset_id][split] += 1

    if build.get("total_episodes") is not None and int(build["total_episodes"]) != len(rows):
        gaps.append(
            {
                "type": "build_total_episode_count_mismatch",
                "build_total": int(build["total_episodes"]),
                "index_total": len(rows),
            }
        )
    build_by_split = dict(build.get("by_split") or {})
    for split in _SPLIT_NAMES:
        if split in build_by_split and int(build_by_split[split]) != observed_split_counts[split]:
            gaps.append(
                {
                    "type": "build_split_episode_count_mismatch",
                    "split": split,
                    "build_count": int(build_by_split[split]),
                    "index_count": observed_split_counts[split],
                }
            )
    for split in materialized_splits:
        if observed_split_counts[split] <= 0:
            gaps.append({"type": "materialized_split_has_no_episodes", "split": split})

    assigned_split_counts = {split: 0 for split in _SPLIT_NAMES}
    assigned_dataset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in _SPLIT_NAMES}
    )
    for group, split in assignments.items():
        assigned_split_counts[split] += 1
        dataset_id = group.split("|", 1)[0]
        assigned_dataset_counts[dataset_id][split] += 1

    nonmaterialized_present = "nonmaterialized_entry_counts" in build
    nonmaterialized_entry_counts = {
        split: int(dict(build.get("nonmaterialized_entry_counts") or {}).get(split, 0) or 0)
        for split in _SPLIT_NAMES
    }
    if strategy == "publisher_explicit_plus_dataset_stratified_complete_groups_v1" and not nonmaterialized_present:
        gaps.append({"type": "nonmaterialized_entry_counts_missing"})
    for split in _SPLIT_NAMES:
        if split in materialized_set:
            if nonmaterialized_entry_counts[split] != 0:
                gaps.append(
                    {
                        "type": "active_split_reported_nonmaterialized_entries",
                        "split": split,
                        "entry_count": nonmaterialized_entry_counts[split],
                    }
                )
        else:
            if observed_split_counts[split] != 0:
                gaps.append(
                    {
                        "type": "nonmaterialized_split_contains_episodes",
                        "split": split,
                        "episode_count": observed_split_counts[split],
                    }
                )
            if assigned_split_counts[split] > 0 and nonmaterialized_present and nonmaterialized_entry_counts[split] <= 0:
                gaps.append(
                    {
                        "type": "assigned_nonmaterialized_split_not_recorded",
                        "split": split,
                        "assigned_group_count": assigned_split_counts[split],
                    }
                )

    # Dataset-level coverage includes publisher-explicit groups such as UPC,
    # which are not present in the nested generated-strata report.
    for dataset_id, assigned_counts in sorted(assigned_dataset_counts.items()):
        for split in materialized_splits:
            if assigned_counts[split] > 0 and int(dataset_counts[dataset_id][split]) <= 0:
                gaps.append(
                    {
                        "type": "assigned_dataset_split_not_materialized",
                        "dataset_id": dataset_id,
                        "split": split,
                        "assigned_group_count": assigned_counts[split],
                    }
                )

    audited_buckets: list[dict[str, Any]] = []
    for bucket in generated_report.get("by_dataset_stratum", []):
        dataset_id = str(bucket["dataset_id"])
        stratum = str(bucket["stratum"])
        group_count = int(bucket["group_count"])
        coverage_limited = bool(bucket.get("coverage_limited", False))
        group_counts = {
            split: int(dict(bucket.get("group_counts") or {}).get(split, 0) or 0)
            for split in _SPLIT_NAMES
        }
        counts = {
            split: int(episode_counts[(dataset_id, stratum, split)])
            for split in _SPLIT_NAMES
        }
        missing = [
            split
            for split in materialized_splits
            if group_counts[split] > 0 and counts[split] == 0
        ]
        item = {
            "dataset_id": dataset_id,
            "stratum": stratum,
            "group_count": group_count,
            "group_counts": group_counts,
            "coverage_limited": coverage_limited,
            "episode_counts": counts,
            "materialize_splits": list(materialized_splits),
            "missing_materialized_splits": missing,
            "intentionally_nonmaterialized_splits": [
                split
                for split in _SPLIT_NAMES
                if split not in materialized_set and group_counts[split] > 0
            ],
        }
        audited_buckets.append(item)
        if missing:
            gaps.append({"type": "assigned_stratum_split_not_materialized", **item})

    for dataset_id, counts in sorted(labelled_dataset_counts.items()):
        if sum(counts.values()) <= 0:
            continue
        missing = [
            split
            for split in materialized_splits
            if assigned_dataset_counts[dataset_id][split] > 0 and int(counts[split]) <= 0
        ]
        if missing:
            gaps.append(
                {
                    "type": "supervised_dataset_missing_labelled_materialized_split",
                    "dataset_id": dataset_id,
                    "missing_splits": missing,
                    "labelled_episode_counts": dict(counts),
                }
            )

    test_assigned_groups = int(assigned_split_counts["test"])
    test_materialized = "test" in materialized_set
    test_boundary = {
        "status": (
            "materialized"
            if test_materialized
            else ("locked_and_not_materialized" if test_assigned_groups > 0 else "unassigned")
        ),
        "assigned": bool(test_assigned_groups > 0),
        "assigned_group_count": test_assigned_groups,
        "materialized": test_materialized,
        "episode_count": int(observed_split_counts["test"]),
        "nonmaterialized_entry_count": int(nonmaterialized_entry_counts["test"]),
        "locked_and_unopened": bool(
            test_assigned_groups > 0
            and not test_materialized
            and observed_split_counts["test"] == 0
            and (not nonmaterialized_present or nonmaterialized_entry_counts["test"] > 0)
        ),
    }

    report = {
        "strategy": strategy,
        "generated_strategy": generated_report.get("strategy"),
        "materialize_splits": list(materialized_splits),
        "materialized_splits": list(materialized_splits),
        "intentionally_nonmaterialized_splits": list(nonmaterialized_splits),
        "assignment_group_counts": assigned_split_counts,
        "nonmaterialized_entry_counts": nonmaterialized_entry_counts,
        "assignment_lock_verified": assignment_lock_verified,
        "test_boundary": test_boundary,
        "group_disjoint": True,
        "episode_count": len(rows),
        "by_split": observed_split_counts,
        "by_dataset": dict(sorted(dataset_counts.items())),
        "labelled_by_dataset": dict(sorted(labelled_dataset_counts.items())),
        "by_dataset_stratum": audited_buckets,
        "coverage_gaps": gaps,
        "passed": not gaps,
        "build_report": str(build_path),
        "assignment_report": str(assignment_path),
    }
    if gaps:
        raise DataContractError(
            "complete-group split audit failed; inspect coverage_gaps: "
            + json.dumps(gaps[:5], sort_keys=True)
        )
    return report
