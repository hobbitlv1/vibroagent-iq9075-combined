from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..exceptions import DataContractError


_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (float(self.train), float(self.validation), float(self.test))
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError(f"split ratios must sum to one, got {sum(values):.12f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "train": float(self.train),
            "validation": float(self.validation),
            "test": float(self.test),
        }


def assign_group(group: str, ratios: SplitRatios, seed: int) -> str:
    """Assign a complete group deterministically without using process hash state.

    This legacy helper remains available for compatibility and small tests. The
    public episode builder uses :func:`assign_dataset_stratified_groups` so each
    supervised dataset and state stratum receives explicit held-out coverage.
    """

    digest = hashlib.sha256(f"{int(seed)}\0{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios.train:
        return "train"
    if value < ratios.train + ratios.validation:
        return "validation"
    return "test"


def _stable_order(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{int(seed)}\0{value}".encode("utf-8")).digest()


def _allocation_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    """Allocate group counts with deterministic minimum held-out coverage.

    When a stratum has at least three groups and all three ratios are active,
    each split receives at least one complete group. With two groups, training
    and test are preferred because test coverage is more important than a
    single-group validation estimate. One-group strata remain training-only and
    are explicitly reported as coverage-limited.
    """

    if total < 1:
        return {name: 0 for name in _SPLITS}
    ratio_map = ratios.as_dict()
    active = [name for name in _SPLITS if ratio_map[name] > 0]
    if not active:
        raise ValueError("at least one split ratio must be positive")

    counts = {name: 0 for name in _SPLITS}
    if total == 1:
        preferred = "train" if "train" in active else active[0]
        counts[preferred] = 1
        return counts
    if total == 2:
        preferred: list[str] = []
        if "train" in active:
            preferred.append("train")
        if "test" in active:
            preferred.append("test")
        if "validation" in active:
            preferred.append("validation")
        for name in active:
            if name not in preferred:
                preferred.append(name)
        for index in range(total):
            counts[preferred[min(index, len(preferred) - 1)]] += 1
        return counts

    # Three or more groups: guarantee every active split one group whenever
    # possible, then distribute the remaining groups by largest remainder.
    minimum_splits = active[: min(total, len(active))]
    for name in minimum_splits:
        counts[name] += 1
    remaining = total - len(minimum_splits)
    if remaining <= 0:
        return counts

    ratio_sum = sum(ratio_map[name] for name in active)
    quotas = {
        name: remaining * ratio_map[name] / ratio_sum
        for name in active
    }
    floors = {name: int(quotas[name]) for name in active}
    for name, value in floors.items():
        counts[name] += value
    leftover = remaining - sum(floors.values())
    order = sorted(
        active,
        key=lambda name: (quotas[name] - floors[name], ratio_map[name], -_SPLITS.index(name)),
        reverse=True,
    )
    for name in order[:leftover]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise AssertionError("internal split allocation error")
    return counts


def assign_dataset_stratified_groups(
    records: Iterable[Mapping[str, object]],
    ratios: SplitRatios,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign complete groups within every dataset/state stratum.

    Required input keys are ``split_group``, ``dataset_id`` and
    ``split_stratum``. Repeated rows for the same group are allowed only when
    dataset and stratum metadata agree. The result is deterministic and avoids
    the previous failure mode where a large dataset could receive no test group.
    """

    group_metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        group = str(record.get("split_group", ""))
        dataset_id = str(record.get("dataset_id", ""))
        stratum = str(record.get("split_stratum", ""))
        if not group or not dataset_id or not stratum:
            raise DataContractError(
                "dataset-stratified splitting requires split_group, dataset_id and split_stratum"
            )
        previous = group_metadata.setdefault(group, (dataset_id, stratum))
        if previous != (dataset_id, stratum):
            raise DataContractError(
                f"split group {group!r} has conflicting dataset/stratum metadata: "
                f"{previous!r} versus {(dataset_id, stratum)!r}"
            )
    if not group_metadata:
        raise DataContractError("cannot assign an empty group collection")

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, metadata in group_metadata.items():
        buckets[metadata].append(group)

    assignments: dict[str, str] = {}
    bucket_report: list[dict[str, Any]] = []
    for (dataset_id, stratum), groups in sorted(buckets.items()):
        ordered = sorted(groups, key=lambda group: _stable_order(group, seed))
        counts = _allocation_counts(len(ordered), ratios)
        cursor = 0
        for split in _SPLITS:
            for group in ordered[cursor : cursor + counts[split]]:
                assignments[group] = split
            cursor += counts[split]
        bucket_report.append(
            {
                "dataset_id": dataset_id,
                "stratum": stratum,
                "group_count": len(ordered),
                "group_counts": dict(counts),
                "coverage_limited": len(ordered) < len([name for name in _SPLITS if ratios.as_dict()[name] > 0]),
                "groups": {
                    split: [group for group in ordered if assignments[group] == split]
                    for split in _SPLITS
                },
            }
        )

    if set(assignments) != set(group_metadata):
        missing = sorted(set(group_metadata) - set(assignments))
        raise AssertionError(f"internal split assignment omitted groups: {missing}")

    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in _SPLITS})
    for group, split in assignments.items():
        dataset_id, _ = group_metadata[group]
        by_dataset[dataset_id][split] += 1

    report: dict[str, Any] = {
        "strategy": "dataset_stratified_complete_groups_v2",
        "seed": int(seed),
        "ratios": ratios.as_dict(),
        "group_count": len(assignments),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_dataset_stratum": bucket_report,
    }
    assert_stratified_group_coverage(group_metadata, assignments, ratios)
    return assignments, report


def assert_stratified_group_coverage(
    group_metadata: Mapping[str, tuple[str, str]],
    assignments: Mapping[str, str],
    ratios: SplitRatios,
) -> None:
    """Fail when an avoidable dataset/stratum coverage gap is present."""

    active = [name for name in _SPLITS if ratios.as_dict()[name] > 0]
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, metadata in group_metadata.items():
        if group not in assignments:
            raise DataContractError(f"split assignment missing group {group!r}")
        split = assignments[group]
        if split not in _SPLITS:
            raise DataContractError(f"invalid split {split!r} for group {group!r}")
        buckets[metadata].append(group)
    for metadata, groups in buckets.items():
        if len(groups) < len(active):
            continue
        present = {assignments[group] for group in groups}
        missing = set(active) - present
        if missing:
            raise DataContractError(
                f"avoidable split-coverage gap for dataset/stratum {metadata!r}: missing {sorted(missing)}"
            )


def assert_disjoint_groups(records: Iterable[Mapping[str, object]]) -> None:
    assignments: dict[str, str] = {}
    for record in records:
        group = str(record.get("split_group", ""))
        split = str(record.get("split", ""))
        if not group or split not in set(_SPLITS):
            raise DataContractError("every episode index record needs split_group and a valid split")
        previous = assignments.setdefault(group, split)
        if previous != split:
            raise DataContractError(
                f"data leakage: split group {group!r} occurs in both {previous!r} and {split!r}"
            )
