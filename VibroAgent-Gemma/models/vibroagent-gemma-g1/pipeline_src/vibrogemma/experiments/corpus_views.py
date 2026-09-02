from __future__ import annotations

"""Deterministic filtered episode views for corpus-composition experiments."""

import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def dataset_id(record: Mapping[str, Any]) -> str:
    candidates = (
        record.get("dataset_id"),
        _nested(record, "provenance", "dataset_id"),
        _nested(record, "metadata", "dataset_id"),
        _nested(record, "episode", "provenance", "dataset_id"),
    )
    for value in candidates:
        if value not in (None, ""):
            return str(value)
    raise KeyError("episode-index row has no dataset_id")


def split_name(record: Mapping[str, Any]) -> str:
    for value in (record.get("split"), _nested(record, "provenance", "split")):
        if value not in (None, ""):
            return str(value)
    for key in ("array_path", "metadata_path"):
        value = record.get(key)
        if value:
            parts = Path(str(value)).parts
            for split in ("train", "validation", "test"):
                if split in parts:
                    return split
    raise KeyError("episode-index row has no split")


def group_id(record: Mapping[str, Any]) -> str:
    candidates = (
        record.get("split_group"),
        record.get("group"),
        _nested(record, "provenance", "split_group"),
        _nested(record, "provenance", "session_id"),
        record.get("session_id"),
        record.get("episode_id"),
    )
    for value in candidates:
        if value not in (None, ""):
            return str(value)
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]


def training_role(record: Mapping[str, Any]) -> str:
    for value in (
        record.get("training_role"),
        _nested(record, "provenance", "training_role"),
        _nested(record, "metadata", "training_role"),
    ):
        if value not in (None, ""):
            return str(value)
    return "supervised"


def inspect_episode_index(index_path: str | Path) -> dict[str, Any]:
    index_path = Path(index_path)
    rows = _read_jsonl(index_path)
    by_dataset_split: Counter[tuple[str, str]] = Counter()
    by_dataset_role: Counter[tuple[str, str]] = Counter()
    groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        did = dataset_id(row)
        split = split_name(row)
        by_dataset_split[(did, split)] += 1
        by_dataset_role[(did, training_role(row))] += 1
        groups[(did, split)].add(group_id(row))
    datasets = sorted({key[0] for key in by_dataset_split})
    return {
        "index_path": str(index_path),
        "row_count": len(rows),
        "datasets": {
            did: {
                "splits": {
                    split: {
                        "episodes": by_dataset_split[(did, split)],
                        "groups": len(groups[(did, split)]),
                    }
                    for split in ("train", "validation", "test")
                },
                "roles": {
                    role: count
                    for (dataset, role), count in sorted(by_dataset_role.items())
                    if dataset == did
                },
            }
            for did in datasets
        },
    }


def _link_support_files(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for child in source_root.iterdir():
        if child.name == "index.jsonl":
            continue
        target = destination_root / child.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())


def _sample_with_replacement(rows: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: (group_id(row), str(row.get("episode_id", "")), json.dumps(row, sort_keys=True)))
    if count <= len(ordered):
        selected = rng.sample(ordered, count)
    else:
        selected = list(ordered)
        selected.extend(rng.choice(ordered) for _ in range(count - len(ordered)))
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        copied = dict(row)
        copied["_corpus_schedule_instance"] = index
        result.append(copied)
    return result


def build_episode_view(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    include_datasets: Sequence[str],
    fixed_train_records: int | None = None,
    fixed_validation_records_per_dataset: int | None = None,
    seed: int = 0,
    include_splits: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, Any]:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    source_index = source_root / "index.jsonl"
    if not source_index.exists():
        raise FileNotFoundError(source_index)
    include = set(map(str, include_datasets))
    allowed_splits = set(map(str, include_splits))
    rows = [
        row for row in _read_jsonl(source_index)
        if dataset_id(row) in include and split_name(row) in allowed_splits
    ]
    if not rows:
        raise RuntimeError(f"No episode rows remain for datasets {sorted(include)}")

    by_split_dataset: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split_dataset[(split_name(row), dataset_id(row))].append(row)

    output: list[dict[str, Any]] = []
    train_quota: dict[str, int] = {}
    if fixed_train_records is not None:
        datasets_with_train = [did for did in sorted(include) if by_split_dataset[("train", did)]]
        if not datasets_with_train:
            raise RuntimeError("No train rows are available")
        base = fixed_train_records // len(datasets_with_train)
        remainder = fixed_train_records % len(datasets_with_train)
        for index, did in enumerate(datasets_with_train):
            train_quota[did] = base + (1 if index < remainder else 0)
            output.extend(_sample_with_replacement(
                by_split_dataset[("train", did)], train_quota[did], seed + 1009 * (index + 1)
            ))
    else:
        for did in sorted(include):
            output.extend(by_split_dataset[("train", did)])

    for split in ("validation", "test"):
        for index, did in enumerate(sorted(include)):
            candidates = by_split_dataset[(split, did)]
            if split == "validation" and fixed_validation_records_per_dataset is not None:
                candidates = _sample_with_replacement(
                    candidates,
                    min(fixed_validation_records_per_dataset, len(candidates)),
                    seed + 7919 + 101 * index,
                )
            output.extend(candidates)

    if destination_root.exists():
        shutil.rmtree(destination_root)
    _link_support_files(source_root, destination_root)
    _write_jsonl(destination_root / "index.jsonl", output)
    report = inspect_episode_index(destination_root / "index.jsonl")
    report.update({
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "include_datasets": sorted(include),
        "fixed_train_records": fixed_train_records,
        "train_quota": train_quota,
        "seed": int(seed),
    })
    (destination_root / "corpus_view_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
