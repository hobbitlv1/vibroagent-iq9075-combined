from __future__ import annotations

import hashlib
import inspect
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from ..config import resolve_from_config
from ..contracts import SixBoardEpisode
from ..data.episodes import EpisodeDataset
from ..signal.multiresolution import (
    MultiResolutionEpisodePreprocessor,
    PreparedEpisodeBatch,
    stack_prepared_episode_batch,
)
from ..signal.prepared_cache import PreparedEpisodeDiskCache

class _GroupedSamplerBase(Sampler[int]):
    """Shared deterministic hierarchy for group-aware supervised samplers."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
        groups_by_dataset: dict[str, dict[str, list[int]]] = {}
        group_states: dict[tuple[str, str], str] = {}
        for index, record in enumerate(records):
            if not bool(record.get("has_label", True)):
                continue
            dataset_id = str(record.get("dataset_id", ""))
            stratum = str(record.get("split_stratum", ""))
            group = str(record.get("sampling_group") or record.get("split_group", ""))
            if not dataset_id or not stratum or not group:
                raise ValueError(
                    "group-aware sampling requires dataset_id, split_stratum and split_group"
                )
            if stratum == "representation_only":
                continue
            state = "normal" if stratum == "healthy" else stratum.removeprefix("changed::")
            previous_state = group_states.setdefault((dataset_id, group), state)
            if previous_state != state:
                raise ValueError(
                    f"group {dataset_id}|{group} spans multiple supervised states"
                )
            hierarchy.setdefault(dataset_id, {}).setdefault(state, {}).setdefault(
                group, []
            ).append(index)
            groups_by_dataset.setdefault(dataset_id, {}).setdefault(group, []).append(index)
        if not hierarchy:
            raise ValueError("group-aware sampler found no supervised records")
        self.hierarchy = hierarchy
        self.groups_by_dataset = groups_by_dataset
        self.datasets = tuple(sorted(hierarchy))
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _draw_state_balanced(self, rng: random.Random, dataset_id: str) -> int:
        states = tuple(sorted(self.hierarchy[dataset_id]))
        state = rng.choice(states)
        groups = tuple(sorted(self.hierarchy[dataset_id][state]))
        group = rng.choice(groups)
        return rng.choice(self.hierarchy[dataset_id][state][group])

    def _draw_group_uniform(self, rng: random.Random, dataset_id: str) -> int:
        groups = tuple(sorted(self.groups_by_dataset[dataset_id]))
        group = rng.choice(groups)
        return rng.choice(self.groups_by_dataset[dataset_id][group])

    def _dataset_report(self) -> dict[str, Any]:
        return {
            dataset_id: {
                state: {
                    "groups": len(groups),
                    "windows": sum(len(indices) for indices in groups.values()),
                }
                for state, groups in sorted(states.items())
            }
            for dataset_id, states in sorted(self.hierarchy.items())
        }


class FactorizedTaskBalancedSampler(Sampler[int]):
    """Balance global and independent-target language tasks without pseudo-labels.

    The sampler chooses global versus target task mass explicitly, then chooses
    dataset, state, complete group and window uniformly. Only QUGS contributes
    target tasks in the final corpus, so UPC/LUMO can never be sampled as target
    negatives or positives.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
        target_task_probability: float = 0.5,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        probability = float(target_task_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("target_task_probability must be in [0, 1]")
        hierarchy: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {}
        for index, record in enumerate(records):
            family = str(record.get("_task_family") or "global")
            if family not in {"global", "target"}:
                continue
            dataset_id = str(record.get("dataset_id") or "")
            state = str(record.get("_task_state") or "")
            group = str(
                record.get("_task_group")
                or record.get("sampling_group")
                or record.get("split_group")
                or ""
            )
            if not dataset_id or not state or not group:
                raise ValueError(
                    "factorized task sampling requires dataset_id, task state and complete group"
                )
            hierarchy.setdefault(family, {}).setdefault(dataset_id, {}).setdefault(
                state, {}
            ).setdefault(group, []).append(index)
        if "global" not in hierarchy:
            raise ValueError("factorized task sampler found no global tasks")
        self.hierarchy = hierarchy
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.target_task_probability = probability if "target" in hierarchy else 0.0
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            family = (
                "target"
                if "target" in self.hierarchy
                and rng.random() < self.target_task_probability
                else "global"
            )
            datasets = tuple(sorted(self.hierarchy[family]))
            dataset_id = rng.choice(datasets)
            states = tuple(sorted(self.hierarchy[family][dataset_id]))
            state = rng.choice(states)
            groups = tuple(sorted(self.hierarchy[family][dataset_id][state]))
            group = rng.choice(groups)
            yield rng.choice(self.hierarchy[family][dataset_id][state][group])

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "factorized_task_dataset_state_group_v1",
            "target_task_probability": self.target_task_probability,
            "num_samples": self.num_samples,
            "support": {
                family: {
                    dataset_id: {
                        state: {
                            "groups": len(groups),
                            "windows": sum(len(indices) for indices in groups.values()),
                        }
                        for state, groups in sorted(states.items())
                    }
                    for dataset_id, states in sorted(datasets.items())
                }
                for family, datasets in sorted(self.hierarchy.items())
            },
        }


class FactorizedTargetIndexBalancedSampler(Sampler[int]):
    """Balance independent target tasks by logical target index and class state.

    Sampling order is task family -> target index -> state -> dataset -> complete
    group -> window.  This prevents the target identifier in the prompt from
    becoming a label prior when one dataset supplies positive support only for a
    subset of targets. Global tasks retain dataset/state/group balancing.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
        target_task_probability: float = 0.7,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        probability = float(target_task_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("target_task_probability must be in [0, 1]")
        global_hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
        target_hierarchy: dict[int, dict[str, dict[str, dict[str, list[int]]]]] = {}
        for index, record in enumerate(records):
            family = str(record.get("_task_family") or "global")
            dataset_id = str(record.get("dataset_id") or "")
            state = str(record.get("_task_state") or "")
            group = str(
                record.get("_task_group")
                or record.get("sampling_group")
                or record.get("split_group")
                or ""
            )
            if not dataset_id or not state or not group:
                raise ValueError(
                    "target-index sampling requires dataset_id, task state and complete group"
                )
            if family == "global":
                global_hierarchy.setdefault(dataset_id, {}).setdefault(state, {}).setdefault(
                    group, []
                ).append(index)
            elif family == "target":
                target_index = int(record.get("_task_target_index") or 0)
                if target_index not in {1, 2, 3, 4, 5}:
                    raise ValueError(
                        "target-index sampling requires _task_target_index=1..5"
                    )
                target_hierarchy.setdefault(target_index, {}).setdefault(state, {}).setdefault(
                    dataset_id, {}
                ).setdefault(group, []).append(index)
        if not global_hierarchy:
            raise ValueError("target-index sampler found no global tasks")
        if target_hierarchy and set(target_hierarchy) != {1, 2, 3, 4, 5}:
            raise ValueError(
                "target-index sampler requires target-task support for all five logical targets"
            )
        self.global_hierarchy = global_hierarchy
        self.target_hierarchy = target_hierarchy
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.target_task_probability = probability if target_hierarchy else 0.0
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            use_target = bool(
                self.target_hierarchy and rng.random() < self.target_task_probability
            )
            if not use_target:
                dataset_id = rng.choice(tuple(sorted(self.global_hierarchy)))
                state = rng.choice(tuple(sorted(self.global_hierarchy[dataset_id])))
                group = rng.choice(
                    tuple(sorted(self.global_hierarchy[dataset_id][state]))
                )
                yield rng.choice(self.global_hierarchy[dataset_id][state][group])
                continue
            target_index = rng.choice((1, 2, 3, 4, 5))
            states = tuple(sorted(self.target_hierarchy[target_index]))
            state = rng.choice(states)
            datasets = tuple(sorted(self.target_hierarchy[target_index][state]))
            dataset_id = rng.choice(datasets)
            groups = tuple(
                sorted(self.target_hierarchy[target_index][state][dataset_id])
            )
            group = rng.choice(groups)
            yield rng.choice(
                self.target_hierarchy[target_index][state][dataset_id][group]
            )

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "factorized_target_index_state_dataset_group_v2",
            "target_task_probability": self.target_task_probability,
            "num_samples": self.num_samples,
            "global_support": {
                dataset_id: {
                    state: {
                        "groups": len(groups),
                        "windows": sum(len(indices) for indices in groups.values()),
                    }
                    for state, groups in sorted(states.items())
                }
                for dataset_id, states in sorted(self.global_hierarchy.items())
            },
            "target_support": {
                f"target_{target_index}": {
                    state: {
                        dataset_id: {
                            "groups": len(groups),
                            "windows": sum(len(indices) for indices in groups.values()),
                        }
                        for dataset_id, groups in sorted(datasets.items())
                    }
                    for state, datasets in sorted(states.items())
                }
                for target_index, states in sorted(self.target_hierarchy.items())
            },
        }


class FactorizedSourceKindTargetIndexBalancedSampler(Sampler[int]):
    """Enforce an exact measured/synthetic draw mix while preserving task balance.

    The schedule is constructed deterministically for every epoch.  It first
    allocates the requested number of measured and OpenSees-derived draws, then
    balances global versus target tasks, logical target index and class state
    inside each source kind.  Dataset, complete group and window are sampled only
    after those factors have been fixed.

    This makes ``synthetic_probability=0.30`` an actual training exposure
    contract rather than an approximation based on manifest size.
    """

    SYNTHETIC_SOURCE_KINDS = frozenset(
        {
            "opensees_measured_conditioned_v017",
            "opensees_measured_calibrated_v018",
            "opensees_lumo_full_inverse_fe_v019",
        }
    )

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
        target_task_probability: float = 0.7,
        synthetic_probability: float = 0.30,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        target_probability = float(target_task_probability)
        synthetic_probability = float(synthetic_probability)
        if not 0.0 <= target_probability <= 1.0:
            raise ValueError("target_task_probability must be in [0, 1]")
        if not 0.0 <= synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in [0, 1]")

        # source -> state -> dataset -> group -> indices
        global_hierarchy: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {}
        # source -> target -> state -> dataset -> group -> indices
        target_hierarchy: dict[
            str, dict[int, dict[str, dict[str, dict[str, list[int]]]]]
        ] = {}

        for index, record in enumerate(records):
            family = str(record.get("_task_family") or "global")
            dataset_id = str(record.get("dataset_id") or "")
            state = str(record.get("_task_state") or "")
            group = str(
                record.get("_task_group")
                or record.get("sampling_group")
                or record.get("split_group")
                or ""
            )
            if not dataset_id or not state or not group:
                raise ValueError(
                    "source-kind target sampling requires dataset_id, task state and complete group"
                )
            source_kind = str(record.get("source_kind") or "")
            synthetic = bool(record.get("_opensees_synthetic_row")) or (
                source_kind in self.SYNTHETIC_SOURCE_KINDS
            )
            source = "synthetic" if synthetic else "measured"

            if family == "global":
                global_hierarchy.setdefault(source, {}).setdefault(state, {}).setdefault(
                    dataset_id, {}
                ).setdefault(group, []).append(index)
            elif family == "target":
                target_index = int(record.get("_task_target_index") or 0)
                if target_index not in {1, 2, 3, 4, 5}:
                    raise ValueError(
                        "source-kind target sampling requires _task_target_index=1..5"
                    )
                target_hierarchy.setdefault(source, {}).setdefault(
                    target_index, {}
                ).setdefault(state, {}).setdefault(dataset_id, {}).setdefault(
                    group, []
                ).append(index)

        if "measured" not in global_hierarchy:
            raise ValueError("source-kind sampler found no measured global tasks")
        requested_sources = {"measured"}
        if synthetic_probability > 0.0:
            requested_sources.add("synthetic")
        missing_global = requested_sources.difference(global_hierarchy)
        if missing_global:
            raise ValueError(f"missing global support for source kinds: {sorted(missing_global)}")
        for source in requested_sources:
            if target_probability > 0.0:
                observed = set(target_hierarchy.get(source, {}))
                if observed != {1, 2, 3, 4, 5}:
                    raise ValueError(
                        f"source kind {source} requires target support 1..5, observed={sorted(observed)}"
                    )

        self.global_hierarchy = global_hierarchy
        self.target_hierarchy = target_hierarchy
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.target_task_probability = target_probability
        self.synthetic_probability = synthetic_probability
        self.epoch = 0
        self._last_schedule_report: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _balanced_values(
        values: Sequence[Any], count: int, rng: random.Random
    ) -> list[Any]:
        values = tuple(values)
        if count < 0 or not values and count:
            raise ValueError("invalid balanced allocation")
        if count == 0:
            return []
        base, remainder = divmod(count, len(values))
        output = [value for value in values for _ in range(base)]
        extras = list(values)
        rng.shuffle(extras)
        output.extend(extras[:remainder])
        rng.shuffle(output)
        return output

    def _source_counts(self) -> dict[str, int]:
        synthetic = int(round(self.num_samples * self.synthetic_probability))
        synthetic = min(max(synthetic, 0), self.num_samples)
        return {"measured": self.num_samples - synthetic, "synthetic": synthetic}

    def _build_schedule(self, rng: random.Random) -> list[tuple[str, str, int | None, str]]:
        schedule: list[tuple[str, str, int | None, str]] = []
        report: dict[str, Any] = {"source_draw_counts": {}, "family_draw_counts": {}, "target_draw_counts": {}, "state_draw_counts": {}}
        for source, source_count in self._source_counts().items():
            if source_count == 0:
                continue
            target_count = int(round(source_count * self.target_task_probability))
            target_count = min(max(target_count, 0), source_count)
            global_count = source_count - target_count
            report["source_draw_counts"][source] = source_count
            report["family_draw_counts"][source] = {"global": global_count, "target": target_count}
            report["target_draw_counts"][source] = {}
            report["state_draw_counts"][source] = {"global": {}, "target": {}}

            global_states = tuple(sorted(self.global_hierarchy[source]))
            for state in self._balanced_values(global_states, global_count, rng):
                schedule.append((source, "global", None, str(state)))
                counts = report["state_draw_counts"][source]["global"]
                counts[str(state)] = counts.get(str(state), 0) + 1

            target_indices = self._balanced_values((1, 2, 3, 4, 5), target_count, rng)
            by_target: dict[int, int] = {}
            for target_index in target_indices:
                by_target[int(target_index)] = by_target.get(int(target_index), 0) + 1
            target_state_totals: dict[str, int] = {}
            for target_index, count in sorted(by_target.items()):
                report["target_draw_counts"][source][f"target_{target_index}"] = count
                states = tuple(sorted(self.target_hierarchy[source][target_index]))
                state_counts: dict[str, int] = {}
                base, remainder = divmod(count, len(states))
                target_states: list[str] = []
                for state in states:
                    target_states.extend([str(state)] * base)
                    target_state_totals[str(state)] = target_state_totals.get(str(state), 0) + base
                extras = list(states)
                rng.shuffle(extras)
                extras.sort(key=lambda state: target_state_totals.get(str(state), 0))
                for state in extras[:remainder]:
                    target_states.append(str(state))
                    target_state_totals[str(state)] = target_state_totals.get(str(state), 0) + 1
                rng.shuffle(target_states)
                for state in target_states:
                    schedule.append((source, "target", target_index, str(state)))
                    state_counts[str(state)] = state_counts.get(str(state), 0) + 1
                report["state_draw_counts"][source]["target"][f"target_{target_index}"] = state_counts

        if len(schedule) != self.num_samples:
            raise RuntimeError(f"source-kind schedule length {len(schedule)} != {self.num_samples}")
        rng.shuffle(schedule)
        self._last_schedule_report = report
        return schedule

    def __iter__(self):
        rng = self._rng()
        for source, family, target_index, state in self._build_schedule(rng):
            if family == "global":
                datasets = tuple(sorted(self.global_hierarchy[source][state]))
                dataset_id = rng.choice(datasets)
                groups = tuple(sorted(self.global_hierarchy[source][state][dataset_id]))
                group = rng.choice(groups)
                yield rng.choice(self.global_hierarchy[source][state][dataset_id][group])
                continue
            assert target_index is not None
            datasets = tuple(sorted(self.target_hierarchy[source][target_index][state]))
            dataset_id = rng.choice(datasets)
            groups = tuple(sorted(self.target_hierarchy[source][target_index][state][dataset_id]))
            group = rng.choice(groups)
            yield rng.choice(
                self.target_hierarchy[source][target_index][state][dataset_id][group]
            )

    @staticmethod
    def _support_report(hierarchy: Mapping[str, Any]) -> dict[str, Any]:
        # Reports are built explicitly below because global and target nesting differ.
        return dict(hierarchy)

    def report(self) -> dict[str, Any]:
        if self._last_schedule_report is None:
            self._build_schedule(self._rng())
        source_counts = self._source_counts()
        return {
            "strategy": "factorized_source_kind_target_index_state_dataset_group_v1",
            "target_task_probability": self.target_task_probability,
            "synthetic_probability": self.synthetic_probability,
            "num_samples": self.num_samples,
            "planned_source_draw_counts": source_counts,
            "last_schedule": self._last_schedule_report,
            "global_support": {
                source: {
                    state: {
                        dataset_id: {
                            "groups": len(groups),
                            "windows": sum(len(indices) for indices in groups.values()),
                        }
                        for dataset_id, groups in sorted(datasets.items())
                    }
                    for state, datasets in sorted(states.items())
                }
                for source, states in sorted(self.global_hierarchy.items())
            },
            "target_support": {
                source: {
                    f"target_{target_index}": {
                        state: {
                            dataset_id: {
                                "groups": len(groups),
                                "windows": sum(len(indices) for indices in groups.values()),
                            }
                            for dataset_id, groups in sorted(datasets.items())
                        }
                        for state, datasets in sorted(states.items())
                    }
                    for target_index, states in sorted(targets.items())
                }
                for source, targets in sorted(self.target_hierarchy.items())
            },
        }


class StructuredSetBalancedSampler(Sampler[int]):
    """Balance complete events without breaking their five-target label set.

    Sampling is source kind -> global state -> complete target-set category ->
    dataset -> independent group -> window. Global normal/anomaly states receive
    equal scheduled mass whenever both are available. Within the anomaly state,
    each single-target identity and each genuine multi-target set is a separate
    category. This prevents independent 50/50 target resampling from destroying
    same-event hard negatives.
    """

    SYNTHETIC_SOURCE_KINDS = FactorizedSourceKindTargetIndexBalancedSampler.SYNTHETIC_SOURCE_KINDS

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
        synthetic_probability: float = 0.10,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        synthetic_probability = float(synthetic_probability)
        if not 0.0 <= synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in [0, 1]")

        # source -> global state -> set category -> dataset -> group -> indices
        hierarchy: dict[
            str,
            dict[str, dict[str, dict[str, dict[str, list[int]]]]],
        ] = {}
        for index, record in enumerate(records):
            if not bool(record.get("has_label", True)):
                continue
            dataset_id = str(record.get("dataset_id") or "")
            group = str(
                record.get("sampling_group")
                or record.get("split_group")
                or ""
            )
            state = str(record.get("global_class_name") or "")
            if state not in {"normal", "unknown_anomaly"}:
                stratum = str(record.get("split_stratum") or "")
                state = "normal" if stratum == "healthy" else "unknown_anomaly"
            if not dataset_id or not group:
                raise ValueError(
                    "structured-set sampling requires dataset_id and complete group"
                )

            source_kind = str(record.get("source_kind") or "")
            synthetic = bool(record.get("_opensees_synthetic_row")) or (
                source_kind in self.SYNTHETIC_SOURCE_KINDS
            )
            source = "synthetic" if synthetic else "measured"
            if bool(record.get("target_supervision_available", False)):
                affected = tuple(
                    sorted(
                        int(value)
                        for value in record.get("affected_target_indices", [])
                    )
                )
                if any(value not in {1, 2, 3, 4, 5} for value in affected):
                    raise ValueError(
                        f"invalid affected target set for structured sampling: {affected}"
                    )
                if not affected:
                    category = "localized::none"
                elif len(affected) == 1:
                    category = f"localized::single::target_{affected[0]}"
                else:
                    category = "localized::multi::" + "+".join(
                        f"target_{value}" for value in affected
                    )
            else:
                category = "global_only"
            hierarchy.setdefault(source, {}).setdefault(state, {}).setdefault(
                category, {}
            ).setdefault(dataset_id, {}).setdefault(group, []).append(index)

        if "measured" not in hierarchy:
            raise ValueError("structured-set sampler found no measured labelled events")
        requested_sources = {"measured"}
        if synthetic_probability > 0.0:
            requested_sources.add("synthetic")
        missing = requested_sources.difference(hierarchy)
        if missing:
            raise ValueError(
                f"structured-set sampler lacks requested source support: {sorted(missing)}"
            )
        for source in requested_sources:
            if set(hierarchy[source]) != {"normal", "unknown_anomaly"}:
                raise ValueError(
                    f"source {source} must support both global states for symmetric training; "
                    f"observed={sorted(hierarchy[source])}"
                )

        self.hierarchy = hierarchy
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.synthetic_probability = synthetic_probability
        self.epoch = 0
        self._last_schedule_report: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _balanced_values(
        values: Sequence[Any], count: int, rng: random.Random
    ) -> list[Any]:
        values = tuple(values)
        if count < 0 or (count and not values):
            raise ValueError("invalid balanced allocation")
        if count == 0:
            return []
        base, remainder = divmod(count, len(values))
        output = [value for value in values for _ in range(base)]
        extras = list(values)
        rng.shuffle(extras)
        output.extend(extras[:remainder])
        rng.shuffle(output)
        return output

    def _source_counts(self) -> dict[str, int]:
        synthetic = int(round(self.num_samples * self.synthetic_probability))
        synthetic = min(max(synthetic, 0), self.num_samples)
        return {"measured": self.num_samples - synthetic, "synthetic": synthetic}

    def _build_schedule(self, rng: random.Random) -> list[tuple[str, str, str]]:
        schedule: list[tuple[str, str, str]] = []
        report: dict[str, Any] = {
            "source_draw_counts": {},
            "global_state_draw_counts": {},
            "set_category_draw_counts": {},
        }
        for source, source_count in self._source_counts().items():
            if source_count == 0:
                continue
            report["source_draw_counts"][source] = source_count
            report["global_state_draw_counts"][source] = {}
            report["set_category_draw_counts"][source] = {}
            states = self._balanced_values(
                tuple(sorted(self.hierarchy[source])), source_count, rng
            )
            state_counts: dict[str, int] = {}
            for state in states:
                state_counts[str(state)] = state_counts.get(str(state), 0) + 1
            for state, count in sorted(state_counts.items()):
                report["global_state_draw_counts"][source][state] = count
                categories = self._balanced_values(
                    tuple(sorted(self.hierarchy[source][state])), count, rng
                )
                category_counts: dict[str, int] = {}
                for category in categories:
                    schedule.append((source, state, str(category)))
                    category_counts[str(category)] = (
                        category_counts.get(str(category), 0) + 1
                    )
                report["set_category_draw_counts"][source][state] = category_counts
        if len(schedule) != self.num_samples:
            raise RuntimeError(
                f"structured-set schedule length {len(schedule)} != {self.num_samples}"
            )
        rng.shuffle(schedule)
        self._last_schedule_report = report
        return schedule

    def __iter__(self):
        rng = self._rng()
        for source, state, category in self._build_schedule(rng):
            datasets = tuple(sorted(self.hierarchy[source][state][category]))
            dataset_id = rng.choice(datasets)
            groups = tuple(
                sorted(self.hierarchy[source][state][category][dataset_id])
            )
            group = rng.choice(groups)
            yield rng.choice(
                self.hierarchy[source][state][category][dataset_id][group]
            )

    def report(self) -> dict[str, Any]:
        if self._last_schedule_report is None:
            self._build_schedule(self._rng())
        return {
            "strategy": "structured_set_global_state_category_dataset_group_v1",
            "synthetic_probability": self.synthetic_probability,
            "num_samples": self.num_samples,
            "planned_source_draw_counts": self._source_counts(),
            "last_schedule": self._last_schedule_report,
            "support": {
                source: {
                    state: {
                        category: {
                            dataset_id: {
                                "groups": len(groups),
                                "windows": sum(
                                    len(indices) for indices in groups.values()
                                ),
                            }
                            for dataset_id, groups in sorted(datasets.items())
                        }
                        for category, datasets in sorted(categories.items())
                    }
                    for state, categories in sorted(states.items())
                }
                for source, states in sorted(self.hierarchy.items())
            },
        }


class HierarchicalBalancedSampler(_GroupedSamplerBase):
    """Dataset -> state -> complete-group -> window balanced sampling.

    This is the v0.8 recipe. It remains available for reproducibility but is no
    longer the default in v0.9 because the measured run over-corrected the class
    prior and regressed several datasets.
    """

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            dataset_id = rng.choice(self.datasets)
            yield self._draw_state_balanced(rng, dataset_id)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "dataset_state_group_window_uniform_v1",
            "num_samples": self.num_samples,
            "datasets": self._dataset_report(),
        }


class DatasetGroupUniformSampler(_GroupedSamplerBase):
    """Choose dataset, complete session group and window uniformly.

    This removes the dominance of long overlapping sessions while preserving the
    natural state prevalence at the independent-group level. It is intentionally
    milder than the v0.8 state-balanced sampler.
    """

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            dataset_id = rng.choice(self.datasets)
            yield self._draw_group_uniform(rng, dataset_id)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "dataset_group_window_uniform_v1",
            "num_samples": self.num_samples,
            "datasets": self._dataset_report(),
        }


class HybridGroupBalancedSampler(_GroupedSamplerBase):
    """Mild mixture of group-uniform and state-balanced supervised draws."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
        state_balance_probability: float = 0.5,
    ) -> None:
        super().__init__(records, num_samples=num_samples, seed=seed)
        probability = float(state_balance_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("state_balance_probability must be in [0, 1]")
        self.state_balance_probability = probability

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            dataset_id = rng.choice(self.datasets)
            if rng.random() < self.state_balance_probability:
                yield self._draw_state_balanced(rng, dataset_id)
            else:
                yield self._draw_group_uniform(rng, dataset_id)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "hybrid_dataset_group_state_v1",
            "state_balance_probability": self.state_balance_probability,
            "num_samples": self.num_samples,
            "datasets": self._dataset_report(),
        }


class RepresentationDatasetCompleteGroupSampler(Sampler[int]):
    """Choose dataset, complete group, then window for encoder pretraining.

    Unlike the supervised samplers, this hierarchy deliberately retains records
    without labels and the ``representation_only`` stratum. Dataset identity is
    used only by the sampler and is never added to the model input tensors.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        hierarchy: dict[str, dict[str, list[int]]] = {}
        for index, record in enumerate(records):
            dataset_id = str(record.get("dataset_id") or "").strip()
            group = str(
                record.get("sampling_group")
                or record.get("split_group")
                or ""
            ).strip()
            if not dataset_id or not group:
                raise ValueError(
                    "representation sampling requires dataset_id and a complete "
                    "sampling_group or split_group"
                )
            hierarchy.setdefault(dataset_id, {}).setdefault(group, []).append(index)
        if not hierarchy:
            raise ValueError("representation sampler found no records")
        self.hierarchy = hierarchy
        self._has_label = {
            index: bool(record.get("has_label", True))
            for index, record in enumerate(records)
        }
        self.datasets = tuple(sorted(hierarchy))
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{self.epoch}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def __iter__(self):
        rng = self._rng()
        for _ in range(self.num_samples):
            dataset_id = rng.choice(self.datasets)
            groups = tuple(sorted(self.hierarchy[dataset_id]))
            group = rng.choice(groups)
            yield rng.choice(self.hierarchy[dataset_id][group])

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "dataset_complete_group_window_uniform_v1",
            "num_samples": self.num_samples,
            "includes_unlabelled_records": True,
            "datasets": {
                dataset_id: {
                    "groups": len(groups),
                    "windows": sum(len(indices) for indices in groups.values()),
                    "labelled_windows": sum(
                        bool(self._record_has_label(index))
                        for indices in groups.values()
                        for index in indices
                    ),
                    "unlabelled_windows": sum(
                        not self._record_has_label(index)
                        for indices in groups.values()
                        for index in indices
                    ),
                }
                for dataset_id, groups in sorted(self.hierarchy.items())
            },
        }

    def _record_has_label(self, index: int) -> bool:
        return self._has_label[index]


def set_loader_epoch(loader: Any, epoch: int) -> None:
    """Forward an epoch number through Accelerate/DataLoader wrappers."""

    if hasattr(loader, "set_epoch"):
        loader.set_epoch(int(epoch))
        return
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def collate_episodes(episodes: Sequence[SixBoardEpisode]) -> list[SixBoardEpisode]:
    """Compatibility collator for callers that intentionally need raw episodes."""

    return list(episodes)


def episode_index_path(config: dict[str, Any]) -> Path:
    return resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"


def pretrain_corpus_options(config: dict[str, Any]) -> dict[str, set[str] | None]:
    """Resolve opt-in encoder corpus filters without changing legacy defaults."""

    pretrain = dict(config.get("training", {}).get("pretrain") or {})
    corpus = dict(pretrain.get("corpus") or {})

    def optional_set(key: str) -> set[str] | None:
        value = corpus.get(key)
        if value is None:
            return None
        return {str(item) for item in value}

    return {
        "include_dataset_ids": optional_set("include_dataset_ids"),
        "include_training_roles": optional_set("include_training_roles"),
        "train_only_dataset_ids": optional_set("train_only_dataset_ids"),
    }


def stage_corpus_options(
    config: dict[str, Any], stage: str | None
) -> dict[str, set[str] | None]:
    """Return loader-side corpus filters for pretrain, alignment, and LoRA."""

    if stage == "pretrain":
        return pretrain_corpus_options(config)
    if stage in {"align", "lora"}:
        stage_cfg = dict(config.get("training", {}).get(stage) or {})
        corpus = dict(stage_cfg.get("corpus") or {})
        include = corpus.get("include_dataset_ids")
        return {
            "include_dataset_ids": (
                None if include is None else {str(item) for item in include}
            ),
            "include_training_roles": None,
            "train_only_dataset_ids": None,
        }
    return {
        "include_dataset_ids": None,
        "include_training_roles": None,
        "train_only_dataset_ids": None,
    }


def stage_training_settings(config: dict[str, Any], stage: str | None) -> dict[str, Any]:
    """Resolve stage-specific loader/accumulation settings with global fallbacks."""

    training = config["training"]
    stage_config = training.get(stage, {}) if stage else {}
    num_workers = int(stage_config.get("num_workers", training.get("num_workers", 0)))
    prefetch_factor = int(stage_config.get("prefetch_factor", training.get("prefetch_factor", 1)))
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1")
    worker_thread_limit = int(
        stage_config.get("worker_thread_limit", training.get("worker_thread_limit", 1))
    )
    if worker_thread_limit < 1:
        raise ValueError("worker_thread_limit must be >= 1")
    validation_num_workers = int(
        stage_config.get(
            "validation_num_workers",
            training.get("validation_num_workers", min(num_workers, 4)),
        )
    )
    return {
        "batch_size": int(stage_config.get("batch_size", training.get("batch_size", 1))),
        "validation_batch_size": int(
            stage_config.get(
                "validation_batch_size",
                stage_config.get("batch_size", training.get("batch_size", 1)),
            )
        ),
        "gradient_accumulation_steps": int(
            stage_config.get(
                "gradient_accumulation_steps",
                training.get("gradient_accumulation_steps", 1),
            )
        ),
        "num_workers": num_workers,
        "validation_num_workers": validation_num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": bool(
            stage_config.get(
                "persistent_workers",
                training.get("persistent_workers", num_workers > 0),
            )
        ),
        "pin_memory": bool(stage_config.get("pin_memory", training.get("pin_memory", True))),
        "worker_thread_limit": worker_thread_limit,
        "in_order": bool(stage_config.get("in_order", training.get("in_order", True))),
    }


def _configure_worker(worker_id: int) -> None:
    """Keep every SciPy/BLAS worker single-threaded to avoid oversubscription."""

    del worker_id
    limit = max(1, int(os.environ.get("VIBROGEMMA_WORKER_THREAD_LIMIT", "1")))
    torch.set_num_threads(limit)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before inter-op work begins.
        pass
    try:
        from threadpoolctl import threadpool_limits

        # The context manager form is temporary; calling the controller and
        # retaining it in the worker process applies the limit for its lifetime.
        controller = threadpool_limits(limits=limit)
        globals()["_VIBROGEMMA_THREADPOOL_CONTROLLER"] = controller
    except (ImportError, RuntimeError):
        pass

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class MultiResolutionBatchCollator:
    """Run expensive multi-resolution preparation inside DataLoader workers."""

    config: dict[str, Any]
    retain_prepared: bool = False
    split: str | Sequence[str] | None = None
    prepared_cache_mode: str = "read_only"
    _preprocessor: MultiResolutionEpisodePreprocessor | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_cache: PreparedEpisodeDiskCache | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __getstate__(self) -> dict[str, Any]:
        # Spawn-based DataLoaders should construct SciPy/preprocessor state in
        # the worker, not pickle it from the parent process.
        state = dict(self.__dict__)
        state["_preprocessor"] = None
        state["_prepared_cache"] = None
        return state

    @property
    def preprocessor(self) -> MultiResolutionEpisodePreprocessor:
        if self._preprocessor is None:
            self._preprocessor = MultiResolutionEpisodePreprocessor.from_config(self.config)
        return self._preprocessor

    @property
    def prepared_cache(self) -> PreparedEpisodeDiskCache | None:
        if self.prepared_cache_mode == "disabled":
            return None
        requested = (
            (self.split,)
            if isinstance(self.split, str)
            else tuple(self.split or ())
        )
        # TEST and split='all' never read or write prepared cache entries.
        if not requested or not set(requested).issubset({"train", "validation"}):
            return None
        root_value = self.config.get("data", {}).get("prepared_cache_root")
        if not root_value:
            return None
        if self._prepared_cache is None:
            self._prepared_cache = PreparedEpisodeDiskCache(
                resolve_from_config(self.config, root_value), self.preprocessor
            )
        return self._prepared_cache

    def __call__(self, episodes: Sequence[SixBoardEpisode]) -> PreparedEpisodeBatch:
        cache = self.prepared_cache
        if self.prepared_cache_mode not in {"disabled", "read_only", "read_write"}:
            raise ValueError(
                "prepared_cache_mode must be 'disabled', 'read_only' or 'read_write'"
            )
        prepared = (
            self.preprocessor.prepare_many(episodes)
            if cache is None
            else [
                cache.prepare(
                    episode, read_only=self.prepared_cache_mode == "read_only"
                )
                for episode in episodes
            ]
        )
        return stack_prepared_episode_batch(
            prepared, device=None, retain_prepared=self.retain_prepared
        )


def build_episode_loader(
    config: dict[str, Any],
    split: str | Sequence[str],
    *,
    shuffle: bool,
    batch_size: int | None = None,
    require_labels: bool = True,
    stage: str | None = None,
    prepare_multiresolution: bool = True,
    retain_prepared: bool = False,
    episode_ids: set[str] | None = None,
    prepared_cache_mode: str = "read_only",
    apply_training_augmentation: bool = False,
) -> DataLoader:
    """Build a loader whose workers can perform the full signal preprocessing.

    With ``prepare_multiresolution=True`` (the training default), each worker
    loads a serialized episode, performs resampling/STFT/physics extraction,
    stacks a CPU tensor batch, and sends that batch to the training process.
    This removes the serial NumPy/SciPy loop from ``model.forward``.
    """

    stage_cfg = dict(config.get("training", {}).get(stage, {})) if stage else {}
    partial_label_cfg = dict(stage_cfg.get("partial_label_tasks") or {})
    structured_set_cfg = dict(
        config.get("training", {}).get("structured_set_training") or {}
    )
    structured_set_enabled = bool(
        require_labels
        and stage in {"align", "lora"}
        and structured_set_cfg.get("enabled", False)
    )
    expand_supervision_tasks = bool(
        require_labels
        and stage in {"align", "lora"}
        and partial_label_cfg.get("enabled", False)
        and not structured_set_enabled
    )
    permutation_cfg = dict(
        config.get("training", {}).get("target_permutation_augmentation") or {}
    )
    permutation_enabled = bool(
        (shuffle or apply_training_augmentation)
        and stage in {"align", "lora"}
        and (expand_supervision_tasks or structured_set_enabled)
        and permutation_cfg.get("enabled", False)
    )
    corpus_options = stage_corpus_options(config, stage)
    dataset = EpisodeDataset(
        episode_index_path(config),
        split,
        require_labels=require_labels,
        episode_ids=episode_ids,
        expand_supervision_tasks=expand_supervision_tasks,
        target_permutation_augmentation=permutation_enabled,
        target_permutation_seed=int(
            permutation_cfg.get("seed", config.get("project", {}).get("seed", 0))
        ),
        **corpus_options,
    )
    if not dataset.records:
        raise RuntimeError(
            f"{stage or 'episode'} loader has no records after split/corpus/role filtering"
        )
    settings = stage_training_settings(config, stage)
    resolved_batch_size = int(
        batch_size
        or (
            settings["validation_batch_size"]
            if not shuffle
            else settings["batch_size"]
        )
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    num_workers = int(
        settings["num_workers"] if shuffle else settings["validation_num_workers"]
    )
    os.environ["VIBROGEMMA_WORKER_THREAD_LIMIT"] = str(settings["worker_thread_limit"])
    sampler = None
    if shuffle and stage == "pretrain":
        sampling_cfg = dict(stage_cfg.get("sampling") or {})
        strategy = str(sampling_cfg.get("strategy", "random"))
        requested = sampling_cfg.get("samples_per_epoch")
        samples_per_epoch = int(requested) if requested is not None else len(dataset)
        if strategy == "dataset_complete_group_window_uniform":
            sampler = RepresentationDatasetCompleteGroupSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=int(config.get("project", {}).get("seed", 0)) + 17,
            )
        elif strategy not in {"random", "shuffle"}:
            raise ValueError(
                f"unsupported encoder pretraining sampling strategy: {strategy}"
            )
    elif shuffle and require_labels and stage in {"align", "lora", "residual_adapter"}:
        stage_cfg = dict(config.get("training", {}).get(stage, {}))
        sampling_cfg = dict(stage_cfg.get("sampling") or {})
        strategy = str(sampling_cfg.get("strategy", "hierarchical_balanced"))
        requested = sampling_cfg.get("samples_per_epoch")
        samples_per_epoch = int(requested) if requested is not None else len(dataset)
        stage_seed_offset = {"align": 101, "lora": 202, "residual_adapter": 303}
        sampler_seed = int(config.get("project", {}).get("seed", 0)) + stage_seed_offset[stage]
        if strategy == "hierarchical_balanced":
            sampler = HierarchicalBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
            )
        elif strategy == "dataset_group_uniform":
            sampler = DatasetGroupUniformSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
            )
        elif strategy == "hybrid_group_balanced":
            sampler = HybridGroupBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
                state_balance_probability=float(
                    sampling_cfg.get("state_balance_probability", 0.5)
                ),
            )
        elif strategy == "factorized_task_balanced":
            sampler = FactorizedTaskBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
                target_task_probability=float(
                    sampling_cfg.get("target_task_probability", 0.5)
                ),
            )
        elif strategy == "factorized_target_index_balanced":
            sampler = FactorizedTargetIndexBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
                target_task_probability=float(
                    sampling_cfg.get("target_task_probability", 0.7)
                ),
            )
        elif strategy == "factorized_source_kind_target_index_balanced":
            sampler = FactorizedSourceKindTargetIndexBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
                target_task_probability=float(
                    sampling_cfg.get("target_task_probability", 0.7)
                ),
                synthetic_probability=float(
                    sampling_cfg.get("synthetic_probability", 0.30)
                ),
            )
        elif strategy == "structured_set_balanced":
            sampler = StructuredSetBalancedSampler(
                dataset.records,
                num_samples=samples_per_epoch,
                seed=sampler_seed,
                synthetic_probability=float(
                    sampling_cfg.get("synthetic_probability", 0.10)
                ),
            )
        elif strategy not in {"random", "shuffle"}:
            raise ValueError(f"unsupported supervised sampling strategy: {strategy}")

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": resolved_batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(settings["pin_memory"]),
        "persistent_workers": bool(settings["persistent_workers"] and num_workers > 0),
        "collate_fn": (
            MultiResolutionBatchCollator(
                config,
                retain_prepared=retain_prepared,
                split=split,
                prepared_cache_mode=prepared_cache_mode,
            )
            if prepare_multiresolution
            else collate_episodes
        ),
        "drop_last": shuffle,
        "worker_init_fn": _configure_worker if num_workers > 0 else None,
    }
    if "in_order" in inspect.signature(DataLoader).parameters:
        loader_kwargs["in_order"] = bool(settings["in_order"])
    elif not bool(settings["in_order"]):
        raise RuntimeError(
            "training.in_order=false requires a PyTorch DataLoader with in_order support"
        )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(settings["prefetch_factor"])
    return DataLoader(**loader_kwargs)


def benchmark_preprocessing_loader(
    config: dict[str, Any],
    *,
    split: str = "train",
    stage: str = "pretrain",
    batches: int = 4,
) -> dict[str, Any]:
    """Measure real worker-side preprocessing throughput on serialized episodes."""

    if batches < 1:
        raise ValueError("batches must be >= 1")
    loader = build_episode_loader(
        config,
        split,
        shuffle=split == "train",
        require_labels=stage != "pretrain",
        stage=stage,
        prepare_multiresolution=True,
        prepared_cache_mode="disabled",
    )
    started = time.perf_counter()
    examples = 0
    completed_batches = 0
    tensor_shapes: dict[str, list[int]] = {}
    for batch in loader:
        if not isinstance(batch, PreparedEpisodeBatch):
            raise TypeError(f"expected PreparedEpisodeBatch, got {type(batch).__name__}")
        examples += len(batch)
        completed_batches += 1
        tensor_shapes = {name: list(tensor.shape) for name, tensor in batch.tensors.items()}
        if completed_batches >= batches:
            break
    elapsed = time.perf_counter() - started
    if completed_batches == 0:
        raise RuntimeError("preprocessing benchmark produced no batches")
    settings = stage_training_settings(config, stage)
    return {
        "split": split,
        "stage": stage,
        "requested_batches": batches,
        "completed_batches": completed_batches,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
        "batches_per_second": completed_batches / max(elapsed, 1e-9),
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        "validation_num_workers": settings["validation_num_workers"],
        "prefetch_factor": settings["prefetch_factor"],
        "worker_thread_limit": settings["worker_thread_limit"],
        "tensor_shapes": tensor_shapes,
    }
