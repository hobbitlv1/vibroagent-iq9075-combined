from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import DataContractError
from ..utils import atomic_write_json, utc_now_iso


@dataclass(slots=True)
class _RunningVectorStats:
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(cls, size: int) -> "_RunningVectorStats":
        return cls(
            count=np.zeros(size, dtype=np.int64),
            mean=np.zeros(size, dtype=np.float64),
            m2=np.zeros(size, dtype=np.float64),
        )

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(vector)
        if vector.shape != self.mean.shape or valid.shape != self.mean.shape:
            raise DataContractError("healthy-profile vector shape mismatch")
        for index in np.flatnonzero(valid):
            self.count[index] += 1
            delta = vector[index] - self.mean[index]
            self.mean[index] += delta / self.count[index]
            delta2 = vector[index] - self.mean[index]
            self.m2[index] += delta * delta2

    def finalize(self, minimum_count: int, std_floor: float) -> dict[str, Any]:
        count_valid = self.count >= int(minimum_count)
        variance = np.zeros_like(self.mean)
        usable = self.count > 1
        variance[usable] = self.m2[usable] / (self.count[usable] - 1)
        std = np.sqrt(np.maximum(variance, float(std_floor) ** 2))
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "count_valid": count_valid.tolist(),
        }


@dataclass(slots=True)
class HealthyProfile:
    feature_names: tuple[str, ...]
    groups: dict[str, dict[str, np.ndarray | int | bool]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def exact_key(dataset_id: str, structure_id: str, target_index: int) -> str:
        return f"dataset:{dataset_id}|structure:{structure_id}|target:{target_index}"

    @staticmethod
    def dataset_key(dataset_id: str, target_index: int) -> str:
        return f"dataset:{dataset_id}|structure:*|target:{target_index}"

    @staticmethod
    def global_key(target_index: int) -> str:
        return f"dataset:*|structure:*|target:{target_index}"

    def _lookup_keys(self, dataset_id: str, structure_id: str, target_index: int) -> tuple[str, ...]:
        keys = [self.exact_key(dataset_id, structure_id, target_index)]
        policy = dict(self.metadata.get("lookup_policy") or {})
        if bool(policy.get("allow_dataset_fallback", True)):
            keys.append(self.dataset_key(dataset_id, target_index))
        if bool(policy.get("allow_global_fallback", False)):
            keys.append(self.global_key(target_index))
        return tuple(keys)

    def lookup(
        self,
        *,
        dataset_id: str,
        structure_id: str,
        target_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        for key in self._lookup_keys(dataset_id, structure_id, target_index):
            if key not in self.groups:
                continue
            group = self.groups[key]
            if not bool(group.get("group_valid", True)):
                continue
            return (
                np.asarray(group["mean"], dtype=np.float32),
                np.asarray(group["std"], dtype=np.float32),
                np.asarray(group["valid"], dtype=bool),
                key,
            )
        return None

    def z_scores(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        dataset_id: str,
        structure_id: str,
        target_index: int,
        clip: float = 12.0,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        vector = np.asarray(values, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if vector.shape != (len(self.feature_names),) or valid.shape != vector.shape:
            raise DataContractError("healthy-profile input does not match feature schema")
        found = self.lookup(
            dataset_id=dataset_id,
            structure_id=structure_id,
            target_index=target_index,
        )
        if found is None:
            return np.zeros_like(vector), np.zeros_like(valid), None
        mean, std, profile_valid, key = found
        usable = valid & profile_valid & np.isfinite(std) & (std > 0)
        z = np.zeros_like(vector)
        z[usable] = np.clip((vector[usable] - mean[usable]) / std[usable], -clip, clip)
        return z, usable, key

    def save(self, path: str | Path) -> None:
        payload = {
            "version": "2.0",
            "created_utc": utc_now_iso(),
            "feature_names": list(self.feature_names),
            "groups": {
                key: {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in group.items()
                }
                for key, group in self.groups.items()
            },
            "metadata": self.metadata,
        }
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "HealthyProfile":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        feature_names = tuple(str(name) for name in payload["feature_names"])
        groups: dict[str, dict[str, np.ndarray | int | bool]] = {}
        for key, value in payload["groups"].items():
            groups[str(key)] = {
                "mean": np.asarray(value["mean"], dtype=np.float32),
                "std": np.asarray(value["std"], dtype=np.float32),
                "valid": np.asarray(value["valid"], dtype=bool),
                "count": np.asarray(value.get("count", [0] * len(feature_names)), dtype=np.int64),
                "group_count": int(value.get("group_count", 0)),
                "group_valid": bool(value.get("group_valid", True)),
            }
        profile = cls(
            feature_names=feature_names,
            groups=groups,
            metadata=dict(payload.get("metadata", {})),
        )
        for key, group in groups.items():
            for field_name in ("mean", "std", "valid", "count"):
                if np.asarray(group[field_name]).shape != (len(feature_names),):
                    raise DataContractError(
                        f"healthy profile group {key!r} has invalid {field_name} shape"
                    )
        return profile


class HealthyProfileBuilder:
    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = tuple(feature_names)
        self._groups: dict[str, _RunningVectorStats] = {}
        self._independent_groups: dict[str, set[str]] = {}
        self.episodes_seen = 0

    def _update_key(
        self,
        key: str,
        values: np.ndarray,
        mask: np.ndarray,
        split_group: str,
    ) -> None:
        stats = self._groups.setdefault(key, _RunningVectorStats.create(len(self.feature_names)))
        stats.update(values, mask)
        self._independent_groups.setdefault(key, set()).add(str(split_group))

    def add(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        dataset_id: str,
        structure_id: str,
        target_index: int,
        split_group: str,
    ) -> None:
        vector = np.asarray(values, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if vector.shape != (len(self.feature_names),) or valid.shape != vector.shape:
            raise DataContractError("healthy profile builder vector shape mismatch")
        keys = (
            HealthyProfile.exact_key(dataset_id, structure_id, target_index),
            HealthyProfile.dataset_key(dataset_id, target_index),
            HealthyProfile.global_key(target_index),
        )
        for key in keys:
            self._update_key(key, vector, valid, split_group)
        self.episodes_seen += 1

    def finalize(
        self,
        *,
        minimum_count: int = 8,
        minimum_independent_groups: int = 2,
        std_floor: float = 1e-3,
        allow_dataset_fallback: bool = True,
        allow_global_fallback: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> HealthyProfile:
        if minimum_independent_groups < 1:
            raise ValueError("minimum_independent_groups must be >= 1")
        groups: dict[str, dict[str, np.ndarray | int | bool]] = {}
        for key, stats in self._groups.items():
            finalized = stats.finalize(minimum_count, std_floor)
            group_count = len(self._independent_groups.get(key, set()))
            group_valid = group_count >= int(minimum_independent_groups)
            count_valid = np.asarray(finalized["count_valid"], dtype=bool)
            groups[key] = {
                "count": np.asarray(finalized["count"], dtype=np.int64),
                "mean": np.asarray(finalized["mean"], dtype=np.float32),
                "std": np.asarray(finalized["std"], dtype=np.float32),
                "valid": count_valid & group_valid,
                "group_count": group_count,
                "group_valid": group_valid,
            }
        return HealthyProfile(
            feature_names=self.feature_names,
            groups=groups,
            metadata={
                "episodes_seen": self.episodes_seen,
                "minimum_independent_groups": int(minimum_independent_groups),
                "lookup_policy": {
                    "allow_dataset_fallback": bool(allow_dataset_fallback),
                    "allow_global_fallback": bool(allow_global_fallback),
                },
                **(metadata or {}),
            },
        )
