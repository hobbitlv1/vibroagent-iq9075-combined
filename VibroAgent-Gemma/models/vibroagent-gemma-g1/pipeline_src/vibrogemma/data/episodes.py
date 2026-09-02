from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from scipy.io import loadmat, whosmat
from torch.utils.data import Dataset

from ..config import resolve_from_config
from ..contracts import (
    ALLOWED_CLASSES,
    ALLOWED_SEVERITIES,
    AXES,
    BoardWindow,
    EpisodeLabel,
    EpisodeProvenance,
    SixBoardEpisode,
    TargetLabel,
)
from ..exceptions import DataContractError, ProvenanceError
from ..signal.quality import assess_quality
from ..utils import append_jsonl, atomic_write_json, iter_jsonl, sha256_file, stable_hash, utc_now_iso
from .manifest import DatasetManifest, load_dataset_manifests
from .serialization import load_episode, save_episode
from .splits import SplitRatios, assign_dataset_stratified_groups, assert_disjoint_groups


class _IndexedRecordTooShort(DataContractError):
    """One strict-index record cannot form even one configured window."""

    def __init__(
        self,
        *,
        dataset_id: str,
        source_file: str,
        experiment_id: str,
        sample_count: int,
        sample_rate_hz: float,
        window_seconds: float,
        window_samples: int,
    ) -> None:
        self.dataset_id = str(dataset_id)
        self.source_file = str(source_file)
        self.experiment_id = str(experiment_id)
        self.sample_count = int(sample_count)
        self.sample_rate_hz = float(sample_rate_hz)
        self.window_seconds = float(window_seconds)
        self.window_samples = int(window_samples)
        duration_seconds = self.sample_count / self.sample_rate_hz
        super().__init__(
            f"{self.dataset_id}:{self.source_file}: indexed record has "
            f"{self.sample_count} samples ({duration_seconds:.6f} s at "
            f"{self.sample_rate_hz:g} Hz), shorter than the configured "
            f"{self.window_seconds:g} s / {self.window_samples}-sample window"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_file": self.source_file,
            "experiment_id": self.experiment_id,
            "reason": "record_shorter_than_configured_window",
            "sample_count": self.sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.sample_count / self.sample_rate_hz,
            "required_window_seconds": self.window_seconds,
            "required_window_samples": self.window_samples,
        }


class EpisodeDataset(Dataset):
    """Load serialized six-board episodes from the deterministic build index."""

    def __init__(
        self,
        index_path: str | Path,
        split: str | Sequence[str],
        *,
        require_labels: bool,
        episode_ids: set[str] | None = None,
        expand_supervision_tasks: bool = False,
        target_permutation_augmentation: bool = False,
        target_permutation_seed: int = 0,
        include_dataset_ids: set[str] | None = None,
        include_training_roles: set[str] | None = None,
        train_only_dataset_ids: set[str] | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        requested = [split] if isinstance(split, str) else list(split)
        if not requested:
            raise ValueError("at least one split is required")
        allowed = {"train", "validation", "test", "all"}
        unknown = set(requested) - allowed
        if unknown:
            raise ValueError(f"unknown split(s): {sorted(unknown)}")
        if "all" in requested and len(requested) != 1:
            raise ValueError("split='all' cannot be combined with named splits")
        self.splits = tuple(requested)
        records = list(iter_jsonl(self.index_path))
        assert_disjoint_groups(records)
        split_records = [
            record
            for record in records
            if "all" in self.splits or str(record["split"]) in self.splits
        ]
        if episode_ids is not None:
            wanted = {str(value) for value in episode_ids}
            split_records = [record for record in split_records if str(record["episode_id"]) in wanted]
        if include_dataset_ids is not None:
            included = {str(value) for value in include_dataset_ids}
            split_records = [
                record
                for record in split_records
                if str(record.get("dataset_id", "")) in included
            ]
        if include_training_roles is not None:
            included_roles = {str(value) for value in include_training_roles}
            missing_role = [
                str(record.get("episode_id", ""))
                for record in split_records
                if not str(record.get("training_role", "")).strip()
            ]
            if missing_role:
                raise DataContractError(
                    "configured training-role filtering requires training_role on every record; "
                    f"missing examples={missing_role[:5]}"
                )
            split_records = [
                record
                for record in split_records
                if str(record["training_role"]) in included_roles
            ]
        if train_only_dataset_ids is not None:
            train_only = {str(value) for value in train_only_dataset_ids}
            split_records = [
                record
                for record in split_records
                if str(record.get("dataset_id", "")) not in train_only
                or str(record.get("split", "")) == "train"
            ]
        self.require_labels = bool(require_labels)
        self.target_permutation_augmentation = bool(target_permutation_augmentation)
        self.target_permutation_seed = int(target_permutation_seed)
        base_records = (
            [record for record in split_records if bool(record.get("has_label", True))]
            if self.require_labels
            else split_records
        )
        if expand_supervision_tasks and self.require_labels:
            expanded: list[dict[str, Any]] = []
            for record in base_records:
                base_sampling_group = str(
                    record.get("sampling_group") or record.get("split_group", "")
                )
                permutation = self._target_permutation(record)
                inverse = {old_index: new_index for new_index, old_index in enumerate(permutation, start=1)}
                global_record = dict(record)
                global_record["_supervision_task"] = "global"
                global_record["_task_family"] = "global"
                global_record["_task_state"] = str(
                    record.get("global_class_name") or "unknown"
                )
                global_record["_task_group"] = f"{base_sampling_group}|task_global"
                global_record["_target_permutation"] = list(permutation)
                expanded.append(global_record)
                if bool(record.get("target_supervision_available", True)):
                    affected = {
                        int(value) for value in record.get("affected_target_indices", [])
                    }
                    for original_target_index in range(1, 6):
                        logical_target_index = inverse[original_target_index]
                        target_record = dict(record)
                        target_record["_supervision_task"] = f"target_{logical_target_index}"
                        target_record["_task_family"] = "target"
                        target_record["_task_target_index"] = logical_target_index
                        target_record["_task_original_target_index"] = original_target_index
                        target_record["_task_state"] = (
                            "unknown_anomaly" if original_target_index in affected else "normal"
                        )
                        target_record["_task_group"] = (
                            f"{base_sampling_group}|task_target_{logical_target_index}"
                        )
                        target_record["_target_permutation"] = list(permutation)
                        expanded.append(target_record)
            self.records = expanded
        else:
            self.records = base_records

    def _target_permutation(self, record: dict[str, Any]) -> tuple[int, int, int, int, int]:
        """Return a reproducible training-only permutation of the five target boards.

        The permutation follows the waveform and every physical metadata field; labels
        are remapped with it.  It therefore changes only arbitrary logical slot order and
        cannot fabricate signal values or target labels.
        """

        identity = (1, 2, 3, 4, 5)
        if not self.target_permutation_augmentation:
            return identity
        key = "\0".join(
            (
                str(self.target_permutation_seed),
                str(record.get("dataset_id", "")),
                str(record.get("episode_id", "")),
                str(record.get("split_group", "")),
            )
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        values = list(identity)
        rng.shuffle(values)
        return tuple(values)  # type: ignore[return-value]

    @staticmethod
    def _apply_target_permutation(
        episode: SixBoardEpisode, permutation: Sequence[int]
    ) -> SixBoardEpisode:
        values = tuple(int(value) for value in permutation)
        if values == (1, 2, 3, 4, 5):
            return episode
        if sorted(values) != [1, 2, 3, 4, 5]:
            raise DataContractError(f"invalid target permutation: {values}")
        episode = copy.deepcopy(episode)
        original_boards = list(episode.boards[1:])
        remapped_boards = []
        for new_index, old_index in enumerate(values, start=1):
            board = original_boards[old_index - 1]
            board.board_id = f"target_{new_index}"
            board.role = "target"
            board.target_index = new_index
            board.metadata = dict(board.metadata)
            board.metadata["original_target_index_before_permutation"] = old_index
            remapped_boards.append(board)
        episode.boards = [episode.boards[0], *remapped_boards]
        if episode.label is not None and episode.label.target_supervision_available:
            original_targets = list(episode.label.targets)
            remapped_targets = []
            for new_index, old_index in enumerate(values, start=1):
                target = original_targets[old_index - 1]
                target.sensor_id = f"target_{new_index}"
                remapped_targets.append(target)
            episode.label.targets = remapped_targets
        episode.metadata = dict(episode.metadata)
        episode.metadata["target_permutation_augmentation"] = list(values)
        episode.metadata["target_permutation_semantics"] = (
            "training_only_logical_slot_permutation_with_waveform_metadata_and_labels_remapped"
        )
        return episode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SixBoardEpisode:
        record = self.records[index]
        base = self.index_path.parent
        array_path = Path(record["array_path"])
        metadata_path = Path(record["metadata_path"])
        if not array_path.is_absolute():
            array_path = base / array_path
        if not metadata_path.is_absolute():
            metadata_path = base / metadata_path
        episode = load_episode(array_path, metadata_path)
        episode = self._apply_target_permutation(
            episode, record.get("_target_permutation") or (1, 2, 3, 4, 5)
        )
        supervision_task = record.get("_supervision_task")
        if supervision_task is not None:
            if episode.label is None:
                raise DataContractError("supervision-task expansion requires an episode label")
            if supervision_task == "global":
                episode.label.targets = []
                episode.label.target_supervision_available = False
                episode.label.supervision_scope = "global_only"
                episode.label.focus_target_index = None
            elif str(supervision_task).startswith("target_"):
                target_index = int(str(supervision_task).split("_", 1)[1])
                if not episode.label.target_supervision_available:
                    raise DataContractError("target task requested for a globally labelled episode")
                episode.label.supervision_scope = "target_only"
                episode.label.focus_target_index = target_index
            else:
                raise DataContractError(f"unknown supervision task: {supervision_task}")
            episode.metadata["supervision_task"] = str(supervision_task)
        episode.validate(require_label=self.require_labels)
        return episode


class _NumericSource:
    def __init__(self, path: Path, options: dict[str, Any] | None = None) -> None:
        self.path = path
        self.suffix = path.suffix.lower()
        self.options = dict(options or {})
        self._handle: Any = None
        self._csv_delimiter: str | None = None
        self._csv_header: list[str] | None = None
        self._text_matrix: np.ndarray | None = None

    def __enter__(self) -> "_NumericSource":
        if self.suffix in {".h5", ".hdf5", ".he5"}:
            self._handle = h5py.File(self.path, "r")
        elif self.suffix == ".mat":
            try:
                self._handle = h5py.File(self.path, "r")
                self.suffix = ".hdf5-mat"
            except OSError:
                self._handle = loadmat(self.path, squeeze_me=True, struct_as_record=False)
        elif self.suffix == ".npz":
            self._handle = np.load(self.path, allow_pickle=False)
        elif self.suffix in {".csv", ".txt"}:
            self._handle = None
        else:
            raise DataContractError(f"unsupported source format for {self.path}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if hasattr(self._handle, "close"):
            self._handle.close()

    @staticmethod
    def _normalise_path(path: str) -> list[str]:
        return [part for part in path.strip("/").split("/") if part]

    def array(self, numeric_path: str, *, csv_has_header: bool = True, delimiter: str = ",") -> np.ndarray:
        if self.suffix in {".h5", ".hdf5", ".he5", ".hdf5-mat"}:
            if numeric_path not in self._handle:
                raise DataContractError(f"{self.path}: HDF5 node not found: {numeric_path}")
            value = np.asarray(self._handle[numeric_path])
        elif self.suffix == ".mat":
            current: Any = self._handle
            for part in self._normalise_path(numeric_path):
                if isinstance(current, dict):
                    if part not in current:
                        raise DataContractError(f"{self.path}: MAT field not found: {numeric_path}")
                    current = current[part]
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    raise DataContractError(f"{self.path}: MAT field not found: {numeric_path}")
                while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
                    current = current.item()
            value = np.asarray(current)
        elif self.suffix == ".npz":
            key = numeric_path.strip("/")
            if key not in self._handle:
                raise DataContractError(f"{self.path}: NPZ key not found: {key}")
            value = np.asarray(self._handle[key])
        else:
            field = numeric_path.strip("/")
            effective_delimiter = str(self.options.get("delimiter", delimiter))
            effective_has_header = bool(self.options.get("has_header", csv_has_header))
            skip_rows = int(self.options.get("skip_rows", 0))
            if field == "__matrix__" or not effective_has_header:
                if self._text_matrix is None:
                    self._text_matrix = np.genfromtxt(
                        self.path,
                        delimiter=effective_delimiter,
                        dtype=np.float64,
                        skip_header=skip_rows,
                        invalid_raise=False,
                    )
                    if self._text_matrix.ndim == 1:
                        self._text_matrix = self._text_matrix.reshape(-1, 1)
                value = self._text_matrix
            else:
                if self._csv_header is None or self._csv_delimiter is None:
                    with self.path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                        sample = handle.read(65536)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                        self._csv_delimiter = dialect.delimiter
                    except csv.Error:
                        self._csv_delimiter = effective_delimiter
                    lines = sample.splitlines()
                    header_row = int(self.options.get("header_row", 0))
                    first_line = lines[header_row] if len(lines) > header_row else ""
                    self._csv_header = [
                        item.strip()
                        for item in next(csv.reader([first_line], delimiter=self._csv_delimiter), [])
                    ]
                if field not in self._csv_header:
                    raise DataContractError(
                        f"{self.path}: exact CSV column not found: {field!r}; available={self._csv_header}"
                    )
                column_index = self._csv_header.index(field)
                values: list[float] = []
                data_start_row = int(self.options.get("data_start_row", 1))
                with self.path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    reader = csv.reader(handle, delimiter=self._csv_delimiter)
                    for _ in range(data_start_row):
                        next(reader, None)
                    for line_number, row in enumerate(reader, start=data_start_row + 1):
                        if not row or all(not item.strip() for item in row):
                            continue
                        if column_index >= len(row):
                            raise DataContractError(
                                f"{self.path}:{line_number}: row has no value for column {field!r}"
                            )
                        text = row[column_index].strip()
                        if self._csv_delimiter != "," and "," in text and "." not in text:
                            text = text.replace(",", ".")
                        try:
                            values.append(float(text))
                        except ValueError as exc:
                            raise DataContractError(
                                f"{self.path}:{line_number}: non-numeric value {row[column_index]!r} "
                                f"in column {field!r}"
                            ) from exc
                value = np.asarray(values, dtype=np.float64)
        if not np.issubdtype(value.dtype, np.number):
            raise DataContractError(f"{self.path}:{numeric_path} is not numeric")
        value = np.asarray(value)
        value = np.squeeze(value)
        if value.ndim == 0:
            raise DataContractError(f"{self.path}:{numeric_path} is scalar, not a signal")
        return value


def _resolve_axis_signal(
    source: _NumericSource,
    board: dict[str, Any],
    axis: str,
    *,
    sample_axis: int,
) -> np.ndarray | None:
    axis_paths = board.get("axis_paths") or {}
    if axis in axis_paths:
        value = source.array(str(axis_paths[axis]))
        if value.ndim != 1:
            raise DataContractError(f"axis path for {axis} must resolve to one-dimensional data")
        return value.astype(np.float32, copy=False)

    available = [str(item).lower() for item in board.get("axes_available", [])]
    if axis not in available:
        return None
    signal_path = board.get("signal_path")
    channel_indices = board.get("channel_indices") or {}
    if signal_path is None or axis not in channel_indices:
        raise DataContractError(f"board {board.get('sensor_id')}: axis {axis} needs axis_paths or signal_path/channel_indices")
    matrix = source.array(str(signal_path))
    if matrix.ndim == 1:
        if int(channel_indices[axis]) != 0:
            raise DataContractError("one-dimensional signal permits only channel index 0")
        return matrix.astype(np.float32, copy=False)
    if matrix.ndim != 2:
        raise DataContractError(f"signal_path must resolve to a 1D or 2D array, got {matrix.shape}")
    channel_axis = 1 - int(sample_axis)
    channel = int(channel_indices[axis])
    if channel < 0 or channel >= matrix.shape[channel_axis]:
        raise DataContractError(f"channel index {channel} is outside source shape {matrix.shape}")
    value = matrix[:, channel] if sample_axis == 0 else matrix[channel, :]
    return np.asarray(value, dtype=np.float32)


def _normalise_affected_targets(values: Any) -> set[int]:
    """Return canonical target indices from strict-index target identifiers.

    Private indexes may contain integer indices (1..5), numeric strings, or
    explicit sensor identifiers (target_1..target_5). Accept those unambiguous
    forms while rejecting booleans, unsupported values, and out-of-range slots.
    """

    if values is None:
        return set()
    if isinstance(values, (str, int, np.integer)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise DataContractError(
            "affected_targets must be a sequence of 1..5 or target_1..target_5 identifiers"
        )

    targets: set[int] = set()
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise DataContractError(
                f"affected_targets contains invalid boolean identifier {raw_value!r}"
            )
        if isinstance(raw_value, (int, np.integer)):
            target_index = int(raw_value)
        elif isinstance(raw_value, str):
            text = raw_value.strip().lower()
            suffix = text.removeprefix("target_")
            if suffix not in {"1", "2", "3", "4", "5"}:
                raise DataContractError(
                    f"affected_targets contains invalid identifier {raw_value!r}; "
                    "expected 1..5 or target_1..target_5"
                )
            target_index = int(suffix)
        else:
            raise DataContractError(
                f"affected_targets contains unsupported identifier {raw_value!r}; "
                "expected 1..5 or target_1..target_5"
            )
        if target_index not in {1, 2, 3, 4, 5}:
            raise DataContractError(
                f"affected_targets contains out-of-range target {target_index}; expected 1..5"
            )
        targets.add(target_index)
    return targets


def _build_label(entry: dict[str, Any], manifest: DatasetManifest) -> EpisodeLabel | None:
    if str(entry.get("training_role", "supervised")) == "representation_only":
        return None
    healthy = bool(entry.get("healthy", False))
    class_name = str(entry.get("class_name", "normal" if healthy else "unknown_anomaly"))
    severity = str(entry.get("severity", "none" if healthy else ""))
    if class_name not in ALLOWED_CLASSES:
        raise DataContractError(f"index class_name {class_name!r} is not in the declared vocabulary")
    if severity not in ALLOWED_SEVERITIES:
        raise DataContractError("changed records require an explicit advisory/warning/critical severity")

    if healthy:
        class_name = "normal"
        severity = "none"
    global_class_name = "normal" if healthy else "unknown_anomaly"
    global_severity = "none" if healthy else severity

    affected_targets = _normalise_affected_targets(entry.get("affected_targets", []))
    target_supervision_value = entry.get("target_supervision_available")
    if target_supervision_value is not None and not isinstance(
        target_supervision_value, (bool, np.bool_)
    ):
        raise DataContractError(
            "target_supervision_available must be a boolean when supplied"
        )
    explicit_target_supervision = (
        bool(target_supervision_value)
        if target_supervision_value is not None
        else None
    )
    declared_scope = str(entry.get("supervision_scope") or "").strip().lower()
    scope_is_global_only = declared_scope == "global_only"
    if scope_is_global_only and explicit_target_supervision is True:
        raise DataContractError(
            "global_only supervision contradicts target_supervision_available=true"
        )
    force_global_only = scope_is_global_only or explicit_target_supervision is False
    if force_global_only:
        if affected_targets:
            raise DataContractError(
                "global-only records must not declare affected_targets"
            )
        return EpisodeLabel(
            targets=[],
            explanation_target=str(entry.get("explanation_target", "")),
            global_class_name=global_class_name,
            global_severity=global_severity,
            target_supervision_available=False,
            supervision_scope="global_only",
        )

    localisation_mode = str(manifest.localisation.get("mode"))
    if healthy and localisation_mode != "global_unknown":
        labels = [
            TargetLabel(
                sensor_id=f"target_{target_index}",
                affected=False,
                class_name="normal",
                severity="none",
                source="publisher_localisation",
            )
            for target_index in range(1, 6)
        ]
        return EpisodeLabel(
            targets=labels,
            explanation_target=str(entry.get("explanation_target", "")),
            global_class_name="normal",
            global_severity="none",
            target_supervision_available=True,
            supervision_scope="joint",
        )

    if not affected_targets:
        if localisation_mode != "global_unknown" and not healthy:
            raise DataContractError(
                "changed records need explicit affected_targets for localised datasets"
            )
        # A global source label establishes that the episode changed but does not
        # establish any target state. Missing target labels are represented as
        # unavailable, never as all-normal or all-affected pseudo-labels.
        return EpisodeLabel(
            targets=[],
            explanation_target=str(entry.get("explanation_target", "")),
            global_class_name=global_class_name,
            global_severity=global_severity,
            target_supervision_available=False,
            supervision_scope="global_only",
        )

    if healthy:
        raise DataContractError("healthy records must not declare affected_targets")
    if not affected_targets.issubset({1, 2, 3, 4, 5}):
        raise DataContractError("affected_targets must be a subset of 1..5")
    labels = []
    for target_index in range(1, 6):
        affected = target_index in affected_targets
        labels.append(
            TargetLabel(
                sensor_id=f"target_{target_index}",
                affected=affected,
                class_name=class_name if affected else "normal",
                severity=severity if affected else "none",
                source="publisher_localisation",
            )
        )
    return EpisodeLabel(
        targets=labels,
        explanation_target=str(entry.get("explanation_target", "")),
        global_class_name=global_class_name,
        global_severity=global_severity,
        target_supervision_available=True,
        supervision_scope="joint",
    )


def _entry_split_stratum(entry: dict[str, Any]) -> str:
    """Return the state stratum used for dataset-aware group assignment.

    A verified adapter may provide an explicit stratum when independent
    publisher conditions need guaranteed train/validation/test coverage.
    """

    training_role = str(entry.get("training_role", "supervised"))
    if training_role == "representation_only":
        return "representation_only"
    explicit = entry.get("split_stratum")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    healthy = bool(entry.get("healthy", False))
    if healthy:
        return "healthy"
    return f"changed::{str(entry.get('class_name', 'unknown_anomaly'))}"


def _inspect_file(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    try:
        if suffix in {".h5", ".hdf5", ".he5"}:
            with h5py.File(path, "r") as handle:
                def visitor(name: str, value: Any) -> None:
                    if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
                        output.append({"path": f"/{name}", "shape": list(value.shape), "dtype": str(value.dtype)})
                handle.visititems(visitor)
        elif suffix == ".mat":
            try:
                with h5py.File(path, "r") as handle:
                    def visitor(name: str, value: Any) -> None:
                        if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
                            output.append({"path": f"/{name}", "shape": list(value.shape), "dtype": str(value.dtype)})
                    handle.visititems(visitor)
            except OSError:
                for name, shape, dtype in whosmat(path):
                    output.append({"path": name, "shape": list(shape), "dtype": dtype})
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    value = archive[name]
                    if np.issubdtype(value.dtype, np.number):
                        output.append({"path": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        elif suffix in {".csv", ".txt"}:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            output.append({"path": "<columns>", "columns": header})
    except Exception as exc:
        output.append({"error": f"{type(exc).__name__}: {exc}"})
    return output


class PublicEpisodeBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.manifests = load_dataset_manifests(config)
        data_cfg = config["data"]
        split_cfg = data_cfg["split"]
        self.ratios = SplitRatios(
            float(split_cfg["train"]),
            float(split_cfg["validation"]),
            float(split_cfg["test"]),
        )
        # Split identity is independent of the optimization seed.  This keeps
        # all three training seeds on one immutable data partition.
        self.seed = int(split_cfg.get("seed", config["project"].get("seed", 0)))
        self.window_seconds = float(data_cfg["window_seconds"])
        self.stride_seconds = float(data_cfg["stride_seconds"])
        self.minimum_valid_fraction = float(data_cfg.get("minimum_valid_fraction", 0.98))
        requested_splits = data_cfg.get("materialize_splits", ["train", "validation", "test"])
        self.materialize_splits = {str(value) for value in requested_splits}
        unknown_splits = self.materialize_splits - {"train", "validation", "test"}
        if not self.materialize_splits or unknown_splits:
            raise DataContractError(
                f"data.materialize_splits must be a non-empty subset of train/validation/test; "
                f"unknown={sorted(unknown_splits)}"
            )
        lock_value = dict(data_cfg.get("split") or {}).get("assignment_lock_path")
        self.split_assignment_lock_path = (
            resolve_from_config(self.config, str(lock_value)) if lock_value else None
        )
        # A large public release can reference the same multi-gigabyte source
        # file from hundreds of strict-index rows.  Verify each physical file
        # once per builder process rather than re-hashing it for every row and
        # every emitted window.
        self._source_checksum_cache: dict[Path, str] = {}
        if bool(config["project"].get("allow_local_training_data", False)):
            raise ProvenanceError("this builder is intentionally public-data-only")

    def _verified_source_checksum(self, path: Path, expected: Any | None = None) -> str:
        resolved = path.resolve()
        actual = self._source_checksum_cache.get(resolved)
        if actual is None:
            actual = sha256_file(path)
            self._source_checksum_cache[resolved] = actual
        if expected is not None and str(expected).lower() != actual.lower():
            raise ProvenanceError(f"source checksum mismatch for {path}")
        return actual

    def verify(self, *, require_licence_markers: bool = True) -> dict[str, Any]:
        records = []
        for manifest in self.manifests:
            root = manifest.raw_root(self.config)
            index_path = manifest.index_path(self.config)
            marker = manifest.licence_marker(self.config)
            receipt = root / "download_receipt.json"
            licence_ok = marker.exists() or (
                not bool(manifest.download.get("requires_manual_acceptance", True)) and receipt.exists()
            )
            status = {
                "dataset_id": manifest.id,
                "training_allowed": manifest.training_allowed,
                "raw_root": str(root),
                "raw_root_exists": root.exists(),
                "index": str(index_path),
                "index_exists": index_path.exists(),
                "licence_marker": str(marker),
                "licence_acknowledged": licence_ok,
            }
            if manifest.training_allowed and require_licence_markers and not licence_ok:
                status["error"] = "licence_not_acknowledged"
            if manifest.training_allowed and not index_path.exists():
                status["error"] = "strict_index_missing"
            if index_path.exists():
                entries = list(iter_jsonl(index_path))
                status["index_entries"] = len(entries)
                for item in entries:
                    self._validate_index_entry(item, manifest)
            records.append(status)
        failures = [record for record in records if "error" in record]
        if failures:
            raise ProvenanceError(f"public dataset verification failed: {failures}")
        return {"verified_utc": utc_now_iso(), "datasets": records}

    def inspect(self) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        for manifest in self.manifests:
            root = manifest.raw_root(self.config)
            glob = str(manifest.reader.get("file_glob", "**/*"))
            for path in sorted(root.glob(glob)) if root.exists() else []:
                if path.is_file():
                    nodes = _inspect_file(path)
                    if nodes:
                        report.append(
                            {
                                "dataset_id": manifest.id,
                                "file": str(path),
                                "sha256": sha256_file(path),
                                "numeric_nodes": nodes,
                            }
                        )
        return report

    @staticmethod
    def _validate_index_entry(entry: dict[str, Any], manifest: DatasetManifest) -> None:
        training_role = str(entry.get("training_role", "supervised"))
        if training_role not in {"supervised", "representation_only"}:
            raise DataContractError("training_role must be supervised or representation_only")
        required = [
            "dataset_id",
            "source_file",
            "structure_id",
            "session_id",
            "experiment_id",
            "sample_rate_hz",
            "unit",
            "boards",
        ]
        if training_role == "supervised":
            required.extend(["healthy", "severity", "class_name"])
        missing = [key for key in required if key not in entry]
        if missing:
            raise DataContractError(f"{manifest.id} index entry is missing {missing}")
        if str(entry["dataset_id"]) != manifest.id:
            raise DataContractError(f"index dataset_id must equal {manifest.id!r}")
        if float(entry["sample_rate_hz"]) <= 0:
            raise DataContractError("sample_rate_hz must be positive")
        if "window_stride_seconds" in entry:
            stride_seconds = float(entry["window_stride_seconds"])
            if not np.isfinite(stride_seconds) or stride_seconds <= 0:
                raise DataContractError("window_stride_seconds must be finite and positive")
        if len(entry["boards"]) != 6:
            raise DataContractError("every indexed record must map six genuine sensor locations")
        identifiers = [str(board.get("sensor_id", "")) for board in entry["boards"]]
        expected = ["reference", "target_1", "target_2", "target_3", "target_4", "target_5"]
        if identifiers != expected:
            raise DataContractError(f"board sensor_id order must be {expected}")
        for board in entry["boards"]:
            axes = {str(axis).lower() for axis in board.get("axes_available", [])}
            if not axes or not axes.issubset(set(AXES)):
                raise DataContractError("axes_available must explicitly list genuine x/y/z axes")
            if not board.get("axis_paths") and not board.get("signal_path"):
                raise DataContractError("each board requires exact axis_paths or signal_path")
        original_sources = entry.get("original_source_files")
        if original_sources is not None:
            if not isinstance(original_sources, list) or not original_sources:
                raise DataContractError(
                    "original_source_files must be a non-empty list when supplied"
                )
            for item in original_sources:
                if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                    raise DataContractError(
                        "each original_source_files item requires path and sha256"
                    )

    def _entry_arrays(self, entry: dict[str, Any], manifest: DatasetManifest) -> tuple[np.ndarray, np.ndarray]:
        source_path = manifest.raw_root(self.config) / str(entry["source_file"])
        if not source_path.exists():
            raise DataContractError(f"source file does not exist: {source_path}")
        sample_axis = int(entry.get("sample_axis", 0))
        if sample_axis not in {0, 1}:
            raise DataContractError("sample_axis must be 0 or 1")
        arrays: list[np.ndarray] = []
        axis_masks: list[np.ndarray] = []
        with _NumericSource(source_path, options=dict(entry.get("source_options") or {})) as source:
            for board in entry["boards"]:
                axis_values = []
                axis_mask = []
                for axis in AXES:
                    signal = _resolve_axis_signal(source, board, axis, sample_axis=sample_axis)
                    axis_mask.append(signal is not None)
                    axis_values.append(signal)
                genuine = [value for value in axis_values if value is not None]
                if not genuine:
                    raise DataContractError("a board has no genuine signal axis")
                lengths = {value.shape[0] for value in genuine}
                if len(lengths) != 1:
                    raise DataContractError("axes mapped to one board have different lengths")
                length = next(iter(lengths))
                matrix = np.zeros((3, length), dtype=np.float32)
                for axis_index, signal in enumerate(axis_values):
                    if signal is not None:
                        matrix[axis_index] = signal
                invalid_abs_threshold = entry.get("invalid_abs_threshold")
                if invalid_abs_threshold is not None:
                    threshold = float(invalid_abs_threshold)
                    matrix[np.abs(matrix) > threshold] = np.nan
                arrays.append(matrix)
                axis_masks.append(np.asarray(axis_mask, dtype=bool))
        lengths = {array.shape[-1] for array in arrays}
        if len(lengths) != 1:
            raise DataContractError("six mapped board signals have different lengths")
        return np.stack(arrays), np.stack(axis_masks)

    def _episodes_from_entry(self, entry: dict[str, Any], manifest: DatasetManifest) -> Iterable[SixBoardEpisode]:
        self._validate_index_entry(entry, manifest)
        values, axis_masks = self._entry_arrays(entry, manifest)
        sample_rate = float(entry["sample_rate_hz"])
        window_samples = int(round(self.window_seconds * sample_rate))
        stride_seconds = float(entry.get("window_stride_seconds", self.stride_seconds))
        stride_samples = int(round(stride_seconds * sample_rate))
        if window_samples <= 0 or stride_samples <= 0:
            raise DataContractError("window and stride must contain at least one sample")
        if values.shape[-1] < window_samples:
            raise _IndexedRecordTooShort(
                dataset_id=manifest.id,
                source_file=str(entry["source_file"]),
                experiment_id=str(entry["experiment_id"]),
                sample_count=int(values.shape[-1]),
                sample_rate_hz=sample_rate,
                window_seconds=self.window_seconds,
                window_samples=window_samples,
            )
        source_path = manifest.raw_root(self.config) / str(entry["source_file"])
        source_checksum = self._verified_source_checksum(source_path, entry.get("source_sha256"))
        group = f"{manifest.id}|{entry['structure_id']}|{entry['session_id']}"
        label = _build_label(entry, manifest)
        training_role = str(entry.get("training_role", "supervised"))
        orientation_default = None
        for start in range(0, values.shape[-1] - window_samples + 1, stride_samples):
            stop = start + window_samples
            episode_key = {
                "dataset": manifest.id,
                "source": str(entry["source_file"]),
                "experiment": str(entry["experiment_id"]),
                "start": start,
                "stop": stop,
                "checksum": source_checksum,
            }
            episode_id = f"{manifest.id}-{stable_hash(episode_key)[:20]}"
            boards: list[BoardWindow] = []
            for board_index, board_cfg in enumerate(entry["boards"]):
                board_values = values[board_index, :, start:stop].copy()
                finite = np.isfinite(board_values[axis_masks[board_index]]).all(axis=0)
                sample_mask = np.ones(window_samples, dtype=bool)
                if finite.shape[0] == window_samples:
                    sample_mask &= finite
                board_values[:, ~sample_mask] = 0.0
                orientation = board_cfg.get("orientation_matrix", orientation_default)
                position = board_cfg.get("sensor_position")
                quality = assess_quality(
                    board_values,
                    sample_mask,
                    axis_masks[board_index],
                    sample_rate_hz=sample_rate,
                    original_sample_rate_hz=sample_rate,
                    notes=["public_dataset", f"source_condition:{entry.get('condition', '')}"],
                )
                quality.usable_band_hz = min(
                    sample_rate / 2.0,
                    float(entry.get("sensor_useful_band_hz", sample_rate / 2.0)),
                )
                boards.append(
                    BoardWindow(
                        board_id="reference" if board_index == 0 else f"target_{board_index}",
                        values=board_values,
                        sample_rate_hz=sample_rate,
                        start_time_s=float(entry.get("start_time_s", 0.0)) + start / sample_rate,
                        axis_mask=axis_masks[board_index].copy(),
                        sample_mask=sample_mask,
                        role="reference" if board_index == 0 else "target",
                        target_index=None if board_index == 0 else board_index,
                        sensor_position=(tuple(float(value) for value in position) if position is not None else None),
                        orientation_matrix=(
                            np.asarray(orientation, dtype=np.float32) if orientation is not None else None
                        ),
                        quality=quality,
                        metadata={
                            "unit": str(entry["unit"]),
                            "amplitude_calibrated": bool(
                                entry.get("amplitude_calibrated", str(entry["unit"]) != "release_native")
                            ),
                            "sensor_useful_band_hz": quality.usable_band_hz,
                            "source_sensor_id": board_cfg["sensor_id"],
                            "source_sensor_name": board_cfg.get("source_sensor_name"),
                            "source_measurement_level": board_cfg.get(
                                "source_measurement_level"
                            ),
                            "sensor_elevation_m": board_cfg.get("sensor_elevation_m"),
                            "physical_sensor_id": board_cfg.get("physical_sensor_id"),
                            "position_basis": board_cfg.get("position_basis"),
                            "sensor_position_m": board_cfg.get("sensor_position_m"),
                            "physical_position_basis": board_cfg.get(
                                "physical_position_basis"
                            ),
                            "fixed_physical_panel": bool(board_cfg.get("fixed_physical_panel", False)),
                            "source_channel_indices": dict(board_cfg.get("channel_indices") or {}),
                            "source_file": str(entry["source_file"]),
                        },
                    )
                )
            original_source = entry.get("original_source_file")
            original_checksum = entry.get("original_source_sha256")
            source_files = [str(source_path)]
            source_checksums = [source_checksum]
            original_sources: list[tuple[str, Any | None]] = []
            if original_source is not None:
                original_sources.append((str(original_source), original_checksum))
            original_sources.extend(
                (str(item["path"]), item["sha256"])
                for item in entry.get("original_source_files", [])
            )
            for original_value, expected_original_checksum in original_sources:
                original_path = manifest.raw_root(self.config) / original_value
                if not original_path.exists():
                    raise DataContractError(f"original source file does not exist: {original_path}")
                actual_original_checksum = self._verified_source_checksum(
                    original_path,
                    expected_original_checksum,
                )
                source_files.append(str(original_path))
                source_checksums.append(actual_original_checksum)
            provenance = EpisodeProvenance(
                dataset_id=manifest.id,
                provenance_type=manifest.provenance_type,
                source_files=source_files,
                source_checksums=source_checksums,
                licence=manifest.licence,
                licence_accepted=True,
                structure_id=str(entry["structure_id"]),
                session_id=str(entry["session_id"]),
                experiment_id=str(entry["experiment_id"]),
                split_group=group,
                derived_operations=[
                    f"window:{start}:{stop}",
                    "missing_axes_zero_storage_with_explicit_axis_mask",
                    f"training_role:{training_role}",
                ],
                local_training_data=False,
            )
            episode = SixBoardEpisode(
                episode_id=episode_id,
                boards=boards,
                provenance=provenance,
                label=label,
                metadata={
                    "condition": entry.get("condition"),
                    "training_role": training_role,
                    "phase_synchronized": bool(entry.get("phase_synchronized", False)),
                    "synchronization_verification_id": entry.get("synchronization_verification_id"),
                    "amplitude_calibrated": bool(
                        entry.get("amplitude_calibrated", str(entry["unit"]) != "release_native")
                    ),
                    "auto_index": entry.get("auto_index"),
                    "audit_metadata": dict(entry.get("audit_metadata") or {}),
                },
            )
            episode.validate(require_label=training_role == "supervised")
            valid_fraction = min(board.quality.valid_fraction for board in episode.boards)
            if valid_fraction >= self.minimum_valid_fraction:
                yield episode

    def build(self, *, overwrite: bool = False) -> dict[str, Any]:
        self.verify(require_licence_markers=True)
        root = resolve_from_config(self.config, self.config["data"]["episode_root"])
        if overwrite and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "index.jsonl"
        provenance_path = resolve_from_config(
            self.config,
            self.config["project"].get("provenance_ledger", "data/episodes/provenance.jsonl"),
        )
        if index_path.exists() and not overwrite:
            raise FileExistsError(f"episode index already exists: {index_path}; pass --overwrite")
        index_path.unlink(missing_ok=True)
        provenance_path.unlink(missing_ok=True)

        # Resolve every complete session group before materialising any waveform.
        # Publisher-locked assignments (UPC) are accepted only when every row in
        # the same split group agrees. Remaining groups use the deterministic
        # dataset/stratum assignment. The optional lock is reused by the later
        # test-only materialisation so the test boundary cannot drift.
        indexed_entries: list[tuple[DatasetManifest, dict[str, Any]]] = []
        split_descriptors: list[dict[str, str]] = []
        explicit_assignments: dict[str, str] = {}
        for manifest in self.manifests:
            if not manifest.training_allowed:
                continue
            for entry in iter_jsonl(manifest.index_path(self.config)):
                self._validate_index_entry(entry, manifest)
                group = f"{manifest.id}|{entry['structure_id']}|{entry['session_id']}"
                indexed_entries.append((manifest, entry))
                explicit_split = entry.get("split")
                if explicit_split is not None:
                    explicit_split = str(explicit_split)
                    if explicit_split not in {"train", "validation", "test"}:
                        raise DataContractError(
                            f"explicit split for {group} is invalid: {explicit_split!r}"
                        )
                    previous = explicit_assignments.setdefault(group, explicit_split)
                    if previous != explicit_split:
                        raise DataContractError(
                            f"split group {group} has conflicting explicit assignments"
                        )
                else:
                    split_descriptors.append(
                        {
                            "split_group": group,
                            "dataset_id": manifest.id,
                            "split_stratum": _entry_split_stratum(entry),
                        }
                    )

        if split_descriptors:
            assigned, generated_report = assign_dataset_stratified_groups(
                split_descriptors,
                self.ratios,
                self.seed,
            )
        else:
            assigned = {}
            generated_report = {
                "strategy": "explicit_locked_complete_groups_v1",
                "assignments": {},
                "datasets": {},
            }
        assignments = {**assigned, **explicit_assignments}
        expected_groups = {
            f"{manifest.id}|{entry['structure_id']}|{entry['session_id']}"
            for manifest, entry in indexed_entries
        }
        if set(assignments) != expected_groups:
            missing = sorted(expected_groups - set(assignments))
            extra = sorted(set(assignments) - expected_groups)
            raise DataContractError(
                f"split assignment coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )

        lock_payload = {
            "schema": "vibrogemma-split-assignment-lock-v1",
            "seed": self.seed,
            "strategy": "publisher_explicit_plus_dataset_stratified_complete_groups_v1",
            "assignments": dict(sorted(assignments.items())),
            "explicit_group_count": len(explicit_assignments),
            "generated_group_count": len(assigned),
        }
        if self.split_assignment_lock_path is not None:
            self.split_assignment_lock_path.parent.mkdir(parents=True, exist_ok=True)
            if self.split_assignment_lock_path.exists():
                existing_lock = json.loads(
                    self.split_assignment_lock_path.read_text(encoding="utf-8")
                )
                existing_assignments = {
                    str(key): str(value)
                    for key, value in dict(existing_lock.get("assignments") or {}).items()
                }
                if existing_assignments != lock_payload["assignments"]:
                    raise DataContractError(
                        "current metadata no longer reproduces the frozen split-assignment lock"
                    )
                assignments = existing_assignments
                lock_payload = existing_lock
            else:
                atomic_write_json(self.split_assignment_lock_path, lock_payload)

        split_report = {
            "strategy": lock_payload["strategy"],
            "materialize_splits": sorted(self.materialize_splits),
            "explicit_group_count": len(explicit_assignments),
            "generated_group_count": len(assigned),
            "assignments": dict(sorted(assignments.items())),
            "generated_assignment_report": generated_report,
            "assignment_lock_path": (
                str(self.split_assignment_lock_path)
                if self.split_assignment_lock_path is not None
                else None
            ),
        }
        atomic_write_json(root / "split_assignment_report.json", split_report)

        counts = {"train": 0, "validation": 0, "test": 0}
        nonmaterialized_entry_counts = {"train": 0, "validation": 0, "test": 0}
        label_counts = {"labelled": 0, "representation_only": 0}
        dataset_counts: dict[str, int] = {}
        dataset_split_counts: dict[str, dict[str, int]] = {}
        dataset_stratum_split_counts: dict[str, dict[str, dict[str, int]]] = {}
        skipped_entries: list[dict[str, Any]] = []
        skipped_by_dataset: dict[str, int] = {}
        skip_path = root / "build_skips.jsonl"
        skip_path.unlink(missing_ok=True)
        records: list[dict[str, Any]] = []

        for manifest, entry in indexed_entries:
            group = f"{manifest.id}|{entry['structure_id']}|{entry['session_id']}"
            split = assignments[group]
            stratum = _entry_split_stratum(entry)
            if split not in self.materialize_splits:
                nonmaterialized_entry_counts[split] += 1
                continue
            try:
                for episode in self._episodes_from_entry(entry, manifest):
                    split_dir = root / split
                    array_path, metadata_path = save_episode(episode, split_dir)
                    record = {
                        "episode_id": episode.episode_id,
                        "dataset_id": manifest.id,
                        "structure_id": episode.provenance.structure_id,
                        "session_id": episode.provenance.session_id,
                        "experiment_id": episode.provenance.experiment_id,
                        "split_group": episode.provenance.split_group,
                        "split_stratum": stratum,
                        "split": split,
                        "array_path": str(array_path.relative_to(root)),
                        "metadata_path": str(metadata_path.relative_to(root)),
                        "has_label": episode.label is not None,
                        "training_role": str(episode.metadata.get("training_role", "supervised")),
                        "global_class_name": (
                            episode.label.global_class_name if episode.label is not None else None
                        ),
                        "target_supervision_available": bool(
                            episode.label is not None
                            and episode.label.target_supervision_available
                        ),
                        "affected_target_indices": (
                            list(episode.label.affected_target_indices)
                            if episode.label is not None
                            and episode.label.target_supervision_available
                            else []
                        ),
                        "sampling_group": (
                            f"{episode.provenance.split_group}|"
                            f"{episode.provenance.experiment_id}"
                        ),
                    }
                    append_jsonl(index_path, record)
                    append_jsonl(provenance_path, {**record, "provenance": asdict(episode.provenance)})
                    records.append(record)
                    counts[split] += 1
                    label_counts["labelled" if episode.label is not None else "representation_only"] += 1
                    dataset_counts[manifest.id] = dataset_counts.get(manifest.id, 0) + 1
                    dataset_split_counts.setdefault(
                        manifest.id, {"train": 0, "validation": 0, "test": 0}
                    )[split] += 1
                    dataset_stratum_split_counts.setdefault(manifest.id, {}).setdefault(
                        stratum, {"train": 0, "validation": 0, "test": 0}
                    )[split] += 1
            except _IndexedRecordTooShort as exc:
                skip_record = exc.as_record()
                append_jsonl(skip_path, skip_record)
                skipped_entries.append(skip_record)
                skipped_by_dataset[manifest.id] = skipped_by_dataset.get(manifest.id, 0) + 1
                continue

        assert_disjoint_groups(records)
        datasets_without_episodes = [
            manifest.id
            for manifest in self.manifests
            if manifest.training_allowed and dataset_counts.get(manifest.id, 0) == 0
        ]
        report = {
            "built_utc": utc_now_iso(),
            "index": str(index_path),
            "split_strategy": split_report["strategy"],
            "split_assignment_report": str(root / "split_assignment_report.json"),
            "split_assignment_lock": (
                str(self.split_assignment_lock_path)
                if self.split_assignment_lock_path is not None
                else None
            ),
            "materialize_splits": sorted(self.materialize_splits),
            "nonmaterialized_entry_counts": nonmaterialized_entry_counts,
            "total_episodes": sum(counts.values()),
            "by_split": counts,
            "by_label_availability": label_counts,
            "by_dataset": dataset_counts,
            "by_dataset_split": dataset_split_counts,
            "by_dataset_stratum_split": dataset_stratum_split_counts,
            "skipped_entries": len(skipped_entries),
            "skipped_by_dataset": skipped_by_dataset,
            "skip_log": str(skip_path) if skipped_entries else None,
            "datasets_without_episodes": datasets_without_episodes,
        }
        atomic_write_json(root / "build_report.json", report)
        if not records:
            raise DataContractError(
                f"no episodes were built; inspect {skip_path} and the strict indexes"
            )
        if datasets_without_episodes:
            raise DataContractError(
                "training datasets produced no complete windows: "
                f"{datasets_without_episodes}; inspect {root / 'build_report.json'}"
            )
        return report

# v0.17.0 train-only OpenSees augmentation hook.  The wrapper is a no-op unless
# VIBROGEMMA_SYNTHETIC_AUGMENTATION_MANIFEST is set and the requested split is
# exactly "train".  Validation and test therefore remain measured-only.
from vibrogemma.simulation.synthetic_dataset_v017 import wrap_episode_dataset_class as _wrap_episode_dataset_v017
EpisodeDataset = _wrap_episode_dataset_v017(EpisodeDataset)
del _wrap_episode_dataset_v017
