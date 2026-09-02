from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ..contracts import QualityReport, SixBoardEpisode
from ..exceptions import DataContractError
from ..utils import json_default, sha256_file, stable_hash
from .multiresolution import MultiResolutionEpisodePreprocessor, PreparedEpisode

CACHE_SCHEMA = "vibrogemma-prepared-multiresolution-v1"
_METADATA_KEY = "__cache_metadata_json__"
_ARRAY_NAMES = (
    "processed_values",
    "processed_axis_mask",
    "processed_sample_mask",
    "modal_values",
    "modal_axis_mask",
    "modal_sample_mask",
    "wideband_features",
    "wideband_frequency_mask",
    "wideband_cross_valid",
    "wideband_frequency_grid_hz",
    "wideband_time_grid_s",
    "physics_features",
    "physics_mask",
    "profile_feature_values",
    "profile_feature_mask",
    "sensor_positions",
    "orientation_features",
    "quality_features",
)
_PREPROCESSING_SOURCE_FILES = (
    "multiresolution.py",
    "orientation.py",
    "physics.py",
    "profile.py",
    "quality.py",
    "resample.py",
    "spectral.py",
    "units.py",
)
_DERIVED_BOARD_METADATA_KEYS = (
    "unit",
    "source_unit",
    "amplitude_calibrated",
    "source_sample_rate_hz",
    "orientation_calibrated",
    "orientation_applied",
    "orientation_reason",
    "sensor_useful_band_hz",
)


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _preprocessing_code_sha256() -> str:
    signal_root = Path(__file__).resolve().parent
    payload = {name: sha256_file(signal_root / name) for name in _PREPROCESSING_SOURCE_FILES}
    return stable_hash(payload)


def _episode_payload_sha256(episode: SixBoardEpisode) -> str:
    """Hash exact physical input and provenance while deliberately excluding labels."""

    episode.validate(require_label=False)
    metadata = dict(episode.metadata)
    # Expanded global/target tasks have identical physical evidence. Keeping the
    # task name out of the payload lets those views share one prepared item.
    metadata.pop("supervision_task", None)
    header = {
        "episode_id": episode.episode_id,
        "provenance": asdict(episode.provenance),
        "metadata": metadata,
        "boards": [
            {
                "board_id": board.board_id,
                "sample_rate_hz": float(board.sample_rate_hz),
                "start_time_s": float(board.start_time_s),
                "role": board.role,
                "target_index": board.target_index,
                "sensor_position": board.sensor_position,
                "orientation_matrix": (
                    None
                    if board.orientation_matrix is None
                    else np.asarray(board.orientation_matrix).tolist()
                ),
                "quality": asdict(board.quality),
                "metadata": board.metadata,
            }
            for board in episode.boards
        ],
        "labels_in_key": False,
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")
    )
    for board in episode.boards:
        for name, value in (
            ("values", board.values),
            ("axis_mask", board.axis_mask),
            ("sample_mask", board.sample_mask),
        ):
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(_array_sha256(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


class PreparedEpisodeDiskCache:
    """Lossless content-addressed cache for deterministic signal preprocessing."""

    def __init__(
        self,
        root: str | Path,
        preprocessor: MultiResolutionEpisodePreprocessor,
    ) -> None:
        self.root = Path(root)
        self.preprocessor = preprocessor
        settings = asdict(preprocessor.settings)
        profile_path = preprocessor.settings.healthy_profile_path
        profile_sha256 = (
            sha256_file(profile_path) if profile_path is not None and profile_path.is_file() else None
        )
        # The profile content hash, rather than its machine-specific path, is the
        # preprocessing dependency. This permits reuse between controlled arms.
        settings["healthy_profile_path"] = None
        self.fingerprint_payload = {
            "schema": CACHE_SCHEMA,
            "settings": settings,
            "healthy_profile_sha256": profile_sha256,
            "preprocessing_code_sha256": _preprocessing_code_sha256(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        }
        self.preprocessing_sha256 = stable_hash(self.fingerprint_payload)

    def path_for(self, episode: SixBoardEpisode) -> tuple[Path, str, str]:
        payload_sha256 = _episode_payload_sha256(episode)
        key = stable_hash(
            {
                "schema": CACHE_SCHEMA,
                "preprocessing_sha256": self.preprocessing_sha256,
                "episode_payload_sha256": payload_sha256,
            }
        )
        path = self.root / CACHE_SCHEMA / self.preprocessing_sha256 / key[:2] / f"{key}.npz"
        return path, key, payload_sha256

    @staticmethod
    def _arrays(prepared: PreparedEpisode) -> dict[str, np.ndarray]:
        return {
            "processed_values": np.stack([board.values for board in prepared.episode.boards]),
            "processed_axis_mask": np.stack([board.axis_mask for board in prepared.episode.boards]),
            "processed_sample_mask": np.stack([board.sample_mask for board in prepared.episode.boards]),
            "modal_values": prepared.modal_values,
            "modal_axis_mask": prepared.modal_axis_mask,
            "modal_sample_mask": prepared.modal_sample_mask,
            "wideband_features": prepared.wideband_features,
            "wideband_frequency_mask": prepared.wideband_frequency_mask,
            "wideband_cross_valid": prepared.wideband_cross_valid,
            "wideband_frequency_grid_hz": prepared.wideband_frequency_grid_hz,
            "wideband_time_grid_s": prepared.wideband_time_grid_s,
            "physics_features": prepared.physics_features,
            "physics_mask": prepared.physics_mask,
            "profile_feature_values": prepared.profile_feature_values,
            "profile_feature_mask": prepared.profile_feature_mask,
            "sensor_positions": prepared.sensor_positions,
            "orientation_features": prepared.orientation_features,
            "quality_features": prepared.quality_features,
        }

    def _metadata(
        self,
        prepared: PreparedEpisode,
        *,
        key: str,
        payload_sha256: str,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "key": key,
            "episode_payload_sha256": payload_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "fingerprint": self.fingerprint_payload,
            "contains_labels": False,
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
            "physics_feature_names": list(prepared.physics_feature_names),
            "profile_feature_names": list(prepared.profile_feature_names),
            "processed_board_quality": [asdict(board.quality) for board in prepared.episode.boards],
            "processed_board_metadata_updates": [
                {
                    name: board.metadata[name]
                    for name in _DERIVED_BOARD_METADATA_KEYS
                    if name in board.metadata
                }
                for board in prepared.episode.boards
            ],
            "evidence": prepared.evidence,
            "evidence_text": prepared.evidence_text,
            "synchronization_valid": prepared.synchronization_valid,
            "phase_synchronization_reason": prepared.episode.metadata.get("phase_synchronization_reason"),
            "amplitude_scale": prepared.amplitude_scale,
            "amplitude_unit": prepared.amplitude_unit,
            "amplitude_calibrated": prepared.amplitude_calibrated,
        }

    def load(self, episode: SixBoardEpisode) -> PreparedEpisode | None:
        path, key, payload_sha256 = self.path_for(episode)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                observed_names = set(archive.files)
                expected_names = {*_ARRAY_NAMES, _METADATA_KEY}
                if observed_names != expected_names:
                    return None
                metadata_bytes = np.asarray(archive[_METADATA_KEY], dtype=np.uint8)
                metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
                if (
                    metadata.get("schema") != CACHE_SCHEMA
                    or metadata.get("key") != key
                    or metadata.get("episode_payload_sha256") != payload_sha256
                    or metadata.get("preprocessing_sha256") != self.preprocessing_sha256
                    or metadata.get("contains_labels") is not False
                ):
                    return None
                arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}
            array_metadata = metadata.get("arrays")
            if not isinstance(array_metadata, dict):
                return None
            for name, value in arrays.items():
                expected = array_metadata.get(name)
                if not isinstance(expected, dict):
                    return None
                if (
                    expected.get("dtype") != str(value.dtype)
                    or expected.get("shape") != list(value.shape)
                    or expected.get("sha256") != _array_sha256(value)
                ):
                    return None
            qualities = metadata["processed_board_quality"]
            board_metadata_updates = metadata["processed_board_metadata_updates"]
            if len(qualities) != 6 or len(board_metadata_updates) != 6:
                return None
            processed_boards = []
            for index, board in enumerate(episode.boards):
                quality_payload = dict(qualities[index])
                quality_payload["axes_present"] = tuple(
                    bool(value) for value in quality_payload["axes_present"]
                )
                processed_boards.append(
                    replace(
                        board,
                        values=arrays["processed_values"][index],
                        axis_mask=arrays["processed_axis_mask"][index],
                        sample_mask=arrays["processed_sample_mask"][index],
                        quality=QualityReport(**quality_payload),
                        metadata={
                            **board.metadata,
                            **dict(board_metadata_updates[index]),
                        },
                    )
                )
            processed_episode = replace(episode, boards=processed_boards)
            processed_episode.metadata = {
                **episode.metadata,
                "phase_synchronized": bool(metadata["synchronization_valid"]),
                "phase_synchronization_reason": metadata["phase_synchronization_reason"],
            }
            prepared = PreparedEpisode(
                episode=processed_episode,
                modal_values=arrays["modal_values"],
                modal_axis_mask=arrays["modal_axis_mask"],
                modal_sample_mask=arrays["modal_sample_mask"],
                wideband_features=arrays["wideband_features"],
                wideband_frequency_mask=arrays["wideband_frequency_mask"],
                wideband_cross_valid=arrays["wideband_cross_valid"],
                wideband_frequency_grid_hz=arrays["wideband_frequency_grid_hz"],
                wideband_time_grid_s=arrays["wideband_time_grid_s"],
                physics_features=arrays["physics_features"],
                physics_mask=arrays["physics_mask"],
                physics_feature_names=tuple(metadata["physics_feature_names"]),
                profile_feature_values=arrays["profile_feature_values"],
                profile_feature_mask=arrays["profile_feature_mask"],
                profile_feature_names=tuple(metadata["profile_feature_names"]),
                sensor_positions=arrays["sensor_positions"],
                orientation_features=arrays["orientation_features"],
                quality_features=arrays["quality_features"],
                evidence=dict(metadata["evidence"]),
                evidence_text=str(metadata["evidence_text"]),
                synchronization_valid=bool(metadata["synchronization_valid"]),
                amplitude_scale=float(metadata["amplitude_scale"]),
                amplitude_unit=str(metadata["amplitude_unit"]),
                amplitude_calibrated=bool(metadata["amplitude_calibrated"]),
            )
            prepared.validate()
            return prepared
        except (DataContractError, EOFError, KeyError, OSError, TypeError, ValueError):
            # A partial/stale/corrupt entry is a miss. The caller recomputes it
            # and atomically replaces the bad file without exposing it to readers.
            return None

    def store(self, episode: SixBoardEpisode, prepared: PreparedEpisode) -> Path:
        prepared.validate()
        path, key, payload_sha256 = self.path_for(episode)
        arrays = self._arrays(prepared)
        metadata = self._metadata(
            prepared,
            key=key,
            payload_sha256=payload_sha256,
            arrays=arrays,
        )
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    **arrays,
                    **{_METADATA_KEY: np.frombuffer(metadata_bytes, dtype=np.uint8)},
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path

    def prepare(self, episode: SixBoardEpisode, *, read_only: bool = True) -> PreparedEpisode:
        cached = self.load(episode)
        if cached is not None:
            return cached
        if read_only:
            path, _, _ = self.path_for(episode)
            raise DataContractError(
                "prepared preprocessing cache is missing or invalid for "
                f"{episode.episode_id}: {path}. Warm TRAIN/VALIDATION before training."
            )
        prepared = self.preprocessor.prepare(episode)
        self.store(episode, prepared)
        return prepared
