from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import (
    BoardWindow,
    EpisodeLabel,
    EpisodeProvenance,
    QualityReport,
    SixBoardEpisode,
    TargetLabel,
)
from ..utils import atomic_write_json


def _board_metadata(board: BoardWindow) -> dict[str, Any]:
    return {
        "board_id": board.board_id,
        "sample_rate_hz": float(board.sample_rate_hz),
        "start_time_s": float(board.start_time_s),
        "role": board.role,
        "target_index": board.target_index,
        "sensor_position": list(board.sensor_position) if board.sensor_position is not None else None,
        "orientation_matrix": (
            np.asarray(board.orientation_matrix, dtype=np.float32).tolist()
            if board.orientation_matrix is not None
            else None
        ),
        "quality": asdict(board.quality),
        "metadata": board.metadata,
    }


def save_episode(episode: SixBoardEpisode, directory: str | Path) -> tuple[Path, Path]:
    episode.validate(require_label=False)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    array_path = destination / f"{episode.episode_id}.npz"
    metadata_path = destination / f"{episode.episode_id}.json"

    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            values=episode.stacked_values(),
            axis_mask=episode.stacked_axis_mask(),
            sample_mask=episode.stacked_sample_mask(),
        )
    temporary.replace(array_path)

    label = None
    if episode.label is not None:
        label = {
            "targets": [
                {
                    "sensor_id": target.sensor_id,
                    "affected": target.affected,
                    "class_name": target.class_name,
                    "severity": target.severity,
                    "source": target.source,
                }
                for target in episode.label.targets
            ],
            "explanation_target": episode.label.explanation_target,
            "global_class_name": episode.label.global_class_name,
            "global_severity": episode.label.global_severity,
            "target_supervision_available": episode.label.target_supervision_available,
            "supervision_scope": episode.label.supervision_scope,
            "focus_target_index": episode.label.focus_target_index,
        }
    atomic_write_json(
        metadata_path,
        {
            "format_version": "2.0",
            "episode_id": episode.episode_id,
            "array_file": array_path.name,
            "boards": [_board_metadata(board) for board in episode.boards],
            "provenance": asdict(episode.provenance),
            "label": label,
            "metadata": episode.metadata,
        },
    )
    return array_path, metadata_path


def load_episode(
    array_path: str | Path,
    metadata_path: str | Path | None = None,
) -> SixBoardEpisode:
    array_source = Path(array_path)
    metadata_source = Path(metadata_path) if metadata_path is not None else array_source.with_suffix(".json")
    payload = json.loads(metadata_source.read_text(encoding="utf-8"))
    with np.load(array_source, allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float32)
        axis_mask = np.asarray(archive["axis_mask"], dtype=bool)
        sample_mask = np.asarray(archive["sample_mask"], dtype=bool)
    if values.ndim != 3 or values.shape[:2] != (6, 3):
        raise ValueError(f"episode values must have shape [6,3,N], got {values.shape}")
    if axis_mask.shape != (6, 3) or sample_mask.shape != (6, values.shape[-1]):
        raise ValueError("serialized episode masks do not match values")

    board_payloads = payload["boards"]
    if len(board_payloads) != 6:
        raise ValueError("serialized episode must describe six boards")
    boards: list[BoardWindow] = []
    for index, item in enumerate(board_payloads):
        quality_payload = dict(item.get("quality") or {})
        if "axes_present" in quality_payload:
            quality_payload["axes_present"] = tuple(bool(value) for value in quality_payload["axes_present"])
        quality = QualityReport(**quality_payload)
        orientation = item.get("orientation_matrix")
        position = item.get("sensor_position")
        boards.append(
            BoardWindow(
                board_id=str(item["board_id"]),
                values=values[index],
                sample_rate_hz=float(item["sample_rate_hz"]),
                start_time_s=float(item["start_time_s"]),
                axis_mask=axis_mask[index],
                sample_mask=sample_mask[index],
                role=str(item["role"]),
                target_index=item.get("target_index"),
                sensor_position=tuple(float(value) for value in position) if position is not None else None,
                orientation_matrix=np.asarray(orientation, dtype=np.float32) if orientation is not None else None,
                quality=quality,
                metadata=dict(item.get("metadata") or {}),
            )
        )
    provenance = EpisodeProvenance(**payload["provenance"])
    label_payload = payload.get("label")
    label = None
    if label_payload is not None:
        label = EpisodeLabel(
            targets=[TargetLabel(**item) for item in label_payload.get("targets", [])],
            explanation_target=str(label_payload.get("explanation_target", "")),
            global_class_name=label_payload.get("global_class_name"),
            global_severity=label_payload.get("global_severity"),
            target_supervision_available=bool(
                label_payload.get("target_supervision_available", True)
            ),
            supervision_scope=str(label_payload.get("supervision_scope", "joint")),
            focus_target_index=label_payload.get("focus_target_index"),
        )
    episode = SixBoardEpisode(
        episode_id=str(payload["episode_id"]),
        boards=boards,
        provenance=provenance,
        label=label,
        metadata=dict(payload.get("metadata") or {}),
    )
    episode.validate(require_label=False)
    return episode
