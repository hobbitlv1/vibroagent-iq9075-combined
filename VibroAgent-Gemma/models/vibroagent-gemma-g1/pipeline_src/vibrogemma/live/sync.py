from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from ..contracts import AXES, BoardWindow, EpisodeProvenance, QualityReport, SixBoardEpisode
from ..exceptions import LiveAcquisitionError
from ..utils import stable_hash
from .ring_buffer import TimestampedRingBuffer


class SynchronizedWindowAssembler:
    """Align six timestamped buffers without claiming sample-level phase sync."""

    def __init__(
        self,
        buffers: dict[str, TimestampedRingBuffer],
        *,
        reference_id: str,
        target_ids: list[str],
        raw_sample_rate_hz: float,
        window_seconds: float,
        maximum_missing_fraction: float,
        max_clock_skew_ms: float,
        clipping_abs_g: float,
        sensor_useful_band_hz: float = 6000.0,
        board_geometry: dict[str, Any] | None = None,
        phase_synchronized: bool = False,
        synchronization_verification_id: str | None = None,
    ) -> None:
        expected = [reference_id, *target_ids]
        if set(expected) != set(buffers):
            raise ValueError("Buffer IDs do not match reference + targets")
        if len(target_ids) != 5:
            raise ValueError("Exactly five target IDs are required")
        if phase_synchronized and not synchronization_verification_id:
            raise ValueError("phase_synchronized=true requires a synchronization_verification_id")
        self.buffers = buffers
        self.reference_id = reference_id
        self.target_ids = target_ids
        self.raw_sample_rate_hz = float(raw_sample_rate_hz)
        self.window_seconds = float(window_seconds)
        self.raw_samples = int(round(raw_sample_rate_hz * window_seconds))
        self.maximum_missing_fraction = float(maximum_missing_fraction)
        self.max_clock_skew_ms = float(max_clock_skew_ms)
        self.clipping_abs_g = float(clipping_abs_g)
        self.sensor_useful_band_hz = min(float(sensor_useful_band_hz), raw_sample_rate_hz / 2.0)
        self.board_geometry = board_geometry or {}
        self.phase_synchronized = bool(phase_synchronized)
        self.synchronization_verification_id = synchronization_verification_id

    def ready(self) -> bool:
        latest = [buffer.latest_timestamp() for buffer in self.buffers.values()]
        earliest = [buffer.earliest_timestamp() for buffer in self.buffers.values()]
        if any(value is None for value in latest) or any(value is None for value in earliest):
            return False
        end = min(float(value) for value in latest if value is not None)
        return all(end - float(value) >= self.window_seconds for value in earliest if value is not None)

    def _geometry(self, board_id: str) -> tuple[tuple[float, float, float] | None, np.ndarray | None, np.ndarray]:
        payload = self.board_geometry.get(board_id, self.board_geometry.get(str(board_id), {}))
        if not isinstance(payload, dict):
            raise LiveAcquisitionError(f"board geometry for {board_id} must be a mapping")
        position_payload = payload.get("position_m")
        position = tuple(float(value) for value in position_payload) if position_payload is not None else None
        orientation_payload = payload.get("orientation_matrix")
        orientation = np.asarray(orientation_payload, dtype=np.float32) if orientation_payload is not None else None
        available_axes = {str(axis).lower() for axis in payload.get("axes_available", AXES)}
        axis_mask = np.asarray([axis in available_axes for axis in AXES], dtype=bool)
        return position, orientation, axis_mask

    def assemble_latest(self) -> SixBoardEpisode:
        latest = {board_id: buffer.latest_timestamp() for board_id, buffer in self.buffers.items()}
        if any(value is None for value in latest.values()):
            raise LiveAcquisitionError("Not every board has data")
        end_time = min(float(value) for value in latest.values() if value is not None)
        skew_ms = {
            board_id: (float(timestamp) - end_time) * 1000.0
            for board_id, timestamp in latest.items()
            if timestamp is not None
        }
        start_time = end_time - self.window_seconds
        ids = [self.reference_id, *self.target_ids]
        boards: list[BoardWindow] = []
        for index, board_id in enumerate(ids):
            sliced = self.buffers[board_id].extract(
                start_time,
                end_time,
                expected_samples=self.raw_samples,
            )
            valid_fraction = float(sliced.sample_mask.mean())
            notes = []
            if 1.0 - valid_fraction > self.maximum_missing_fraction:
                notes.append("excessive_missing_samples")
            if abs(skew_ms[board_id]) > self.max_clock_skew_ms:
                notes.append("excessive_clock_skew")
            if not self.phase_synchronized:
                notes.append("cross_board_phase_not_verified")
            position, orientation, axis_mask = self._geometry(board_id)
            values = sliced.values.copy()
            values[~axis_mask] = 0.0
            boards.append(
                BoardWindow(
                    board_id="reference" if index == 0 else f"target_{index}",
                    values=values,
                    sample_rate_hz=self.raw_sample_rate_hz,
                    start_time_s=start_time,
                    axis_mask=axis_mask,
                    sample_mask=sliced.sample_mask,
                    role="reference" if index == 0 else "target",
                    target_index=None if index == 0 else index,
                    sensor_position=position,
                    orientation_matrix=orientation,
                    quality=QualityReport(
                        valid_fraction=valid_fraction,
                        missing_fraction=1.0 - valid_fraction,
                        clock_skew_ms=skew_ms[board_id],
                        axes_present=tuple(bool(value) for value in axis_mask),
                        usable_band_hz=self.sensor_useful_band_hz,
                        packet_loss_count=sliced.packet_loss_count,
                        notes=notes,
                    ),
                    metadata={
                        "unit": "g",
                        "physical_board_id": board_id,
                        "clipping_abs_g": self.clipping_abs_g,
                        "sensor_useful_band_hz": self.sensor_useful_band_hz,
                    },
                )
            )
        start_utc = datetime.fromtimestamp(start_time, UTC).isoformat()
        episode_id = stable_hash({"live_start": start_time, "ids": ids})[:24]
        episode = SixBoardEpisode(
            episode_id=episode_id,
            boards=boards,
            provenance=EpisodeProvenance(
                dataset_id="live_stwin",
                provenance_type="live_unlabelled",
                source_files=[],
                source_checksums=[],
                licence="local live stream; inference only",
                licence_accepted=True,
                structure_id="live_installation",
                session_id=datetime.now(UTC).strftime("%Y%m%d"),
                experiment_id="live",
                split_group="live",
                derived_operations=["timestamp_window_alignment"],
                local_training_data=True,
            ),
            label=None,
            metadata={
                "start_utc": start_utc,
                "end_time_s": end_time,
                "phase_synchronized": self.phase_synchronized,
                "synchronization_verification_id": self.synchronization_verification_id,
                "host_window_alignment_max_skew_ms": max(abs(value) for value in skew_ms.values()),
            },
        )
        episode.validate(require_label=False)
        return episode
