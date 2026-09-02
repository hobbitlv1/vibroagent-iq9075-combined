from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import resolve_from_config
from ..contracts import BoardWindow, SixBoardEpisode
from ..exceptions import DataContractError
from .orientation import apply_orientation, orientation_feature_vector
from .physics import PhysicsFeatureExtractor, PhysicsFeatureResult
from .profile import HealthyProfile
from .quality import assess_quality
from .resample import resample_masked_signal
from .spectral import WidebandFeatureResult, compute_wideband_features
from .units import acceleration_to_analysis_units


@dataclass(frozen=True, slots=True)
class MultiResolutionSettings:
    window_seconds: float
    modal_sample_rate_hz: float
    minimum_valid_fraction: float
    remove_linear_trend: bool
    robust_scale_floor_g: float
    sensor_useful_band_hz: float
    maximum_phase_skew_ms: float
    wideband_max_frequency_hz: float
    wideband_frequency_bins: int
    wideband_time_bins: int
    wideband_frame_seconds: float
    wideband_overlap_fraction: float
    wideband_max_nfft: int
    physics_bands_hz: tuple[tuple[float, float], ...]
    dominant_peaks_per_axis: int
    modal_peak_count: int
    modal_band_hz: tuple[float, float]
    welch_segment_seconds: float
    minimum_peak_prominence_db: float
    healthy_profile_path: Path | None
    require_healthy_profile: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MultiResolutionSettings":
        model = config.get("model", {})
        data = config.get("data", {})
        multi = model.get("multiresolution", {})
        modal = multi.get("modal", {})
        wideband = multi.get("wideband", {})
        physics = multi.get("physics", {})
        profile_value = physics.get("healthy_profile_path")
        profile_path = resolve_from_config(config, profile_value) if profile_value else None
        bands = tuple(
            (float(pair[0]), float(pair[1]))
            for pair in physics.get(
                "bands_hz",
                [
                    [0.1, 1.0],
                    [1.0, 5.0],
                    [5.0, 20.0],
                    [20.0, 50.0],
                    [50.0, 200.0],
                    [200.0, 1000.0],
                    [1000.0, 3000.0],
                    [3000.0, 6000.0],
                ],
            )
        )
        return cls(
            window_seconds=float(data.get("window_seconds", model.get("window_seconds", 10.0))),
            modal_sample_rate_hz=float(
                modal.get("sample_rate_hz", 1024.0)
            ),
            minimum_valid_fraction=float(data.get("minimum_valid_fraction", 0.98)),
            remove_linear_trend=bool(multi.get("remove_linear_trend", True)),
            robust_scale_floor_g=float(multi.get("robust_scale_floor_g", 1e-7)),
            sensor_useful_band_hz=float(wideband.get("sensor_useful_band_hz", 6000.0)),
            maximum_phase_skew_ms=float(wideband.get("maximum_phase_skew_ms", 0.05)),
            wideband_max_frequency_hz=float(wideband.get("max_frequency_hz", 6000.0)),
            wideband_frequency_bins=int(wideband.get("frequency_bins", 192)),
            wideband_time_bins=int(wideband.get("time_bins", 64)),
            wideband_frame_seconds=float(wideband.get("frame_seconds", 0.08)),
            wideband_overlap_fraction=float(wideband.get("overlap_fraction", 0.75)),
            wideband_max_nfft=int(wideband.get("max_nfft", 8192)),
            physics_bands_hz=bands,
            dominant_peaks_per_axis=int(physics.get("dominant_peaks_per_axis", 3)),
            modal_peak_count=int(physics.get("modal_peak_count", 3)),
            modal_band_hz=tuple(float(value) for value in physics.get("modal_band_hz", [0.1, 100.0])),
            welch_segment_seconds=float(physics.get("welch_segment_seconds", 2.0)),
            minimum_peak_prominence_db=float(physics.get("minimum_peak_prominence_db", 3.0)),
            healthy_profile_path=profile_path,
            require_healthy_profile=bool(physics.get("require_healthy_profile", False)),
        )


@dataclass(slots=True)
class PreparedEpisode:
    episode: SixBoardEpisode
    modal_values: np.ndarray
    modal_axis_mask: np.ndarray
    modal_sample_mask: np.ndarray
    wideband_features: np.ndarray
    wideband_frequency_mask: np.ndarray
    wideband_cross_valid: np.ndarray
    wideband_frequency_grid_hz: np.ndarray
    wideband_time_grid_s: np.ndarray
    physics_features: np.ndarray
    physics_mask: np.ndarray
    physics_feature_names: tuple[str, ...]
    profile_feature_values: np.ndarray
    profile_feature_mask: np.ndarray
    profile_feature_names: tuple[str, ...]
    sensor_positions: np.ndarray
    orientation_features: np.ndarray
    quality_features: np.ndarray
    evidence: dict[str, Any]
    evidence_text: str
    synchronization_valid: bool
    amplitude_scale: float
    amplitude_unit: str
    amplitude_calibrated: bool

    def validate(self) -> None:
        self.episode.validate(require_label=False)
        if self.modal_values.ndim != 3 or self.modal_values.shape[:2] != (6, 3):
            raise DataContractError("modal_values must have shape [6,3,samples]")
        if self.modal_axis_mask.shape != (6, 3):
            raise DataContractError("modal_axis_mask must have shape [6,3]")
        if self.modal_sample_mask.shape != (6, self.modal_values.shape[-1]):
            raise DataContractError("modal_sample_mask shape mismatch")
        if self.wideband_features.ndim != 5 or self.wideband_features.shape[:2] != (6, 3):
            raise DataContractError("wideband_features must have shape [6,3,channels,frequency,time]")
        if self.wideband_frequency_mask.shape != (6, self.wideband_features.shape[-2]):
            raise DataContractError("wideband_frequency_mask shape mismatch")
        if self.wideband_cross_valid.shape != (6, 3):
            raise DataContractError("wideband_cross_valid must have shape [6,3]")
        if self.wideband_frequency_grid_hz.shape != (self.wideband_features.shape[-2],):
            raise DataContractError("wideband frequency grid shape mismatch")
        if self.wideband_time_grid_s.shape != (self.wideband_features.shape[-1],):
            raise DataContractError("wideband time grid shape mismatch")
        if self.physics_features.shape != self.physics_mask.shape:
            raise DataContractError("physics feature/mask shape mismatch")
        if self.physics_features.shape != (6, len(self.physics_feature_names)):
            raise DataContractError("physics feature schema mismatch")
        if self.profile_feature_values.shape != self.profile_feature_mask.shape:
            raise DataContractError("profile feature/mask shape mismatch")
        if self.profile_feature_values.shape != (6, len(self.profile_feature_names)):
            raise DataContractError("profile feature schema mismatch")
        if self.sensor_positions.shape != (6, 3):
            raise DataContractError("sensor_positions must have shape [6,3]")
        if self.orientation_features.shape != (6, 11):
            raise DataContractError("orientation_features must have shape [6,11]")
        if self.quality_features.shape != (6, 13):
            raise DataContractError("quality_features must have shape [6,13]")
        if not np.isfinite(self.modal_values).all():
            raise DataContractError("modal_values contain non-finite data")
        if not np.isfinite(self.wideband_features).all():
            raise DataContractError("wideband_features contain non-finite data")
        if not np.isfinite(self.physics_features).all():
            raise DataContractError("physics_features contain non-finite data")


@dataclass(frozen=True, slots=True)
class PreparedEpisodeContext:
    """The small subset of preparation metadata needed by Gemma prompting."""

    episode: SixBoardEpisode
    evidence_text: str
    evidence: dict[str, Any]


@dataclass(slots=True)
class PreparedEpisodeBatch:
    """A worker-prepared batch with CPU/GPU tensors and prompt metadata.

    DataLoader workers construct this object after the expensive NumPy/SciPy
    preprocessing. PyTorch and Accelerate recognise ``pin_memory`` and ``to``
    methods, so the stacked tensors can move through shared memory and then to
    the accelerator without re-running preprocessing in the training process.
    """

    tensors: dict[str, torch.Tensor]
    contexts: list[PreparedEpisodeContext]
    prepared: list[PreparedEpisode] | None = None

    def __len__(self) -> int:
        return len(self.contexts)

    @property
    def episodes(self) -> list[SixBoardEpisode]:
        return [item.episode for item in self.contexts]

    @property
    def evidence_texts(self) -> list[str]:
        return [item.evidence_text for item in self.contexts]

    @property
    def evidences(self) -> list[dict[str, Any]]:
        return [item.evidence for item in self.contexts]

    def require_full_prepared(self) -> list[PreparedEpisode]:
        if self.prepared is None:
            raise DataContractError(
                "this worker-prepared training batch does not retain full per-episode arrays"
            )
        return self.prepared

    def pin_memory(self) -> "PreparedEpisodeBatch":
        return PreparedEpisodeBatch(
            tensors={name: tensor.pin_memory() for name, tensor in self.tensors.items()},
            contexts=self.contexts,
            prepared=self.prepared,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "PreparedEpisodeBatch":
        target = torch.device(device)
        if all(tensor.device == target for tensor in self.tensors.values()):
            return self
        return PreparedEpisodeBatch(
            tensors={
                name: tensor.to(target, non_blocking=non_blocking)
                for name, tensor in self.tensors.items()
            },
            contexts=self.contexts,
            prepared=self.prepared,
        )


class MultiResolutionEpisodePreprocessor:
    """Prepare native-rate, modal-rate, and physics views without losing XYZ axes."""

    def __init__(self, settings: MultiResolutionSettings) -> None:
        self.settings = settings
        healthy_profile: HealthyProfile | None = None
        if settings.healthy_profile_path is not None and settings.healthy_profile_path.exists():
            healthy_profile = HealthyProfile.load(settings.healthy_profile_path)
        elif settings.require_healthy_profile:
            raise DataContractError(
                f"Required healthy profile does not exist: {settings.healthy_profile_path}"
            )
        self.physics_extractor = PhysicsFeatureExtractor(
            bands_hz=settings.physics_bands_hz,
            dominant_peaks_per_axis=settings.dominant_peaks_per_axis,
            modal_peak_count=settings.modal_peak_count,
            modal_band_hz=settings.modal_band_hz,
            welch_segment_seconds=settings.welch_segment_seconds,
            minimum_peak_prominence_db=settings.minimum_peak_prominence_db,
            max_frequency_hz=settings.wideband_max_frequency_hz,
            healthy_profile=healthy_profile,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MultiResolutionEpisodePreprocessor":
        return cls(MultiResolutionSettings.from_config(config))

    @staticmethod
    def _detrend_masked(values: np.ndarray, sample_mask: np.ndarray, axis_mask: np.ndarray, linear: bool) -> np.ndarray:
        output = np.asarray(values, dtype=np.float32).copy()
        valid = np.asarray(sample_mask, dtype=bool)
        present = np.asarray(axis_mask, dtype=bool)
        sample_index = np.arange(output.shape[-1], dtype=np.float64)
        for axis in range(3):
            if not present[axis] or valid.sum() < 2:
                output[axis] = 0.0
                continue
            axis_values = output[axis].astype(np.float64)
            axis_values -= float(axis_values[valid].mean())
            if linear and valid.sum() >= 3:
                coefficient = np.polyfit(sample_index[valid], axis_values[valid], 1)
                axis_values -= np.polyval(coefficient, sample_index)
            axis_values[~valid] = 0.0
            output[axis] = axis_values.astype(np.float32)
        return output

    def _amplitude_scale(self, values: np.ndarray, masks: np.ndarray, axes: np.ndarray) -> float:
        collected: list[np.ndarray] = []
        for board in range(6):
            if axes[board].any() and masks[board].any():
                collected.append(np.abs(values[board, axes[board]][:, masks[board]]).reshape(-1))
        if not collected:
            return self.settings.robust_scale_floor_g
        combined = np.concatenate(collected)
        median = float(np.median(combined))
        mad = float(np.median(np.abs(combined - median)))
        percentile = float(np.quantile(combined, 0.95))
        return max(1.4826 * mad, percentile / 3.0, self.settings.robust_scale_floor_g)

    def _phase_synchronization_valid(self, episode: SixBoardEpisode, sample_masks: np.ndarray) -> tuple[bool, str]:
        explicit = episode.metadata.get("phase_synchronized")
        if explicit is not True:
            return False, "phase_synchronization_not_explicitly_verified"
        if any(abs(float(board.quality.clock_skew_ms)) > self.settings.maximum_phase_skew_ms for board in episode.boards):
            return False, "clock_skew_exceeds_phase_limit"
        if any(float(mask.mean()) < self.settings.minimum_valid_fraction for mask in sample_masks):
            return False, "missing_samples_prevent_phase_comparison"
        verification_id = episode.metadata.get("synchronization_verification_id")
        if not verification_id:
            return False, "synchronization_verification_id_missing"
        return True, "sample_level_phase_synchronization_verified"

    def prepare(self, episode: SixBoardEpisode) -> PreparedEpisode:
        episode.validate(require_label=False)
        source_rates = {round(float(board.sample_rate_hz), 9) for board in episode.boards}
        lengths = {int(board.values.shape[-1]) for board in episode.boards}
        if len(source_rates) != 1 or len(lengths) != 1:
            raise DataContractError("multi-resolution preparation requires aligned boards with one sample rate and length")
        source_rate = float(episode.boards[0].sample_rate_hz)
        expected_duration = episode.boards[0].values.shape[-1] / source_rate
        if not np.isclose(expected_duration, self.settings.window_seconds, rtol=0.0, atol=2.0 / source_rate):
            raise DataContractError(
                f"episode duration {expected_duration:.6f}s does not match configured {self.settings.window_seconds:.6f}s"
            )

        native_values = np.zeros((6, 3, episode.boards[0].values.shape[-1]), dtype=np.float32)
        sample_masks = np.zeros((6, episode.boards[0].values.shape[-1]), dtype=bool)
        axis_masks = np.zeros((6, 3), dtype=bool)
        positions = np.full((6, 3), np.nan, dtype=np.float32)
        orientations = np.zeros((6, 11), dtype=np.float32)
        usable_bands = np.zeros(6, dtype=np.float32)
        amplitude_calibration = np.zeros(6, dtype=bool)
        amplitude_units: list[str] = []
        processed_boards: list[BoardWindow] = []

        for board_index, board in enumerate(episode.boards):
            unit = str(board.metadata.get("unit", "g"))
            analysis_values, calibrated, canonical_unit = acceleration_to_analysis_units(board.values, unit)
            declared_calibration = board.metadata.get("amplitude_calibrated")
            if declared_calibration is not None and bool(declared_calibration) != calibrated:
                raise DataContractError(
                    f"{board.board_id}: unit {unit!r} conflicts with amplitude_calibrated={declared_calibration!r}"
                )
            amplitude_calibration[board_index] = calibrated
            amplitude_units.append(canonical_unit)
            orientation = apply_orientation(analysis_values, board.axis_mask, board.orientation_matrix)
            processed = self._detrend_masked(
                orientation.values,
                board.sample_mask,
                orientation.axis_mask,
                self.settings.remove_linear_trend,
            )
            native_values[board_index] = processed
            sample_masks[board_index] = board.sample_mask.astype(bool)
            axis_masks[board_index] = orientation.axis_mask
            if board.sensor_position is not None:
                positions[board_index] = np.asarray(board.sensor_position, dtype=np.float32)
            orientations[board_index] = orientation_feature_vector(
                orientation.matrix,
                calibrated=orientation.calibrated,
                applied=orientation.applied,
            )
            declared_band = board.quality.usable_band_hz
            if declared_band is None:
                declared_band = board.metadata.get("sensor_useful_band_hz", source_rate / 2.0)
            usable_band = min(
                float(declared_band),
                source_rate / 2.0,
                self.settings.sensor_useful_band_hz,
                self.settings.wideband_max_frequency_hz,
            )
            usable_bands[board_index] = max(0.0, usable_band)
            quality = assess_quality(
                processed,
                board.sample_mask,
                orientation.axis_mask,
                sample_rate_hz=source_rate,
                original_sample_rate_hz=source_rate,
                clipping_abs_g=board.metadata.get("clipping_abs_g"),
                packet_loss_count=board.quality.packet_loss_count,
                clock_skew_ms=board.quality.clock_skew_ms,
                notes=[*board.quality.notes, orientation.reason],
            )
            quality.usable_band_hz = usable_band
            if quality.valid_fraction < self.settings.minimum_valid_fraction:
                quality.notes.append("below_minimum_valid_fraction")
            processed_boards.append(
                replace(
                    board,
                    values=processed,
                    axis_mask=orientation.axis_mask,
                    quality=quality,
                    metadata={
                        **board.metadata,
                        "unit": "g" if calibrated else "release_native",
                        "source_unit": canonical_unit,
                        "amplitude_calibrated": calibrated,
                        "source_sample_rate_hz": source_rate,
                        "orientation_calibrated": orientation.calibrated,
                        "orientation_applied": orientation.applied,
                        "orientation_reason": orientation.reason,
                        "sensor_useful_band_hz": usable_band,
                    },
                )
            )

        processed_episode = replace(episode, boards=processed_boards)
        if len(set(amplitude_units)) != 1 or len(set(amplitude_calibration.tolist())) != 1:
            raise DataContractError(
                "all six boards in one episode must share one amplitude calibration regime and source unit"
            )
        episode_amplitude_calibrated = bool(amplitude_calibration.all())
        episode_amplitude_unit = "g" if episode_amplitude_calibrated else "release_native"
        phase_valid, phase_reason = self._phase_synchronization_valid(processed_episode, sample_masks)
        processed_episode.metadata = {
            **processed_episode.metadata,
            "phase_synchronized": phase_valid,
            "phase_synchronization_reason": phase_reason,
        }

        amplitude_scale = self._amplitude_scale(native_values, sample_masks, axis_masks)
        normalized_native = (native_values / amplitude_scale).astype(np.float32)
        modal_samples = int(round(self.settings.modal_sample_rate_hz * self.settings.window_seconds))
        modal_values_g = np.zeros((6, 3, modal_samples), dtype=np.float32)
        modal_normalized = np.zeros_like(modal_values_g)
        modal_masks = np.zeros((6, modal_samples), dtype=bool)
        for board in range(6):
            physical_modal, modal_mask = resample_masked_signal(
                native_values[board],
                sample_masks[board],
                source_rate,
                self.settings.modal_sample_rate_hz,
                target_samples=modal_samples,
            )
            physical_modal *= axis_masks[board, :, None]
            modal_values_g[board] = physical_modal
            modal_normalized[board] = physical_modal / amplitude_scale
            modal_masks[board] = modal_mask

        wideband: WidebandFeatureResult = compute_wideband_features(
            normalized_native,
            sample_masks,
            axis_masks,
            sample_rate_hz=source_rate,
            window_seconds=self.settings.window_seconds,
            max_frequency_hz=self.settings.wideband_max_frequency_hz,
            frequency_bins=self.settings.wideband_frequency_bins,
            time_bins=self.settings.wideband_time_bins,
            frame_seconds=self.settings.wideband_frame_seconds,
            overlap_fraction=self.settings.wideband_overlap_fraction,
            max_nfft=self.settings.wideband_max_nfft,
            synchronized=phase_valid,
            usable_band_hz=usable_bands,
        )
        physics: PhysicsFeatureResult = self.physics_extractor.extract(
            native_values_g=native_values,
            native_sample_masks=sample_masks,
            axis_masks=axis_masks,
            sample_rate_hz=source_rate,
            modal_values_g=modal_values_g,
            modal_sample_masks=modal_masks,
            modal_rate_hz=self.settings.modal_sample_rate_hz,
            usable_band_hz=usable_bands,
            synchronized=phase_valid,
            provenance=episode.provenance,
            amplitude_calibrated=amplitude_calibration,
        )

        quality_features = np.zeros((6, 13), dtype=np.float32)
        for board_index, board in enumerate(processed_boards):
            quality_features[board_index] = np.asarray(
                [
                    board.quality.valid_fraction,
                    board.quality.missing_fraction,
                    board.quality.clipped_fraction,
                    float(board.quality.timestamp_monotonic),
                    min(abs(board.quality.clock_skew_ms) / 100.0, 10.0),
                    float(sum(board.quality.axes_present)) / 3.0,
                    min(float(board.quality.usable_band_hz or 0.0) / self.settings.wideband_max_frequency_hz, 1.0),
                    min(np.log1p(board.quality.packet_loss_count) / 10.0, 10.0),
                    min(source_rate / 26700.0, 4.0),
                    orientations[board_index, -2],
                    orientations[board_index, -1],
                    float(phase_valid),
                    float(amplitude_calibration[board_index]),
                ],
                dtype=np.float32,
            )

        evidence = {
            **physics.evidence,
            "preprocessing": {
                "source_sample_rate_hz": source_rate,
                "modal_sample_rate_hz": self.settings.modal_sample_rate_hz,
                "wideband_max_frequency_hz": self.settings.wideband_max_frequency_hz,
                "amplitude_scale": amplitude_scale,
                "amplitude_unit": episode_amplitude_unit,
                "amplitude_calibrated": episode_amplitude_calibrated,
                "phase_synchronization_reason": phase_reason,
                "orientation": {
                    ("reference" if index == 0 else f"target_{index}"): {
                        "calibrated": bool(orientations[index, -2]),
                        "rotated_to_building_frame": bool(orientations[index, -1]),
                    }
                    for index in range(6)
                },
            },
        }
        prepared = PreparedEpisode(
            episode=processed_episode,
            modal_values=modal_normalized,
            modal_axis_mask=axis_masks,
            modal_sample_mask=modal_masks,
            wideband_features=wideband.features,
            wideband_frequency_mask=wideband.frequency_mask,
            wideband_cross_valid=wideband.cross_spectral_valid,
            wideband_frequency_grid_hz=wideband.frequency_grid_hz,
            wideband_time_grid_s=wideband.time_grid_s,
            physics_features=physics.values,
            physics_mask=physics.mask,
            physics_feature_names=physics.feature_names,
            profile_feature_values=physics.profile_feature_values,
            profile_feature_mask=physics.profile_feature_mask,
            profile_feature_names=physics.profile_feature_names,
            sensor_positions=positions,
            orientation_features=orientations,
            quality_features=quality_features,
            evidence=evidence,
            evidence_text=physics.evidence_text,
            synchronization_valid=phase_valid,
            amplitude_scale=amplitude_scale,
            amplitude_unit=episode_amplitude_unit,
            amplitude_calibrated=episode_amplitude_calibrated,
        )
        prepared.validate()
        return prepared

    def prepare_many(self, episodes: Sequence[SixBoardEpisode | PreparedEpisode]) -> list[PreparedEpisode]:
        output: list[PreparedEpisode] = []
        for item in episodes:
            if isinstance(item, PreparedEpisode):
                item.validate()
                output.append(item)
            else:
                output.append(self.prepare(item))
        return output


def stack_prepared_episode_batch(
    prepared: Sequence[PreparedEpisode],
    *,
    device: torch.device | str | None = None,
    retain_prepared: bool = True,
) -> PreparedEpisodeBatch:
    """Stack already prepared episodes without executing signal preprocessing."""

    items = list(prepared)
    if not items:
        raise DataContractError("cannot stack an empty prepared-episode batch")
    for item in items:
        item.validate()
    physics_names = {item.physics_feature_names for item in items}
    if len(physics_names) != 1:
        raise DataContractError("physics feature schema differs across a batch")
    tensors = {
        "modal_values": torch.from_numpy(np.stack([item.modal_values for item in items])),
        "modal_axis_mask": torch.from_numpy(np.stack([item.modal_axis_mask for item in items])),
        "modal_sample_mask": torch.from_numpy(np.stack([item.modal_sample_mask for item in items])),
        "wideband_features": torch.from_numpy(np.stack([item.wideband_features for item in items])),
        "wideband_frequency_mask": torch.from_numpy(
            np.stack([item.wideband_frequency_mask for item in items])
        ),
        "wideband_cross_valid": torch.from_numpy(
            np.stack([item.wideband_cross_valid for item in items])
        ),
        "physics_features": torch.from_numpy(np.stack([item.physics_features for item in items])),
        "physics_mask": torch.from_numpy(np.stack([item.physics_mask for item in items])),
        "sensor_positions": torch.from_numpy(np.stack([item.sensor_positions for item in items])),
        "orientation_features": torch.from_numpy(
            np.stack([item.orientation_features for item in items])
        ),
        "quality_features": torch.from_numpy(np.stack([item.quality_features for item in items])),
    }
    contexts = [
        PreparedEpisodeContext(
            episode=item.episode,
            evidence_text=item.evidence_text,
            evidence=item.evidence,
        )
        for item in items
    ]
    batch = PreparedEpisodeBatch(
        tensors=tensors,
        contexts=contexts,
        prepared=items if retain_prepared else None,
    )
    return batch if device is None else batch.to(device, non_blocking=True)


def prepared_episode_batch(
    episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
    *,
    preprocessor: MultiResolutionEpisodePreprocessor | None,
    device: torch.device | str,
) -> PreparedEpisodeBatch:
    """Return a device batch, reusing worker-prepared tensors when available."""

    if isinstance(episodes, PreparedEpisodeBatch):
        return episodes.to(device, non_blocking=True)
    if preprocessor is None:
        if not all(isinstance(item, PreparedEpisode) for item in episodes):
            raise DataContractError(
                "a multi-resolution preprocessor is required for raw SixBoardEpisode inputs"
            )
        prepared = [item for item in episodes if isinstance(item, PreparedEpisode)]
    else:
        prepared = preprocessor.prepare_many(episodes)
    return stack_prepared_episode_batch(prepared, device=device)
