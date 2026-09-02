from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..data.lumo_geometry import (
    LUMO_REFERENCE_MEASUREMENT_LEVEL,
    LUMO_TARGET_MEASUREMENT_LEVELS,
)


@dataclass(frozen=True)
class DatasetCalibration:
    sample_rate_hz: float
    dominant_frequency_hz: float
    board_rms: tuple[float, float, float, float, float, float]
    stiffness_scale: float
    duration_seconds: float


MEASURED_TRANSFER_CALIBRATION_SCHEMA = "vibrogemma-measured-transfer-calibration-v0.18.1"


@dataclass(frozen=True)
class MeasuredTransferCalibration:
    dataset_id: str
    band_edges_hz: tuple[float, ...]
    gains_by_target: dict[int, tuple[tuple[float, ...], ...]]
    log_gain_mad_by_target: dict[int, tuple[tuple[float, ...], ...]]
    basis_by_target: dict[int, str]
    source_targets_by_target: dict[int, tuple[int, ...]]
    dispersion_estimable_by_target: dict[int, bool]
    geometry_consistency_error_by_target: dict[int, float]
    derivation_contract: dict[str, Any] | None
    healthy_group_count: int
    target_group_counts: dict[int, int]
    source_group_hashes: dict[str, str]
    source_splits: tuple[str, ...] = ("train",)

    def validate(self) -> None:
        band_count = len(self.band_edges_hz) - 1
        if band_count < 1 or any(
            right <= left for left, right in zip(self.band_edges_hz, self.band_edges_hz[1:])
        ):
            raise ValueError("calibration band edges must be strictly increasing")
        if set(self.gains_by_target) != set(range(1, 6)):
            raise ValueError("measured calibration requires target_1..target_5")
        if set(self.basis_by_target) != set(range(1, 6)) or set(
            self.source_targets_by_target
        ) != set(range(1, 6)):
            raise ValueError("calibration target provenance is incomplete")
        if set(self.dispersion_estimable_by_target) != set(range(1, 6)):
            raise ValueError("calibration dispersion provenance is incomplete")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self.geometry_consistency_error_by_target.values()
        ):
            raise ValueError("calibration geometry-consistency error is invalid")
        for target, boards in self.gains_by_target.items():
            if len(boards) != 6 or any(len(values) != band_count for values in boards):
                raise ValueError(f"target_{target} calibration must have shape [6,{band_count}]")
            if not np.isfinite(np.asarray(boards, dtype=np.float64)).all():
                raise ValueError(f"target_{target} calibration contains non-finite gains")
            if np.any(np.asarray(boards, dtype=np.float64) <= 0.0):
                raise ValueError(f"target_{target} calibration contains nonpositive gains")
            mad = self.log_gain_mad_by_target.get(target)
            if mad is None or np.asarray(mad).shape != (6, band_count):
                raise ValueError(f"target_{target} calibration MAD must have shape [6,{band_count}]")
            if not np.isfinite(np.asarray(mad, dtype=np.float64)).all() or np.any(
                np.asarray(mad, dtype=np.float64) < 0.0
            ):
                raise ValueError(f"target_{target} calibration MAD is invalid")
            sources = self.source_targets_by_target[target]
            if not sources or any(source not in {1, 2, 3, 4, 5} for source in sources):
                raise ValueError(f"target_{target} calibration sources are invalid")
            basis = self.basis_by_target[target]
            count = self.target_group_counts[target]
            if basis == "measured_train_groups":
                if sources != (target,) or count < 1:
                    raise ValueError(f"target_{target} measured provenance is invalid")
            elif basis == "derived_same_dataset_ordinal_shift_prior":
                expected = {1: (2,), 4: (3, 5)}.get(target)
                if self.dataset_id != "lumo" or sources != expected or count != 0:
                    raise ValueError(f"target_{target} derived provenance is invalid")
            else:
                raise ValueError(f"target_{target} calibration basis is unknown: {basis}")
        derived_targets = {
            target
            for target, basis in self.basis_by_target.items()
            if basis == "derived_same_dataset_ordinal_shift_prior"
        }
        if derived_targets:
            if derived_targets != {1, 4} or not self.derivation_contract:
                raise ValueError("LUMO derivation contract is incomplete")
            if (
                self.derivation_contract.get("schema")
                != "vibrogemma-lumo-ordinal-target-derivation-v0.18.1"
                or not self.derivation_contract.get("sha256")
            ):
                raise ValueError("LUMO derivation contract identity is invalid")
            contract_payload = dict(self.derivation_contract)
            observed_contract_hash = str(contract_payload.pop("sha256"))
            expected_contract_hash = hashlib.sha256(
                json.dumps(contract_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if observed_contract_hash != expected_contract_hash:
                raise ValueError("LUMO derivation contract hash is invalid")
            if set(self.geometry_consistency_error_by_target) != {2, 3, 5}:
                raise ValueError("LUMO geometry-consistency coverage is incomplete")
        elif self.derivation_contract is not None:
            raise ValueError("measured-only calibration cannot carry a derivation contract")
        if self.source_splits != ("train",):
            raise ValueError("measured calibration must be fitted from train only")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MEASURED_TRANSFER_CALIBRATION_SCHEMA,
            "dataset_id": self.dataset_id,
            "method": "group_median_common_mode_relative_log_band_magnitude_v1",
            "phase_calibrated": False,
            "band_edges_hz": list(self.band_edges_hz),
            "gains_by_target": {
                f"target_{target}": [list(values) for values in boards]
                for target, boards in sorted(self.gains_by_target.items())
            },
            "log_gain_mad_by_target": {
                f"target_{target}": [list(values) for values in boards]
                for target, boards in sorted(self.log_gain_mad_by_target.items())
            },
            "basis_by_target": {
                f"target_{target}": basis
                for target, basis in sorted(self.basis_by_target.items())
            },
            "source_targets_by_target": {
                f"target_{target}": list(sources)
                for target, sources in sorted(self.source_targets_by_target.items())
            },
            "dispersion_estimable_by_target": {
                f"target_{target}": value
                for target, value in sorted(self.dispersion_estimable_by_target.items())
            },
            "geometry_consistency_error_by_target": {
                f"target_{target}": value
                for target, value in sorted(self.geometry_consistency_error_by_target.items())
            },
            "derivation_contract": self.derivation_contract,
            "healthy_group_count": self.healthy_group_count,
            "target_group_counts": {
                f"target_{target}": count
                for target, count in sorted(self.target_group_counts.items())
            },
            "source_group_hashes": dict(sorted(self.source_group_hashes.items())),
            "source_splits": list(self.source_splits),
        }


def _sample_axis(values: np.ndarray) -> int:
    if values.ndim == 1:
        return 0
    if values.ndim != 2:
        raise ValueError(f"board waveform must be 1D/2D, observed {values.shape}")
    # Structural windows have many more samples than axes.
    return int(np.argmax(values.shape))


def canonical_time_axis(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, None], False
    axis = _sample_axis(values)
    if axis == 0:
        return values, False
    return values.T, True


def restore_time_axis(values: np.ndarray, transposed: bool, original_ndim: int) -> np.ndarray:
    if original_ndim == 1:
        return values[:, 0]
    return values.T if transposed else values


def estimate_dominant_frequency(values: np.ndarray, sample_rate_hz: float, max_hz: float = 100.0) -> float:
    series, _ = canonical_time_axis(np.asarray(values, dtype=np.float64))
    series = series - np.median(series, axis=0, keepdims=True)
    if series.shape[0] < 32 or np.allclose(series, 0.0):
        return min(5.0, sample_rate_hz / 8.0)
    spectrum = np.sum(
        np.abs(np.fft.rfft(series * np.hanning(series.shape[0])[:, None], axis=0)) ** 2,
        axis=1,
    )
    frequencies = np.fft.rfftfreq(series.shape[0], d=1.0 / sample_rate_hz)
    valid = (frequencies >= 0.25) & (frequencies <= min(max_hz, sample_rate_hz / 2.0))
    if not np.any(valid):
        return min(5.0, sample_rate_hz / 8.0)
    selected = np.flatnonzero(valid)
    return float(frequencies[selected[int(np.argmax(spectrum[valid]))]])


def estimate_dataset_calibration(
    *,
    board_values: Sequence[np.ndarray],
    sample_rate_hz: float,
    duration_seconds: float,
    max_frequency_hz: float = 100.0,
) -> DatasetCalibration:
    if len(board_values) != 6:
        raise ValueError("calibration requires one reference plus five targets")
    rms_values: list[float] = []
    peaks: list[float] = []
    for board in board_values:
        matrix, _ = canonical_time_axis(np.asarray(board, dtype=np.float64))
        rms_values.append(float(np.sqrt(np.mean(np.square(matrix))) + 1e-12))
        peaks.append(
            estimate_dominant_frequency(
                matrix, sample_rate_hz, max_hz=max_frequency_hz
            )
        )
    dominant = float(np.median(peaks))
    # For unit median mass, k≈(2πf)^2.  Randomized masses preserve the scale.
    stiffness_scale = float((2.0 * math.pi * max(dominant, 0.25)) ** 2)
    return DatasetCalibration(
        sample_rate_hz=float(sample_rate_hz),
        dominant_frequency_hz=dominant,
        board_rms=tuple(rms_values),
        stiffness_scale=stiffness_scale,
        duration_seconds=float(duration_seconds),
    )


def _calibration_band_edges(max_frequency_hz: float) -> tuple[float, ...]:
    if max_frequency_hz <= 0.5:
        raise ValueError("measured calibration requires max_frequency_hz > 0.5")
    interior = (0.25, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0, 150.0)
    return tuple([value for value in interior if value < max_frequency_hz] + [float(max_frequency_hz)])


def _spatial_log_band_signature(
    board_values: Sequence[np.ndarray],
    sample_rate_hz: float,
    band_edges_hz: Sequence[float],
) -> np.ndarray:
    if len(board_values) != 6:
        raise ValueError("measured calibration requires one reference plus five targets")
    if sample_rate_hz / 2.0 < float(band_edges_hz[-1]):
        raise ValueError("measured calibration band exceeds the waveform Nyquist limit")
    energies: list[list[float]] = []
    for board in board_values:
        matrix, _ = canonical_time_axis(np.asarray(board, dtype=np.float64))
        if matrix.shape[0] < 32:
            raise ValueError("measured calibration waveform is too short")
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
        spectrum = np.fft.rfft(matrix * np.hanning(matrix.shape[0])[:, None], axis=0)
        power = np.sum(np.abs(spectrum) ** 2, axis=1)
        frequencies = np.fft.rfftfreq(matrix.shape[0], d=1.0 / sample_rate_hz)
        board_energy: list[float] = []
        for band_index, (low, high) in enumerate(
            zip(band_edges_hz, band_edges_hz[1:])
        ):
            upper = frequencies <= high if band_index == len(band_edges_hz) - 2 else frequencies < high
            selected = (frequencies >= low) & upper
            if not np.any(selected):
                raise ValueError(f"measured calibration band [{low}, {high}] has no FFT bins")
            board_energy.append(float(np.sqrt(np.mean(power[selected]) + 1e-18)))
        energies.append(board_energy)
    log_energy = np.log(np.maximum(np.asarray(energies, dtype=np.float64), 1e-12))
    return log_energy - np.median(log_energy, axis=0, keepdims=True)


def fit_measured_transfer_calibration(
    *,
    dataset_id: str,
    examples: Iterable[tuple[int | None, str, Sequence[np.ndarray], float]],
    max_frequency_hz: float = 100.0,
    minimum_target_groups: int = 2,
    gain_clip: tuple[float, float] = (2.0 / 3.0, 1.5),
    missing_target_policy: str = "fail",
    expected_measured_targets: Sequence[int] | None = None,
    maximum_geometry_consistency_error: float = 0.35,
) -> MeasuredTransferCalibration:
    """Fit train-only target×board magnitude calibration from independent groups.

    ``examples`` contain ``(affected_target, complete_group, six_boards, sample_rate)``.
    Affected target ``None`` denotes a genuinely healthy target-supervised episode.
    Windows are median-aggregated inside each complete group before any target profile
    is fitted, so repeated windows cannot masquerade as independent experiments.
    """

    if minimum_target_groups < 1:
        raise ValueError("minimum_target_groups must be positive")
    if not 0.0 < gain_clip[0] <= 1.0 <= gain_clip[1]:
        raise ValueError("gain_clip must bracket 1.0")
    if missing_target_policy not in {"fail", "lumo_ordinal_derived"}:
        raise ValueError("missing_target_policy must be fail or lumo_ordinal_derived")
    if maximum_geometry_consistency_error <= 0.0:
        raise ValueError("maximum_geometry_consistency_error must be positive")
    band_edges = _calibration_band_edges(max_frequency_hz)
    grouped: dict[int | None, dict[str, list[np.ndarray]]] = {}
    for target, group, board_values, sample_rate_hz in examples:
        if target is not None and target not in {1, 2, 3, 4, 5}:
            raise ValueError(f"invalid calibration target: {target}")
        if not group:
            raise ValueError("measured calibration example lacks a complete group")
        signature = _spatial_log_band_signature(board_values, sample_rate_hz, band_edges)
        grouped.setdefault(target, {}).setdefault(str(group), []).append(signature)

    healthy_groups = grouped.get(None, {})
    if not healthy_groups:
        raise ValueError(f"{dataset_id}: measured calibration has no healthy train groups")
    measured_targets = [
        target for target in range(1, 6)
        if len(grouped.get(target, {})) >= minimum_target_groups
    ]
    if expected_measured_targets is not None and set(measured_targets) != {
        int(value) for value in expected_measured_targets
    }:
        raise ValueError(
            f"{dataset_id}: measured calibration target support mismatch; "
            f"expected={sorted(int(value) for value in expected_measured_targets)}, "
            f"observed={measured_targets}"
        )
    missing = [target for target in range(1, 6) if target not in measured_targets]
    if missing and missing_target_policy == "fail":
        raise ValueError(
            f"{dataset_id}: measured calibration lacks target groups "
            f"{missing}; minimum={minimum_target_groups}"
        )
    if missing and (
        dataset_id != "lumo"
        or measured_targets != [2, 3, 5]
        or missing != [1, 4]
    ):
        raise ValueError(
            f"{dataset_id}: ordinal derivation requires measured targets [2, 3, 5] "
            f"and missing targets [1, 4]"
        )

    def group_medians(values: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, list[str]]:
        names = sorted(values)
        medians = [np.median(np.stack(values[name]), axis=0) for name in names]
        return np.stack(medians), names

    healthy, healthy_names = group_medians(healthy_groups)
    healthy_profile = np.median(healthy, axis=0)
    lower, upper = np.log(gain_clip[0]), np.log(gain_clip[1])
    gains_by_target: dict[int, tuple[tuple[float, ...], ...]] = {}
    log_gain_mad_by_target: dict[int, tuple[tuple[float, ...], ...]] = {}
    target_group_counts: dict[int, int] = {}
    basis_by_target: dict[int, str] = {}
    source_targets_by_target: dict[int, tuple[int, ...]] = {}
    dispersion_estimable_by_target: dict[int, bool] = {}
    geometry_consistency_error_by_target: dict[int, float] = {}
    log_gain_profiles: dict[int, np.ndarray] = {}
    log_gain_mad_profiles: dict[int, np.ndarray] = {}
    source_group_hashes = {
        "healthy": hashlib.sha256("\n".join(healthy_names).encode("utf-8")).hexdigest()
    }
    for target in measured_targets:
        damaged, names = group_medians(grouped[target])
        group_log_gain = damaged - healthy_profile[None, :, :]
        group_log_gain -= np.median(group_log_gain, axis=1, keepdims=True)
        log_gain = np.median(group_log_gain, axis=0)
        log_gain_mad = np.median(np.abs(group_log_gain - log_gain[None, :, :]), axis=0)
        gain = np.exp(np.clip(log_gain, lower, upper))
        log_gain_profiles[target] = np.clip(log_gain, lower, upper)
        log_gain_mad_profiles[target] = log_gain_mad
        gains_by_target[target] = tuple(tuple(float(value) for value in row) for row in gain)
        log_gain_mad_by_target[target] = tuple(
            tuple(float(value) for value in row) for row in log_gain_mad
        )
        target_group_counts[target] = len(names)
        basis_by_target[target] = "measured_train_groups"
        source_targets_by_target[target] = (target,)
        dispersion_estimable_by_target[target] = len(names) >= 2
        source_group_hashes[f"target_{target}"] = hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest()

    def shifted(
        profile: np.ndarray, source: int, destination: int, *, recenter: bool
    ) -> np.ndarray:
        output = np.zeros_like(profile)
        output[0] = profile[0]
        offset = destination - source
        for destination_board in range(1, 6):
            source_board = destination_board - offset
            if 1 <= source_board <= 5:
                output[destination_board] = profile[source_board]
        if recenter:
            output -= np.median(output, axis=0, keepdims=True)
        return output

    def neighbor_sources(target: int, candidates: Sequence[int]) -> tuple[int, ...]:
        left = [value for value in candidates if value < target]
        right = [value for value in candidates if value > target]
        return tuple(
            value
            for value in (
                max(left) if left else None,
                min(right) if right else None,
            )
            if value is not None
        )

    def derived_profile(target: int, sources: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        shifted_profiles = [
            shifted(log_gain_profiles[source], source, target, recenter=True)
            for source in sources
        ]
        shifted_mads = [
            shifted(log_gain_mad_profiles[source], source, target, recenter=False)
            for source in sources
        ]
        log_gain = np.mean(np.stack(shifted_profiles), axis=0)
        disagreement = np.mean(
            np.abs(np.stack(shifted_profiles) - log_gain[None, :, :]), axis=0
        )
        log_gain_mad = np.mean(np.stack(shifted_mads), axis=0) + disagreement
        return log_gain, log_gain_mad

    derivation_contract = None
    if missing:
        for target in measured_targets:
            sources = neighbor_sources(
                target, [value for value in measured_targets if value != target]
            )
            predicted, _ = derived_profile(target, sources)
            error = float(np.mean(np.abs(predicted - log_gain_profiles[target])))
            geometry_consistency_error_by_target[target] = error
            if error > maximum_geometry_consistency_error:
                raise ValueError(
                    f"{dataset_id}: ordinal-shift consistency error for target_{target} "
                    f"is {error:.6f}, above {maximum_geometry_consistency_error:.6f}"
                )
        derivation_contract = {
            "schema": "vibrogemma-lumo-ordinal-target-derivation-v0.18.1",
            "reference": LUMO_REFERENCE_MEASUREMENT_LEVEL,
            "ordered_targets": [
                LUMO_TARGET_MEASUREMENT_LEVELS[f"target_{index}"]
                for index in range(1, 6)
            ],
            "measured_damage_mapping": {"DAM3": 2, "DAM4": 3, "DAM6": 5},
            "rules": {
                "target_1": {"sources": [2], "offsets": [-1], "weights": [1.0]},
                "target_4": {
                    "sources": [3, 5],
                    "offsets": [1, -1],
                    "weights": [0.5, 0.5],
                },
            },
            "profile_space": "common_mode_relative_log_band_magnitude",
            "boundary_rule": "unity_gain_for_shifted_positions_without_same_dataset_support",
            "phase_calibrated": False,
            "maximum_leave_one_target_out_error": maximum_geometry_consistency_error,
            "derived_log_gain_mad_floor": 0.1,
        }
        derivation_contract["sha256"] = hashlib.sha256(
            json.dumps(derivation_contract, sort_keys=True).encode("utf-8")
        ).hexdigest()

    fixed_sources = {1: (2,), 4: (3, 5)}
    for target in missing:
        sources = fixed_sources[target]
        log_gain, log_gain_mad = derived_profile(target, sources)
        log_gain_mad = np.maximum(log_gain_mad, 0.1)
        gain = np.exp(np.clip(log_gain, lower, upper))
        gains_by_target[target] = tuple(tuple(float(value) for value in row) for row in gain)
        log_gain_mad_by_target[target] = tuple(
            tuple(float(value) for value in row) for row in log_gain_mad
        )
        target_group_counts[target] = 0
        basis_by_target[target] = "derived_same_dataset_ordinal_shift_prior"
        source_targets_by_target[target] = sources
        dispersion_estimable_by_target[target] = False
        source_group_hashes[f"target_{target}"] = hashlib.sha256(
            json.dumps(
                {
                    "contract_sha256": derivation_contract["sha256"],
                    "dataset_id": dataset_id,
                    "target": target,
                    "source_target_group_hashes": {
                        str(source): source_group_hashes[f"target_{source}"]
                        for source in sources
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    calibration = MeasuredTransferCalibration(
        dataset_id=str(dataset_id),
        band_edges_hz=band_edges,
        gains_by_target=gains_by_target,
        log_gain_mad_by_target=log_gain_mad_by_target,
        basis_by_target=basis_by_target,
        source_targets_by_target=source_targets_by_target,
        dispersion_estimable_by_target=dispersion_estimable_by_target,
        geometry_consistency_error_by_target=geometry_consistency_error_by_target,
        derivation_contract=derivation_contract,
        healthy_group_count=len(healthy_names),
        target_group_counts=target_group_counts,
        source_group_hashes=source_group_hashes,
    )
    calibration.validate()
    return calibration


def measured_transfer_calibration_error(
    *,
    calibration: MeasuredTransferCalibration,
    target: int,
    healthy_board_values: Sequence[np.ndarray],
    damaged_board_values: Sequence[np.ndarray],
    sample_rate_hz: float,
) -> float:
    calibration.validate()
    healthy = _spatial_log_band_signature(
        healthy_board_values, sample_rate_hz, calibration.band_edges_hz
    )
    damaged = _spatial_log_band_signature(
        damaged_board_values, sample_rate_hz, calibration.band_edges_hz
    )
    observed = damaged - healthy
    observed -= np.median(observed, axis=0, keepdims=True)
    expected = np.log(np.asarray(calibration.gains_by_target[target], dtype=np.float64))
    return float(np.mean(np.abs(observed - expected)))


def _smooth_complex(values: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    if width == 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    real = np.convolve(values.real, kernel, mode="same")
    imag = np.convolve(values.imag, kernel, mode="same")
    return real + 1j * imag


def apply_simulated_transfer(
    *,
    real_values: np.ndarray,
    simulated_healthy: np.ndarray,
    simulated_damaged: np.ndarray,
    real_sample_rate_hz: float,
    simulated_sample_rate_hz: float,
    max_transfer_hz: float = 100.0,
    magnitude_clip: tuple[float, float] = (0.5, 2.0),
    smoothing_bins: int = 7,
    measured_band_gains: Sequence[float] | None = None,
    calibration_band_edges_hz: Sequence[float] | None = None,
    calibration_strength: float = 1.0,
) -> np.ndarray:
    """Condition a real healthy waveform with a simulated damage transfer.

    The measured waveform remains the carrier.  OpenSees supplies only a bounded
    low-frequency healthy-to-damaged transfer; frequencies beyond the declared
    simulation support are left unchanged.
    """

    real = np.asarray(real_values)
    matrix, transposed = canonical_time_axis(real.astype(np.float64, copy=False))
    n = matrix.shape[0]
    sim_h, _ = canonical_time_axis(np.asarray(simulated_healthy, dtype=np.float64))
    sim_d, _ = canonical_time_axis(np.asarray(simulated_damaged, dtype=np.float64))
    if sim_h.shape != sim_d.shape or sim_h.shape[0] < 16:
        raise ValueError("simulated healthy/damaged signals are incompatible")
    if sim_h.shape[1] > matrix.shape[1]:
        raise ValueError("simulated axes cannot exceed the measured waveform axes")

    window = np.hanning(sim_h.shape[0])[:, None]
    h_fft = np.fft.rfft(sim_h * window, axis=0)
    d_fft = np.fft.rfft(sim_d * window, axis=0)
    regularizer = np.maximum(np.median(np.abs(h_fft), axis=0) * 1e-3, 1e-10)
    phase_supported = np.abs(h_fft) >= np.maximum(
        np.max(np.abs(h_fft), axis=0) * 1e-4, regularizer
    )[None, :]
    ratio = d_fft * np.conj(h_fft) / (np.square(np.abs(h_fft)) + regularizer**2)
    ratio = np.column_stack(
        [_smooth_complex(ratio[:, axis], smoothing_bins) for axis in range(ratio.shape[1])]
    )
    magnitude = np.clip(np.abs(ratio), magnitude_clip[0], magnitude_clip[1])
    phase = np.where(phase_supported, np.angle(ratio), 0.0)
    ratio = magnitude * np.exp(1j * phase)

    sim_freq = np.fft.rfftfreq(sim_h.shape[0], d=1.0 / simulated_sample_rate_hz)
    real_freq = np.fft.rfftfreq(n, d=1.0 / real_sample_rate_hz)
    real_ratio = np.column_stack(
        [
            np.interp(real_freq, sim_freq, ratio[:, axis].real, left=1.0, right=1.0)
            + 1j
            * np.interp(real_freq, sim_freq, ratio[:, axis].imag, left=0.0, right=0.0)
            for axis in range(ratio.shape[1])
        ]
    )
    if real_ratio.shape[1] == 1 and matrix.shape[1] > 1:
        real_ratio = np.repeat(real_ratio, matrix.shape[1], axis=1)
    elif real_ratio.shape[1] < matrix.shape[1]:
        # LUMO supplies calibrated x/y FE modes while BoardWindow retains a
        # masked z row. Preserve unsupported measured axes exactly.
        padding = np.ones(
            (real_ratio.shape[0], matrix.shape[1] - real_ratio.shape[1]),
            dtype=np.complex128,
        )
        real_ratio = np.column_stack((real_ratio, padding))
    if measured_band_gains is not None:
        if calibration_band_edges_hz is None:
            raise ValueError("calibration band edges are required with measured gains")
        edges = tuple(float(value) for value in calibration_band_edges_hz)
        gains = tuple(float(value) for value in measured_band_gains)
        if len(edges) != len(gains) + 1 or not 0.0 <= calibration_strength <= 1.0:
            raise ValueError("invalid measured calibration gains/edges/strength")
        calibrated = np.ones_like(real_freq)
        centers = np.sqrt(np.asarray(edges[:-1]) * np.asarray(edges[1:]))
        valid = (real_freq >= edges[0]) & (real_freq <= edges[-1])
        calibrated[valid] = np.exp(
            np.interp(
                np.log(real_freq[valid]),
                np.log(centers),
                np.log(np.maximum(gains, 1e-12)),
            )
        )
        magnitude = np.power(np.maximum(np.abs(real_ratio), 1e-12), 1.0 - calibration_strength)
        magnitude *= np.power(calibrated[:, None], calibration_strength)
        magnitude = np.clip(magnitude, magnitude_clip[0], magnitude_clip[1])
        real_ratio = magnitude * np.exp(1j * np.angle(real_ratio))
    supported = real_freq <= min(max_transfer_hz, simulated_sample_rate_hz / 2.0)
    real_ratio[~supported, :] = 1.0 + 0.0j
    real_ratio[0, :] = 1.0 + 0.0j

    transformed = np.empty_like(matrix, dtype=np.float64)
    for axis in range(matrix.shape[1]):
        spectrum = np.fft.rfft(matrix[:, axis])
        transformed[:, axis] = np.fft.irfft(spectrum * real_ratio[:, axis], n=n)

    restored = restore_time_axis(transformed, transposed, real.ndim)
    return restored.astype(real.dtype, copy=False)
