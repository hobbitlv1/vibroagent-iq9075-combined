from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.signal import coherence, csd, find_peaks, peak_widths, welch
from scipy.stats import kurtosis

from ..contracts import EpisodeProvenance
from ..exceptions import DataContractError
from .profile import HealthyProfile

AXIS_NAMES = ("x", "y", "z")
COVARIANCE_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@dataclass(frozen=True, slots=True)
class PhysicsFeatureResult:
    values: np.ndarray  # [6, feature]
    mask: np.ndarray  # [6, feature]
    feature_names: tuple[str, ...]
    profile_feature_values: np.ndarray  # [6, profile_feature]
    profile_feature_mask: np.ndarray
    profile_feature_names: tuple[str, ...]
    evidence: dict[str, Any]
    evidence_text: str


@dataclass(frozen=True, slots=True)
class PhysicsFeatureSchema:
    bands_hz: tuple[tuple[float, float], ...]
    dominant_peaks_per_axis: int
    modal_peak_count: int

    @property
    def band_labels(self) -> tuple[str, ...]:
        return tuple(f"{low:g}-{high:g}Hz" for low, high in self.bands_hz)

    @property
    def local_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for axis in AXIS_NAMES:
            names.extend(
                (
                    f"rms_g::{axis}",
                    f"peak_abs_g::{axis}",
                    f"crest_factor::{axis}",
                    f"kurtosis::{axis}",
                    f"spectral_entropy::{axis}",
                )
            )
            for label in self.band_labels:
                names.append(f"band_energy_g2::{axis}::{label}")
            for peak_index in range(1, self.dominant_peaks_per_axis + 1):
                names.extend(
                    (
                        f"dominant_peak_frequency_hz::{axis}::p{peak_index}",
                        f"dominant_peak_width_hz::{axis}::p{peak_index}",
                        f"dominant_peak_prominence_db::{axis}::p{peak_index}",
                    )
                )
        for first, second in COVARIANCE_PAIRS:
            names.append(f"cross_axis_covariance_g2::{AXIS_NAMES[first]}{AXIS_NAMES[second]}")
        for peak_index in range(1, self.modal_peak_count + 1):
            names.extend(
                (
                    f"modal_frequency_hz::p{peak_index}",
                    f"modal_damping_ratio::p{peak_index}",
                    f"modal_prominence_db::p{peak_index}",
                )
            )
        return tuple(names)

    @property
    def relation_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for axis in AXIS_NAMES:
            for label in self.band_labels:
                names.extend(
                    (
                        f"transmissibility_db::{axis}::{label}",
                        f"cross_power_g2::{axis}::{label}",
                        f"coherence::{axis}::{label}",
                    )
                )
        return tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *self.local_feature_names,
            *self.relation_feature_names,
            *(f"normal_change_z::{name}" for name in self.local_feature_names),
        )


class PhysicsFeatureExtractor:
    """Compute auditable time-, frequency-, modal-, and reference-domain features.

    This module creates numeric evidence only. It emits no class, anomaly score,
    severity, threshold decision, or alert state.
    """

    def __init__(
        self,
        *,
        bands_hz: Iterable[Iterable[float]] = (
            (0.1, 1.0),
            (1.0, 5.0),
            (5.0, 20.0),
            (20.0, 50.0),
            (50.0, 200.0),
            (200.0, 1000.0),
            (1000.0, 3000.0),
            (3000.0, 6000.0),
        ),
        dominant_peaks_per_axis: int = 3,
        modal_peak_count: int = 3,
        modal_band_hz: tuple[float, float] = (0.1, 100.0),
        welch_segment_seconds: float = 2.0,
        minimum_peak_prominence_db: float = 3.0,
        max_frequency_hz: float = 6000.0,
        healthy_profile: HealthyProfile | None = None,
    ) -> None:
        bands = tuple((float(pair[0]), float(pair[1])) for pair in bands_hz)
        if not bands or any(low < 0 or high <= low for low, high in bands):
            raise DataContractError("physics bands must be increasing positive intervals")
        self.schema = PhysicsFeatureSchema(
            bands_hz=bands,
            dominant_peaks_per_axis=int(dominant_peaks_per_axis),
            modal_peak_count=int(modal_peak_count),
        )
        self.modal_band_hz = (float(modal_band_hz[0]), float(modal_band_hz[1]))
        self.welch_segment_seconds = float(welch_segment_seconds)
        self.minimum_peak_prominence_db = float(minimum_peak_prominence_db)
        self.max_frequency_hz = float(max_frequency_hz)
        self.healthy_profile = healthy_profile
        if healthy_profile is not None and healthy_profile.feature_names != self.schema.local_feature_names:
            raise DataContractError("healthy profile feature schema does not match the configured physics extractor")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.schema.feature_names

    @property
    def profile_feature_names(self) -> tuple[str, ...]:
        return self.schema.local_feature_names

    @staticmethod
    def _fill_signal(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).copy()
        mask = np.asarray(valid, dtype=bool)
        if vector.ndim != 1 or mask.shape != vector.shape:
            raise DataContractError("physics signal/mask shape mismatch")
        indices = np.arange(vector.size)
        valid_indices = indices[mask]
        if valid_indices.size == 0:
            return np.zeros_like(vector)
        if valid_indices.size == 1:
            return np.full_like(vector, vector[valid_indices[0]])
        if not mask.all():
            vector[~mask] = np.interp(indices[~mask], valid_indices, vector[mask])
        return vector

    def _psd(self, values: np.ndarray, valid: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
        filled = self._fill_signal(values, valid)
        nominal = max(64, int(round(rate_hz * self.welch_segment_seconds)))
        nperseg = min(filled.size, nominal)
        if nperseg < 8:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        frequencies, density = welch(
            filled,
            fs=rate_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="linear",
            scaling="density",
        )
        return frequencies.astype(np.float64), np.maximum(density.astype(np.float64), 0.0)

    @staticmethod
    def _integrate_band(frequencies: np.ndarray, density: np.ndarray, low: float, high: float) -> tuple[float, bool]:
        selector = (frequencies >= low) & (frequencies < high)
        indices = np.flatnonzero(selector)
        if indices.size == 0:
            return 0.0, False
        if indices.size == 1:
            if frequencies.size < 2:
                return float(density[indices[0]]), True
            spacing = float(np.median(np.diff(frequencies)))
            return float(density[indices[0]] * spacing), True
        return float(np.trapezoid(density[indices], frequencies[indices])), True

    @staticmethod
    def _spectral_entropy(density: np.ndarray) -> tuple[float, bool]:
        total = float(np.sum(density))
        if density.size < 2 or total <= 0:
            return 0.0, False
        probabilities = density / total
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-18))) / np.log(density.size)
        return float(np.clip(entropy, 0.0, 1.0)), True

    def _dominant_peaks(
        self,
        frequencies: np.ndarray,
        density: np.ndarray,
        usable_band_hz: float,
        *,
        amplitude_calibrated: bool,
    ) -> list[dict[str, float]]:
        valid = (frequencies > 0.0) & (frequencies <= usable_band_hz) & (density > 0)
        if valid.sum() < 4:
            return []
        selected_frequency = frequencies[valid]
        selected_density = density[valid]
        density_db = 10.0 * np.log10(selected_density + 1e-24)
        peak_indices, properties = find_peaks(density_db, prominence=self.minimum_peak_prominence_db)
        if peak_indices.size == 0:
            peak_indices = np.asarray([int(np.argmax(density_db))])
            prominences = np.asarray([0.0])
        else:
            prominences = np.asarray(properties.get("prominences", np.zeros(peak_indices.size)))
        widths = peak_widths(density_db, peak_indices, rel_height=0.5)[0]
        frequency_spacing = float(np.median(np.diff(selected_frequency))) if selected_frequency.size > 1 else 0.0
        order = np.argsort(prominences + density_db[peak_indices] * 1e-3)[::-1]
        peaks: list[dict[str, float]] = []
        for rank in order[: self.schema.dominant_peaks_per_axis]:
            index = int(peak_indices[rank])
            item = {
                "frequency_hz": float(selected_frequency[index]),
                "width_hz": float(widths[rank] * frequency_spacing),
                "prominence_db": float(prominences[rank]),
            }
            if amplitude_calibrated:
                item["power_density_g2_per_hz"] = float(selected_density[index])
            peaks.append(item)
        return peaks

    def _modal_peaks(
        self,
        modal_values: np.ndarray,
        modal_sample_mask: np.ndarray,
        axis_mask: np.ndarray,
        modal_rate_hz: float,
        usable_band_hz: float,
    ) -> list[dict[str, float]]:
        densities: list[np.ndarray] = []
        frequencies: np.ndarray | None = None
        for axis in range(3):
            if not axis_mask[axis]:
                continue
            freq, density = self._psd(modal_values[axis], modal_sample_mask, modal_rate_hz)
            if density.size:
                frequencies = freq
                densities.append(density)
        if not densities or frequencies is None:
            return []
        combined = np.mean(np.stack(densities), axis=0)
        low, configured_high = self.modal_band_hz
        high = min(configured_high, usable_band_hz, modal_rate_hz / 2.0)
        valid = (frequencies >= low) & (frequencies <= high) & (combined > 0)
        if valid.sum() < 4:
            return []
        frequency = frequencies[valid]
        power = combined[valid]
        power_db = 10.0 * np.log10(power + 1e-24)
        indices, properties = find_peaks(power_db, prominence=self.minimum_peak_prominence_db)
        if indices.size == 0:
            indices = np.asarray([int(np.argmax(power_db))])
            prominences = np.asarray([0.0])
        else:
            prominences = np.asarray(properties.get("prominences", np.zeros(indices.size)))
        order = np.argsort(prominences + power_db[indices] * 1e-3)[::-1]
        output: list[dict[str, float]] = []
        for rank in order[: self.schema.modal_peak_count]:
            peak = int(indices[rank])
            half_power = power[peak] / 2.0
            left = peak
            while left > 0 and power[left] > half_power:
                left -= 1
            right = peak
            while right < power.size - 1 and power[right] > half_power:
                right += 1
            centre = float(frequency[peak])
            bandwidth = max(0.0, float(frequency[right] - frequency[left]))
            damping = bandwidth / (2.0 * centre) if centre > 0 else 0.0
            output.append(
                {
                    "frequency_hz": centre,
                    "damping_ratio": float(np.clip(damping, 0.0, 1.0)),
                    "prominence_db": float(prominences[rank]),
                }
            )
        return output

    def _normalize(self, name: str, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if name.startswith(("rms_g::", "peak_abs_g::")):
            return float(np.clip(np.log10(1.0 + abs(value) / 1e-7), 0.0, 12.0))
        if name.startswith(("band_energy_g2::", "cross_power_g2::")):
            return float(np.clip(np.log10(1.0 + abs(value) / 1e-14), 0.0, 16.0))
        if name.startswith("cross_axis_covariance_g2::"):
            return float(np.clip(np.sign(value) * np.log10(1.0 + abs(value) / 1e-14), -16.0, 16.0))
        if "frequency_hz" in name or "width_hz" in name:
            return float(np.clip(value / max(self.max_frequency_hz, 1.0), 0.0, 2.0))
        if name.startswith("crest_factor::"):
            return float(np.clip(np.log1p(max(value, 0.0)) / 3.0, 0.0, 4.0))
        if name.startswith("kurtosis::"):
            return float(np.clip(np.sign(value) * np.log1p(abs(value)) / 4.0, -4.0, 4.0))
        if name.startswith("spectral_entropy::") or name.startswith("coherence::"):
            return float(np.clip(value, 0.0, 1.0))
        if "prominence_db" in name:
            return float(np.clip(value / 60.0, 0.0, 2.0))
        if name.startswith("modal_damping_ratio::"):
            return float(np.clip(value / 0.20, 0.0, 5.0))
        if name.startswith("transmissibility_db::"):
            return float(np.clip(value / 40.0, -3.0, 3.0))
        if name.startswith("normal_change_z::"):
            return float(np.clip(value / 6.0, -2.0, 2.0))
        return float(np.clip(value, -20.0, 20.0))

    def extract(
        self,
        *,
        native_values_g: np.ndarray,
        native_sample_masks: np.ndarray,
        axis_masks: np.ndarray,
        sample_rate_hz: float,
        modal_values_g: np.ndarray,
        modal_sample_masks: np.ndarray,
        modal_rate_hz: float,
        usable_band_hz: np.ndarray,
        synchronized: bool,
        provenance: EpisodeProvenance,
        amplitude_calibrated: np.ndarray | bool = True,
        reference_index: int = 0,
    ) -> PhysicsFeatureResult:
        native = np.asarray(native_values_g, dtype=np.float32)
        native_masks = np.asarray(native_sample_masks, dtype=bool)
        axes = np.asarray(axis_masks, dtype=bool)
        modal = np.asarray(modal_values_g, dtype=np.float32)
        modal_masks = np.asarray(modal_sample_masks, dtype=bool)
        usable = np.asarray(usable_band_hz, dtype=np.float32)
        calibration = np.asarray(amplitude_calibrated, dtype=bool)
        if calibration.ndim == 0:
            calibration = np.full(6, bool(calibration), dtype=bool)
        if native.ndim != 3 or native.shape[:2] != (6, 3):
            raise DataContractError("native_values_g must have shape [6,3,samples]")
        if native_masks.shape != (6, native.shape[-1]) or axes.shape != (6, 3):
            raise DataContractError("native mask shape mismatch")
        if modal.ndim != 3 or modal.shape[:2] != (6, 3):
            raise DataContractError("modal_values_g must have shape [6,3,samples]")
        if modal_masks.shape != (6, modal.shape[-1]) or usable.shape != (6,):
            raise DataContractError("modal/usable-band shape mismatch")
        if calibration.shape != (6,):
            raise DataContractError("amplitude_calibrated must be a scalar or shape [6]")

        local_names = self.schema.local_feature_names
        relation_names = self.schema.relation_feature_names
        base_names = (*local_names, *relation_names)
        base_index = {name: index for index, name in enumerate(base_names)}
        local_index = {name: index for index, name in enumerate(local_names)}
        base_raw = np.zeros((6, len(base_names)), dtype=np.float64)
        base_mask = np.zeros((6, len(base_names)), dtype=bool)
        evidence_boards: dict[str, Any] = {}
        psd_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        band_energy_cache: dict[tuple[int, int, int], tuple[float, bool]] = {}

        def set_feature(board: int, name: str, value: float, valid: bool) -> None:
            index = base_index[name]
            if valid and np.isfinite(value):
                base_raw[board, index] = float(value)
                base_mask[board, index] = True

        for board in range(6):
            board_name = "reference" if board == reference_index else f"target_{board}"
            calibrated = bool(calibration[board])
            board_evidence: dict[str, Any] = {
                "amplitude_calibrated": calibrated,
                "rms_g": {},
                "peak_abs_g": {},
                "release_native_rms": {},
                "release_native_peak_abs": {},
                "crest_factor": {},
                "kurtosis": {},
                "spectral_entropy": {},
                "band_energy_g2": {},
                "dominant_peaks": {},
                "cross_axis_covariance_g2": {},
                "modal_estimates": [],
            }
            valid_sample = native_masks[board]
            for axis, axis_name in enumerate(AXIS_NAMES):
                if not axes[board, axis] or valid_sample.sum() < 4:
                    continue
                signal = native[board, axis, valid_sample].astype(np.float64)
                rms = float(np.sqrt(np.mean(signal**2)))
                peak = float(np.max(np.abs(signal)))
                crest = peak / rms if rms > 0 else 0.0
                kurt = float(kurtosis(signal, fisher=False, bias=False)) if signal.size >= 8 else 0.0
                set_feature(board, f"rms_g::{axis_name}", rms, calibrated)
                set_feature(board, f"peak_abs_g::{axis_name}", peak, calibrated)
                set_feature(board, f"crest_factor::{axis_name}", crest, rms > 0)
                set_feature(board, f"kurtosis::{axis_name}", kurt, signal.size >= 8)
                if calibrated:
                    board_evidence["rms_g"][axis_name] = rms
                    board_evidence["peak_abs_g"][axis_name] = peak
                else:
                    board_evidence["release_native_rms"][axis_name] = rms
                    board_evidence["release_native_peak_abs"][axis_name] = peak
                board_evidence["crest_factor"][axis_name] = crest
                board_evidence["kurtosis"][axis_name] = kurt

                frequencies, density = self._psd(native[board, axis], valid_sample, sample_rate_hz)
                psd_cache[(board, axis)] = (frequencies, density)
                entropy, entropy_valid = self._spectral_entropy(
                    density[(frequencies <= min(float(usable[board]), self.max_frequency_hz))]
                )
                set_feature(board, f"spectral_entropy::{axis_name}", entropy, entropy_valid)
                if entropy_valid:
                    board_evidence["spectral_entropy"][axis_name] = entropy
                board_evidence["band_energy_g2"][axis_name] = {}
                for band_index, ((low, high), label) in enumerate(zip(self.schema.bands_hz, self.schema.band_labels, strict=True)):
                    energy, valid = self._integrate_band(
                        frequencies,
                        density,
                        low,
                        min(high, float(usable[board]) + 1e-9),
                    )
                    if high > float(usable[board]) + 1e-9:
                        valid = False
                    band_energy_cache[(board, axis, band_index)] = (energy, valid)
                    set_feature(board, f"band_energy_g2::{axis_name}::{label}", energy, valid and calibrated)
                    if valid and calibrated:
                        board_evidence["band_energy_g2"][axis_name][label] = energy
                peaks = self._dominant_peaks(
                    frequencies,
                    density,
                    min(float(usable[board]), self.max_frequency_hz),
                    amplitude_calibrated=calibrated,
                )
                board_evidence["dominant_peaks"][axis_name] = peaks
                for peak_index in range(1, self.schema.dominant_peaks_per_axis + 1):
                    available = peak_index <= len(peaks)
                    peak_data = peaks[peak_index - 1] if available else {}
                    set_feature(
                        board,
                        f"dominant_peak_frequency_hz::{axis_name}::p{peak_index}",
                        float(peak_data.get("frequency_hz", 0.0)),
                        available,
                    )
                    set_feature(
                        board,
                        f"dominant_peak_width_hz::{axis_name}::p{peak_index}",
                        float(peak_data.get("width_hz", 0.0)),
                        available,
                    )
                    set_feature(
                        board,
                        f"dominant_peak_prominence_db::{axis_name}::p{peak_index}",
                        float(peak_data.get("prominence_db", 0.0)),
                        available,
                    )

            common_axes = np.flatnonzero(axes[board])
            if common_axes.size and valid_sample.sum() >= 4:
                covariance = np.cov(native[board][:, valid_sample], bias=False)
                for first, second in COVARIANCE_PAIRS:
                    pair_name = f"{AXIS_NAMES[first]}{AXIS_NAMES[second]}"
                    valid = bool(axes[board, first] and axes[board, second] and calibrated)
                    value = float(covariance[first, second]) if valid else 0.0
                    set_feature(board, f"cross_axis_covariance_g2::{pair_name}", value, valid)
                    if valid:
                        board_evidence["cross_axis_covariance_g2"][pair_name] = value

            modal_peaks = self._modal_peaks(
                modal[board],
                modal_masks[board],
                axes[board],
                modal_rate_hz,
                min(float(usable[board]), self.modal_band_hz[1]),
            )
            board_evidence["modal_estimates"] = modal_peaks
            for peak_index in range(1, self.schema.modal_peak_count + 1):
                available = peak_index <= len(modal_peaks)
                peak_data = modal_peaks[peak_index - 1] if available else {}
                set_feature(
                    board,
                    f"modal_frequency_hz::p{peak_index}",
                    float(peak_data.get("frequency_hz", 0.0)),
                    available,
                )
                set_feature(
                    board,
                    f"modal_damping_ratio::p{peak_index}",
                    float(peak_data.get("damping_ratio", 0.0)),
                    available,
                )
                set_feature(
                    board,
                    f"modal_prominence_db::p{peak_index}",
                    float(peak_data.get("prominence_db", 0.0)),
                    available,
                )
            evidence_boards[board_name] = board_evidence

        # Reference-conditioned transmissibility is always safe in magnitude.
        # Cross power and coherence require explicit sample-level synchronization.
        for board in range(6):
            board_name = "reference" if board == reference_index else f"target_{board}"
            relation_evidence: dict[str, Any] = {
                "transmissibility_db": {},
                "cross_power_g2": {},
                "coherence": {},
            }
            for axis, axis_name in enumerate(AXIS_NAMES):
                relation_evidence["transmissibility_db"][axis_name] = {}
                relation_evidence["cross_power_g2"][axis_name] = {}
                relation_evidence["coherence"][axis_name] = {}
                axis_relation_valid = bool(axes[board, axis] and axes[reference_index, axis])
                cross_frequency: np.ndarray | None = None
                cross_density: np.ndarray | None = None
                coherence_values: np.ndarray | None = None
                if synchronized and axis_relation_valid:
                    target_filled = self._fill_signal(native[board, axis], native_masks[board])
                    reference_filled = self._fill_signal(native[reference_index, axis], native_masks[reference_index])
                    nperseg = min(
                        target_filled.size,
                        max(64, int(round(sample_rate_hz * self.welch_segment_seconds))),
                    )
                    cross_frequency, cross_density = csd(
                        target_filled,
                        reference_filled,
                        fs=sample_rate_hz,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=nperseg // 2,
                        detrend="linear",
                        scaling="density",
                    )
                    coherence_frequency, coherence_values = coherence(
                        target_filled,
                        reference_filled,
                        fs=sample_rate_hz,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=nperseg // 2,
                        detrend="linear",
                    )
                    if not np.allclose(cross_frequency, coherence_frequency):
                        coherence_values = np.interp(cross_frequency, coherence_frequency, coherence_values)
                for band_index, ((low, high), label) in enumerate(zip(self.schema.bands_hz, self.schema.band_labels, strict=True)):
                    target_energy, target_valid = band_energy_cache.get((board, axis, band_index), (0.0, False))
                    reference_energy, reference_valid = band_energy_cache.get(
                        (reference_index, axis, band_index), (0.0, False)
                    )
                    ratio_valid = axis_relation_valid and target_valid and reference_valid and reference_energy > 0
                    ratio_db = 10.0 * np.log10((target_energy + 1e-24) / (reference_energy + 1e-24)) if ratio_valid else 0.0
                    name = f"transmissibility_db::{axis_name}::{label}"
                    set_feature(board, name, ratio_db, ratio_valid)
                    if ratio_valid:
                        relation_evidence["transmissibility_db"][axis_name][label] = float(ratio_db)

                    cross_valid = bool(
                        synchronized
                        and calibration[board]
                        and calibration[reference_index]
                        and axis_relation_valid
                        and cross_frequency is not None
                        and cross_density is not None
                        and coherence_values is not None
                        and high <= min(float(usable[board]), float(usable[reference_index])) + 1e-9
                    )
                    cross_power = 0.0
                    coherence_mean = 0.0
                    if cross_valid:
                        cross_power, cross_valid = self._integrate_band(
                            cross_frequency,
                            np.abs(cross_density),
                            low,
                            high,
                        )
                        selector = (cross_frequency >= low) & (cross_frequency < high)
                        if selector.any():
                            coherence_mean = float(np.mean(coherence_values[selector]))
                        else:
                            cross_valid = False
                    set_feature(board, f"cross_power_g2::{axis_name}::{label}", cross_power, cross_valid)
                    set_feature(board, f"coherence::{axis_name}::{label}", coherence_mean, cross_valid)
                    if cross_valid:
                        relation_evidence["cross_power_g2"][axis_name][label] = float(cross_power)
                        relation_evidence["coherence"][axis_name][label] = float(coherence_mean)
            evidence_boards[board_name]["reference_relation"] = relation_evidence

        normalized_base = np.zeros_like(base_raw, dtype=np.float32)
        for feature_index, name in enumerate(base_names):
            for board in range(6):
                if base_mask[board, feature_index]:
                    normalized_base[board, feature_index] = self._normalize(name, base_raw[board, feature_index])

        profile_indices = np.asarray([base_index[name] for name in local_names], dtype=np.int64)
        profile_values = normalized_base[:, profile_indices]
        profile_mask = base_mask[:, profile_indices]
        z_values = np.zeros_like(profile_values, dtype=np.float32)
        z_mask = np.zeros_like(profile_mask, dtype=bool)
        profile_keys: dict[str, str | None] = {}
        for board in range(1, 6):
            board_name = f"target_{board}"
            profile_key: str | None = None
            if self.healthy_profile is not None:
                z_values[board], z_mask[board], profile_key = self.healthy_profile.z_scores(
                    profile_values[board],
                    profile_mask[board],
                    dataset_id=provenance.dataset_id,
                    structure_id=provenance.structure_id,
                    target_index=board,
                )
            profile_keys[board_name] = profile_key
            valid_z = np.flatnonzero(z_mask[board])
            top_changes: list[dict[str, Any]] = []
            if valid_z.size:
                order = valid_z[np.argsort(np.abs(z_values[board, valid_z]))[::-1][:5]]
                for feature_index in order:
                    top_changes.append(
                        {
                            "feature": local_names[int(feature_index)],
                            "z": float(z_values[board, feature_index]),
                        }
                    )
            evidence_boards[board_name]["normal_distribution_change"] = {
                "profile_key": profile_key,
                "top_absolute_z": top_changes,
            }

        normalized_z = np.zeros_like(z_values)
        for board in range(6):
            for index, value in enumerate(z_values[board]):
                if z_mask[board, index]:
                    normalized_z[board, index] = self._normalize(
                        f"normal_change_z::{local_names[index]}", float(value)
                    )
        final_values = np.concatenate((normalized_base, normalized_z), axis=1).astype(np.float32)
        final_mask = np.concatenate((base_mask, z_mask), axis=1).astype(bool)
        evidence = {
            "units": {
                "acceleration": "g when amplitude_calibrated=true; otherwise release-native scale",
                "frequency": "Hz",
                "band_energy": "g^2 only when amplitude_calibrated=true",
                "cross_power": "g^2 only when amplitude_calibrated=true",
                "covariance": "g^2 only when amplitude_calibrated=true",
                "transmissibility": "dB",
            },
            "amplitude_calibrated_by_board": {
                ("reference" if board == reference_index else f"target_{board}"): bool(calibration[board])
                for board in range(6)
            },
            "synchronization": {
                "phase_valid": bool(synchronized),
                "coherence_emitted": bool(synchronized),
                "cross_power_emitted": bool(
                    synchronized and calibration[reference_index] and calibration.all()
                ),
                "cross_power_and_coherence_emitted": bool(
                    synchronized and calibration[reference_index] and calibration.all()
                ),
            },
            "boards": evidence_boards,
            "feature_schema": list(self.feature_names),
            "healthy_profile_keys": profile_keys,
        }
        evidence_text = self.format_evidence_text(evidence)
        return PhysicsFeatureResult(
            values=final_values,
            mask=final_mask,
            feature_names=self.feature_names,
            profile_feature_values=profile_values.astype(np.float32),
            profile_feature_mask=profile_mask.astype(bool),
            profile_feature_names=local_names,
            evidence=evidence,
            evidence_text=evidence_text,
        )

    @staticmethod
    def _strongest_axis_value(mapping: dict[str, float]) -> tuple[str, float] | None:
        if not mapping:
            return None
        axis = max(mapping, key=lambda key: abs(float(mapping[key])))
        return axis, float(mapping[axis])

    def format_evidence_text(self, evidence: dict[str, Any]) -> str:
        """Create a compact, numeric, source-grounded evidence section for Gemma."""

        sync = bool(evidence["synchronization"]["phase_valid"])
        all_calibrated = all(bool(value) for value in evidence["amplitude_calibrated_by_board"].values())
        amplitude_text = (
            "absolute acceleration is calibrated in g"
            if all_calibrated
            else "absolute acceleration is uncalibrated release-native scale; g/g^2 features are masked"
        )
        lines = [
            f"Auditable vibration evidence; {amplitude_text}; frequencies are Hz and transmissibility is dB. "
            f"Cross-phase/coherence evidence valid={'yes' if sync else 'no'}."
        ]
        boards = evidence["boards"]
        reference = boards["reference"]
        reference_rms = self._strongest_axis_value(reference.get("rms_g", {}))
        reference_modal = reference.get("modal_estimates", [])
        if reference_rms is not None:
            modal_text = (
                f", strongest modal estimate={reference_modal[0]['frequency_hz']:.3g} Hz"
                if reference_modal
                else ""
            )
            lines.append(f"reference: max-axis RMS={reference_rms[1]:.3g} g ({reference_rms[0]}){modal_text}.")
        elif reference_modal:
            lines.append(f"reference: strongest modal estimate={reference_modal[0]['frequency_hz']:.3g} Hz.")
        for target_index in range(1, 6):
            name = f"target_{target_index}"
            board = boards[name]
            rms = self._strongest_axis_value(board.get("rms_g", {}))
            modal = board.get("modal_estimates", [])
            relation = board.get("reference_relation", {})
            ratios: list[tuple[str, float]] = []
            for axis, values in relation.get("transmissibility_db", {}).items():
                for band, value in values.items():
                    ratios.append((f"{axis}/{band}", float(value)))
            strongest_ratio = max(ratios, key=lambda item: abs(item[1])) if ratios else None
            coherences = [
                float(value)
                for values in relation.get("coherence", {}).values()
                for value in values.values()
            ]
            normal_changes = board.get("normal_distribution_change", {}).get("top_absolute_z", [])
            pieces = []
            if rms is not None:
                pieces.append(f"max-axis RMS={rms[1]:.3g} g ({rms[0]})")
            if modal:
                pieces.append(
                    f"modal={modal[0]['frequency_hz']:.3g} Hz, damping={modal[0]['damping_ratio']:.3g}"
                )
            if strongest_ratio is not None:
                pieces.append(f"max |transmissibility|={strongest_ratio[1]:.3g} dB at {strongest_ratio[0]}")
            if coherences:
                pieces.append(f"mean coherence={float(np.mean(coherences)):.3g}")
            if normal_changes:
                pieces.append(
                    f"largest normal-change z={normal_changes[0]['z']:.3g} ({normal_changes[0]['feature']})"
                )
            lines.append(f"{name}: " + (", ".join(pieces) if pieces else "physics evidence unavailable") + ".")
        return "\n".join(lines)
