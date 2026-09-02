from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter
from scipy.signal import stft

from ..exceptions import DataContractError


WIDEBAND_CHANNEL_NAMES = (
    "log_power",
    "complex_phase_cos",
    "complex_phase_sin",
    "reference_cross_log_magnitude",
    "reference_cross_phase_cos",
    "reference_cross_phase_sin",
    "reference_coherence",
    "cross_spectral_valid",
    "frequency_valid",
)


@dataclass(frozen=True, slots=True)
class WidebandFeatureResult:
    features: np.ndarray  # [6, 3, channels, frequency, time]
    frequency_mask: np.ndarray  # [6, frequency]
    cross_spectral_valid: np.ndarray  # [6, 3]
    frequency_grid_hz: np.ndarray
    time_grid_s: np.ndarray
    channel_names: tuple[str, ...] = WIDEBAND_CHANNEL_NAMES


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _fill_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    if matrix.ndim != 2 or mask.shape != (matrix.shape[-1],):
        raise DataContractError("wideband signal/mask shape mismatch")
    sample_index = np.arange(matrix.shape[-1], dtype=np.float64)
    for axis in range(matrix.shape[0]):
        valid_index = sample_index[mask]
        if valid_index.size == 0:
            matrix[axis] = 0.0
        elif valid_index.size == 1:
            matrix[axis] = matrix[axis, int(valid_index[0])]
        elif not mask.all():
            matrix[axis, ~mask] = np.interp(sample_index[~mask], valid_index, matrix[axis, mask])
    return matrix


def _resample_complex_grid(
    spectrum: np.ndarray,
    source_frequency_hz: np.ndarray,
    source_time_s: np.ndarray,
    target_frequency_hz: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    if spectrum.shape[-2:] != (source_frequency_hz.size, source_time_s.size):
        raise DataContractError("STFT spectrum dimensions do not match its coordinate grids")
    frequency_interpolator = interp1d(
        source_frequency_hz,
        spectrum,
        axis=-2,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    frequency_resampled = frequency_interpolator(target_frequency_hz)
    if source_time_s.size == 1:
        return np.repeat(frequency_resampled, target_time_s.size, axis=-1)
    # scipy interp1d does not reliably broadcast tuple fill values for arbitrary
    # leading dimensions. Clip to the measured time support, then interpolate.
    clipped_time = np.clip(target_time_s, source_time_s[0], source_time_s[-1])
    time_interpolator = interp1d(
        source_time_s,
        frequency_resampled,
        axis=-1,
        bounds_error=True,
        assume_sorted=True,
    )
    return time_interpolator(clipped_time)


def compute_wideband_features(
    values: np.ndarray,
    sample_masks: np.ndarray,
    axis_masks: np.ndarray,
    *,
    sample_rate_hz: float,
    window_seconds: float,
    max_frequency_hz: float = 6000.0,
    frequency_bins: int = 192,
    time_bins: int = 64,
    frame_seconds: float = 0.08,
    overlap_fraction: float = 0.75,
    max_nfft: int = 8192,
    synchronized: bool = False,
    usable_band_hz: np.ndarray | None = None,
    reference_index: int = 0,
    epsilon: float = 1e-12,
) -> WidebandFeatureResult:
    """Create fixed-size numeric wideband representations from native-rate signals.

    The native waveform is never first collapsed to the modal sample rate. A
    complex STFT is calculated at the physical source rate and then interpolated
    onto a fixed 0..max_frequency_hz grid. Reference cross spectra and coherence
    are emitted only when sample-level synchronization has been explicitly
    validated.
    """

    signal = np.asarray(values, dtype=np.float32)
    valid_samples = np.asarray(sample_masks, dtype=bool)
    present_axes = np.asarray(axis_masks, dtype=bool)
    if signal.ndim != 3 or signal.shape[:2] != (6, 3):
        raise DataContractError(f"wideband values must have shape [6,3,samples], got {signal.shape}")
    if valid_samples.shape != (6, signal.shape[-1]):
        raise DataContractError("wideband sample_masks must have shape [6,samples]")
    if present_axes.shape != (6, 3):
        raise DataContractError("wideband axis_masks must have shape [6,3]")
    if not 0 <= reference_index < 6:
        raise DataContractError("reference_index must identify one of six boards")
    if sample_rate_hz <= 0 or window_seconds <= 0:
        raise DataContractError("sample_rate_hz and window_seconds must be positive")
    if frequency_bins < 8 or time_bins < 4:
        raise DataContractError("wideband grid is too small")

    filled = np.stack([_fill_missing(signal[index], valid_samples[index]) for index in range(6)], axis=0)
    filled *= present_axes[:, :, None]

    nominal_segment = max(32, int(round(sample_rate_hz * frame_seconds)))
    nperseg = min(signal.shape[-1], max(32, min(_next_power_of_two(nominal_segment), int(max_nfft))))
    if nperseg < 8:
        raise DataContractError("wideband window has too few samples for an STFT")
    noverlap = min(nperseg - 1, int(round(nperseg * overlap_fraction)))
    nfft = min(int(max_nfft), _next_power_of_two(nperseg))
    source_frequency_hz, source_time_s, complex_spectrum = stft(
        filled,
        fs=float(sample_rate_hz),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )
    if complex_spectrum.shape[-1] == 0:
        # This is possible only for a very small pathological input. Preserve a
        # deterministic one-frame representation rather than guessing data.
        complex_spectrum = np.zeros((*signal.shape[:2], source_frequency_hz.size, 1), dtype=np.complex64)
        source_time_s = np.asarray([window_seconds / 2.0], dtype=np.float64)

    upper_frequency = min(float(max_frequency_hz), float(sample_rate_hz) / 2.0)
    target_frequency_hz = np.linspace(0.0, float(max_frequency_hz), int(frequency_bins), dtype=np.float64)
    target_time_s = np.linspace(0.0, float(window_seconds), int(time_bins), dtype=np.float64)
    spectrum = _resample_complex_grid(
        complex_spectrum,
        source_frequency_hz,
        source_time_s,
        target_frequency_hz,
        target_time_s,
    ).astype(np.complex64, copy=False)
    # Remove only the exact DC bin. Structural quasi-static/modal content is
    # retained by the dedicated modal branch, while this prevents frame means
    # from dominating the wideband tokenization.
    spectrum[..., target_frequency_hz == 0.0, :] = 0.0

    if usable_band_hz is None:
        board_usable_band = np.full(6, upper_frequency, dtype=np.float32)
    else:
        board_usable_band = np.asarray(usable_band_hz, dtype=np.float32)
        if board_usable_band.shape != (6,):
            raise DataContractError("usable_band_hz must have shape [6]")
        board_usable_band = np.minimum(board_usable_band, upper_frequency)
    frequency_mask = target_frequency_hz[None, :] <= board_usable_band[:, None] + 1e-9

    power = np.abs(spectrum).astype(np.float64) ** 2
    magnitude = np.sqrt(power + epsilon)
    local_phase_cos = spectrum.real / magnitude
    local_phase_sin = spectrum.imag / magnitude
    log_power = np.log1p(power)

    reference = spectrum[reference_index : reference_index + 1]
    raw_cross = spectrum * np.conj(reference)
    smooth_target_power = uniform_filter(power, size=(1, 1, 3, 5), mode="nearest")
    smooth_reference_power = uniform_filter(
        power[reference_index : reference_index + 1], size=(1, 1, 3, 5), mode="nearest"
    )
    smooth_cross_real = uniform_filter(raw_cross.real, size=(1, 1, 3, 5), mode="nearest")
    smooth_cross_imag = uniform_filter(raw_cross.imag, size=(1, 1, 3, 5), mode="nearest")
    smooth_cross = smooth_cross_real + 1j * smooth_cross_imag
    cross_magnitude = np.abs(smooth_cross)
    cross_phase_cos = smooth_cross.real / (cross_magnitude + epsilon)
    cross_phase_sin = smooth_cross.imag / (cross_magnitude + epsilon)
    coherence = np.clip(
        cross_magnitude**2 / (smooth_target_power * smooth_reference_power + epsilon),
        0.0,
        1.0,
    )
    cross_log_magnitude = np.log1p(cross_magnitude)

    reference_axis_present = present_axes[reference_index]
    reference_valid_fraction = float(valid_samples[reference_index].mean())
    cross_valid = np.zeros((6, 3), dtype=bool)
    if synchronized:
        for board in range(6):
            board_valid_fraction = float(valid_samples[board].mean())
            cross_valid[board] = (
                present_axes[board]
                & reference_axis_present
                & (board_valid_fraction >= 0.98)
                & (reference_valid_fraction >= 0.98)
            )

    frequency_valid_map = frequency_mask[:, None, :, None] & present_axes[:, :, None, None]
    cross_valid_map = cross_valid[:, :, None, None]
    shape = (6, 3, len(WIDEBAND_CHANNEL_NAMES), int(frequency_bins), int(time_bins))
    features = np.zeros(shape, dtype=np.float32)
    features[:, :, 0] = log_power.astype(np.float32)
    features[:, :, 1] = local_phase_cos.astype(np.float32)
    features[:, :, 2] = local_phase_sin.astype(np.float32)
    features[:, :, 3] = np.where(cross_valid_map, cross_log_magnitude, 0.0).astype(np.float32)
    features[:, :, 4] = np.where(cross_valid_map, cross_phase_cos, 0.0).astype(np.float32)
    features[:, :, 5] = np.where(cross_valid_map, cross_phase_sin, 0.0).astype(np.float32)
    features[:, :, 6] = np.where(cross_valid_map, coherence, 0.0).astype(np.float32)
    features[:, :, 7] = np.broadcast_to(cross_valid_map, (6, 3, frequency_bins, time_bins)).astype(np.float32)
    features[:, :, 8] = np.broadcast_to(frequency_valid_map, (6, 3, frequency_bins, time_bins)).astype(np.float32)

    # Never let unavailable frequencies or absent axes leak interpolated values.
    features[:, :, :7] *= frequency_valid_map[:, :, None, :, :].astype(np.float32)

    return WidebandFeatureResult(
        features=features,
        frequency_mask=frequency_mask.astype(bool, copy=False),
        cross_spectral_valid=cross_valid,
        frequency_grid_hz=target_frequency_hz.astype(np.float32),
        time_grid_s=target_time_s.astype(np.float32),
    )
