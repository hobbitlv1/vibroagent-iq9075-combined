"""Numerically stable vibration-spectrum helpers used by the webchat.

The module intentionally depends only on NumPy.  It provides a raw modified
periodogram (the former "FFT PSD" behaviour), a Welch estimate for an
easier-to-read lower-variance display, and optional zero-phase post-acquisition
filters shared by the waveform and spectrum views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_TINY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class PsdEstimate:
    frequencies_hz: np.ndarray
    psd_g2_per_hz: np.ndarray
    estimator: str
    window_name: str
    record_length: int
    segment_length: int
    overlap_samples: int
    segment_count: int
    sampling_rate_hz: float
    reference_fft: np.ndarray | None = None
    reference_fft_frequencies_hz: np.ndarray | None = None

    @property
    def resolution_hz(self) -> float:
        return float(self.sampling_rate_hz / self.segment_length)

    @property
    def record_resolution_hz(self) -> float:
        return float(self.sampling_rate_hz / self.record_length)

    @property
    def segment_duration_s(self) -> float:
        return float(self.segment_length / self.sampling_rate_hz)

    @property
    def overlap_fraction(self) -> float:
        if self.segment_length <= 0:
            return 0.0
        return float(self.overlap_samples / self.segment_length)


def _signal_and_rate(signal: Any, sampling_rate_hz: float) -> tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("PSD requires at least two samples")
    if not np.all(np.isfinite(values)):
        # Removing invalid samples would close time gaps and create a spectrum
        # for a different, artificially time-compressed signal.  Reject the
        # window instead so missing/corrupt samples cannot create false peaks.
        raise ValueError("PSD signal contains non-finite samples; repair or reacquire the window")
    try:
        rate = float(sampling_rate_hz)
    except (TypeError, ValueError):
        raise ValueError("sampling rate must be a finite positive number") from None
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling rate must be a finite positive number")
    return values, rate


def periodic_window(window_name: str, length: int) -> np.ndarray:
    """Return a DFT-periodic analysis window.

    NumPy's Hann and Hamming helpers are symmetric; evaluating them at
    ``length + 1`` and dropping the final sample produces the periodic form
    used by FFT spectral estimators such as
    ``scipy.signal.get_window(..., fftbins=True)``.
    """

    if length <= 0:
        raise ValueError("FFT window length must be positive")
    normalized = str(window_name or "").strip().lower()
    if normalized in {"hamming", "hamm"}:
        if length == 1:
            return np.ones(1, dtype=np.float64)
        return np.hamming(length + 1).astype(np.float64)[:-1]
    if normalized in {"hann", "hanning"}:
        if length == 1:
            return np.ones(1, dtype=np.float64)
        return np.hanning(length + 1).astype(np.float64)[:-1]
    if normalized in {"rect", "rectangular", "boxcar", "none"}:
        return np.ones(length, dtype=np.float64)
    raise ValueError(f"unsupported FFT PSD window: {window_name}")


def normalize_frequency_filter(filter_type: str | None) -> str:
    """Return the canonical post-acquisition filter name.

    These filters are intended for the interactive waveform/PSD viewer.  They
    do not replace the sensor's analogue/digital anti-alias filtering and are
    deliberately disabled unless the caller selects one.
    """

    normalized = str(filter_type or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "highpass": "highpass",
        "high_pass": "highpass",
        "hp": "highpass",
        "lowpass": "lowpass",
        "low_pass": "lowpass",
        "lp": "lowpass",
        "bandpass": "bandpass",
        "band_pass": "bandpass",
        "bp": "bandpass",
        "notch": "notch",
        "bandstop": "notch",
        "band_stop": "notch",
        "notch_50": "notch_50",
        "notch50": "notch_50",
        "mains_50": "notch_50",
        "50hz": "notch_50",
        "notch_60": "notch_60",
        "notch60": "notch_60",
        "mains_60": "notch_60",
        "60hz": "notch_60",
    }
    canonical = aliases.get(normalized)
    if canonical is None:
        raise ValueError(
            "filter_type must be one of: none, highpass, lowpass, bandpass, notch, notch_50, notch_60"
        )
    return canonical


def _filter_float(value: Any, *, name: str, default: float) -> float:
    if value is None or value == "":
        result = float(default)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _filter_order(value: Any) -> int:
    if value is None or value == "":
        return 4
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("filter_order must be an integer from 1 to 12") from None
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("filter_order must be an integer from 1 to 12")
    order = int(numeric)
    if not 1 <= order <= 12:
        raise ValueError("filter_order must be an integer from 1 to 12")
    return order


def _butterworth_lowpass_response(frequencies_hz: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        ratio_power = np.power(frequencies_hz / float(cutoff_hz), 2 * int(order))
    response = 1.0 / np.sqrt(1.0 + ratio_power)
    response[0] = 1.0
    return np.asarray(response, dtype=np.float64)


def _butterworth_highpass_response(frequencies_hz: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
    response = np.zeros_like(frequencies_hz, dtype=np.float64)
    positive = frequencies_hz > 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio_power = np.power(float(cutoff_hz) / frequencies_hz[positive], 2 * int(order))
    response[positive] = 1.0 / np.sqrt(1.0 + ratio_power)
    return response


def _notch_response(
    frequencies_hz: np.ndarray,
    *,
    center_hz: float,
    width_hz: float,
    order: int,
) -> np.ndarray:
    """Return a smooth zero-phase notch with the requested -3 dB full width."""

    normalized_distance = 2.0 * np.abs(frequencies_hz - float(center_hz)) / float(width_hz)
    with np.errstate(over="ignore", invalid="ignore"):
        shape = np.power(normalized_distance, 2 * int(order))
    # At |f-f0| = width/2, shape == 1 and the power gain is 0.5 (-3.01 dB).
    power_gain = -np.expm1(-np.log(2.0) * shape)
    return np.sqrt(np.clip(power_gain, 0.0, 1.0)).astype(np.float64)


def _filter_padding_samples(
    *,
    sample_count: int,
    sampling_rate_hz: float,
    characteristic_hz: float,
    maximum: int = 262_144,
) -> int:
    if sample_count <= 2:
        return 0
    # Eight periods of the slowest transition component materially reduces the
    # circular edge interaction while keeping the edge-device memory bound.
    desired = int(np.ceil(8.0 * float(sampling_rate_hz) / max(float(characteristic_hz), _TINY)))
    desired = max(256, desired)
    return max(0, min(sample_count - 1, int(maximum), desired))


def _next_fast_fft_length(minimum: int) -> int:
    """Return the smallest 2/3/5-smooth FFT length at or above ``minimum``.

    NumPy may use a Bluestein transform for awkward record lengths.  On long
    IIS3DWB windows that can need several times more temporary memory than a
    nearby smooth length, so the display filter deliberately pads to a length
    made only of factors 2, 3, and 5.
    """

    target = max(1, int(minimum))
    best: int | None = None
    power_2 = 1
    while power_2 < target * 2:
        power_3 = power_2
        while power_3 < target * 2:
            power_5 = power_3
            while power_5 < target:
                power_5 *= 5
            if best is None or power_5 < best:
                best = power_5
            power_3 *= 3
        power_2 *= 2
    return int(best if best is not None else target)


def apply_frequency_filter(
    signal: Any,
    sampling_rate_hz: float,
    *,
    filter_type: str | None = "none",
    low_cutoff_hz: float | None = None,
    high_cutoff_hz: float | None = None,
    notch_frequency_hz: float | None = None,
    notch_width_hz: float | None = None,
    filter_order: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an optional zero-phase display filter to a vibration record.

    A smooth real-valued transfer function is applied to a reflection-padded
    FFT and transformed back to the time domain.  The same returned signal can
    therefore feed both the waveform and the PSD without phase displacement or
    inconsistent preprocessing.  This is an *offline analysis/display* filter,
    not a causal acquisition filter and not an anti-alias filter.
    """

    values, rate = _signal_and_rate(signal, sampling_rate_hz)
    canonical = normalize_frequency_filter(filter_type)
    order = _filter_order(filter_order)
    nyquist_hz = 0.5 * rate
    centered = values - float(np.mean(values))

    effective_type = canonical
    preset: str | None = None
    if canonical == "notch_50":
        effective_type = "notch"
        preset = "50_hz_mains"
        notch_frequency_hz = 50.0
    elif canonical == "notch_60":
        effective_type = "notch"
        preset = "60_hz_mains"
        notch_frequency_hz = 60.0

    default_low_hz = min(1.0, 0.1 * nyquist_hz)
    default_high_hz = min(6000.0, 0.9 * nyquist_hz)
    if default_low_hz >= default_high_hz:
        default_low_hz = 0.05 * nyquist_hz
        default_high_hz = 0.9 * nyquist_hz

    low_hz: float | None = None
    high_hz: float | None = None
    notch_hz: float | None = None
    width_hz: float | None = None
    characteristic_hz = max(rate / float(values.size), _TINY)

    if effective_type in {"highpass", "bandpass"}:
        low_hz = _filter_float(low_cutoff_hz, name="low_cutoff_hz", default=default_low_hz)
        if not 0.0 < low_hz < nyquist_hz:
            raise ValueError(f"low_cutoff_hz must be greater than 0 and below Nyquist ({nyquist_hz:g} Hz)")
        characteristic_hz = low_hz

    if effective_type in {"lowpass", "bandpass"}:
        high_hz = _filter_float(high_cutoff_hz, name="high_cutoff_hz", default=default_high_hz)
        if not 0.0 < high_hz < nyquist_hz:
            raise ValueError(f"high_cutoff_hz must be greater than 0 and below Nyquist ({nyquist_hz:g} Hz)")
        if effective_type == "lowpass":
            characteristic_hz = high_hz

    if effective_type == "bandpass" and low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValueError("band-pass low_cutoff_hz must be lower than high_cutoff_hz")

    if effective_type == "notch":
        notch_hz = _filter_float(notch_frequency_hz, name="notch_frequency_hz", default=50.0)
        if not 0.0 < notch_hz < nyquist_hz:
            raise ValueError(
                f"notch_frequency_hz must be greater than 0 and below Nyquist ({nyquist_hz:g} Hz)"
            )
        width_hz = _filter_float(notch_width_hz, name="notch_width_hz", default=2.0)
        maximum_width_hz = 2.0 * min(notch_hz, nyquist_hz - notch_hz)
        if not 0.0 < width_hz < maximum_width_hz:
            raise ValueError(
                "notch_width_hz must be positive and keep the -3 dB notch edges inside 0..Nyquist"
            )
        characteristic_hz = width_hz

    if effective_type == "none":
        metadata = {
            "type": "none",
            "preset": None,
            "label": "None · mean removed",
            "applied": False,
            "low_cutoff_hz": None,
            "high_cutoff_hz": None,
            "notch_frequency_hz": None,
            "notch_width_hz": None,
            "notch_width_definition": None,
            "order": None,
            "phase_response": "not applicable",
            "implementation": "mean removal only",
            "padding_samples": 0,
            "padding_left_samples": 0,
            "padding_right_samples": 0,
            "filter_fft_length": 0,
            "mean_removed": True,
            "anti_alias_filter": False,
            "scope": "analysis/display only; raw acquisition unchanged",
        }
        return np.asarray(centered, dtype=np.float64), metadata

    minimum_padding_samples = _filter_padding_samples(
        sample_count=int(values.size),
        sampling_rate_hz=rate,
        characteristic_hz=characteristic_hz,
    )
    filter_fft_length = _next_fast_fft_length(int(values.size) + 2 * minimum_padding_samples)
    total_padding = max(0, filter_fft_length - int(values.size))
    padding_left_samples = total_padding // 2
    padding_right_samples = total_padding - padding_left_samples
    padded = np.pad(
        centered,
        (padding_left_samples, padding_right_samples),
        mode="reflect",
    )
    frequencies = np.fft.rfftfreq(padded.size, d=1.0 / rate)

    if effective_type == "highpass":
        assert low_hz is not None
        response = _butterworth_highpass_response(frequencies, low_hz, order)
        label = f"High-pass {low_hz:g} Hz"
        implementation = "zero-phase FFT-domain Butterworth high-pass magnitude"
    elif effective_type == "lowpass":
        assert high_hz is not None
        response = _butterworth_lowpass_response(frequencies, high_hz, order)
        label = f"Low-pass {high_hz:g} Hz"
        implementation = "zero-phase FFT-domain Butterworth low-pass magnitude"
    elif effective_type == "bandpass":
        assert low_hz is not None and high_hz is not None
        response = _butterworth_highpass_response(frequencies, low_hz, order)
        response *= _butterworth_lowpass_response(frequencies, high_hz, order)
        label = f"Band-pass {low_hz:g}–{high_hz:g} Hz"
        implementation = "zero-phase FFT-domain Butterworth band-pass magnitude"
    else:
        assert notch_hz is not None and width_hz is not None
        response = _notch_response(
            frequencies,
            center_hz=notch_hz,
            width_hz=width_hz,
            order=order,
        )
        label = f"Notch {notch_hz:g} Hz · {width_hz:g} Hz width"
        implementation = "zero-phase FFT-domain smooth -3 dB-width notch"

    spectrum = np.fft.rfft(padded)
    spectrum *= response
    filtered_padded = np.fft.irfft(spectrum, n=padded.size)
    filtered = filtered_padded[padding_left_samples : padding_left_samples + values.size]
    # Reflection padding can leave a tiny numerical DC residual after cropping.
    filtered = np.asarray(filtered, dtype=np.float64)
    filtered -= float(np.mean(filtered))

    metadata = {
        "type": effective_type,
        "preset": preset,
        "label": label,
        "applied": True,
        "low_cutoff_hz": float(low_hz) if low_hz is not None else None,
        "high_cutoff_hz": float(high_hz) if high_hz is not None else None,
        "notch_frequency_hz": float(notch_hz) if notch_hz is not None else None,
        "notch_width_hz": float(width_hz) if width_hz is not None else None,
        "notch_width_definition": "full width between -3 dB power points" if width_hz is not None else None,
        "order": int(order),
        "phase_response": "zero phase",
        "implementation": implementation,
        "padding_samples": int(max(padding_left_samples, padding_right_samples)),
        "padding_left_samples": int(padding_left_samples),
        "padding_right_samples": int(padding_right_samples),
        "filter_fft_length": int(filter_fft_length),
        "mean_removed": True,
        "anti_alias_filter": False,
        "scope": "analysis/display only; raw acquisition unchanged",
    }
    return filtered, metadata


def _one_sided_density(spectrum: np.ndarray, *, sampling_rate_hz: float, window: np.ndarray, n: int) -> np.ndarray:
    window_power = float(np.dot(window, window))
    if not np.isfinite(window_power) or window_power <= 0:
        raise ValueError("FFT analysis window has zero power")
    psd = (np.abs(spectrum) ** 2) / (float(sampling_rate_hz) * window_power)
    if psd.size > 1:
        if n % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
    return np.asarray(psd, dtype=np.float64)


def compute_periodogram_psd(
    signal: Any,
    sampling_rate_hz: float,
    *,
    window_name: str = "hann",
) -> PsdEstimate:
    """Return a one-sided modified periodogram with density scaling.

    The normalization is ``|rFFT(x*w)|² / (fs * sum(w²))`` with the negative
    frequency energy folded into the positive side, excluding DC and Nyquist.
    """

    values, rate = _signal_and_rate(signal, sampling_rate_hz)
    centered = values - float(np.mean(values))
    n = int(centered.size)
    window = periodic_window(window_name, n)
    spectrum = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(n, d=1.0 / rate)
    psd = _one_sided_density(spectrum, sampling_rate_hz=rate, window=window, n=n)
    return PsdEstimate(
        frequencies_hz=frequencies,
        psd_g2_per_hz=psd,
        estimator="periodogram",
        window_name=window_name,
        record_length=n,
        segment_length=n,
        overlap_samples=0,
        segment_count=1,
        sampling_rate_hz=rate,
        reference_fft=spectrum,
        reference_fft_frequencies_hz=frequencies,
    )


def _nearest_power_of_two(value: float, *, maximum: int) -> int:
    if maximum < 2:
        return maximum
    target = max(2.0, min(float(maximum), float(value)))
    exponent = int(round(np.log2(target)))
    candidate = 1 << max(1, exponent)
    if candidate > maximum:
        candidate = 1 << int(np.floor(np.log2(maximum)))
    return max(2, min(maximum, candidate))


def compute_welch_psd(
    signal: Any,
    sampling_rate_hz: float,
    *,
    window_name: str = "hann",
    target_resolution_hz: float = 0.25,
    overlap_fraction: float = 0.5,
    minimum_segment_count: int = 3,
) -> PsdEstimate:
    """Estimate a one-sided PSD by averaging overlapping periodograms.

    Segment length is selected near the requested frequency resolution and
    rounded to a power of two for predictable edge-device FFT performance.
    """

    values, rate = _signal_and_rate(signal, sampling_rate_hz)
    n = int(values.size)
    try:
        target_resolution = float(target_resolution_hz)
    except (TypeError, ValueError):
        target_resolution = 0.25
    if not np.isfinite(target_resolution) or target_resolution <= 0:
        target_resolution = 0.25

    try:
        overlap = float(overlap_fraction)
    except (TypeError, ValueError):
        overlap = 0.5
    if not np.isfinite(overlap):
        overlap = 0.5
    overlap = min(0.9, max(0.0, overlap))

    # A Welch label is useful only when the record is actually averaged.  Cap
    # nperseg so short IIS3DWB captures still contain at least three segments
    # whenever the sample count permits it.  The requested resolution remains
    # a target; the returned ``resolution_hz`` reports the achieved value.
    try:
        required_segments = max(1, int(minimum_segment_count))
    except (TypeError, ValueError):
        required_segments = 3
    if n < 8:
        required_segments = 1

    denominator = 1.0 + max(0, required_segments - 1) * (1.0 - overlap)
    maximum_for_averaging = max(2, min(n, int(np.floor(n / max(1.0, denominator)))))
    target_length = rate / target_resolution
    segment_length = _nearest_power_of_two(target_length, maximum=maximum_for_averaging)

    overlap_samples = min(segment_length - 1, int(round(segment_length * overlap)))
    step = max(1, segment_length - overlap_samples)
    starts = list(range(0, n - segment_length + 1, step))
    # Rounding to a power of two normally gives at least the requested count.
    # Reduce once more if unusual overlap rounding prevents that guarantee.
    while len(starts) < required_segments and segment_length > 2:
        segment_length //= 2
        overlap_samples = min(segment_length - 1, int(round(segment_length * overlap)))
        step = max(1, segment_length - overlap_samples)
        starts = list(range(0, n - segment_length + 1, step))
    if not starts:
        starts = [0]
        segment_length = n
        overlap_samples = 0

    window = periodic_window(window_name, segment_length)
    accumulated = np.zeros(segment_length // 2 + 1, dtype=np.float64)
    for start in starts:
        segment = np.asarray(values[start : start + segment_length], dtype=np.float64)
        segment = segment - float(np.mean(segment))
        spectrum = np.fft.rfft(segment * window)
        accumulated += _one_sided_density(
            spectrum,
            sampling_rate_hz=rate,
            window=window,
            n=segment_length,
        )

    psd = accumulated / float(len(starts))
    frequencies = np.fft.rfftfreq(segment_length, d=1.0 / rate)
    return PsdEstimate(
        frequencies_hz=frequencies,
        psd_g2_per_hz=psd,
        estimator="welch",
        window_name=window_name,
        record_length=n,
        segment_length=segment_length,
        overlap_samples=overlap_samples,
        segment_count=len(starts),
        sampling_rate_hz=rate,
    )


def aggregate_waveform_for_plot(
    signal: Any,
    sampling_rate_hz: float,
    *,
    timestamps_s: Any | None = None,
    max_points: int = 900,
    center: bool = True,
) -> dict[str, Any]:
    """Return a bounded, peak-preserving time-domain representation.

    The PSD is always computed from the full-rate signal.  This helper is only
    for browser rendering: long records are grouped into time buckets and each
    bucket returns its minimum, maximum, and mean.  Consequently, short impacts
    are not hidden by naïve point skipping and no display decimation can alter
    the spectrum.
    """

    values, rate = _signal_and_rate(signal, sampling_rate_hz)
    displayed = values - float(np.mean(values)) if center else values.copy()
    n = int(displayed.size)
    try:
        point_limit = int(max_points)
    except (TypeError, ValueError):
        point_limit = 900
    point_limit = max(2, point_limit)

    timestamp_source = "sampling_rate"
    relative_times: np.ndarray
    if timestamps_s is not None:
        candidate = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
        if (
            candidate.size == n
            and np.all(np.isfinite(candidate))
            and np.all(np.diff(candidate) >= 0.0)
            and (candidate[-1] > candidate[0] or n == 1)
        ):
            relative_times = candidate - float(candidate[0])
            timestamp_source = "sdk_timestamps"
        else:
            relative_times = np.arange(n, dtype=np.float64) / rate
    else:
        relative_times = np.arange(n, dtype=np.float64) / rate

    if n <= point_limit:
        return {
            "time_s": relative_times,
            "min_g": displayed.copy(),
            "max_g": displayed.copy(),
            "mean_g": displayed.copy(),
            "point_count": n,
            "source_sample_count": n,
            "downsampled": False,
            "centered": bool(center),
            "timestamp_source": timestamp_source,
            "display_method": "full_rate_line",
        }

    edges = np.linspace(0, n, point_limit + 1, dtype=np.int64)
    plot_time = np.empty(point_limit, dtype=np.float64)
    plot_min = np.empty(point_limit, dtype=np.float64)
    plot_max = np.empty(point_limit, dtype=np.float64)
    plot_mean = np.empty(point_limit, dtype=np.float64)
    for index in range(point_limit):
        start = int(edges[index])
        end = max(start + 1, int(edges[index + 1]))
        chunk = displayed[start:end]
        plot_time[index] = 0.5 * (relative_times[start] + relative_times[end - 1])
        plot_min[index] = float(np.min(chunk))
        plot_max[index] = float(np.max(chunk))
        plot_mean[index] = float(np.mean(chunk))

    return {
        "time_s": plot_time,
        "min_g": plot_min,
        "max_g": plot_max,
        "mean_g": plot_mean,
        "point_count": point_limit,
        "source_sample_count": n,
        "downsampled": True,
        "centered": bool(center),
        "timestamp_source": timestamp_source,
        "display_method": "min_max_mean_envelope",
    }


def integrated_psd_power(frequencies_hz: Any, psd: Any) -> float:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    density = np.asarray(psd, dtype=np.float64).reshape(-1)
    if frequencies.size != density.size or frequencies.size < 2:
        return 0.0
    spacing = float(np.median(np.diff(frequencies)))
    if not np.isfinite(spacing) or spacing <= 0:
        return 0.0
    finite = np.isfinite(density) & (density >= 0)
    return float(np.sum(density[finite]) * spacing)


def _parabolic_peak(frequencies: np.ndarray, psd: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= psd.size - 1:
        return float(frequencies[index]), float(psd[index])
    y0, y1, y2 = np.log(np.maximum(psd[index - 1 : index + 2], _TINY))
    denominator = y0 - 2.0 * y1 + y2
    if not np.isfinite(denominator) or abs(denominator) < 1e-18:
        return float(frequencies[index]), float(psd[index])
    offset = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
    spacing = float(frequencies[index + 1] - frequencies[index])
    refined_frequency = float(frequencies[index] + offset * spacing)
    refined_log_power = float(y1 - 0.25 * (y0 - y2) * offset)
    return refined_frequency, float(np.exp(refined_log_power))


def find_distinct_peaks(
    frequencies_hz: Any,
    psd: Any,
    *,
    limit: int = 8,
    min_distance_hz: float | None = None,
    min_prominence_db: float = 3.0,
    relative_floor_db: float = -50.0,
    reference_fft: Any | None = None,
    reference_fft_frequencies_hz: Any | None = None,
) -> list[dict[str, float | None]]:
    """Return distinct local maxima rather than several bins from one lobe."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    density = np.asarray(psd, dtype=np.float64).reshape(-1)
    if frequencies.size != density.size or frequencies.size < 3:
        return []

    valid = np.isfinite(frequencies) & np.isfinite(density) & (frequencies > 0) & (density > 0)
    local = np.zeros(density.size, dtype=bool)
    local[1:-1] = (density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])
    candidates = np.flatnonzero(valid & local)
    valid_indexes = np.flatnonzero(valid)
    if valid_indexes.size == 0:
        return []
    global_max = float(np.max(density[valid_indexes]))
    try:
        relative_threshold = global_max * (10.0 ** (float(relative_floor_db) / 10.0))
    except (TypeError, ValueError):
        relative_threshold = global_max * 1e-5
    candidates = candidates[density[candidates] >= max(_TINY, relative_threshold)]
    if candidates.size == 0:
        candidates = np.asarray([valid_indexes[int(np.argmax(density[valid_indexes]))]], dtype=np.int64)

    spacing = float(np.median(np.diff(frequencies)))
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 0.0
    if min_distance_hz is None:
        min_distance = max(0.25, 4.0 * spacing)
    else:
        try:
            min_distance = max(0.0, float(min_distance_hz))
        except (TypeError, ValueError):
            min_distance = max(0.25, 4.0 * spacing)

    ordered = candidates[np.argsort(density[candidates])[::-1]]
    selected: list[int] = []
    for index in ordered:
        frequency = float(frequencies[index])
        if all(abs(frequency - float(frequencies[other])) >= min_distance for other in selected):
            selected.append(int(index))
        if len(selected) >= max(1, int(limit)):
            break

    reference_spectrum = None if reference_fft is None else np.asarray(reference_fft).reshape(-1)
    reference_frequencies = (
        None
        if reference_fft_frequencies_hz is None
        else np.asarray(reference_fft_frequencies_hz, dtype=np.float64).reshape(-1)
    )

    peaks: list[dict[str, float | None]] = []
    neighborhood_bins = max(12, int(round(max(min_distance, spacing) / max(spacing, _TINY))))
    for index in selected:
        refined_frequency, refined_power = _parabolic_peak(frequencies, density, index)
        lo = max(0, index - neighborhood_bins)
        hi = min(density.size, index + neighborhood_bins + 1)
        surrounding = np.concatenate((density[lo : max(lo, index - 2)], density[min(hi, index + 3) : hi]))
        surrounding = surrounding[np.isfinite(surrounding) & (surrounding > 0)]
        baseline = float(np.median(surrounding)) if surrounding.size else _TINY
        prominence_db = float(10.0 * np.log10(max(refined_power, _TINY) / max(baseline, _TINY)))

        phase_degrees: float | None = None
        if reference_spectrum is not None and reference_frequencies is not None and reference_spectrum.size:
            ref_index = int(np.argmin(np.abs(reference_frequencies - refined_frequency)))
            if 0 <= ref_index < reference_spectrum.size:
                phase_degrees = float(np.angle(reference_spectrum[ref_index], deg=True))

        half_band_hz = max(2.0 * spacing, min_distance * 0.5)
        band_mask = np.abs(frequencies - refined_frequency) <= half_band_hz
        band_power = integrated_psd_power(frequencies[band_mask], density[band_mask]) if np.count_nonzero(band_mask) >= 2 else 0.0
        if prominence_db < float(min_prominence_db):
            continue
        peaks.append(
            {
                "frequency_hz": refined_frequency,
                "bin_frequency_hz": float(frequencies[index]),
                "psd_g2_per_hz": refined_power,
                "psd_db_re_1_g2_per_hz": float(10.0 * np.log10(max(refined_power, _TINY))),
                "asd_g_per_sqrt_hz": float(np.sqrt(max(refined_power, 0.0))),
                "band_rms_g": float(np.sqrt(max(band_power, 0.0))),
                "prominence_db": prominence_db,
                "phase_degrees": phase_degrees,
            }
        )
    return peaks


def analyze_impact_ringdown(
    signal: Any,
    sampling_rate_hz: float,
    *,
    window_name: str = "hann",
    minimum_frequency_hz: float = 0.5,
    maximum_frequency_hz: float = 200.0,
    impact_skip_s: float = 0.005,
    minimum_ringdown_s: float = 0.5,
    candidate_durations_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    event_block_s: float = 0.02,
    clipping_suspected: bool = False,
) -> dict[str, Any]:
    """Detect an impact and estimate a resonant frequency from its ring-down.

    A long Welch average is intentionally insensitive to a brief impact: only
    one or two segments contain the event, and the direct impulse is broadband.
    For an impact test the physically useful frequency is normally found in the
    decaying response *after* the contact impulse.  This helper therefore keeps
    the normal PSD untouched and performs a separate, bounded multi-duration
    periodogram analysis on the post-impact ring-down.

    The function is conservative.  It reports no ring-down peak when the event
    is only a broadband impulse, when there is too little post-event data, or
    when the post-event energy does not decay above the robust background.
    """

    values, rate = _signal_and_rate(signal, sampling_rate_hz)
    centered = values - float(np.median(values))
    sample_count = int(centered.size)
    tiny = np.finfo(np.float64).tiny

    try:
        block_duration_s = float(event_block_s)
    except (TypeError, ValueError):
        block_duration_s = 0.02
    if not np.isfinite(block_duration_s) or block_duration_s <= 0:
        block_duration_s = 0.02
    block_length = max(4, int(round(rate * block_duration_s)))
    block_starts = np.arange(0, sample_count, block_length, dtype=np.int64)
    squared = centered * centered
    block_sums = np.add.reduceat(squared, block_starts)
    block_counts = np.minimum(block_length, sample_count - block_starts)
    block_rms = np.sqrt(block_sums / np.maximum(block_counts, 1))

    background_rms = float(np.median(block_rms)) if block_rms.size else 0.0
    block_mad = float(np.median(np.abs(block_rms - background_rms))) if block_rms.size else 0.0
    robust_scale = 1.4826 * block_mad
    event_block_index = int(np.argmax(block_rms)) if block_rms.size else 0
    event_block_start = int(block_starts[event_block_index]) if block_starts.size else 0
    event_block_stop = min(sample_count, event_block_start + block_length)
    event_index = event_block_start + int(
        np.argmax(np.abs(centered[event_block_start:event_block_stop]))
    )

    event_block_rms = float(block_rms[event_block_index]) if block_rms.size else 0.0
    event_rms_ratio = float(event_block_rms / max(background_rms, tiny))
    event_robust_z = float(
        (event_block_rms - background_rms)
        / max(robust_scale, 0.05 * background_rms, tiny)
    )
    overall_rms = float(np.sqrt(np.mean(centered * centered)))
    absolute_peak_g = float(np.max(np.abs(centered)))
    crest_factor = float(absolute_peak_g / max(overall_rms, tiny))
    event_detected = bool(
        (event_rms_ratio >= 4.0 and event_robust_z >= 8.0)
        or (event_rms_ratio >= 2.5 and crest_factor >= 8.0)
    )

    result: dict[str, Any] = {
        "detected": event_detected,
        "status": "no_transient",
        "message": "No isolated impact was detected in this analysis window.",
        "event_index": int(event_index),
        "event_time_s": float(event_index / rate),
        "event_block_duration_s": float(block_length / rate),
        "event_block_rms_g": event_block_rms,
        "background_rms_g": background_rms,
        "event_rms_ratio": event_rms_ratio,
        "event_robust_z": event_robust_z,
        "crest_factor": crest_factor,
        "absolute_peak_g": absolute_peak_g,
        "clipping_suspected": bool(clipping_suspected),
        "ringdown_start_index": None,
        "ringdown_start_s": None,
        "available_post_event_s": 0.0,
        "analysis_duration_s": None,
        "early_ringdown_rms_g": None,
        "late_ringdown_rms_g": None,
        "ringdown_energy_ratio": None,
        "decay_ratio": None,
        "peak_frequency_hz": None,
        "peak_bin_frequency_hz": None,
        "peak_psd_g2_per_hz": None,
        "peak_psd_db_re_1_g2_per_hz": None,
        "peak_prominence_db": None,
        "peak_reliable": False,
        "supporting_duration_count": 0,
        "supporting_durations_s": [],
        "frequencies_hz": np.asarray([], dtype=np.float64),
        "psd_g2_per_hz": np.asarray([], dtype=np.float64),
    }
    if not event_detected:
        return result

    try:
        skip_s = float(impact_skip_s)
    except (TypeError, ValueError):
        skip_s = 0.005
    if not np.isfinite(skip_s) or skip_s < 0:
        skip_s = 0.005
    ringdown_start = min(sample_count, event_index + max(1, int(round(skip_s * rate))))
    available_post_event_s = float((sample_count - ringdown_start) / rate)
    result.update(
        {
            "status": "impact_detected",
            "message": "Impact detected; checking the decaying response after contact.",
            "ringdown_start_index": int(ringdown_start),
            "ringdown_start_s": float(ringdown_start / rate),
            "available_post_event_s": available_post_event_s,
        }
    )

    try:
        minimum_duration_s = float(minimum_ringdown_s)
    except (TypeError, ValueError):
        minimum_duration_s = 0.5
    if not np.isfinite(minimum_duration_s) or minimum_duration_s <= 0:
        minimum_duration_s = 0.5
    if available_post_event_s < minimum_duration_s:
        result.update(
            {
                "status": "need_more_post_event_data",
                "message": (
                    "Impact detected too close to the end of the window; wait briefly "
                    "for the ring-down and run the spectrum again."
                ),
            }
        )
        return result

    try:
        minimum_frequency = max(0.0, float(minimum_frequency_hz))
    except (TypeError, ValueError):
        minimum_frequency = 0.5
    try:
        maximum_frequency = min(0.5 * rate, float(maximum_frequency_hz))
    except (TypeError, ValueError):
        maximum_frequency = min(0.5 * rate, 200.0)
    if not np.isfinite(minimum_frequency):
        minimum_frequency = 0.5
    if not np.isfinite(maximum_frequency) or maximum_frequency <= minimum_frequency:
        maximum_frequency = min(0.5 * rate, 200.0)

    durations: list[float] = []
    for raw_duration in candidate_durations_s:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite(duration)
            and duration >= minimum_duration_s
            and duration <= available_post_event_s + 1e-12
            and duration not in durations
        ):
            durations.append(duration)
    durations.sort()
    if not durations:
        durations = [available_post_event_s]

    candidates: list[dict[str, Any]] = []
    estimates: dict[float, tuple[np.ndarray, np.ndarray, float, float, float, float]] = {}
    for duration_s in durations:
        duration_samples = min(
            sample_count - ringdown_start,
            max(2, int(round(duration_s * rate))),
        )
        segment = np.asarray(
            centered[ringdown_start : ringdown_start + duration_samples],
            dtype=np.float64,
        )
        if segment.size < 2:
            continue
        edge_count = max(1, int(segment.size // 5))
        early_rms = float(np.sqrt(np.mean(segment[:edge_count] ** 2)))
        late_rms = float(np.sqrt(np.mean(segment[-edge_count:] ** 2)))
        segment_rms = float(np.sqrt(np.mean(segment * segment)))
        energy_ratio = float(early_rms / max(background_rms, tiny))
        decay_ratio = float(early_rms / max(late_rms, tiny))

        # A direct impulse followed only by background noise is broadband and
        # must not manufacture a resonance from a random noise-bin maximum.
        if energy_ratio < 1.35 or decay_ratio < 1.10:
            continue

        estimate = compute_periodogram_psd(segment, rate, window_name=window_name)
        effective_duration_s = float(segment.size / rate)
        lower_hz = max(minimum_frequency, 1.0 / max(effective_duration_s, tiny))
        mask = (
            (estimate.frequencies_hz >= lower_hz)
            & (estimate.frequencies_hz <= maximum_frequency)
        )
        frequencies = estimate.frequencies_hz[mask]
        density = estimate.psd_g2_per_hz[mask]
        if frequencies.size < 3:
            continue
        peaks = find_distinct_peaks(
            frequencies,
            density,
            limit=5,
            min_distance_hz=max(0.5, 4.0 / max(effective_duration_s, tiny)),
            min_prominence_db=3.0,
        )
        estimates[duration_s] = (
            np.asarray(frequencies, dtype=np.float64),
            np.asarray(density, dtype=np.float64),
            early_rms,
            late_rms,
            energy_ratio,
            decay_ratio,
        )
        for peak in peaks:
            frequency_hz = float(peak["frequency_hz"])
            cycles = float(frequency_hz * effective_duration_s)
            if cycles < 3.0:
                continue
            prominence_db = float(peak.get("prominence_db") or 0.0)
            score = float(
                prominence_db
                + 6.0 * np.log10(max(energy_ratio, 1.0))
                + 2.0 * np.log10(max(decay_ratio, 1.0))
            )
            candidates.append(
                {
                    **peak,
                    "duration_s": duration_s,
                    "effective_duration_s": effective_duration_s,
                    "cycles": cycles,
                    "early_rms_g": early_rms,
                    "late_rms_g": late_rms,
                    "segment_rms_g": segment_rms,
                    "energy_ratio": energy_ratio,
                    "decay_ratio": decay_ratio,
                    "score": score,
                }
            )

    clusters: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
        frequency_hz = float(candidate["frequency_hz"])
        tolerance_hz = max(1.0, 2.0 / max(float(candidate["effective_duration_s"]), tiny))
        matched: dict[str, Any] | None = None
        for cluster in clusters:
            if abs(frequency_hz - float(cluster["frequency_hz"])) <= max(
                tolerance_hz, float(cluster["tolerance_hz"])
            ):
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    "frequency_hz": frequency_hz,
                    "tolerance_hz": tolerance_hz,
                    "items": [candidate],
                }
            )
            continue
        matched["items"].append(candidate)
        weights = np.asarray(
            [max(1.0, float(item.get("prominence_db") or 0.0)) for item in matched["items"]],
            dtype=np.float64,
        )
        matched["frequency_hz"] = float(
            np.average(
                [float(item["frequency_hz"]) for item in matched["items"]],
                weights=weights,
            )
        )
        matched["tolerance_hz"] = max(float(matched["tolerance_hz"]), tolerance_hz)

    for cluster in clusters:
        items = cluster["items"]
        unique_durations = sorted({float(item["duration_s"]) for item in items})
        best = max(items, key=lambda item: float(item["score"]))
        cluster["best"] = best
        cluster["supporting_durations_s"] = unique_durations
        cluster["supporting_duration_count"] = len(unique_durations)
        cluster["score"] = float(best["score"] + 4.0 * max(0, len(unique_durations) - 1))
    clusters.sort(key=lambda item: float(item["score"]), reverse=True)

    if not clusters:
        result.update(
            {
                "status": "broadband_impact_no_ringdown_peak",
                "message": (
                    "Impact detected, but the post-impact response is broadband or "
                    "does not contain a repeatable resonant peak."
                ),
            }
        )
        return result

    selected_cluster = clusters[0]
    selected = selected_cluster["best"]
    selected_duration = float(selected["duration_s"])
    selected_frequencies, selected_density = estimates[selected_duration][:2]
    support_count = int(selected_cluster["supporting_duration_count"])
    reliable = bool(
        (
            support_count >= 2
            or (
                float(selected.get("prominence_db") or 0.0) >= 8.0
                and float(selected["energy_ratio"]) >= 2.0
            )
        )
        and float(selected["decay_ratio"]) >= 1.10
        and float(selected["cycles"]) >= 3.0
        and not bool(clipping_suspected)
    )
    status = "ringdown_peak" if reliable else "ringdown_peak_low_confidence"
    if clipping_suspected:
        status = "ringdown_peak_clipping_suspected"
    message = (
        "Impact ring-down resonance estimated from the decaying post-impact response."
        if reliable
        else "A ring-down frequency was estimated, but treat it as low confidence."
    )
    if clipping_suspected:
        message = (
            "A ring-down frequency was estimated, but the impact may have clipped "
            "the sensor; repeat with a gentler tap."
        )

    result.update(
        {
            "status": status,
            "message": message,
            "analysis_duration_s": float(selected["effective_duration_s"]),
            "early_ringdown_rms_g": float(selected["early_rms_g"]),
            "late_ringdown_rms_g": float(selected["late_rms_g"]),
            "ringdown_energy_ratio": float(selected["energy_ratio"]),
            "decay_ratio": float(selected["decay_ratio"]),
            "peak_frequency_hz": float(selected_cluster["frequency_hz"]),
            "peak_bin_frequency_hz": float(selected["bin_frequency_hz"]),
            "peak_psd_g2_per_hz": float(selected["psd_g2_per_hz"]),
            "peak_psd_db_re_1_g2_per_hz": float(selected["psd_db_re_1_g2_per_hz"]),
            "peak_prominence_db": float(selected["prominence_db"]),
            "peak_reliable": reliable,
            "supporting_duration_count": support_count,
            "supporting_durations_s": list(selected_cluster["supporting_durations_s"]),
            "frequencies_hz": np.asarray(selected_frequencies, dtype=np.float64),
            "psd_g2_per_hz": np.asarray(selected_density, dtype=np.float64),
        }
    )
    return result


def _plot_edges(point_count: int, max_bins: int, frequency_scale: str) -> np.ndarray:
    bin_count = max(1, min(int(max_bins), int(point_count)))
    if bin_count >= point_count:
        return np.arange(point_count + 1, dtype=np.int64)
    if frequency_scale == "log":
        # Logarithmic index edges guarantee non-empty buckets while allocating
        # substantially more visual detail to low frequencies.
        edges = np.rint(np.geomspace(1.0, float(point_count), bin_count + 1)).astype(np.int64) - 1
        edges[0] = 0
        edges[-1] = point_count
        edges = np.unique(edges)
        if edges[-1] != point_count:
            edges = np.append(edges, point_count)
        return edges
    return np.linspace(0, point_count, bin_count + 1, dtype=np.int64)


def aggregate_spectrum_for_plot(
    frequencies_hz: Any,
    psd: Any,
    *,
    max_bins: int,
    frequency_scale: str = "log",
) -> dict[str, np.ndarray]:
    """Downsample a spectrum without replacing the noise floor with maxima.

    Each bucket returns a median floor value and a separate peak envelope.  The
    old implementation returned only the bucket maximum and connected those
    maxima, which visually overstated broadband vibration.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    density = np.asarray(psd, dtype=np.float64).reshape(-1)
    if frequencies.size != density.size:
        raise ValueError("frequency and PSD arrays must have the same length")
    if frequencies.size == 0:
        return {
            "frequency_hz": np.asarray([], dtype=np.float64),
            "psd_g2_per_hz": np.asarray([], dtype=np.float64),
            "peak_frequency_hz": np.asarray([], dtype=np.float64),
            "peak_psd_g2_per_hz": np.asarray([], dtype=np.float64),
        }

    normalized_scale = "log" if str(frequency_scale).strip().lower() == "log" else "linear"
    edges = _plot_edges(frequencies.size, max_bins, normalized_scale)
    floor_frequencies: list[float] = []
    floor_values: list[float] = []
    peak_frequencies: list[float] = []
    peak_values: list[float] = []

    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment_f = frequencies[start:stop]
        segment_p = density[start:stop]
        finite = np.isfinite(segment_f) & np.isfinite(segment_p) & (segment_p >= 0)
        if not np.any(finite):
            continue
        segment_f = segment_f[finite]
        segment_p = segment_p[finite]
        positive = segment_p[segment_p > 0]
        floor = float(np.median(positive)) if positive.size else 0.0
        peak_index = int(np.argmax(segment_p))
        if normalized_scale == "log" and np.all(segment_f > 0):
            center_frequency = float(np.exp(np.mean(np.log(segment_f))))
        else:
            center_frequency = float(np.mean(segment_f))
        floor_frequencies.append(center_frequency)
        floor_values.append(floor)
        peak_frequencies.append(float(segment_f[peak_index]))
        peak_values.append(float(segment_p[peak_index]))

    return {
        "frequency_hz": np.asarray(floor_frequencies, dtype=np.float64),
        "psd_g2_per_hz": np.asarray(floor_values, dtype=np.float64),
        "peak_frequency_hz": np.asarray(peak_frequencies, dtype=np.float64),
        "peak_psd_g2_per_hz": np.asarray(peak_values, dtype=np.float64),
    }


def frequency_band_summary(frequencies_hz: Any, psd: Any) -> list[dict[str, float | str]]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    density = np.asarray(psd, dtype=np.float64).reshape(-1)
    if frequencies.size != density.size or frequencies.size < 2:
        return []
    nyquist = float(np.max(frequencies))
    edges = [0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0, nyquist]
    cleaned: list[float] = []
    for edge in edges:
        value = min(nyquist, max(0.0, edge))
        if not cleaned or value > cleaned[-1]:
            cleaned.append(value)
    if len(cleaned) < 2:
        return []
    total_power = integrated_psd_power(frequencies, density)
    result: list[dict[str, float | str]] = []
    for index, (low, high) in enumerate(zip(cleaned[:-1], cleaned[1:])):
        include_high = index == len(cleaned) - 2
        mask = (frequencies >= low) & (frequencies <= high if include_high else frequencies < high)
        power = integrated_psd_power(frequencies[mask], density[mask]) if np.count_nonzero(mask) >= 2 else 0.0
        label = f"{low:g}-{high:g} Hz" if high < nyquist else f"{low:g}-{nyquist:g} Hz"
        result.append(
            {
                "label": label,
                "low_hz": float(low),
                "high_hz": float(high),
                "power_g2": float(power),
                "rms_g": float(np.sqrt(max(power, 0.0))),
                "fraction": float(power / total_power) if total_power > 0 else 0.0,
            }
        )
    return result
