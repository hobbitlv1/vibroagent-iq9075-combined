"""Signal pre-processing and building-monitoring feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class ProcessedSignal:
    raw: np.ndarray
    centered: np.ndarray
    standardized: np.ndarray
    sampling_rate_hz: int


def preprocess_signal(signal: np.ndarray, sampling_rate_hz: int) -> ProcessedSignal:
    """Clean and normalize a vibration signal.

    The returned `centered` signal keeps amplitude scale, while `standardized`
    is unitless and is useful for shape/spectral operations.
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 32:
        raise ValueError("Signal must contain at least 32 finite samples")

    # Conservative de-meaning. Real deployments can add band-pass filtering here.
    centered = x - float(np.mean(x))
    std = float(np.std(centered))
    if std < EPS:
        standardized = np.zeros_like(centered)
    else:
        standardized = centered / std

    return ProcessedSignal(raw=x, centered=centered, standardized=standardized, sampling_rate_hz=sampling_rate_hz)


def compute_features(processed: ProcessedSignal) -> dict[str, Any]:
    """Compute compact time- and frequency-domain features for building monitoring."""
    x = processed.centered
    z = processed.standardized
    sr = processed.sampling_rate_hz

    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    mean_abs = float(np.mean(abs_x))
    std = float(np.std(x))
    variance = float(np.var(x))
    abs_peak = float(np.max(abs_x))
    peak_to_peak = float(np.ptp(x))

    if std < EPS:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))  # Normal distribution is approximately 3.

    crest_factor = float(abs_peak / (rms + EPS))
    impulse_factor = float(abs_peak / (mean_abs + EPS))
    shape_factor = float(rms / (mean_abs + EPS))
    zero_crossing_rate = float(np.mean(np.diff(np.signbit(x)).astype(np.float64)))
    energy = float(np.sum(x**2))

    fft = compute_fft_features(x, sr)

    return {
        "time_domain": {
            "num_samples": int(x.size),
            "duration_s": float(x.size / sr),
            "mean": float(np.mean(processed.raw)),
            "rms": rms,
            "std": std,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "crest_factor": crest_factor,
            "impulse_factor": impulse_factor,
            "shape_factor": shape_factor,
            "abs_peak": abs_peak,
            "peak_to_peak": peak_to_peak,
            "zero_crossing_rate": zero_crossing_rate,
            "energy": energy,
        },
        "frequency_domain": fft,
    }


def compute_fft_features(signal_centered: np.ndarray, sampling_rate_hz: int, top_k: int = 5) -> dict[str, Any]:
    """Compute FFT-based summary features."""
    x = np.asarray(signal_centered, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 32:
        raise ValueError("Signal must contain at least 32 samples")

    window = np.hanning(n)
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    mag = np.abs(spectrum)

    # Ignore DC for vibration dominant-frequency estimates.
    if mag.size > 1:
        mag_no_dc = mag.copy()
        mag_no_dc[0] = 0.0
    else:
        mag_no_dc = mag

    total_mag = float(np.sum(mag_no_dc) + EPS)
    dominant_idx = int(np.argmax(mag_no_dc))
    dominant_frequency_hz = float(freqs[dominant_idx])
    dominant_magnitude = float(mag_no_dc[dominant_idx])

    spectral_centroid_hz = float(np.sum(freqs * mag_no_dc) / total_mag)
    spectral_bandwidth_hz = float(np.sqrt(np.sum(((freqs - spectral_centroid_hz) ** 2) * mag_no_dc) / total_mag))
    spectral_flatness = float(np.exp(np.mean(np.log(mag_no_dc + EPS))) / (np.mean(mag_no_dc + EPS)))

    top_k = max(1, min(top_k, mag_no_dc.size))
    # argpartition is efficient; sort selected peaks descending afterwards.
    peak_indices = np.argpartition(mag_no_dc, -top_k)[-top_k:]
    peak_indices = peak_indices[np.argsort(mag_no_dc[peak_indices])[::-1]]
    top_peaks = [
        {
            "frequency_hz": float(freqs[i]),
            "magnitude": float(mag_no_dc[i]),
        }
        for i in peak_indices
    ]

    # Coarse band powers for compact LLM explanation.
    nyquist = sampling_rate_hz / 2.0
    candidate_bands = [(0, 100), (100, 300), (300, 800), (800, 2000), (2000, nyquist)]
    bands = []
    for lo, hi in candidate_bands:
        if lo >= nyquist:
            continue
        bands.append((lo, min(hi, nyquist)))

    power = mag_no_dc**2
    total_power = float(np.sum(power) + EPS)
    band_powers: dict[str, float] = {}
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        key = f"{int(lo)}_{int(hi)}_hz"
        band_powers[key] = float(np.sum(power[mask]) / total_power)

    return {
        "dominant_frequency_hz": dominant_frequency_hz,
        "dominant_magnitude": dominant_magnitude,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "spectral_flatness": spectral_flatness,
        "top_peaks": top_peaks,
        "relative_band_powers": band_powers,
    }


def compute_time_frequency_summary(
    processed: ProcessedSignal,
    frame_size: int = 512,
    hop_size: int = 256,
) -> dict[str, Any]:
    """Return a compact spectrogram-like summary.

    This intentionally avoids returning a large image/matrix to the LLM. In a
    real multimodal classifier this branch can be replaced with CWT/scalogram
    generation or a CNN feature extractor.
    """
    x = processed.centered
    sr = processed.sampling_rate_hz
    if x.size < frame_size:
        frame_size = max(32, int(2 ** np.floor(np.log2(x.size))))
        hop_size = max(16, frame_size // 2)

    frames = []
    for start in range(0, x.size - frame_size + 1, hop_size):
        frame = x[start : start + frame_size]
        frames.append(np.abs(np.fft.rfft(frame * np.hanning(frame_size))) ** 2)

    if not frames:
        frames = [np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2]
        frame_size = x.size

    spec = np.stack(frames, axis=0)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    total = float(np.sum(spec) + EPS)
    freq_energy = np.sum(spec, axis=0)
    time_energy = np.sum(spec, axis=1)

    max_freq_idx = int(np.argmax(freq_energy))
    max_time_idx = int(np.argmax(time_energy))

    # Entropy: lower means energy is concentrated; higher means spread out.
    p = spec.reshape(-1) / total
    entropy = float(-np.sum(p * np.log(p + EPS)))

    return {
        "representation": "stft_spectrogram_summary",
        "note": "Replace with CWT/scalogram/CNN features for a full image branch.",
        "num_frames": int(spec.shape[0]),
        "num_frequency_bins": int(spec.shape[1]),
        "max_energy_frequency_hz": float(freqs[max_freq_idx]),
        "max_energy_frame_index": int(max_time_idx),
        "spectrogram_entropy": entropy,
        "energy_concentration_ratio": float(np.max(spec) / total),
    }
