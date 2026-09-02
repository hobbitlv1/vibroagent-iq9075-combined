from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from ..exceptions import DataContractError


def rational_ratio(source_rate_hz: float, target_rate_hz: float, max_denominator: int = 100_000) -> tuple[int, int]:
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sample rates must be positive")
    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(max_denominator)
    return ratio.numerator, ratio.denominator


def _interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    if matrix.ndim != 2:
        raise DataContractError("values must have shape [axes, samples]")
    x = np.arange(matrix.shape[1])
    for axis in range(matrix.shape[0]):
        if mask.all():
            continue
        valid_indices = x[mask]
        if valid_indices.size == 0:
            matrix[axis] = 0.0
        elif valid_indices.size == 1:
            matrix[axis] = matrix[axis, valid_indices[0]]
        else:
            matrix[axis, ~mask] = np.interp(x[~mask], valid_indices, matrix[axis, mask])
    return matrix


def resample_masked_signal(
    values: np.ndarray,
    sample_mask: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
    *,
    target_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    if matrix.ndim != 2 or mask.shape != (matrix.shape[1],):
        raise DataContractError("Signal/mask shape mismatch")
    if abs(source_rate_hz - target_rate_hz) <= max(source_rate_hz, target_rate_hz) * 1e-9:
        output = matrix.copy()
        output_mask = mask.copy()
    else:
        up, down = rational_ratio(source_rate_hz, target_rate_hz)
        filled = _interpolate_missing(matrix, mask)
        output = resample_poly(filled, up, down, axis=1, padtype="line").astype(np.float32)
        if mask.all():
            output_mask = np.ones(output.shape[1], dtype=bool)
        else:
            mask_float = resample_poly(mask.astype(np.float32)[None, :], up, down, axis=1, padtype="constant")[0]
            # A resampled position is valid only when virtually its complete
            # anti-alias support was supplied by genuine input samples.
            output_mask = mask_float >= 0.999
    if target_samples is not None:
        if output.shape[1] < target_samples:
            pad = target_samples - output.shape[1]
            output = np.pad(output, ((0, 0), (0, pad)), mode="constant")
            output_mask = np.pad(output_mask, (0, pad), mode="constant", constant_values=False)
        elif output.shape[1] > target_samples:
            output = output[:, :target_samples]
            output_mask = output_mask[:target_samples]
    return output, output_mask


class StatefulFIRDecimator:
    """Causal streaming decimator for integer source/target ratios.

    Used only when a deployment explicitly needs causal integer-ratio resampling.
    The model pipeline itself prepares complete ten-second windows with
    `resample_masked_signal`, preserving deterministic anti-alias behavior.
    """

    def __init__(self, source_rate_hz: float, target_rate_hz: float, channels: int = 3, taps: int = 511) -> None:
        ratio = source_rate_hz / target_rate_hz
        rounded = round(ratio)
        if not np.isclose(ratio, rounded, rtol=0, atol=1e-9):
            raise ValueError("StatefulFIRDecimator requires an integer decimation ratio")
        self.factor = int(rounded)
        if taps % 2 == 0:
            taps += 1
        cutoff = 0.8 / self.factor
        self.kernel = firwin(taps, cutoff, window="hamming")
        self.zi = np.zeros((channels, taps - 1), dtype=np.float64)
        self.phase = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        matrix = np.asarray(chunk, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.zi.shape[0]:
            raise DataContractError("chunk must have shape [channels, samples]")
        filtered = np.empty_like(matrix)
        for channel in range(matrix.shape[0]):
            filtered[channel], self.zi[channel] = lfilter(self.kernel, [1.0], matrix[channel], zi=self.zi[channel])
        first = (-self.phase) % self.factor
        output = filtered[:, first:: self.factor]
        self.phase = (self.phase + matrix.shape[1]) % self.factor
        return output.astype(np.float32)
