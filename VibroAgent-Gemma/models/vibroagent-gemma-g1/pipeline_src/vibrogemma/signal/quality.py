from __future__ import annotations

from typing import Iterable

import numpy as np

from ..contracts import QualityReport


def assess_quality(
    values: np.ndarray,
    sample_mask: np.ndarray,
    axis_mask: np.ndarray,
    *,
    sample_rate_hz: float,
    original_sample_rate_hz: float | None = None,
    timestamps_s: np.ndarray | None = None,
    clipping_abs_g: float | None = None,
    packet_loss_count: int = 0,
    clock_skew_ms: float = 0.0,
    notes: Iterable[str] = (),
) -> QualityReport:
    matrix = np.asarray(values)
    valid_samples = np.asarray(sample_mask, dtype=bool)
    present_axes = np.asarray(axis_mask, dtype=bool)
    valid_fraction = float(valid_samples.mean()) if valid_samples.size else 0.0
    clipped_fraction = 0.0
    if clipping_abs_g is not None and present_axes.any() and valid_samples.any():
        valid_values = matrix[present_axes][:, valid_samples]
        clipped_fraction = float((np.abs(valid_values) >= clipping_abs_g).mean()) if valid_values.size else 0.0
    monotonic = True
    quality_notes = list(notes)
    if timestamps_s is not None:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        if timestamps.size > 1:
            deltas = np.diff(timestamps)
            monotonic = bool(np.all(deltas > 0))
            expected = 1.0 / float(original_sample_rate_hz or sample_rate_hz)
            if monotonic:
                gross_gap_count = int(np.sum(deltas > expected * 1.5))
                if gross_gap_count:
                    quality_notes.append(f"timestamp_gaps={gross_gap_count}")
                    packet_loss_count += gross_gap_count
    usable_band_hz = min(float(sample_rate_hz), float(original_sample_rate_hz or sample_rate_hz)) / 2.0
    if not present_axes.all():
        missing = [axis for axis, present in zip(("x", "y", "z"), present_axes, strict=True) if not present]
        quality_notes.append(f"missing_axes={','.join(missing)}")
    if not monotonic:
        quality_notes.append("non_monotonic_timestamps")
    if clipped_fraction > 0:
        quality_notes.append(f"clipped_fraction={clipped_fraction:.6f}")
    return QualityReport(
        valid_fraction=valid_fraction,
        missing_fraction=1.0 - valid_fraction,
        clipped_fraction=clipped_fraction,
        timestamp_monotonic=monotonic,
        clock_skew_ms=float(clock_skew_ms),
        axes_present=tuple(bool(value) for value in present_axes),
        usable_band_hz=usable_band_hz,
        packet_loss_count=int(packet_loss_count),
        notes=quality_notes,
    )
