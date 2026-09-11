"""Acquisition-domain quality checks, before filtering or synthetic overlays."""
import os

import numpy as np


def raw_window_quality(values, *, axis_mask=None, clipping_by_axis=None):
    values = np.asarray(values)
    axes = np.ones(3, dtype=bool) if axis_mask is None else np.asarray(axis_mask, dtype=bool)
    if values.ndim != 2 or values.shape[0] != 3 or axes.shape != (3,) or not axes.any():
        raise ValueError("quality assessment requires XYZ and at least one measured axis")
    threshold = float(os.environ.get("VIBROGEMMA_FLATLINE_MAX_PTP_G", "0.0000001"))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("VIBROGEMMA_FLATLINE_MAX_PTP_G must be finite and non-negative")
    measured = values[axes]
    valid = np.isfinite(measured).all(axis=0)
    flatline = bool(valid.any() and np.all(np.ptp(measured[:, valid], axis=1) <= threshold))
    limits = [float((clipping_by_axis or {}).get(axis) or 0) for axis, present in zip("xyz", axes) if present]
    clipped = None
    if limits and all(np.isfinite(limit) and limit > 0 for limit in limits) and valid.any():
        clipped = float((np.abs(measured[:, valid]) >= np.asarray(limits)[:, None]).mean())
    fraction = float(valid.mean()) if valid.size else 0.0
    return {"valid_fraction": fraction, "missing_fraction": 1.0 - fraction,
            "clipped_fraction": clipped, "flatline": flatline, "flatline_max_ptp_g": threshold}
