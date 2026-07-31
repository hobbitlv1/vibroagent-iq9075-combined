"""Synthetic vibration signals for local demos.

These signals are not a physical bearing simulator. They are deterministic,
lightweight examples that make the MCP pipeline testable without hardware.
"""

from __future__ import annotations

import numpy as np

from .mock_vibrometer import read_mock_vibrometer_window, stable_seed


def generate_demo_signal(
    machine_id: str,
    duration_s: float = 5.0,
    sampling_rate_hz: int = 12_000,
) -> np.ndarray:
    """Generate a deterministic demo vibration signal.

    Names containing these substrings force the scenario:
    - healthy
    - outer
    - inner
    - imbalance / loose
    - degrading

    This wrapper preserves the original demo API. Internally it now uses the
    mock vibrometer simulator so local demos, MCP tools, and tests share the
    same synthetic data source.
    """
    return read_mock_vibrometer_window(
        machine_id=machine_id,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
    ).signal
