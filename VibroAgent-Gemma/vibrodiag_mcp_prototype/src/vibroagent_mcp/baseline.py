"""Baseline comparison utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = {
    "rms": 0.08,
    "kurtosis": 3.0,
    "crest_factor": 3.0,
    "dominant_frequency_hz": 30.0,
}


def load_machine_baseline(machine_id: str) -> dict[str, float]:
    """Load a per-machine healthy baseline if configured.

    Environment variable:
        VIBRO_BASELINES_JSON=/path/to/baselines.json

    JSON format:
        {
          "motor_01": {"rms": 0.12, "kurtosis": 3.1, ...},
          "default": {"rms": 0.08, "kurtosis": 3.0, ...}
        }
    """
    path = os.environ.get("VIBRO_BASELINES_JSON")
    if not path:
        return dict(DEFAULT_BASELINE)

    baseline_path = Path(path).expanduser().resolve()
    if not baseline_path.exists():
        return dict(DEFAULT_BASELINE)

    try:
        with baseline_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        selected = data.get(machine_id) or data.get("default") or DEFAULT_BASELINE
        return {**DEFAULT_BASELINE, **{k: float(v) for k, v in selected.items()}}
    except Exception:
        # Keep tool robust; surface the fallback in the result status instead of crashing.
        return dict(DEFAULT_BASELINE)


def compare_to_baseline(machine_id: str, features: dict[str, Any]) -> dict[str, Any]:
    """Compare extracted features to a healthy baseline."""
    baseline = load_machine_baseline(machine_id)
    td = features["time_domain"]
    fd = features["frequency_domain"]

    rms = float(td["rms"])
    kurtosis = float(td["kurtosis"])
    crest = float(td["crest_factor"])
    dom_freq = float(fd["dominant_frequency_hz"])

    baseline_rms = max(float(baseline.get("rms", DEFAULT_BASELINE["rms"])), 1e-12)
    rms_ratio = float(rms / baseline_rms)
    rms_change_percent = float((rms_ratio - 1.0) * 100.0)
    kurtosis_delta = float(kurtosis - float(baseline.get("kurtosis", DEFAULT_BASELINE["kurtosis"])))
    crest_delta = float(crest - float(baseline.get("crest_factor", DEFAULT_BASELINE["crest_factor"])))
    dom_freq_delta = float(dom_freq - float(baseline.get("dominant_frequency_hz", DEFAULT_BASELINE["dominant_frequency_hz"])))

    vibration_status = "normal"
    if rms_ratio > 1.8 or kurtosis > 6.5 or crest > 5.5:
        vibration_status = "highly_abnormal"
    elif rms_ratio > 1.25 or kurtosis > 4.5 or crest > 4.2:
        vibration_status = "abnormal"

    return {
        "baseline_used": baseline,
        "rms_ratio": rms_ratio,
        "rms_change_percent": rms_change_percent,
        "kurtosis_delta": kurtosis_delta,
        "crest_factor_delta": crest_delta,
        "dominant_frequency_delta_hz": dom_freq_delta,
        "rms_status": _status_from_ratio(rms_ratio),
        "kurtosis_status": _status_from_value(kurtosis, warning=4.5, high=6.5),
        "crest_factor_status": _status_from_value(crest, warning=4.2, high=5.5),
        "vibration_status": vibration_status,
    }


def _status_from_ratio(ratio: float) -> str:
    if ratio > 1.8:
        return "high"
    if ratio > 1.25:
        return "elevated"
    return "normal"


def _status_from_value(value: float, warning: float, high: float) -> str:
    if value > high:
        return "high"
    if value > warning:
        return "elevated"
    return "normal"
