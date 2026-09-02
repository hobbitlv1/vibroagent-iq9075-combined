"""Small bearing-fault classifier layer.

The default classifier is a deterministic heuristic so the prototype can run
without a trained model. Replace it with ONNX/PyTorch/XGBoost in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassifierPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[str]


class HeuristicBearingClassifier:
    """Tiny interpretable classifier for demos and edge prototyping.

    This is not a substitute for a trained model. It maps common vibration
    indicators to plausible labels so the MCP/LLM workflow can be tested.
    """

    labels = ["healthy", "imbalance_or_looseness", "outer_race_fault", "inner_race_fault", "unknown_abnormal"]

    def predict(self, features: dict[str, Any], baseline: dict[str, Any]) -> ClassifierPrediction:
        td = features["time_domain"]
        fd = features["frequency_domain"]

        rms = float(td["rms"])
        kurtosis = float(td["kurtosis"])
        crest = float(td["crest_factor"])
        skewness = abs(float(td["skewness"]))
        dom_freq = float(fd["dominant_frequency_hz"])
        spectral_flatness = float(fd.get("spectral_flatness", 0.0))

        rms_ratio = float(baseline.get("rms_ratio", 1.0))
        kurtosis_delta = float(baseline.get("kurtosis_delta", kurtosis - 3.0))
        crest_delta = float(baseline.get("crest_factor_delta", crest - 3.0))

        evidence: list[str] = []
        if rms_ratio > 1.25:
            evidence.append(f"RMS is elevated versus baseline (ratio={rms_ratio:.2f}).")
        if kurtosis > 5.0:
            evidence.append(f"Kurtosis is high ({kurtosis:.2f}), suggesting impulsive vibration.")
        if crest > 4.5:
            evidence.append(f"Crest factor is high ({crest:.2f}), indicating sharp peaks.")
        if dom_freq > 0:
            evidence.append(f"Dominant frequency is {dom_freq:.1f} Hz.")
        if spectral_flatness > 0.65 and rms_ratio > 1.20:
            evidence.append("Spectrum is relatively broadband/flat, so the fault type is less specific.")

        # Severity score in [0, 1]. Emphasizes impulsiveness and baseline deviation.
        severity_raw = (
            0.35 * _soft_threshold(rms_ratio, center=1.25, scale=0.25)
            + 0.30 * _soft_threshold(kurtosis, center=5.0, scale=1.5)
            + 0.20 * _soft_threshold(crest, center=4.5, scale=1.0)
            + 0.15 * _soft_threshold(skewness, center=1.0, scale=0.7)
        )
        severity = max(0.0, min(1.0, severity_raw))

        if severity < 0.32 and rms_ratio < 1.35 and kurtosis < 4.5:
            label = "healthy"
        elif spectral_flatness > 0.65 and rms_ratio > 1.20 and dom_freq >= 80.0:
            # Broadband abnormal patterns are kept as unknown instead of forcing
            # them into a specific bearing-location class.
            label = "unknown_abnormal"
        elif rms_ratio > 1.35 and kurtosis < 4.8 and dom_freq < 120.0:
            label = "imbalance_or_looseness"
        elif kurtosis > 5.0 or crest > 4.5 or rms_ratio > 2.0:
            # This simple mapping uses dominant frequency bands as a proxy. A real
            # classifier should use bearing geometry and speed to compute BPFO/BPFI.
            if dom_freq >= 200.0:
                label = "inner_race_fault"
            elif 80.0 <= dom_freq < 200.0:
                label = "outer_race_fault"
            elif dom_freq < 80.0 and rms_ratio > 1.35:
                label = "imbalance_or_looseness"
            else:
                label = "unknown_abnormal"
        else:
            label = "unknown_abnormal"

        probabilities = _probabilities_from_label(label, severity)
        confidence = probabilities[label]

        if not evidence:
            evidence.append("No strong abnormal indicators were detected by the heuristic classifier.")

        return ClassifierPrediction(
            label=label,
            confidence=float(confidence),
            probabilities=probabilities,
            evidence=evidence,
        )


def _soft_threshold(value: float, center: float, scale: float) -> float:
    """Smoothly maps values above center toward 1."""
    if scale <= 0:
        scale = 1.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def _probabilities_from_label(label: str, severity: float) -> dict[str, float]:
    labels = HeuristicBearingClassifier.labels
    base = {name: 0.04 for name in labels}

    if label == "healthy":
        base["healthy"] = 0.75 + 0.15 * (1.0 - severity)
        base["unknown_abnormal"] = 0.08 + 0.12 * severity
    else:
        base[label] = 0.62 + 0.25 * severity
        base["healthy"] = max(0.02, 0.18 * (1.0 - severity))
        base["unknown_abnormal"] = max(base.get("unknown_abnormal", 0.04), 0.10)

    total = sum(base.values())
    return {name: float(value / total) for name, value in base.items()}
