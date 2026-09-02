from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ..utils import atomic_write_json, utc_now_iso


def _ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float:
    if correct.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(correct.size)
    value = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        value += float(mask.sum()) / total * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return value


@dataclass(frozen=True, slots=True)
class ConfidenceCalibrator:
    """Validation-fitted confidence curves for independent target and global tasks.

    ``x/y/operational`` remain aliases for the target curve so v1 bundles load
    unchanged. Global confidence has a separate curve and operational gate; one
    task can never borrow the other's calibration evidence.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    operational: bool
    validation_auroc: float | None
    validation_ece_before: float | None
    validation_ece_after: float | None
    global_x: tuple[float, ...] = ()
    global_y: tuple[float, ...] = ()
    global_operational: bool = False
    global_validation_auroc: float | None = None
    global_validation_ece_before: float | None = None
    global_validation_ece_after: float | None = None
    source_path: str | None = None
    schema_version: str = "2.0"

    @staticmethod
    def _apply(value: float | None, x: tuple[float, ...], y: tuple[float, ...]) -> float | None:
        if value is None or not x:
            return None
        calibrated = float(np.interp(float(value), np.asarray(x), np.asarray(y)))
        return min(max(calibrated, 0.0), 1.0)

    def calibrate(self, value: float | None) -> float | None:
        """Calibrate one target-row choice probability."""

        return self._apply(value, self.x, self.y)

    def calibrate_global(self, value: float | None) -> float | None:
        """Calibrate the independent global choice probability."""

        return self._apply(value, self.global_x, self.global_y)

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        target = dict(payload.get("target") or {})
        global_curve = dict(payload.get("global") or {})
        # Backward-compatible v1 target-only calibrator.
        if not target:
            target = payload
        return cls(
            x=tuple(float(value) for value in target.get("x", [])),
            y=tuple(float(value) for value in target.get("y", [])),
            operational=bool(target.get("operational", payload.get("operational", False))),
            validation_auroc=(
                float(target["validation_auroc"])
                if target.get("validation_auroc") is not None
                else None
            ),
            validation_ece_before=(
                float(target["validation_ece_before"])
                if target.get("validation_ece_before") is not None
                else None
            ),
            validation_ece_after=(
                float(target["validation_ece_after"])
                if target.get("validation_ece_after") is not None
                else None
            ),
            global_x=tuple(float(value) for value in global_curve.get("x", [])),
            global_y=tuple(float(value) for value in global_curve.get("y", [])),
            global_operational=bool(global_curve.get("operational", False)),
            global_validation_auroc=(
                float(global_curve["validation_auroc"])
                if global_curve.get("validation_auroc") is not None
                else None
            ),
            global_validation_ece_before=(
                float(global_curve["validation_ece_before"])
                if global_curve.get("validation_ece_before") is not None
                else None
            ),
            global_validation_ece_after=(
                float(global_curve["validation_ece_after"])
                if global_curve.get("validation_ece_after") is not None
                else None
            ),
            source_path=str(source),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


def _target_prediction_rows(path: Path) -> Iterable[tuple[float, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            truth_payload = row["ground_truth"]
            if isinstance(truth_payload, dict):
                if not bool(truth_payload.get("target_supervision_available", False)):
                    continue
                truth = list(truth_payload.get("targets") or [])
            else:  # v1 prediction file
                truth = list(truth_payload)
            predicted = list((row.get("prediction") or {}).get("targets") or [])
            if len(truth) != len(predicted):
                continue
            for target_truth, target_prediction in zip(truth, predicted, strict=True):
                raw = target_prediction.get("raw_choice_probability")
                if raw is None:
                    raw = target_prediction.get("choice_probability")
                if raw is None:
                    continue
                correct = int(
                    bool(target_truth["affected"]) == bool(target_prediction["affected"])
                    and str(target_truth["class"]) == str(target_prediction["class"])
                    and str(target_truth["severity"]) == str(target_prediction["severity"])
                )
                yield float(raw), correct


def _global_prediction_rows(path: Path) -> Iterable[tuple[float, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            truth_payload = row["ground_truth"]
            if isinstance(truth_payload, dict):
                truth_global = dict(truth_payload.get("global") or {})
                truth_class = truth_global.get("class")
            else:
                truth_class = (
                    "unknown_anomaly"
                    if any(str(target.get("class")) not in {"normal", "data_invalid"} for target in truth_payload)
                    else "normal"
                )
            model_payload = dict((row.get("prediction") or {}).get("model") or {})
            predicted_class = model_payload.get("global_class")
            raw = model_payload.get("raw_global_choice_probability")
            if raw is None:
                raw = model_payload.get("global_choice_probability")
            if truth_class is None or predicted_class is None or raw is None:
                continue
            yield float(raw), int(str(truth_class) == str(predicted_class))


def _fit_curve(
    rows: list[tuple[float, int]],
    *,
    minimum_auroc: float,
    minimum_errors: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "correct_count": 0,
            "error_count": 0,
            "validation_auroc": None,
            "validation_ece_before": None,
            "validation_ece_after": None,
            "operational": False,
            "x": [],
            "y": [],
            "reason": "no_labelled_confidence_rows",
        }
    raw = np.clip(
        np.asarray([item[0] for item in rows], dtype=np.float64), 0.0, 1.0
    )
    correct = np.asarray([item[1] for item in rows], dtype=np.int64)
    auroc: float | None
    if len(np.unique(correct)) == 2 and len(np.unique(raw)) > 1:
        auroc = float(roc_auc_score(correct, raw))
    else:
        auroc = None

    if len(np.unique(raw)) > 1 and len(np.unique(correct)) > 1:
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrated = isotonic.fit_transform(raw, correct.astype(np.float64))
        x = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
        y = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    else:
        mean_correct = float(correct.mean())
        x = np.asarray([0.0, 1.0], dtype=np.float64)
        y = np.asarray([mean_correct, mean_correct], dtype=np.float64)
        calibrated = np.full_like(raw, mean_correct, dtype=np.float64)

    error_count = int((correct == 0).sum())
    operational = bool(
        auroc is not None
        and auroc >= float(minimum_auroc)
        and error_count >= int(minimum_errors)
    )
    return {
        "sample_count": int(correct.size),
        "correct_count": int(correct.sum()),
        "error_count": error_count,
        "validation_auroc": auroc,
        "validation_ece_before": _ece(correct, raw),
        "validation_ece_after": _ece(correct, calibrated),
        "operational": operational,
        "x": [float(value) for value in x],
        "y": [float(value) for value in y],
        "reason": None if operational else "discrimination_or_error_support_gate_failed",
    }


def fit_confidence_calibrator(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    minimum_auroc: float = 0.60,
    minimum_errors: int = 20,
) -> dict[str, Any]:
    """Fit separate validation-only target and global confidence curves."""

    source = Path(predictions_path)
    target = _fit_curve(
        list(_target_prediction_rows(source)),
        minimum_auroc=minimum_auroc,
        minimum_errors=minimum_errors,
    )
    global_curve = _fit_curve(
        list(_global_prediction_rows(source)),
        minimum_auroc=minimum_auroc,
        minimum_errors=minimum_errors,
    )
    if target["sample_count"] == 0 and global_curve["sample_count"] == 0:
        raise ValueError(f"No target or global confidence values were found in {source}")
    payload = {
        "schema_version": "2.0",
        "created_utc": utc_now_iso(),
        "source_predictions": str(source),
        "minimum_auroc": float(minimum_auroc),
        "minimum_errors": int(minimum_errors),
        "target": target,
        "global": global_curve,
        # v1 aliases retained for existing consumers/tests.
        "sample_count": target["sample_count"],
        "correct_count": target["correct_count"],
        "error_count": target["error_count"],
        "validation_auroc": target["validation_auroc"],
        "validation_ece_before": target["validation_ece_before"],
        "validation_ece_after": target["validation_ece_after"],
        "operational": target["operational"],
        "global_operational": global_curve["operational"],
        "any_operational": bool(target["operational"] or global_curve["operational"]),
        "operational_note": (
            "global and target confidence are independently gated; only operational curves may be displayed"
        ),
        "x": target["x"],
        "y": target["y"],
    }
    atomic_write_json(output_path, payload)
    return payload
