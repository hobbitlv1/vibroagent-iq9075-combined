from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..contracts import EpisodeLabel, TargetPrediction
from ..model.records import GlobalPrediction


def _derived_global_prediction(values: Sequence[TargetPrediction]) -> GlobalPrediction:
    anomalies = [
        prediction
        for prediction in values
        if prediction.class_name not in {"normal", "data_invalid"}
    ]
    if not anomalies:
        return GlobalPrediction("normal", "none", None)
    order = {"none": 0, "advisory": 1, "warning": 2, "critical": 3}
    severity = max(
        (prediction.severity for prediction in anomalies),
        key=lambda value: order.get(value, -1),
    )
    return GlobalPrediction("unknown_anomaly", severity, None)


def flatten_targets(
    truth: Sequence[EpisodeLabel], predictions: Sequence[Sequence[TargetPrediction]]
) -> tuple[list[str], list[str], list[str], list[str], list[int], list[int]]:
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    true_severities: list[str] = []
    predicted_severities: list[str] = []
    true_affected: list[int] = []
    predicted_affected: list[int] = []
    for episode_truth, episode_prediction in zip(truth, predictions, strict=True):
        if not episode_truth.target_supervision_available:
            continue
        for target_truth, target_prediction in zip(
            episode_truth.targets, episode_prediction, strict=True
        ):
            true_classes.append(target_truth.class_name)
            predicted_classes.append(target_prediction.class_name)
            true_severities.append(target_truth.severity)
            predicted_severities.append(target_prediction.severity)
            true_affected.append(int(target_truth.affected))
            predicted_affected.append(int(target_prediction.affected))
    return (
        true_classes,
        predicted_classes,
        true_severities,
        predicted_severities,
        true_affected,
        predicted_affected,
    )


def _target_localization_metrics(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
) -> dict[str, Any]:
    exact_sets: list[bool] = []
    single_target_all_five: list[bool] = []
    single_target_all_zero: list[bool] = []
    single_target_exact: list[bool] = []
    single_target_partial: list[bool] = []
    normal_any_affected: list[bool] = []
    predicted_count_histogram: Counter[int] = Counter()
    per_target_true: dict[int, list[int]] = defaultdict(list)
    per_target_pred: dict[int, list[int]] = defaultdict(list)

    supervised_episode_count = 0
    for episode_truth, episode_prediction in zip(truth, predictions, strict=True):
        predicted_indices = {
            index
            for index, prediction in enumerate(episode_prediction, start=1)
            if prediction.affected
        }
        predicted_count_histogram[len(predicted_indices)] += 1
        if not episode_truth.target_supervision_available:
            continue
        supervised_episode_count += 1
        true_indices = set(episode_truth.affected_target_indices)
        exact_sets.append(true_indices == predicted_indices)
        if len(true_indices) == 1:
            single_target_all_five.append(len(predicted_indices) == 5)
            single_target_all_zero.append(len(predicted_indices) == 0)
            single_target_exact.append(true_indices == predicted_indices)
            single_target_partial.append(0 < len(predicted_indices) < 5)
        if not true_indices:
            normal_any_affected.append(bool(predicted_indices))
        for target_index in range(1, 6):
            per_target_true[target_index].append(int(target_index in true_indices))
            per_target_pred[target_index].append(int(target_index in predicted_indices))

    per_target: dict[str, dict[str, float | int]] = {}
    for target_index in range(1, 6):
        values_true = per_target_true.get(target_index, [])
        values_pred = per_target_pred.get(target_index, [])
        if not values_true:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            values_true,
            values_pred,
            average="binary",
            zero_division=0,
        )
        per_target[f"target_{target_index}"] = {
            "support": int(sum(values_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return {
        "target_supervised_episode_count": supervised_episode_count,
        "target_unsupervised_episode_count": len(truth) - supervised_episode_count,
        "target_exact_set_match": float(np.mean(exact_sets)) if exact_sets else None,
        "single_target_episode_count": len(single_target_exact),
        "single_target_exact_set_match": (
            float(np.mean(single_target_exact)) if single_target_exact else None
        ),
        "single_target_all_five_rate": (
            float(np.mean(single_target_all_five)) if single_target_all_five else None
        ),
        "single_target_all_zero_rate": (
            float(np.mean(single_target_all_zero)) if single_target_all_zero else None
        ),
        "single_target_partial_prediction_rate": (
            float(np.mean(single_target_partial)) if single_target_partial else None
        ),
        "target_normal_episode_false_positive_rate": (
            float(np.mean(normal_any_affected)) if normal_any_affected else None
        ),
        "predicted_affected_count_histogram_all_episodes": {
            str(key): int(value)
            for key, value in sorted(predicted_count_histogram.items())
        },
        "per_target": per_target,
    }


def compute_metrics(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
    *,
    window_seconds: float,
    global_predictions: Sequence[GlobalPrediction] | None = None,
) -> dict[str, Any]:
    if global_predictions is None:
        global_predictions = [
            _derived_global_prediction(values) for values in predictions
        ]
    if not (len(truth) == len(predictions) == len(global_predictions)):
        raise ValueError("truth, predictions and global_predictions must have equal length")

    true_global_classes = [str(label.global_class_name) for label in truth]
    predicted_global_classes = [value.class_name for value in global_predictions]
    true_global_severities = [str(label.global_severity) for label in truth]
    predicted_global_severities = [value.severity for value in global_predictions]
    global_labels = sorted(set(true_global_classes) | set(predicted_global_classes))
    severity_order = ["none", "advisory", "warning", "critical"]
    severity_support = set(true_global_severities) | set(predicted_global_severities)
    severity_labels = [value for value in severity_order if value in severity_support]

    (
        true_target_classes,
        predicted_target_classes,
        true_target_severities,
        predicted_target_severities,
        true_affected,
        predicted_affected,
    ) = flatten_targets(truth, predictions)

    if true_affected:
        affected_precision, affected_recall, affected_f1, _ = (
            precision_recall_fscore_support(
                true_affected,
                predicted_affected,
                average="binary",
                zero_division=0,
            )
        )
        target_class_labels = sorted(
            set(true_target_classes) | set(predicted_target_classes)
        )
        target_class_accuracy = float(
            accuracy_score(true_target_classes, predicted_target_classes)
        )
        target_class_macro_f1 = float(
            f1_score(
                true_target_classes,
                predicted_target_classes,
                average="macro",
                zero_division=0,
            )
        )
    else:
        affected_precision = affected_recall = affected_f1 = None
        target_class_labels = []
        target_class_accuracy = target_class_macro_f1 = None

    episode_exact: list[bool] = []
    false_alert: list[bool] = []
    healthy_episode: list[bool] = []
    global_target_consistency_all: list[bool] = []
    global_target_consistency_target_supervised: list[bool] = []
    for episode_truth, episode_prediction, global_prediction in zip(
        truth, predictions, global_predictions, strict=True
    ):
        global_correct = (
            episode_truth.global_class_name == global_prediction.class_name
            and episode_truth.global_severity == global_prediction.severity
        )
        if episode_truth.target_supervision_available:
            target_correct = all(
                target_truth.affected == target_prediction.affected
                and target_truth.class_name == target_prediction.class_name
                and target_truth.severity == target_prediction.severity
                for target_truth, target_prediction in zip(
                    episode_truth.targets, episode_prediction, strict=True
                )
            )
        else:
            target_correct = True
        episode_exact.append(global_correct and target_correct)
        healthy = episode_truth.global_class_name == "normal"
        healthy_episode.append(healthy)
        false_alert.append(healthy and global_prediction.class_name != "normal")
        predicted_any_target = any(item.affected for item in episode_prediction)
        consistency = bool(
            (global_prediction.class_name == "normal" and not predicted_any_target)
            or (
                global_prediction.class_name == "unknown_anomaly"
                and predicted_any_target
            )
        )
        global_target_consistency_all.append(consistency)
        if episode_truth.target_supervision_available:
            global_target_consistency_target_supervised.append(consistency)

    healthy_window_hours = sum(healthy_episode) * window_seconds / 3600.0
    false_alert_rate = (
        sum(false_alert) / healthy_window_hours if healthy_window_hours > 0 else None
    )
    localization = _target_localization_metrics(truth, predictions)
    return {
        "episode_count": len(truth),
        "target_count": len(true_affected),
        "predicted_target_decision_count": len(truth) * 5,
        "episode_exact_match": float(np.mean(episode_exact)) if episode_exact else 0.0,
        # Backward-compatible names now refer to the scientifically available
        # episode-level global label, not fabricated five-row pseudo-labels.
        "class_accuracy": float(
            accuracy_score(true_global_classes, predicted_global_classes)
        ),
        "class_macro_f1": float(
            f1_score(
                true_global_classes,
                predicted_global_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "class_weighted_f1": float(
            f1_score(
                true_global_classes,
                predicted_global_classes,
                average="weighted",
                zero_division=0,
            )
        ),
        "severity_accuracy": float(
            accuracy_score(true_global_severities, predicted_global_severities)
        ),
        "severity_macro_f1": float(
            f1_score(
                true_global_severities,
                predicted_global_severities,
                labels=severity_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "affected_precision": (
            float(affected_precision) if affected_precision is not None else None
        ),
        "affected_recall": (
            float(affected_recall) if affected_recall is not None else None
        ),
        "affected_f1": float(affected_f1) if affected_f1 is not None else None,
        "target_class_accuracy": target_class_accuracy,
        "target_class_macro_f1": target_class_macro_f1,
        "target_class_labels": target_class_labels,
        # The headline consistency metric is restricted to episodes with genuine
        # target labels. Global-only sources must not force arbitrary target rows.
        "global_target_consistency_rate": (
            float(np.mean(global_target_consistency_target_supervised))
            if global_target_consistency_target_supervised
            else None
        ),
        "all_output_global_target_consistency_rate": (
            float(np.mean(global_target_consistency_all))
            if global_target_consistency_all
            else None
        ),
        "localization": localization,
        "healthy_evaluated_window_hours": healthy_window_hours,
        "false_alert_window_count": int(sum(false_alert)),
        "raw_false_alert_windows_per_hour": false_alert_rate,
        "healthy_evaluated_hours": healthy_window_hours,
        "false_alert_count": int(sum(false_alert)),
        "false_alerts_per_hour": false_alert_rate,
        "class_labels": global_labels,
        "class_confusion_matrix": confusion_matrix(
            true_global_classes,
            predicted_global_classes,
            labels=global_labels,
        ).tolist(),
        "severity_labels": severity_labels,
        "severity_confusion_matrix": confusion_matrix(
            true_global_severities,
            predicted_global_severities,
            labels=severity_labels,
        ).tolist(),
    }


def bootstrap_intervals(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
    *,
    window_seconds: float,
    samples: int,
    confidence_level: float,
    seed: int,
    global_predictions: Sequence[GlobalPrediction] | None = None,
) -> dict[str, dict[str, float]]:
    if not truth:
        return {}
    if global_predictions is None:
        global_predictions = [
            _derived_global_prediction(values) for values in predictions
        ]
    rng = np.random.default_rng(seed)
    metric_names = [
        "episode_exact_match",
        "class_accuracy",
        "class_macro_f1",
        "severity_accuracy",
        "affected_f1",
    ]
    values: dict[str, list[float]] = defaultdict(list)
    n = len(truth)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        sampled_truth = [truth[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]
        sampled_global = [global_predictions[index] for index in indices]
        metrics = compute_metrics(
            sampled_truth,
            sampled_predictions,
            window_seconds=window_seconds,
            global_predictions=sampled_global,
        )
        for name in metric_names:
            value = metrics.get(name)
            if value is not None:
                values[name].append(float(value))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        name: {
            "lower": float(np.quantile(metric_values, alpha)),
            "upper": float(np.quantile(metric_values, 1.0 - alpha)),
        }
        for name, metric_values in values.items()
        if metric_values
    }


def _confidence_arrays(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
    *,
    raw: bool,
) -> tuple[np.ndarray, np.ndarray]:
    confidences: list[float] = []
    correctness: list[float] = []
    for episode_truth, episode_prediction in zip(truth, predictions, strict=True):
        if not episode_truth.target_supervision_available:
            continue
        for target_truth, target_prediction in zip(
            episode_truth.targets, episode_prediction, strict=True
        ):
            confidence = (
                target_prediction.raw_choice_probability
                if raw
                else target_prediction.choice_probability
            )
            if confidence is None:
                continue
            confidences.append(float(confidence))
            correctness.append(
                float(
                    target_truth.affected == target_prediction.affected
                    and target_truth.class_name == target_prediction.class_name
                    and target_truth.severity == target_prediction.severity
                )
            )
    return np.asarray(confidences, dtype=np.float64), np.asarray(
        correctness, dtype=np.float64
    )


def expected_calibration_error(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
    bins: int = 10,
    *,
    raw: bool = False,
) -> float | None:
    confidence_array, correctness_array = _confidence_arrays(
        truth, predictions, raw=raw
    )
    if confidence_array.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence_array >= lower) & (
            confidence_array <= upper if upper == 1.0 else confidence_array < upper
        )
        if not selected.any():
            continue
        ece += selected.mean() * abs(
            correctness_array[selected].mean() - confidence_array[selected].mean()
        )
    return float(ece)


def confidence_correctness_auroc(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[Sequence[TargetPrediction]],
    *,
    raw: bool = False,
) -> float | None:
    confidence_array, correctness_array = _confidence_arrays(
        truth, predictions, raw=raw
    )
    if (
        confidence_array.size == 0
        or len(np.unique(correctness_array)) < 2
        or len(np.unique(confidence_array)) < 2
    ):
        return None
    return float(roc_auc_score(correctness_array, confidence_array))


def global_expected_calibration_error(
    truth: Sequence[EpisodeLabel],
    predictions: Sequence[GlobalPrediction],
    bins: int = 10,
) -> float | None:
    values = [
        (float(prediction.choice_probability), float(label.global_class_name == prediction.class_name))
        for label, prediction in zip(truth, predictions, strict=True)
        if prediction.choice_probability is not None
    ]
    if not values:
        return None
    confidence_array = np.asarray([value[0] for value in values], dtype=np.float64)
    correctness_array = np.asarray([value[1] for value in values], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence_array >= lower) & (
            confidence_array <= upper if upper == 1.0 else confidence_array < upper
        )
        if selected.any():
            ece += selected.mean() * abs(
                correctness_array[selected].mean() - confidence_array[selected].mean()
            )
    return float(ece)
