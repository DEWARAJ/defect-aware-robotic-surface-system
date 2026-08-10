from __future__ import annotations

import numpy as np


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have identical shapes")
    return {
        "true_positive": int(np.logical_and(pred, truth).sum()),
        "true_negative": int(np.logical_and(~pred, ~truth).sum()),
        "false_positive": int(np.logical_and(pred, ~truth).sum()),
        "false_negative": int(np.logical_and(~pred, truth).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]

    def safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "iou": safe_div(tp, tp + fp + fn),
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "pixel_accuracy": safe_div(tp + tn, tp + tn + fp + fn),
    }


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics_from_counts(confusion_counts(prediction, target))


def add_counts(total: dict[str, int], counts: dict[str, int]) -> dict[str, int]:
    return {key: int(total.get(key, 0) + value) for key, value in counts.items()}

