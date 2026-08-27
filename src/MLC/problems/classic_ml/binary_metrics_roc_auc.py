"""Binary classification metrics and rank-based ROC-AUC.

Interview prompt: compute accuracy, precision, recall, F1, and ROC-AUC without
metrics libraries; define zero-denominator and tied-score behavior explicitly.
"""

from __future__ import annotations

import numpy as np


def binary_classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from binary labels."""
    target_values = np.asarray(targets)
    prediction_values = np.asarray(predictions)
    if target_values.ndim != 1 or target_values.shape != prediction_values.shape:
        raise ValueError("targets and predictions must be equal-length vectors")
    if not np.all(np.isin(target_values, (0, 1))) or not np.all(
        np.isin(prediction_values, (0, 1))
    ):
        raise ValueError("targets and predictions must be binary")
    y_true = target_values.astype(bool)
    y_pred = prediction_values.astype(bool)

    true_positive = int(np.sum(y_true & y_pred))
    true_negative = int(np.sum(~y_true & ~y_pred))
    false_positive = int(np.sum(~y_true & y_pred))
    false_negative = int(np.sum(y_true & ~y_pred))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": _safe_divide(true_positive + true_negative, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC using average ranks, including tied scores."""
    y = np.asarray(targets)
    predictions = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != predictions.shape:
        raise ValueError("targets and scores must be equal-length vectors")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("targets must be binary")

    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative examples")

    order = np.argsort(predictions, kind="mergesort")
    sorted_scores = predictions[order]
    ranks = np.empty(len(predictions), dtype=np.float64)
    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0

