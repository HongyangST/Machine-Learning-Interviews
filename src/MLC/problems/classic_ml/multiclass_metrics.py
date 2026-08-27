"""Multiclass confusion matrix and macro-averaged metrics.

Interview prompt: extend binary precision, recall, and F1 to multiclass labels.
State how absent classes and macro versus micro averaging are handled.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, *, num_classes: int | None = None
) -> np.ndarray:
    """Return a matrix whose rows are true classes and columns are predictions."""
    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be equal-length vectors")
    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(
        y_pred.dtype, np.integer
    ):
        raise TypeError("class labels must be integers")
    inferred = int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1
    classes = inferred if num_classes is None else num_classes
    if classes < inferred or classes < 1 or np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("labels must be in [0, num_classes)")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_precision_recall_f1(matrix: np.ndarray) -> dict[str, float]:
    """Compute unweighted per-class precision, recall, and F1 averages."""
    counts = np.asarray(matrix, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or np.any(counts < 0):
        raise ValueError("matrix must be a non-negative square confusion matrix")
    true_positive = np.diag(counts)
    precision = np.divide(
        true_positive, counts.sum(axis=0), out=np.zeros_like(true_positive), where=counts.sum(axis=0) > 0
    )
    recall = np.divide(
        true_positive, counts.sum(axis=1), out=np.zeros_like(true_positive), where=counts.sum(axis=1) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())}

