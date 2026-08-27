"""One residual-fitting step for squared-error gradient boosting.

Interview prompt: fit a regression stump to current residuals and explain why
the residual is the negative gradient of squared-error loss.
"""

from __future__ import annotations

import numpy as np


def fit_residual_stump(
    features: np.ndarray, targets: np.ndarray, predictions: np.ndarray
) -> tuple[int, float, float, float]:
    """Return feature, threshold, and mean residual on each side of the split."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    current = np.asarray(predictions, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or current.shape != y.shape:
        raise ValueError("expected features [N, D], targets [N], predictions [N]")
    residuals = y - current
    best: tuple[float, int, float, float, float] | None = None
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            left = x[:, feature] <= threshold
            if left.all() or (~left).all():
                continue
            left_value = float(residuals[left].mean())
            right_value = float(residuals[~left].mean())
            fitted = np.where(left, left_value, right_value)
            candidate = (float(np.mean((residuals - fitted) ** 2)), feature, float(threshold), left_value, right_value)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("at least one feature must admit a non-empty split")
    _, feature, threshold, left_value, right_value = best
    return feature, threshold, left_value, right_value


def apply_stump(features: np.ndarray, stump: tuple[int, float, float, float]) -> np.ndarray:
    """Return this weak learner's residual prediction for every row."""
    feature, threshold, left_value, right_value = stump
    x = np.asarray(features, dtype=np.float64)
    return np.where(x[:, feature] <= threshold, left_value, right_value)

