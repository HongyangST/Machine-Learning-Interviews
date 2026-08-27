"""Binary logistic regression trained with batch gradient descent.

Interview prompt: implement training without a framework, including a stable
sigmoid and validation for binary targets.
"""

from __future__ import annotations

import numpy as np


def logistic_regression_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float = 0.1,
    steps: int = 1_000,
) -> tuple[np.ndarray, float]:
    """Fit binary logistic regression with batch gradient descent."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    if y.shape != (x.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("targets must be a finite vector with one value per row")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("targets must be binary")
    if learning_rate <= 0 or steps < 1:
        raise ValueError("learning_rate and steps must be positive")

    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        errors = probabilities - y
        weights -= learning_rate * (x.T @ errors) / x.shape[0]
        bias -= learning_rate * float(errors.mean())
    return weights, bias


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result

