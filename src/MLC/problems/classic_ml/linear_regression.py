"""Linear regression trained with batch gradient descent.

Interview prompt: derive the MSE gradients, implement the optimizer, and explain
why feature scale and learning rate affect convergence.
"""

from __future__ import annotations

import numpy as np


def linear_regression_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float = 0.05,
    steps: int = 1_000,
) -> tuple[np.ndarray, float]:
    """Fit linear regression with batch gradient descent and MSE loss."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    if y.shape != (x.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("targets must be a finite vector with one value per row")
    if learning_rate <= 0 or steps < 1:
        raise ValueError("learning_rate and steps must be positive")

    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        errors = x @ weights + bias - y
        weights -= learning_rate * (2.0 / x.shape[0]) * (x.T @ errors)
        bias -= learning_rate * 2.0 * float(errors.mean())
    return weights, bias

