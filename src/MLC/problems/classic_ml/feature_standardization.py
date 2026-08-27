"""Feature standardization fit on training data only.

Interview prompt: fit per-feature statistics on the training split, transform
later splits with the same state, and handle constant columns safely.
"""

from __future__ import annotations

import numpy as np


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return training-set means and safe population standard deviations."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    return mean, np.where(scale == 0.0, 1.0, scale)


def transform_standardized(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply previously fitted training statistics to a feature matrix."""
    x = np.asarray(features, dtype=np.float64)
    mean_values = np.asarray(mean, dtype=np.float64)
    scale_values = np.asarray(scale, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != mean_values.shape or mean_values.shape != scale_values.shape:
        raise ValueError("feature width and fitted statistics must match")
    if np.any(scale_values <= 0):
        raise ValueError("scale values must be positive")
    return (x - mean_values) / scale_values

