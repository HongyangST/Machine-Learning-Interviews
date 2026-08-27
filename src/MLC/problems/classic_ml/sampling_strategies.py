"""Uniform, weighted, and stratified sampling helpers.

Interview prompt: implement reproducible sampling strategies and explain when
weights or class stratification change the estimator or training distribution.
"""

from __future__ import annotations

import numpy as np


def sample_indices(
    size: int,
    k: int,
    *,
    weights: np.ndarray | None = None,
    replace: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Sample row indices uniformly or in proportion to non-negative weights."""
    if size < 0 or k < 0 or (not replace and k > size):
        raise ValueError("invalid population or sample size")
    probabilities = None
    if weights is not None:
        probabilities = np.asarray(weights, dtype=np.float64)
        if probabilities.shape != (size,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("weights must be a non-negative vector with positive sum")
        probabilities = probabilities / probabilities.sum()
    return np.random.default_rng(seed).choice(size, size=k, replace=replace, p=probabilities)


def stratified_sample_indices(
    labels: np.ndarray, per_class: int, *, seed: int = 0
) -> np.ndarray:
    """Sample the same number of rows without replacement from every class."""
    y = np.asarray(labels)
    if y.ndim != 1 or per_class < 0:
        raise ValueError("labels must be a vector and per_class non-negative")
    generator = np.random.default_rng(seed)
    groups = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        if per_class > len(candidates):
            raise ValueError("per_class exceeds the size of at least one class")
        groups.append(generator.choice(candidates, size=per_class, replace=False))
    result = np.concatenate(groups).astype(np.int64, copy=False)
    generator.shuffle(result)
    return result

