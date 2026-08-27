"""Single-head scaled dot-product attention.

Interview prompt: compute attention from Q, K, and V, including scaling and an
optional boolean mask that cannot leave a query with no valid key.
"""

from __future__ import annotations

import math

import numpy as np


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-head scaled dot-product attention and attention weights."""
    q = _feature_matrix(queries, "queries")
    k = _feature_matrix(keys, "keys")
    v = _feature_matrix(values, "values")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q/K widths and K/V sequence lengths must match")

    scores = q @ k.T / math.sqrt(q.shape[1])
    if mask is not None:
        allowed = np.asarray(mask, dtype=bool)
        if allowed.shape != scores.shape:
            raise ValueError("mask must have shape [num_queries, num_keys]")
        if np.any(~allowed.any(axis=1)):
            raise ValueError("every query must be allowed to attend to at least one key")
        scores = np.where(allowed, scores, -np.inf)
    weights = _softmax(scores)
    return weights @ v, weights


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite [N, D] matrix")
    return matrix


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)

