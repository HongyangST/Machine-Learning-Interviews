"""Symmetric in-batch contrastive loss for paired embeddings.

Interview prompt: normalize two embedding batches, build the similarity matrix,
and compute both retrieval directions with a numerically stable cross-entropy.
"""

from __future__ import annotations

import numpy as np


def symmetric_contrastive_loss(
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray,
    *,
    temperature: float = 0.07,
) -> float:
    """Return mean left-to-right and right-to-left InfoNCE loss."""
    left = _normalize(left_embeddings)
    right = _normalize(right_embeddings)
    if left.shape != right.shape or temperature <= 0:
        raise ValueError("paired embedding shapes must match and temperature be positive")
    logits = left @ right.T / temperature
    targets = np.arange(len(left))
    return 0.5 * (_cross_entropy(logits, targets) + _cross_entropy(logits.T, targets))


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("embeddings must be a non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length embeddings cannot be normalized")
    return matrix / norms


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(len(targets)), targets].mean())

