"""Stable softmax and multiclass cross-entropy.

Interview prompt: implement both operations from logits without numerical overflow.
The important move is subtracting the largest logit before exponentiating.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax over the last dimension."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Return mean multiclass cross-entropy for integer class targets."""
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError("expected logits [N, C] and targets [N]")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("targets must contain integer class indices")
    if np.any((labels < 0) | (labels >= scores.shape[1])):
        raise ValueError("target class index is out of range")

    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(scores.shape[0]), labels].mean())

