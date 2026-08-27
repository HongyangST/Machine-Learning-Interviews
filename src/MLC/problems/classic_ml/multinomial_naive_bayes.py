"""Multinomial Naive Bayes for small text-classification interviews.

Interview prompt: estimate smoothed class and token probabilities, score new
documents in log space, and explain the conditional-independence assumption.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fit_multinomial_naive_bayes(
    documents: Sequence[str], labels: np.ndarray, *, alpha: float = 1.0
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return vocabulary, classes, log priors, and class-token log probabilities."""
    y = np.asarray(labels)
    if y.shape != (len(documents),) or not len(documents):
        raise ValueError("labels must contain one value per non-empty document set")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    tokenized = [document.lower().split() for document in documents]
    vocabulary = sorted({token for tokens in tokenized for token in tokens})
    if not vocabulary:
        raise ValueError("at least one token is required")
    classes, class_counts = np.unique(y, return_counts=True)
    token_index = {token: index for index, token in enumerate(vocabulary)}
    counts = np.full((len(classes), len(vocabulary)), alpha, dtype=np.float64)
    for tokens, label in zip(tokenized, y, strict=True):
        row = int(np.searchsorted(classes, label))
        for token in tokens:
            counts[row, token_index[token]] += 1.0
    log_prior = np.log(class_counts / len(y))
    log_likelihood = np.log(counts / counts.sum(axis=1, keepdims=True))
    return vocabulary, classes, log_prior, log_likelihood


def predict_multinomial_naive_bayes(
    documents: Sequence[str],
    vocabulary: Sequence[str],
    classes: np.ndarray,
    log_prior: np.ndarray,
    log_likelihood: np.ndarray,
) -> np.ndarray:
    """Predict labels, ignoring tokens that were unseen during fitting."""
    token_index = {token: index for index, token in enumerate(vocabulary)}
    scores = np.tile(np.asarray(log_prior, dtype=np.float64), (len(documents), 1))
    for row, document in enumerate(documents):
        for token in document.lower().split():
            if token in token_index:
                scores[row] += log_likelihood[:, token_index[token]]
    return np.asarray(classes)[scores.argmax(axis=1)]

