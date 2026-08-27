"""Temperature, top-k, and nucleus filtering for token sampling.

Interview prompt: transform logits into a valid filtered distribution and sample
reproducibly, while keeping at least one candidate under extreme thresholds.
"""

from __future__ import annotations

import numpy as np


def filtered_probabilities(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> np.ndarray:
    """Return normalized probabilities after temperature, top-k, and top-p."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("logits must be a non-empty finite vector")
    if temperature <= 0 or (top_k is not None and not 1 <= top_k <= len(scores)):
        raise ValueError("temperature and top_k are out of range")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    filtered = scores / temperature
    if top_k is not None:
        cutoff = np.partition(filtered, -top_k)[-top_k]
        filtered = np.where(filtered >= cutoff, filtered, -np.inf)
    probabilities = _softmax(filtered)
    if top_p is not None:
        order = np.argsort(-probabilities)
        remove = np.cumsum(probabilities[order]) - probabilities[order] >= top_p
        probabilities[order[remove]] = 0.0
        probabilities /= probabilities.sum()
    return probabilities


def sample_token(logits: np.ndarray, *, seed: int = 0, **filter_options: object) -> int:
    """Sample one token index from filtered logits."""
    probabilities = filtered_probabilities(logits, **filter_options)
    return int(np.random.default_rng(seed).choice(len(probabilities), p=probabilities))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()

