"""Low-rank matrix factorization for explicit recommendations.

Interview prompt: factorize only observed ratings with SGD and explain latent
dimension, regularization, cold-start behavior, and scale ambiguity.
"""

from __future__ import annotations

import numpy as np


def matrix_factorization_sgd(
    ratings: np.ndarray,
    rank: int,
    *,
    learning_rate: float = 0.02,
    regularization: float = 0.01,
    epochs: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize a matrix where NaN denotes an unobserved user-item rating."""
    values = np.asarray(ratings, dtype=np.float64)
    observed = np.argwhere(np.isfinite(values))
    if values.ndim != 2 or not len(observed):
        raise ValueError("ratings must be a 2D matrix with at least one observation")
    if rank < 1 or learning_rate <= 0 or regularization < 0 or epochs < 1:
        raise ValueError("invalid optimization parameters")
    generator = np.random.default_rng(seed)
    users = generator.normal(0.0, 0.1, size=(values.shape[0], rank))
    items = generator.normal(0.0, 0.1, size=(values.shape[1], rank))
    for _ in range(epochs):
        for user, item in observed[generator.permutation(len(observed))]:
            error = values[user, item] - users[user] @ items[item]
            old_user = users[user].copy()
            users[user] += learning_rate * (error * items[item] - regularization * users[user])
            items[item] += learning_rate * (error * old_user - regularization * items[item])
    return users, items

