"""Leakage-safe train, validation, and test index splitting.

Interview prompt: shuffle or stratify examples reproducibly, return disjoint
partitions, and explain why preprocessing must be fit on the training split.
"""

from __future__ import annotations

import numpy as np


def train_validation_test_indices(
    size: int,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shuffled, disjoint train/validation/test row indices."""
    if size < 3 or validation_fraction < 0 or test_fraction < 0:
        raise ValueError("size must be at least 3 and fractions non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    order = np.random.default_rng(seed).permutation(size)
    test_count = round(size * test_fraction)
    validation_count = round(size * validation_fraction)
    if test_count + validation_count >= size:
        raise ValueError("fractions leave no training examples")
    test = order[:test_count]
    validation = order[test_count : test_count + validation_count]
    train = order[test_count + validation_count :]
    return train, validation, test


def stratified_train_validation_test_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each class independently, then shuffle each combined partition."""
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y):
        raise ValueError("labels must be a non-empty vector")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    generator = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[], [], []]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        generator.shuffle(indices)
        test_count = round(len(indices) * test_fraction)
        validation_count = round(len(indices) * validation_fraction)
        partitions[2].append(indices[:test_count])
        partitions[1].append(indices[test_count : test_count + validation_count])
        partitions[0].append(indices[test_count + validation_count :])
    result = []
    for groups in partitions:
        combined = np.concatenate(groups).astype(np.int64, copy=False)
        generator.shuffle(combined)
        result.append(combined)
    return result[0], result[1], result[2]

