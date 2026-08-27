"""Vectorized k-nearest-neighbors classification.

Interview prompt: predict labels for a batch of queries and discuss the cost of
the distance matrix, tie-breaking, feature scaling, and choosing k.
"""

from __future__ import annotations

import numpy as np


def knn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    """Predict labels using vectorized squared Euclidean distances."""
    train_x = _feature_matrix(train_features, "train_features")
    query_x = _feature_matrix(query_features, "query_features")
    labels = np.asarray(train_labels)
    if labels.shape != (train_x.shape[0],):
        raise ValueError("train_labels must have one value per training row")
    if query_x.shape[1] != train_x.shape[1]:
        raise ValueError("training and query feature widths must match")
    if not 1 <= k <= train_x.shape[0]:
        raise ValueError("k must be between 1 and the number of training rows")

    distances = ((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    neighbor_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    predictions = []
    for indices in neighbor_indices:
        values, counts = np.unique(labels[indices], return_counts=True)
        predictions.append(values[np.argmax(counts)])
    return np.asarray(predictions, dtype=labels.dtype)


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite [N, D] matrix")
    return matrix

