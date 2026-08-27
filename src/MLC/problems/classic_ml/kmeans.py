"""K-means clustering with Lloyd's algorithm.

Interview prompt: implement assignment and update steps, then handle convergence
and an empty cluster instead of assuming every cluster receives a point.
"""

from __future__ import annotations

import numpy as np


def kmeans(
    features: np.ndarray,
    k: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rows with Lloyd's algorithm and deterministic initialization."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    if not 1 <= k <= x.shape[0]:
        raise ValueError("k must be between 1 and the number of rows")
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("max_iterations must be positive and tolerance non-negative")

    generator = np.random.default_rng(seed)
    centers = x[generator.choice(x.shape[0], size=k, replace=False)].copy()
    for _ in range(max_iterations):
        squared_distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared_distances.argmin(axis=1)
        new_centers = centers.copy()
        nearest_distance = squared_distances[np.arange(x.shape[0]), labels].copy()
        for cluster_index in range(k):
            members = x[labels == cluster_index]
            if len(members):
                new_centers[cluster_index] = members.mean(axis=0)
            else:
                farthest_index = int(np.argmax(nearest_distance))
                new_centers[cluster_index] = x[farthest_index]
                nearest_distance[farthest_index] = -1.0
        if np.linalg.norm(new_centers - centers) <= tolerance:
            centers = new_centers
            break
        centers = new_centers

    final_distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return centers, final_distances.argmin(axis=1)

