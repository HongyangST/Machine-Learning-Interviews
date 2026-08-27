"""Principal component analysis via singular-value decomposition.

Interview prompt: center a data matrix, compute its principal directions, and
return both the projection and explained variance.
"""

from __future__ import annotations

import numpy as np


def principal_component_analysis(
    features: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected data, principal axes, and explained variances."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    if not 1 <= n_components <= min(x.shape):
        raise ValueError("n_components must not exceed min(num_rows, num_features)")

    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:n_components]
    transformed = centered @ components.T
    denominator = max(x.shape[0] - 1, 1)
    explained_variance = singular_values[:n_components] ** 2 / denominator
    return transformed, components, explained_variance

