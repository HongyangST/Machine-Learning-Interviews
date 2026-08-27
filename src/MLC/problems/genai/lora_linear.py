"""Low-rank adaptation applied to a frozen linear layer.

Interview prompt: add a trainable rank-r update to a frozen weight matrix and
explain parameter savings, scaling, initialization, and merge-at-inference.
"""

from __future__ import annotations

import numpy as np


def lora_linear(
    inputs: np.ndarray,
    weight: np.ndarray,
    adapter_a: np.ndarray,
    adapter_b: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute xW^T + (alpha/r) xA^T B^T for batch-major inputs."""
    x = np.asarray(inputs, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    a = np.asarray(adapter_a, dtype=np.float64)
    b = np.asarray(adapter_b, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("all inputs must be matrices")
    rank = a.shape[0]
    if rank < 1 or x.shape[1] != w.shape[1] or a.shape[1] != w.shape[1] or b.shape != (w.shape[0], rank):
        raise ValueError("incompatible base and adapter shapes")
    return x @ w.T + (alpha / rank) * ((x @ a.T) @ b.T)

