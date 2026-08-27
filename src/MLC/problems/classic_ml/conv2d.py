"""Single-channel valid 2D cross-correlation.

Interview prompt: implement the operation usually called convolution in deep
learning libraries, including output-shape and stride calculations.
"""

from __future__ import annotations

import numpy as np


def conv2d_valid(
    image: np.ndarray, kernel: np.ndarray, *, stride: int = 1
) -> np.ndarray:
    """Compute a single-channel valid 2D cross-correlation."""
    x = np.asarray(image, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional")
    if stride < 1 or np.any(np.asarray(weights.shape) > np.asarray(x.shape)):
        raise ValueError("stride must be positive and kernel must fit inside image")

    output_height = 1 + (x.shape[0] - weights.shape[0]) // stride
    output_width = 1 + (x.shape[1] - weights.shape[1]) // stride
    output = np.empty((output_height, output_width), dtype=np.float64)
    for row in range(output_height):
        for column in range(output_width):
            row_start = row * stride
            column_start = column * stride
            window = x[
                row_start : row_start + weights.shape[0],
                column_start : column_start + weights.shape[1],
            ]
            output[row, column] = np.sum(window * weights)
    return output

