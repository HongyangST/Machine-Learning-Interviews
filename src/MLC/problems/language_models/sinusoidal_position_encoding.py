"""Sinusoidal positional encodings from the Transformer architecture.

Interview prompt: generate the position-by-dimension matrix and explain why sine
and cosine pairs give the model deterministic relative-position information.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_position_encoding(length: int, width: int) -> np.ndarray:
    """Return a float64 matrix with shape [length, width]."""
    if length < 0 or width < 1:
        raise ValueError("length must be non-negative and width positive")
    positions = np.arange(length, dtype=np.float64)[:, None]
    even_dimensions = np.arange(0, width, 2, dtype=np.float64)
    angles = positions / np.power(10_000.0, even_dimensions / width)
    encoding = np.zeros((length, width), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding

