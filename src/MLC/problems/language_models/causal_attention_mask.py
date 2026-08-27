"""Causal and padding masks for autoregressive attention.

Interview prompt: build an allowed-position mask where a query can attend only
to real keys at or before its own position, with clear shape semantics.
"""

from __future__ import annotations

import numpy as np


def causal_attention_mask(
    sequence_length: int, *, key_is_valid: np.ndarray | None = None
) -> np.ndarray:
    """Return a boolean [query, key] mask for left-to-right self-attention."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    allowed = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))
    if key_is_valid is not None:
        valid = np.asarray(key_is_valid, dtype=bool)
        if valid.shape != (sequence_length,):
            raise ValueError("key_is_valid must have shape [sequence_length]")
        allowed &= valid[None, :]
    return allowed

