"""Pad variable-length token sequences and build a validity mask.

Interview prompt: produce batch-major token IDs and a boolean mask while
handling empty sequences and a caller-specified padding ID.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pad_sequences(
    sequences: Sequence[Sequence[int]], *, padding_id: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded IDs and True-at-real-token mask with shape [batch, length]."""
    if not sequences:
        raise ValueError("at least one sequence is required")
    max_length = max(map(len, sequences))
    ids = np.full((len(sequences), max_length), padding_id, dtype=np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    for row, sequence in enumerate(sequences):
        values = np.asarray(sequence, dtype=np.int64)
        ids[row, : len(values)] = values
        valid[row, : len(values)] = True
    return ids, valid

