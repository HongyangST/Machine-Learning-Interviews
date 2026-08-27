"""A minimal append-only key/value cache for autoregressive decoding.

Interview prompt: store projected keys and values by decoding step, enforce a
fixed capacity, and explain how caching avoids recomputing the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KVCache:
    max_length: int
    _keys: list[np.ndarray] = field(default_factory=list)
    _values: list[np.ndarray] = field(default_factory=list)

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one [heads, width] key/value pair for the next token."""
        key_array = np.asarray(key, dtype=np.float64)
        value_array = np.asarray(value, dtype=np.float64)
        if self.max_length < 1 or key_array.ndim != 2 or key_array.shape != value_array.shape:
            raise ValueError("capacity must be positive and key/value shapes must match")
        if self._keys and key_array.shape != self._keys[0].shape:
            raise ValueError("all cached steps must have the same shape")
        if len(self._keys) >= self.max_length:
            raise OverflowError("KV cache capacity exceeded")
        self._keys.append(key_array.copy())
        self._values.append(value_array.copy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return cached tensors with shape [heads, sequence, width]."""
        if not self._keys:
            raise ValueError("cache is empty")
        return np.stack(self._keys, axis=1), np.stack(self._values, axis=1)

