"""Uniform reservoir sampling over a stream.

Interview prompt: select k items uniformly when the stream length is unknown,
using O(k) memory and one pass.
"""

from __future__ import annotations

from collections.abc import Iterable
import random


def reservoir_sample(items: Iterable[object], k: int, *, seed: int = 0) -> list[object]:
    """Select k uniformly distributed items from a stream of unknown length."""
    if k < 0:
        raise ValueError("k must be non-negative")
    generator = random.Random(seed)
    reservoir: list[object] = []
    for index, item in enumerate(items):
        if index < k:
            reservoir.append(item)
            continue
        replacement_index = generator.randint(0, index)
        if replacement_index < k:
            reservoir[replacement_index] = item
    return reservoir

