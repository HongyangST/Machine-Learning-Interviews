"""A small, deterministic byte-pair-encoding learner.

Interview prompt: repeatedly merge the most frequent adjacent symbol pair, then
apply the learned merges in order to an unseen word. Production tokenizers also
define byte handling, normalization, special tokens, and serialization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


Pair = tuple[str, str]


def learn_bpe(words: Sequence[str], num_merges: int) -> list[Pair]:
    """Learn character-level BPE merges from whitespace-free training words."""
    if num_merges < 0 or any(not word for word in words):
        raise ValueError("num_merges must be non-negative and words non-empty")
    vocabulary = Counter(tuple(word) + ("</w>",) for word in words)
    merges: list[Pair] = []
    for _ in range(num_merges):
        pair_counts: Counter[Pair] = Counter()
        for symbols, count in vocabulary.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        pair = sorted(pair_counts, key=lambda item: (-pair_counts[item], item))[0]
        merges.append(pair)
        updated: Counter[tuple[str, ...]] = Counter()
        for symbols, count in vocabulary.items():
            updated[_merge_pair(symbols, pair)] += count
        vocabulary = updated
    return merges


def encode_bpe(word: str, merges: Sequence[Pair]) -> list[str]:
    """Encode one word by replaying a learned merge sequence."""
    if not word:
        return []
    symbols = tuple(word) + ("</w>",)
    for pair in merges:
        symbols = _merge_pair(symbols, pair)
    return list(symbols)


def _merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)

