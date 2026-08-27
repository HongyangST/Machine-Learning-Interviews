"""A compact TF-IDF feature builder.

Interview prompt: tokenize documents, build a deterministic vocabulary, and
compute smoothed inverse document frequency without a text library.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def tfidf(documents: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Build a basic lowercase whitespace-tokenized TF-IDF matrix."""
    tokenized = [document.lower().split() for document in documents]
    vocabulary = sorted({token for document in tokenized for token in document})
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    if not vocabulary:
        return matrix, vocabulary

    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
    for row, document in enumerate(tokenized):
        if not document:
            continue
        counts: dict[str, int] = {}
        for token in document:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            column = token_to_index[token]
            matrix[row, column] = count / len(document)
            document_frequency[column] += 1

    inverse_document_frequency = np.log(
        (1.0 + len(documents)) / (1.0 + document_frequency)
    ) + 1.0
    return matrix * inverse_document_frequency, vocabulary

