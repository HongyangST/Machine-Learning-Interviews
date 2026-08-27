"""Exact cosine top-k retrieval for a small RAG index.

Interview prompt: rank document embeddings for a query, handle zero vectors, and
explain how a production system adds ANN search, metadata/ACL filters, reranking,
provenance, and retrieval evaluation.
"""

from __future__ import annotations

import numpy as np


def cosine_top_k(
    query: np.ndarray, document_embeddings: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return document indices and cosine scores in descending order."""
    q = np.asarray(query, dtype=np.float64)
    documents = np.asarray(document_embeddings, dtype=np.float64)
    if q.ndim != 1 or documents.ndim != 2 or documents.shape[1] != len(q):
        raise ValueError("expected query [D] and document_embeddings [N, D]")
    if not 1 <= k <= len(documents):
        raise ValueError("k must be between 1 and the number of documents")
    q_norm = np.linalg.norm(q)
    document_norms = np.linalg.norm(documents, axis=1)
    if q_norm == 0 or np.any(document_norms == 0):
        raise ValueError("cosine similarity is undefined for zero vectors")
    scores = documents @ q / (document_norms * q_norm)
    indices = np.argsort(-scores, kind="stable")[:k]
    return indices, scores[indices]

