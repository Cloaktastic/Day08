"""
Task 5 - Semantic search over the local vector index.

The index is produced by Task 4:
    data/index/chunks.json
    data/index/embeddings.npy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .task4_chunking_indexing import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    hashing_embedding,
    run_pipeline,
)

_CHUNKS_CACHE: list[dict] | None = None
_EMBEDDINGS_CACHE: np.ndarray | None = None


def ensure_index_exists() -> None:
    """Build the local index if Task 4 has not been run yet."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        run_pipeline()


def load_index() -> tuple[list[dict], np.ndarray]:
    """Load chunks and embeddings from data/index/ with a small memory cache."""
    global _CHUNKS_CACHE, _EMBEDDINGS_CACHE

    ensure_index_exists()

    if _CHUNKS_CACHE is None:
        _CHUNKS_CACHE = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = np.load(EMBEDDINGS_PATH).astype(np.float32)

    return _CHUNKS_CACHE, _EMBEDDINGS_CACHE


def cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and all chunk vectors.
    Task 4 embeddings are normalized, but this function also handles non-normalized
    vectors to keep the module robust.
    """
    query_norm = float(np.linalg.norm(query_embedding))
    if query_norm == 0 or embeddings.size == 0:
        return np.zeros(len(embeddings), dtype=np.float32)

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    denominator = embedding_norms * query_norm
    denominator[denominator == 0] = 1.0

    return (embeddings @ query_embedding) / denominator


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search relevant chunks using vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted
        by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    chunks, embeddings = load_index()
    if not chunks or embeddings.size == 0:
        return []

    query_embedding = np.array(hashing_embedding(query, dim=embeddings.shape[1]), dtype=np.float32)
    scores = cosine_scores(query_embedding, embeddings)

    limit = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:limit]

    results: list[dict] = []
    for idx in top_indices:
        chunk = chunks[int(idx)]
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(scores[int(idx)]),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return results


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for result in results:
        preview = result["content"][:100].encode("ascii", "ignore").decode("ascii")
        print(f"[{result['score']:.3f}] {preview}...")
