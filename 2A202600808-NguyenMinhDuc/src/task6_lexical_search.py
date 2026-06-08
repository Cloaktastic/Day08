"""
Task 6 - Lexical Search Module using BM25.

BM25 complements semantic search by matching exact legal/news terms such as
"Dieu 249", "tang tru", "cai nghien", or "ma tuy".
"""

from __future__ import annotations

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import CHUNKS_PATH, run_pipeline

CORPUS: list[dict] = []
_BM25_CACHE: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text with a simple Unicode-aware regex."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def ensure_corpus_exists() -> None:
    """Build Task 4 local index if chunks.json is missing."""
    if not CHUNKS_PATH.exists():
        run_pipeline()


def load_corpus() -> list[dict]:
    """Load chunk corpus from data/index/chunks.json."""
    global CORPUS

    ensure_corpus_exists()

    if not CORPUS:
        records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        CORPUS = [
            {
                "content": record.get("content", ""),
                "metadata": record.get("metadata", {}),
            }
            for record in records
            if record.get("content")
        ]

    return CORPUS


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_bm25_index() -> tuple[list[dict], BM25Okapi]:
    """Return cached corpus + BM25 index."""
    global _BM25_CACHE

    corpus = load_corpus()
    if _BM25_CACHE is None:
        _BM25_CACHE = build_bm25_index(corpus)

    return corpus, _BM25_CACHE


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search relevant chunks using BM25 keyword matching.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted
        by BM25 score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus, bm25 = get_bm25_index()
    if not corpus:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    limit = min(top_k, len(corpus))
    top_indices = np.argsort(scores)[::-1][:limit]

    results: list[dict] = []
    for idx in top_indices:
        score = float(scores[int(idx)])
        if score <= 0:
            continue

        doc = corpus[int(idx)]
        results.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": doc.get("metadata", {}),
            }
        )

    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 249 tang tru trai phep chat ma tuy", top_k=5)
    for result in results:
        preview = result["content"][:100].encode("ascii", "ignore").decode("ascii")
        print(f"[{result['score']:.3f}] {preview}...")
