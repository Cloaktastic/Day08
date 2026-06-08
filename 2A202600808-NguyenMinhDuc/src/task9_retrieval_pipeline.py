"""
Task 9 - Complete retrieval pipeline.

Pipeline:
    semantic_search + lexical_search -> RRF merge -> rerank -> PageIndex fallback
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def normalize_result(item: dict, source: str) -> dict:
    """Ensure every retrieval result has the lab-required shape."""
    return {
        "content": item.get("content", ""),
        "score": float(item.get("score", 0.0)),
        "metadata": item.get("metadata", {}),
        "source": source,
    }


def safe_search(search_fn, query: str, top_k: int) -> list[dict]:
    """Run one retriever without letting failures crash the full pipeline."""
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"{search_fn.__name__} failed: {exc}")
        return []


def fallback_to_pageindex(query: str, top_k: int) -> list[dict]:
    """Run PageIndex fallback and normalize source markers."""
    fallback = safe_search(pageindex_search, query, top_k)
    return [normalize_result(item, "pageindex") for item in fallback[:top_k]]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline with fallback logic.

    Args:
        query: User query.
        top_k: Number of final results.
        score_threshold: If the best hybrid score is below this, use PageIndex.
        use_reranking: Whether to re-score merged hybrid results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'hybrid' | 'pageindex'}.
    """
    if top_k <= 0 or not query.strip():
        return []

    candidate_k = max(top_k * 2, top_k)

    dense_results = safe_search(semantic_search, query, candidate_k)
    sparse_results = safe_search(lexical_search, query, candidate_k)

    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    merged = [normalize_result(item, "hybrid") for item in merged]

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = [normalize_result(item, "hybrid") for item in final_results]
    else:
        final_results = merged[:top_k]

    if not final_results:
        return fallback_to_pageindex(query, top_k)

    best_score = float(final_results[0].get("score", 0.0))
    if best_score < score_threshold:
        fallback = fallback_to_pageindex(query, top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "hinh phat cho toi tang tru trai phep chat ma tuy",
        "nghe si nao bi bat vi su dung ma tuy",
        "luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retrieve(query, top_k=3)
        for index, result in enumerate(results, 1):
            preview = result["content"][:80].encode("ascii", "ignore").decode("ascii")
            print(f"  {index}. [{result['score']:.3f}] [{result['source']}] {preview}...")
