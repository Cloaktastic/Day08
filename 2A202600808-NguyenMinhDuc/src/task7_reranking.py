"""
Task 7 - Reranking Module.

Default behavior:
- Use Jina Reranker API when JINA_API_KEY is configured.
- Fall back to local keyword-overlap reranking if the API is unavailable.
- Provide RRF for merging ranked lists from semantic and lexical search.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from math import sqrt

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
JINA_TIMEOUT_SECONDS = 60


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/English text using Unicode word characters."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def normalize_score(score: float, max_score: float) -> float:
    """Normalize a score into [0, 1] using the max score in the candidate list."""
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_score))


def keyword_overlap_score(query: str, content: str) -> float:
    """
    Score how many query terms are covered by the content.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    content_counts = Counter(tokenize(content))

    matched = 0
    total = sum(query_counts.values())
    for token, count in query_counts.items():
        matched += min(count, content_counts.get(token, 0))

    return matched / total if total else 0.0


def keyword_overlap_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Local fallback reranker based on query-token overlap and original score."""
    if top_k <= 0 or not candidates:
        return []

    max_original_score = max(float(item.get("score", 0)) for item in candidates) or 1.0
    reranked: list[dict] = []

    for item in candidates:
        original = normalize_score(float(item.get("score", 0)), max_original_score)
        overlap = keyword_overlap_score(query, item.get("content", ""))
        new_score = 0.65 * overlap + 0.35 * original

        reranked_item = item.copy()
        metadata = dict(reranked_item.get("metadata", {}))
        metadata["rerank_method"] = "keyword_overlap"
        metadata["original_score"] = float(item.get("score", 0))
        metadata["keyword_overlap"] = float(overlap)
        reranked_item["metadata"] = metadata
        reranked_item["score"] = float(new_score)
        reranked.append(reranked_item)

    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates using Jina Reranker API.

    Falls back to keyword-overlap reranking when the key is missing or the API
    returns an error, so tests and demos remain stable.
    """
    if top_k <= 0 or not candidates:
        return []

    if not JINA_API_KEY:
        return keyword_overlap_rerank(query, candidates, top_k=top_k)

    documents = [candidate.get("content", "") for candidate in candidates]
    try:
        response = requests.post(
            JINA_RERANK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
            timeout=JINA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        results: list[dict] = []
        for result in payload.get("results", []):
            index = int(result.get("index", -1))
            if index < 0 or index >= len(candidates):
                continue

            item = candidates[index].copy()
            metadata = dict(item.get("metadata", {}))
            metadata["rerank_method"] = "jina"
            metadata["rerank_model"] = JINA_RERANK_MODEL
            metadata["original_score"] = float(candidates[index].get("score", 0.0))
            item["metadata"] = metadata
            item["score"] = float(result.get("relevance_score", 0.0))
            results.append(item)

        if results:
            return results[:top_k]
    except Exception as exc:
        print(f"Jina reranker failed, fallback to keyword rerank: {exc}")

    return keyword_overlap_rerank(query, candidates, top_k=top_k)


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity for MMR when candidates include embeddings."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: balance relevance and diversity.
    """
    if top_k <= 0 or not candidates:
        return []

    if not all("embedding" in candidate for candidate in candidates):
        return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[:top_k]

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx].get("embedding", []))
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_sim(candidates[idx].get("embedding", []), candidates[sel].get("embedding", []))
                    for sel in selected
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0))
        results.append(item)
    return results


def result_key(item: dict) -> str:
    """Stable key for deduplicating retrieval results."""
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    if source != "" and chunk_index != "":
        return f"{source}::{chunk_index}"
    return item.get("content", "")


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum over rankers of 1 / (k + rank).
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = result_key(item)
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    results: list[dict] = []
    for key in sorted_keys[:top_k]:
        item = content_map[key].copy()
        item["score"] = float(rrf_scores[key])
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    method:
        - cross_encoder: Jina reranker with keyword fallback.
        - keyword_overlap: local fallback directly.
        - original_score: sort by existing score.
    """
    if top_k <= 0 or not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)

    if method == "keyword_overlap":
        return keyword_overlap_rerank(query, candidates, top_k=top_k)

    if method == "original_score":
        return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[:top_k]

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Hinh phat tu 2-7 nam cho toi tang tru", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content']}")
