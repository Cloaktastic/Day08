"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_BM25 = None
_CORPUS_LOADED = None

def load_corpus() -> list[dict]:
    """Tự động tải corpus từ vectorstore hoặc chạy task 4."""
    import json
    from pathlib import Path
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.json"
    if not vectorstore_path.exists():
        print(f"[WARN] Vector store not found, running task 4 pipeline first...")
        from src.task4_chunking_indexing import run_pipeline
        run_pipeline()
    if vectorstore_path.exists():
        with open(vectorstore_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_bm25_index():
    """Lấy cached BM25 index."""
    global _BM25, _CORPUS_LOADED
    if _BM25 is None:
        _CORPUS_LOADED = load_corpus()
        if not _CORPUS_LOADED:
            return None, []
        _BM25 = build_bm25_index(_CORPUS_LOADED)
    return _BM25, _CORPUS_LOADED


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    # Tokenize - split đơn giản, hiệu quả cho tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = get_bm25_index()
    if not bm25 or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"]
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
