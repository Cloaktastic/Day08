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

import json
import sys
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

# Reconfigure output streams to handle UTF-8 printing safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
META_PATH = STANDARDIZED_DIR / "vectorstore_meta.json"

# CORPUS: List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = []
_bm25_index = None


def load_corpus() -> list[dict]:
    """Đọc corpus từ file metadata."""
    global CORPUS
    if not CORPUS:
        if META_PATH.exists():
            try:
                CORPUS = json.loads(META_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading corpus metadata: {e}")
                CORPUS = []
        else:
            print(f"Metadata file {META_PATH} does not exist. Please run task 4 first.")
            CORPUS = []
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize - cho tiếng Việt nên dùng underthesea hoặc đơn giản split()
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def get_bm25_index():
    """Lấy hoặc khởi tạo BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        corpus = load_corpus()
        if corpus:
            _bm25_index = build_bm25_index(corpus)
    return _bm25_index


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
    bm25 = get_bm25_index()
    corpus = load_corpus()
    if bm25 is None or not corpus:
        print("BM25 index could not be built because corpus is empty.")
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get indices of documents, sorted by score descending
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        # Only keep matches that have a positive BM25 score
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"]
            })

    # Explicitly ensure the returned list is sorted descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
