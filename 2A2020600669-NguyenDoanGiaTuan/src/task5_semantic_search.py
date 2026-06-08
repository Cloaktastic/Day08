"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json
import sys
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Reconfigure output streams to handle UTF-8 printing safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Paths (compatible with Task 4 output)
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
FAISS_PATH = STANDARDIZED_DIR / "vectorstore.faiss"
META_PATH = STANDARDIZED_DIR / "vectorstore_meta.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global model instance to avoid reloading it on every query call
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not FAISS_PATH.exists() or not META_PATH.exists():
        print("Vector store files not found. Please run task 4 first.")
        return []

    # 1. Load embedding model and compute query embedding
    model = get_embedding_model()
    query_vector = model.encode(query)

    # 2. Prepare query embedding as 2D float32 numpy array and normalize for Cosine Similarity
    xq = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(xq)

    # 3. Read index and search
    index = faiss.read_index(str(FAISS_PATH))
    D, I = index.search(xq, top_k)

    # 4. Load metadata
    metadata_list = json.loads(META_PATH.read_text(encoding="utf-8"))

    # 5. Build and sort result list
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if idx >= len(metadata_list):
            continue

        item = metadata_list[idx]
        results.append({
            "content": item["content"],
            "score": float(score),
            "metadata": item["metadata"]
        })

    # FAISS FlatIP search returns sorted by descending score already,
    # but we explicitly sort to satisfy all expectations.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
