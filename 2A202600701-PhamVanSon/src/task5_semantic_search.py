"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


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
    import json
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    # Đường dẫn vectorstore
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore.json"
    if not vectorstore_path.exists():
        print(f"[WARN] Vector store not found, running task 4 pipeline first...")
        from src.task4_chunking_indexing import run_pipeline
        run_pipeline()
        if not vectorstore_path.exists():
            return []

    with open(vectorstore_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return []

    # Encode query sử dụng cùng model ở Task 4
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query, show_progress_bar=False)

    # Tính toán Cosine Similarity
    query_emb = np.array(query_embedding)
    chunk_embs = np.array([c["embedding"] for c in chunks])
    
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        query_norm = 1e-10
    query_emb_norm = query_emb / query_norm
    
    chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    chunk_norms[chunk_norms == 0] = 1e-10
    chunk_embs_norm = chunk_embs / chunk_norms
    
    scores = np.dot(chunk_embs_norm, query_emb_norm)

    results = []
    for chunk, score in zip(chunks, scores):
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk["metadata"]
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
