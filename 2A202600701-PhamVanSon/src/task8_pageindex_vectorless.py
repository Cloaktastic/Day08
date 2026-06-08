"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("[WARN] PAGEINDEX_API_KEY is not set. Skipping upload.")
        return

    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            if md_file.is_file():
                content = md_file.read_text(encoding="utf-8")
                pi.upload(
                    content=content,
                    metadata={"filename": md_file.name, "type": md_file.parent.name}
                )
                print(f"  [OK] Uploaded: {md_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to upload documents: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        # Fallback về semantic search nếu không cấu hình PageIndex
        from src.task5_semantic_search import semantic_search
        local_results = semantic_search(query, top_k=top_k)
        return [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"],
                "source": "pageindex"
            }
            for r in local_results
        ]

    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)

        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex"
            }
            for r in results
        ]
    except Exception as e:
        print(f"[ERROR] PageIndex failed: {e}. Falling back to local search.")
        from src.task5_semantic_search import semantic_search
        local_results = semantic_search(query, top_k=top_k)
        return [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"],
                "source": "pageindex"
            }
            for r in local_results
        ]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[WARN] PAGEINDEX_API_KEY not found in .env, using local fallback.")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
