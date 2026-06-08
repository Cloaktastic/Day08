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
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Reconfigure output streams to handle UTF-8 printing safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ documents dạng PDF lên PageIndex.
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not configured in .env file.")
        return

    client = PageIndexClient(PAGEINDEX_API_KEY)

    # Lấy danh sách document hiện tại trên PageIndex để tránh upload trùng lặp
    try:
        existing_docs = client.list_documents().get("documents", [])
        existing_names = {d["name"] for d in existing_docs}
    except Exception as e:
        print(f"Error checking existing documents on PageIndex: {e}")
        existing_names = set()

    # Tìm các file PDF trong data/landing/legal/
    legal_dir = Path(__file__).parent.parent / "data" / "landing" / "legal"
    
    for pdf_file in legal_dir.glob("*.pdf"):
        if pdf_file.name in existing_names:
            print(f"Document {pdf_file.name} is already uploaded. Skipping.")
            continue

        print(f"Uploading PDF: {pdf_file.name} ...")
        try:
            result = client.submit_document(str(pdf_file))
            print(f"  ✓ Uploaded: {pdf_file.name} -> doc_id: {result.get('doc_id')}")
        except Exception as e:
            print(f"  ✗ Failed to upload {pdf_file.name}: {e}")


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
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not configured.")
        return []

    client = PageIndexClient(PAGEINDEX_API_KEY)

    # 1. Lấy danh sách các doc_id đang có trên PageIndex
    try:
        docs = client.list_documents().get("documents", [])
    except Exception as e:
        print(f"Error listing documents on PageIndex: {e}")
        return []

    if not docs:
        print("No documents found in PageIndex. Please upload documents first.")
        return []

    doc_ids = [d["id"] for d in docs]

    # 2. Thực hiện chat completions có citation để lấy câu trả lời suy luận hoàn chỉnh
    try:
        response = client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=doc_ids,
            enable_citations=True
        )
        answer = response["choices"][0]["message"]["content"]
        
        # Đóng gói kết quả phù hợp với định dạng trả về của retrieval
        return [{
            "content": answer,
            "score": 1.0,
            "metadata": {"doc_ids": doc_ids},
            "source": "pageindex"
        }]
    except Exception as e:
        print(f"Error querying PageIndex: {e}")
        raise e


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:300]}...")
