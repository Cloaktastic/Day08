import os
import json
import sys
from pathlib import Path

# Thử import các thư viện cần thiết, hướng dẫn nếu chưa cài đặt
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] Chưa cài đặt đủ thư viện 'langchain-text-splitters' hoặc 'sentence-transformers'.")
    print("Vui lòng chạy lệnh: pip install langchain-text-splitters sentence-transformers")
    sys.exit(1)

# Thiết lập đường dẫn tương thích với cấu trúc của group_project
BASE_DIR = Path(__file__).parent
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
VECTORSTORE_PATH = BASE_DIR / "data" / "vectorstore.json"

# Cấu hình Chunking & Embedding
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    """Đọc toàn bộ file Markdown đã chuẩn hóa từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"[WARN] Thư mục standardized không tồn tại: {STANDARDIZED_DIR}")
        return documents

    # Đọc đệ quy tất cả các file .md trong thư mục standardized/
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file():
            content = md_file.read_text(encoding="utf-8")
            # Xác định loại tài liệu dựa trên đường dẫn thư mục con
            doc_type = "legal" if "legal" in str(md_file.parts) else "news"
            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "title": content.split("\n")[0].replace("#", "").strip() if content.startswith("#") else md_file.stem
                }
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Cắt nhỏ tài liệu thành các phân đoạn (chunks) tuần tự."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Tính toán vector embedding cho các phân đoạn văn bản."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [c["content"] for c in chunks]
    print(f"Calculating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def save_to_vectorstore(chunks: list[dict]):
    """Lưu trữ cơ sở dữ liệu vector xuống file JSON cục bộ."""
    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Ghi dữ liệu ra file JSON
    with open(VECTORSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"  [OK] Đã cập nhật cơ sở dữ liệu tại: {VECTORSTORE_PATH.relative_to(BASE_DIR)}")


def run_indexing():
    """Chạy toàn bộ quy trình Indexing dữ liệu."""
    print("=" * 60)
    print(" Bắt đầu lập chỉ mục dữ liệu mới (Chunking & Embedding) ")
    print("=" * 60)

    # 1. Đọc file
    docs = load_documents()
    print(f"[1/4] Đã đọc: {len(docs)} tài liệu Markdown")
    if not docs:
        print("[WARN] Không tìm thấy dữ liệu Markdown nào. Quy trình dừng lại.")
        return

    # 2. Cắt nhỏ (Chunking)
    chunks = chunk_documents(docs)
    print(f"[2/4] Đã cắt thành: {len(chunks)} phân đoạn")

    # 3. Tạo vector (Embedding)
    chunks_with_embeddings = embed_chunks(chunks)
    print(f"[3/4] Đã vector hóa thành công: {len(chunks_with_embeddings)} phân đoạn")

    # 4. Lưu lại
    save_to_vectorstore(chunks_with_embeddings)
    print("\n[OK] Hoàn thành lập chỉ mục! Dữ liệu mới đã sẵn sàng cho Chatbot hoạt động.")


if __name__ == "__main__":
    run_indexing()
