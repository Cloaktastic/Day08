"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
import sys
from pathlib import Path
import faiss
import numpy as np

# Reconfigure output streams to handle UTF-8 printing safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn RecursiveCharacterTextSplitter vì đây là phương pháp chia nhỏ văn bản an toàn nhất,
# giữ các đoạn thông tin tự nhiên liền mạch qua các dấu ngắt dòng (\n\n, \n) và dấu câu.
# Chọn chunk size là 500 ký tự để giữ cho ngữ cảnh của mỗi đoạn vừa đủ thông tin,
# tránh việc thông tin bị cắt quá nhỏ hoặc quá lớn làm loãng kết quả tìm kiếm ngữ nghĩa.
# Chọn overlap là 50 ký tự để không bị mất thông tin giữa ranh giới các chunk liền kề.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Chọn sentence-transformers/all-MiniLM-L6-v2 vì đây là model embedding cục bộ (local) rất nhẹ,
# chỉ có 384 dimensions, tốc độ sinh embedding cực nhanh và chiếm rất ít bộ nhớ,
# phù hợp hoàn hảo cho việc chạy thử nghiệm và chạy local trên máy cá nhân mà không cần GPU hay API keys.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chọn FAISS làm vector store vì tính đơn giản, hoạt động hoàn toàn offline, lưu trực tiếp dưới dạng file
# trong thư mục dự án và không yêu cầu cài đặt Docker hay chạy dịch vụ Weaviate server bên ngoài,
# giúp cho toàn bộ pipeline của bài cá nhân chạy độc lập và mượt mà.
VECTOR_STORE = "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    embeddings = [c["embedding"] for c in chunks]
    metadata_list = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]

    # Convert embeddings to numpy array
    xb = np.array(embeddings).astype('float32')

    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(xb)

    # Create index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(xb)

    # Save index and metadata
    faiss_path = STANDARDIZED_DIR / "vectorstore.faiss"
    meta_path = STANDARDIZED_DIR / "vectorstore_meta.json"

    faiss.write_index(index, str(faiss_path))
    meta_path.write_text(json.dumps(metadata_list, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
