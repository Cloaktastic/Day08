"""
Task 4 - Chunking and local indexing.

This implementation keeps the lab runnable without Docker or a cloud vector DB:
it loads Markdown files, creates bounded-size chunks, embeds them, and stores a
local index under data/index/. Task 5 can reuse that index for semantic search.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recursive chunking is a safe default for mixed Markdown from legal PDFs and
# news JSON: it first respects paragraphs/headings, then lines, then words.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# For the individual lab we use a lightweight local hashing embedding by
# default. It is deterministic, offline, and avoids blocking on model downloads.
# If USE_SENTENCE_TRANSFORMERS is set True and the model is available, the same
# pipeline can use a real multilingual embedding model.
EMBEDDING_MODEL = "local-hashing-v1"
EMBEDDING_DIM = 384
USE_SENTENCE_TRANSFORMERS = False
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Local vector store: chunks.json + embeddings.npy. This is simple to inspect,
# easy to rebuild, and sufficient for Task 5 semantic search.
VECTOR_STORE = "local"


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def detect_doc_type(md_file: Path) -> str:
    """Infer document type from its folder name."""
    parts = {part.lower() for part in md_file.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        rel_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": rel_path,
                    "type": detect_doc_type(md_file),
                },
            }
        )

    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def split_oversized_text(text: str) -> list[str]:
    """Strictly split text so no chunk exceeds CHUNK_SIZE."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def fallback_recursive_split(text: str) -> list[str]:
    """
    Small local recursive splitter used when langchain-text-splitters is absent.
    It greedily packs paragraphs/lines/sentences while respecting CHUNK_SIZE.
    """
    units = re.split(r"(\n\n+|\n|(?<=[.!?])\s+)", text)
    pieces: list[str] = []
    buffer = ""

    for unit in units:
        if not unit:
            continue

        candidate = f"{buffer}{unit}" if buffer else unit
        if len(candidate) <= CHUNK_SIZE:
            buffer = candidate
            continue

        if buffer.strip():
            pieces.extend(split_oversized_text(buffer))
        buffer = unit

    if buffer.strip():
        pieces.extend(split_oversized_text(buffer))

    final_chunks: list[str] = []
    previous_tail = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if previous_tail and len(piece) + len(previous_tail) + 1 <= CHUNK_SIZE:
            piece = f"{previous_tail} {piece}"
        final_chunks.append(piece)
        previous_tail = piece[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""

    return final_chunks


def split_text(text: str) -> list[str]:
    """Split text with LangChain when available, otherwise use local fallback."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        splits = splitter.split_text(text)
    except Exception:
        splits = fallback_recursive_split(text)

    bounded: list[str] = []
    for split in splits:
        bounded.extend(split_oversized_text(split))
    return [chunk for chunk in bounded if chunk.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using the configured recursive strategy.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks: list[dict] = []

    for doc_index, doc in enumerate(documents):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        splits = split_text(content)

        for chunk_index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


# =============================================================================
# EMBEDDING
# =============================================================================

def tokenize(text: str) -> list[str]:
    """Simple tokenization that works for Vietnamese text with whitespace."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Deterministic local embedding based on hashed tokens.

    This is not as semantic as a transformer model, but it gives a stable vector
    index for the lab and keeps Task 5 runnable without network/model downloads.
    """
    vector = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm

    return vector.tolist()


def embed_with_sentence_transformers(chunks: list[dict]) -> list[list[float]] | None:
    """Try a real multilingual embedding model if explicitly enabled."""
    if not USE_SENTENCE_TRANSFORMERS:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.astype(np.float32).tolist()
    except Exception as exc:
        print(f"SentenceTransformer unavailable; using local hashing embeddings. Error: {exc}")
        return None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' key to every chunk.

    Returns:
        The same chunk dictionaries with embedding: list[float].
    """
    if not chunks:
        return []

    embeddings = embed_with_sentence_transformers(chunks)
    if embeddings is None:
        embeddings = [hashing_embedding(chunk["content"]) for chunk in chunks]

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded = {
            "content": chunk["content"],
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding,
        }
        embedded_chunks.append(embedded)

    return embedded_chunks


# =============================================================================
# LOCAL INDEXING
# =============================================================================

def index_to_vectorstore(chunks: list[dict]) -> None:
    """
    Save chunks and embeddings to a local vector store under data/index/.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunk_records = []
    embeddings = []
    for chunk_id, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")
        if embedding is None:
            embedding = hashing_embedding(chunk["content"])

        chunk_records.append(
            {
                "id": chunk_id,
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
        )
        embeddings.append(embedding)

    embedding_array = np.array(embeddings, dtype=np.float32)
    CHUNKS_PATH.write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(EMBEDDINGS_PATH, embedding_array)

    metadata = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_method": CHUNKING_METHOD,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": int(embedding_array.shape[1]) if embedding_array.size else EMBEDDING_DIM,
        "vector_store": VECTOR_STORE,
        "num_chunks": len(chunk_records),
    }
    INDEX_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline() -> None:
    """Run the full pipeline: load -> chunk -> embed -> index."""
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

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")

    index_to_vectorstore(embedded_chunks)
    print(f"Indexed to local vector store: {INDEX_DIR}")


if __name__ == "__main__":
    run_pipeline()
