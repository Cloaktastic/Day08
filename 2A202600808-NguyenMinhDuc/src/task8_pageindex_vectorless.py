"""
Task 8 - PageIndex Vectorless RAG.

This module uses the real PageIndex API when PAGEINDEX_API_KEY is configured:
- POST /doc/ uploads local legal PDFs and stores returned doc_id values.
- GET /doc/{doc_id}/?type=tree checks whether retrieval is ready.
- POST /retrieval/ and GET /retrieval/{retrieval_id}/ retrieve relevant nodes.

If PageIndex is unavailable or documents are still processing, it falls back to
a local vectorless keyword/structure search so Task 8 and Task 9 remain usable.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .task4_chunking_indexing import CHUNKS_PATH, STANDARDIZED_DIR, run_pipeline

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_API_BASE = "https://api.pageindex.ai"
PROJECT_DIR = Path(__file__).parent.parent
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_MANIFEST_PATH = PROJECT_DIR / "data" / "index" / "pageindex_manifest.json"
REQUEST_TIMEOUT = 60
POLL_INTERVAL_SECONDS = 3
MAX_RETRIEVAL_POLL_SECONDS = 30


def api_headers(content_type_json: bool = False) -> dict[str, str]:
    """Headers required by the PageIndex API."""
    headers = {"api_key": PAGEINDEX_API_KEY}
    if content_type_json:
        headers["Content-Type"] = "application/json"
    return headers


def safe_console(text: str) -> str:
    """Return text safe for Windows terminals using legacy encodings."""
    return text.encode("ascii", "ignore").decode("ascii")


def tokenize(text: str) -> list[str]:
    """Simple Unicode-aware tokenization for Vietnamese and English."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def load_chunks() -> list[dict]:
    """Load Task 4 chunks, rebuilding the local index if needed."""
    if not CHUNKS_PATH.exists():
        run_pipeline()
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def load_manifest() -> dict[str, Any]:
    """Load the local PageIndex manifest."""
    if not PAGEINDEX_MANIFEST_PATH.exists():
        return {"mode": "pageindex_api" if PAGEINDEX_API_KEY else "local_fallback", "documents": []}
    return json.loads(PAGEINDEX_MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    """Persist the PageIndex manifest."""
    PAGEINDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def local_pdf_files() -> list[Path]:
    """PDF files that can be uploaded to PageIndex's document endpoint."""
    if not LANDING_LEGAL_DIR.exists():
        return []
    return sorted(path for path in LANDING_LEGAL_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def list_pageindex_documents() -> list[dict]:
    """List existing PageIndex documents in the user's account."""
    response = requests.get(
        f"{PAGEINDEX_API_BASE}/docs",
        headers=api_headers(),
        params={"limit": 100, "offset": 0},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("documents", [])


def upload_pdf_to_pageindex(pdf_path: Path) -> str:
    """Upload one PDF and return its PageIndex doc_id."""
    with pdf_path.open("rb") as file:
        response = requests.post(
            f"{PAGEINDEX_API_BASE}/doc/",
            headers=api_headers(),
            files={"file": (pdf_path.name, file, "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    data = response.json()
    doc_id = data.get("doc_id") or data.get("id")
    if not doc_id:
        raise ValueError(f"PageIndex upload response missing doc_id for {pdf_path.name}: {data}")
    return doc_id


def upload_documents() -> list[dict]:
    """
    Upload local legal PDFs to PageIndex and save a manifest.

    If a document with the same filename already exists in PageIndex, its id is
    reused to avoid repeated uploads.
    """
    documents: list[dict] = []

    if not PAGEINDEX_API_KEY:
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            if md_file.is_file():
                documents.append(
                    {
                        "filename": md_file.name,
                        "path": md_file.relative_to(STANDARDIZED_DIR).as_posix(),
                        "type": md_file.parent.name,
                        "status": "local_only",
                    }
                )
        save_manifest({"mode": "local_fallback", "documents": documents})
        return documents

    existing_docs = {}
    try:
        for doc in list_pageindex_documents():
            name = doc.get("name") or doc.get("filename")
            if name:
                existing_docs[name] = doc
    except Exception as exc:
        print(f"Could not list PageIndex docs, will try direct upload. Error: {exc}")

    for pdf_path in local_pdf_files():
        existing = existing_docs.get(pdf_path.name)
        if existing:
            doc_id = existing.get("id") or existing.get("doc_id")
            status = existing.get("status", "existing")
            error = None
        else:
            print(f"Uploading to PageIndex: {safe_console(pdf_path.name)}")
            try:
                doc_id = upload_pdf_to_pageindex(pdf_path)
                status = "uploaded"
                error = None
            except Exception as exc:
                doc_id = None
                status = "upload_failed"
                error = str(exc)
                print(f"  Upload failed, keeping local fallback: {exc}")

        record = {
            "filename": pdf_path.name,
            "path": pdf_path.relative_to(PROJECT_DIR).as_posix(),
            "type": "legal_pdf",
            "doc_id": doc_id,
            "status": status,
        }
        if error:
            record["error"] = error
        documents.append(record)

    manifest = {
        "mode": "pageindex_api",
        "api_base": PAGEINDEX_API_BASE,
        "documents": documents,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_manifest(manifest)
    return documents


def is_retrieval_ready(doc_id: str) -> bool:
    """Check if a PageIndex document is ready for retrieval."""
    response = requests.get(
        f"{PAGEINDEX_API_BASE}/doc/{doc_id}/",
        headers=api_headers(),
        params={"type": "tree"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return bool(data.get("retrieval_ready")) or data.get("status") == "completed"


def retrieve_from_doc(doc_id: str, query: str) -> list[dict]:
    """Run legacy PageIndex retrieval for one ready document."""
    response = requests.post(
        f"{PAGEINDEX_API_BASE}/retrieval/",
        headers=api_headers(content_type_json=True),
        json={"doc_id": doc_id, "query": query, "thinking": False},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    retrieval_id = response.json().get("retrieval_id")
    if not retrieval_id:
        return []

    deadline = time.time() + MAX_RETRIEVAL_POLL_SECONDS
    while time.time() < deadline:
        result_response = requests.get(
            f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/",
            headers=api_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        result_response.raise_for_status()
        data = result_response.json()
        if data.get("status") == "completed":
            return parse_retrieval_nodes(data, doc_id)
        if data.get("status") in {"failed", "error"}:
            return []
        time.sleep(POLL_INTERVAL_SECONDS)

    return []


def parse_retrieval_nodes(data: dict, doc_id: str) -> list[dict]:
    """Convert PageIndex retrieved_nodes into lab-compatible result dicts."""
    results: list[dict] = []
    nodes = data.get("retrieved_nodes", []) or []

    for node_rank, node in enumerate(nodes, 1):
        title = node.get("title", "")
        node_id = node.get("node_id", "")
        contents = node.get("relevant_contents", []) or []
        for content_rank, content in enumerate(contents, 1):
            text = content.get("relevant_content") or content.get("text") or ""
            if not text.strip():
                continue
            score = 1.0 / (node_rank + content_rank - 1)
            results.append(
                {
                    "content": text.strip(),
                    "score": float(score),
                    "metadata": {
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "title": title,
                        "page_index": content.get("page_index"),
                        "retrieval_id": data.get("retrieval_id"),
                        "query": data.get("query"),
                    },
                    "source": "pageindex",
                }
            )

    return results


def pageindex_api_search(query: str, top_k: int = 5) -> list[dict]:
    """Search PageIndex documents with the real API."""
    manifest = load_manifest()
    docs = manifest.get("documents", [])
    if not docs:
        docs = upload_documents()

    results: list[dict] = []
    for doc in docs:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        try:
            if not is_retrieval_ready(doc_id):
                continue
            doc_results = retrieve_from_doc(doc_id, query)
            for result in doc_results:
                result["metadata"]["source"] = doc.get("filename", doc_id)
            results.extend(doc_results)
        except Exception as exc:
            print(f"PageIndex retrieval skipped for {doc.get('filename', doc_id)}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def vectorless_score(query: str, content: str, metadata: dict) -> float:
    """Local vectorless fallback score based on token overlap and structure."""
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    content_counts = Counter(content_tokens)
    overlap = sum(min(count, content_counts.get(token, 0)) for token, count in query_counts.items())
    coverage = overlap / max(1, sum(query_counts.values()))

    query_text = " ".join(query_tokens)
    doc_type = str(metadata.get("type", ""))
    boost = 0.0
    if doc_type == "legal" and any(term in query_text for term in ["luat", "dieu", "nghi", "phat", "cai"]):
        boost += 0.08
    if doc_type == "news" and any(term in query_text for term in ["nghe", "si", "dien", "vien", "ca", "bat"]):
        boost += 0.08

    return float(min(1.0, coverage + boost))


def local_pageindex_fallback(query: str, top_k: int = 5) -> list[dict]:
    """Fallback retrieval over local chunks with PageIndex-compatible output."""
    chunks = load_chunks()
    scored = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        score = vectorless_score(query, chunk.get("content", ""), metadata)
        if score <= 0:
            continue
        scored.append(
            {
                "content": chunk.get("content", ""),
                "score": score,
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex API when possible.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'pageindex'}.
    """
    if top_k <= 0 or not query.strip():
        return []

    if PAGEINDEX_API_KEY:
        try:
            api_results = pageindex_api_search(query, top_k=top_k)
            if api_results:
                return api_results
        except Exception as exc:
            print(f"PageIndex API fallback activated: {exc}")

    return local_pageindex_fallback(query, top_k=top_k)


if __name__ == "__main__":
    docs = upload_documents()
    print(f"PageIndex manifest documents: {len(docs)}")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for result in results:
        preview = result["content"][:100].encode("ascii", "ignore").decode("ascii")
        print(f"[{result['score']:.3f}] {preview}...")
