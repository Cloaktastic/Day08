"""
Task 10 - Generation with citations.

This module retrieves context with Task 9, reorders chunks to reduce
"lost in the middle", formats source-labelled context, and generates an answer.
It calls OpenAI when OPENAI_API_KEY is present, with an extractive fallback so
tests and demos still run if the API is unavailable.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()


# top_k=5 gives enough evidence while keeping context short and focused.
TOP_K = 5

# top_p=0.9 allows some natural phrasing while avoiding very broad sampling.
TOP_P = 0.9

# temperature=0.3 keeps RAG factual and conservative.
TEMPERATURE = 0.3

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source, for example [73-2021-qh14-445185.md].

If the information is not explicitly stated in the provided context, state
'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than guessing.

Rules:
- Only use information from the provided context.
- Every factual claim must have a citation.
- Keep the answer concise, clear, and grounded in the sources."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Input sorted by score: [1, 2, 3, 4, 5]
    Output pattern:        [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    return front + list(reversed(back))


def source_label(chunk: dict, fallback: str = "unknown") -> str:
    """Extract a citation-friendly source label from chunk metadata."""
    metadata = chunk.get("metadata", {}) or {}
    return (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("title")
        or metadata.get("doc_id")
        or fallback
    )


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into source-labelled context for the prompt.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = source_label(chunk, fallback=f"Source {index}")
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        content = chunk.get("content", "").strip()
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{content}\n"
        )

    return "\n---\n".join(context_parts)


def build_user_message(query: str, context: str) -> str:
    """Build the user prompt sent to the LLM."""
    return f"""Context:
{context}

---

Question: {query}
"""


def call_openai(query: str, context: str) -> str | None:
    """Call OpenAI if an API key is configured; return None on failure."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(query, context)},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or None
    except Exception as exc:
        print(f"OpenAI generation fallback activated: {exc}")
        return None


def split_sentences(text: str) -> list[str]:
    """Small sentence splitter for fallback extractive answers."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?。])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extractive_answer(query: str, chunks: list[dict]) -> str:
    """
    Fallback answer that quotes the most relevant retrieved snippets with
    citations. It avoids unsupported claims when LLM generation is unavailable.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = []
    for chunk in chunks[:3]:
        source = source_label(chunk)
        sentences = split_sentences(chunk.get("content", ""))
        snippet = sentences[0] if sentences else chunk.get("content", "")[:300].strip()
        if not snippet:
            continue
        answer_parts.append(f"{snippet} [{source}]")

    if not answer_parts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return " ".join(answer_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = call_openai(query, context)
    if not answer:
        answer = extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
        "Nhung nghe si nao da bi bat vi lien quan toi ma tuy?",
        "Quy trinh cai nghien bat buoc theo Luat Phong chong ma tuy 2021?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print("=" * 70)
        result = generate_with_citation(query)
        safe_answer = result["answer"].encode("ascii", "ignore").decode("ascii")
        print(f"\nA: {safe_answer}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
