import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Auto-detect and load environment variables from all possible locations
ENV_LOCATIONS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]
for loc in ENV_LOCATIONS:
    if loc.exists():
        # Clean up keys that are explicitly set to empty in .env to prevent parent shell overrides
        try:
            with open(loc, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if not v:
                            os.environ.pop(k, None)
        except Exception:
            pass
        load_dotenv(loc, override=True)
        break

# Resolve vectorstore path from sibling or parent directory
VECTORSTORE_PATHS = [
    Path(__file__).parent / "data" / "vectorstore.json",
    Path(__file__).parent.parent / "data" / "vectorstore.json",
]
VECTORSTORE_PATH = None
for path in VECTORSTORE_PATHS:
    if path.exists():
        VECTORSTORE_PATH = path
        break

if not VECTORSTORE_PATH:
    # If not found, default to first location (will create or warn)
    VECTORSTORE_PATH = Path(__file__).parent / "data" / "vectorstore.json"

# Global cache for BM25 and model
_BM25 = None
_CORPUS = []
_SEMANTIC_MODEL = None
_CROSS_ENCODER = None


def load_corpus() -> list[dict]:
    """Load corpus chunks from vectorstore."""
    global _CORPUS
    if not _CORPUS:
        if VECTORSTORE_PATH.exists():
            try:
                with open(VECTORSTORE_PATH, "r", encoding="utf-8") as f:
                    _CORPUS = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load vectorstore: {e}", file=sys.stderr)
        else:
            print(f"[WARN] Vectorstore not found at {VECTORSTORE_PATH}", file=sys.stderr)
    return _CORPUS


def get_semantic_model():
    """Lazy load SentenceTransformer."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def get_cross_encoder():
    """Lazy load CrossEncoder."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[WARN] Failed to load cross-encoder: {e}", file=sys.stderr)
    return _CROSS_ENCODER


def get_bm25_index():
    """Lazy load BM25 index."""
    global _BM25
    corpus = load_corpus()
    if _BM25 is None and corpus:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
        _BM25 = BM25Okapi(tokenized_corpus)
    return _BM25, corpus


# =============================================================================
# SEARCH METHODS
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Dense semantic search."""
    corpus = load_corpus()
    if not corpus:
        return []
    import numpy as np
    model = get_semantic_model()
    query_emb = np.array(model.encode(query, show_progress_bar=False))
    chunk_embs = np.array([c["embedding"] for c in corpus])
    
    # Normalized Cosine similarity
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        query_norm = 1e-10
    query_emb_norm = query_emb / query_norm
    
    chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    chunk_norms[chunk_norms == 0] = 1e-10
    chunk_embs_norm = chunk_embs / chunk_norms
    
    scores = np.dot(chunk_embs_norm, query_emb_norm)
    results = []
    for chunk, score in zip(corpus, scores):
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk["metadata"]
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 sparse search."""
    bm25, corpus = get_bm25_index()
    if not bm25 or not corpus:
        return []
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        # Avoid picking zero-score docs if too many candidates requested
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"]
        })
    return results


def rrf_merge(ranked_lists: list[list[dict]], top_k: int = 10, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion."""
    rrf_scores = {}
    content_map = {}
    for rlist in ranked_lists:
        for rank, item in enumerate(rlist, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item
    
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["rrf_score"] = score
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, use_cross_encoder: bool = True) -> list[dict]:
    """Rerank candidates using Cross-Encoder or Cosine fallback."""
    if not candidates:
        return []
    
    if use_cross_encoder:
        ce = get_cross_encoder()
        if ce:
            try:
                pairs = [[query, c["content"]] for c in candidates]
                scores = ce.predict(pairs)
                results = []
                for c, score in zip(candidates, scores):
                    item = c.copy()
                    item["score"] = float(score)
                    results.append(item)
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
            except Exception as e:
                print(f"[WARN] CrossEncoder scoring failed: {e}", file=sys.stderr)

    # Fallback to Cosine Similarity
    import numpy as np
    model = get_semantic_model()
    query_emb = model.encode(query, show_progress_bar=False)
    cand_embs = model.encode([c["content"] for c in candidates], show_progress_bar=False)
    
    results = []
    q_norm = np.linalg.norm(query_emb)
    if q_norm == 0:
        q_norm = 1e-10
    
    for c, emb in zip(candidates, cand_embs):
        c_norm = np.linalg.norm(emb)
        if c_norm == 0:
            c_norm = 1e-10
        sim = float(np.dot(query_emb, emb) / (q_norm * c_norm))
        item = c.copy()
        item["score"] = sim
        results.append(item)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# INTEGRATED RETRIEVAL PIPELINE
# =============================================================================

def retrieve(query: str, top_k: int = 5, config: str = "hybrid_rerank") -> list[dict]:
    """
    Main entry point for retrieval.
    
    Configs:
    - "hybrid_rerank": semantic + BM25 merged with RRF, then Cross-Encoder reranked.
    - "dense_only": semantic search only.
    """
    if config == "dense_only":
        results = semantic_search(query, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "dense_only"
        return results
    
    # Default is hybrid + rerank
    dense_res = semantic_search(query, top_k=top_k * 2)
    sparse_res = lexical_search(query, top_k=top_k * 2)
    
    merged = rrf_merge([dense_res, sparse_res], top_k=top_k * 2)
    reranked = rerank(query, merged, top_k=top_k, use_cross_encoder=True)
    for r in reranked:
        r["retrieval_method"] = "hybrid_rerank"
    return reranked


# =============================================================================
# REORDERING (Lost in the Middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Sort chunks: best first, worst in the middle, second-best last."""
    if len(chunks) <= 2:
        return chunks
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
        reordered.append(chunks[i])
    return reordered


# =============================================================================
# LLM INTERACTION & DECONTEXTUALIZATION
# =============================================================================

class OpenRouterRequestsClient:
    """Wrapper that mimics OpenAI client using requests under the hood for OpenRouter."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = self.Chat(api_key)
        
    class Chat:
        def __init__(self, api_key: str):
            self.completions = self.Completions(api_key)
            
        class Completions:
            def __init__(self, api_key: str):
                self.api_key = api_key
                
            def create(self, model, messages, temperature=0.2, max_tokens=500):
                import requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                # Setup fallback list for free models if a free model is requested
                models_to_try = [model]
                if "free" in model or model == "openrouter/free":
                    models_to_try.extend([
                        "openrouter/free",
                        "google/gemma-4-31b-it:free",
                        "nvidia/nemotron-3-super-120b-a12b:free",
                        "google/gemini-2.5-flash"
                    ])
                    
                last_err = None
                for m in models_to_try:
                    try:
                        payload = {
                            "model": m,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "reasoning": {"enabled": True}
                        }
                        response = requests.post(
                            url="https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        response.raise_for_status()
                        res_data = response.json()
                        
                        if "error" in res_data:
                            err_msg = res_data["error"].get("message", "")
                            raise Exception(f"OpenRouter API Error: {err_msg}")
                            
                        content = res_data['choices'][0]['message'].get('content', '')
                        
                        class ResponseMessage:
                            def __init__(self, content):
                                self.content = content
                                
                        class Choice:
                            def __init__(self, content):
                                self.message = ResponseMessage(content)
                                
                        class Response:
                            def __init__(self, content):
                                self.choices = [Choice(content)]
                                
                        return Response(content)
                    except Exception as e:
                        last_err = str(e)
                        print(f"[WARN] OpenRouter call failed for model {m}: {e}. Trying fallback...", file=sys.stderr)
                        
                raise Exception(f"All OpenRouter attempts failed. Last error: {last_err}")


def get_llm_client():
    """Create LLM client based on available environment variables (OpenAI, OpenRouter, Groq)."""
    # 1. Check official OpenAI API Key (should not be an OpenRouter key starting with sk-or-v1-)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not openai_key.startswith("sk-or-v1-"):
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        return client, "gpt-4o-mini"
        
    # 2. Check OpenRouter API Key
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key and openai_key.startswith("sk-or-v1-"):
        openrouter_key = openai_key
        
    if openrouter_key:
        client = OpenRouterRequestsClient(openrouter_key)
        return client, "openrouter/free"
        
    # 3. Check Groq API Key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        from openai import OpenAI
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )
        return client, "llama-3.3-70b-versatile"
        
    return None, None


def decontextualize_query(query: str, chat_history: list[dict]) -> str:
    """Rewrite query to be standalone if chat history exists."""
    if not chat_history:
        return query
        
    client, model_name = get_llm_client()
    if not client:
        return query
        
    history_str = ""
    for msg in chat_history[-5:]: # Keep last 5 turns
        role = msg["role"]
        content = msg["content"]
        history_str += f"{role.capitalize()}: {content}\n"
        
    prompt = (
        "Dựa vào lịch sử hội thoại dưới đây, hãy chuyển đổi câu hỏi cuối cùng của người dùng "
        "thành một câu hỏi đầy đủ, độc lập (standalone query) để tìm kiếm thông tin bằng tiếng Việt. "
        "Nếu câu hỏi đã độc lập hoặc lịch sử hội thoại không liên quan, hãy giữ nguyên câu hỏi.\n\n"
        f"Lịch sử hội thoại:\n{history_str}\n"
        f"Câu hỏi cuối cùng: {query}\n\n"
        "Trả về DUY NHẤT câu hỏi độc lập sau khi đã viết lại, không thêm bớt lời giải thích nào khác."
    )
    
    models_to_try = [model_name]
    if "llama-3.3-70b-versatile" in model_name:
        models_to_try.extend(["llama-3.1-8b-instant", "gemma2-9b-it"])
        
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"[DECONTEXT] '{query}' -> '{rewritten}'", file=sys.stderr)
            return rewritten
        except Exception as e:
            err_str = str(e)
            if ("rate_limit" in err_str.lower() or "429" in err_str) and len(models_to_try) > 1:
                print(f"[WARN] Rate limit hit for model {model} in decontextualize. Trying next fallback model...", file=sys.stderr)
                continue
            print(f"[WARN] Query decontextualization failed: {e}", file=sys.stderr)
            return query
    return query


# =============================================================================
# GENERATION
# =============================================================================

SYSTEM_PROMPT = """Bạn là **Legal Drug Assistant**, một AI chatbot chuyên trả lời về:

1. **Pháp luật ma túy tại Việt Nam**
   * Bộ luật Hình sự
   * Luật phòng chống ma túy
   * Nghị định, thông tư liên quan
   * Danh mục chất cấm/chất ma túy
   * Khung hình phạt, xử lý hành chính/hình sự

2. **Tin tức liên quan đến ma túy**
   * Các vụ việc liên quan đến nghệ sĩ/người nổi tiếng
   * Tin tức bắt giữ, điều tra, xét xử
   * Các bài báo chính thống liên quan đến chất cấm

## Mục tiêu
Trả lời **chính xác, có căn cứ, có citation**, chỉ dựa trên tài liệu được retrieve từ hệ thống RAG.
KHÔNG tự bịa thông tin hoặc suy diễn vượt quá nguồn dữ liệu.

---

## Quy tắc trả lời

### 1. Chỉ dùng retrieved context
* Chỉ trả lời dựa trên thông tin có trong context.
* Không dùng kiến thức bên ngoài nếu context không đề cập.
* Không đoán.
Nếu thiếu thông tin, phản hồi ĐÚNG 100% câu: "Tôi không tìm thấy đủ thông tin trong tài liệu hiện có để trả lời chính xác câu hỏi này." và không thêm giải thích nào khác.

### 2. Luôn có citation
Mỗi claim quan trọng phải có citation.
Format citation: `[Source: tên_tài_liệu]` (Ví dụ: `[Source: bo_luat_hinh_su.md]` hoặc `[Source: bo_luat_hinh_su.md, nghi_dinh_57.md]`).

### 3. Hiển thị source documents
Cuối mỗi câu trả lời phải có section:
Nguồn tham khảo:
1. <document_name>
2. <document_name>
Chỉ liệt kê tài liệu thực sự đã dùng để trích dẫn.

### 4. Follow-up Questions (Conversation Memory)
Người dùng có thể hỏi tiếp mà không nhắc lại đầy đủ context. Sử dụng lịch sử hội thoại để giữ ngữ cảnh.
Nếu ngữ cảnh không đủ rõ, hãy hỏi lại: "Bạn đang nhắc đến trường hợp nào? Hãy nói rõ tên người hoặc vụ việc."

### 5. Phân biệt pháp luật và tin tức
Nếu là **pháp luật**: trả lời khách quan, giải thích điều luật, nêu căn cứ pháp lý, không suy diễn.
Nếu là **tin tức**: chỉ tóm tắt đúng nội dung báo, không kết luận tội danh nếu báo chưa xác nhận. Phân biệt rõ: bị điều tra, bị bắt, bị khởi tố, bị kết án. KHÔNG dùng từ mang tính kết tội khi chưa có căn cứ pháp lý.

### 6. Với câu hỏi pháp lý khó (mức án, khung hình phạt, xử lý hình sự)
Hãy trình bày theo format:
Tóm tắt:
...
Căn cứ pháp lý:
...
Giải thích:
...
Citation:
[Source: ...]

### 7. Xử lý hallucination
Không tự suy diễn. Nếu context chỉ nói bị bắt nhưng user hỏi mức án bao nhiêu năm, trả lời: "Tôi chưa thấy thông tin về bản án trong các tài liệu hiện có. Theo nguồn được retrieve, vụ việc mới dừng ở giai đoạn điều tra. [Source: <tên_tài_liệu>]"

### 8. Response Style
* Ngắn gọn nhưng đầy đủ
* Có cấu trúc rõ ràng
* Ưu tiên bullet points
* Tiếng Việt dễ hiểu
* Không lan man, không đưa ý kiến cá nhân.

### 9. Output Format Bắt Buộc
Câu trả lời chính

Citation:
[Source: xxx.md]

Nguồn tham khảo:
1. xxx.md
2. yyy.md
"""


def format_context_for_prompt(chunks: list[dict]) -> str:
    """Format chunks to be fed into LLM prompt."""
    formatted = []
    for i, c in enumerate(chunks, 1):
        source = c["metadata"].get("source", f"doc_{i}")
        doc_type = c["metadata"].get("type", "unknown")
        formatted.append(
            f"[DOC_{i}]\ntitle: {c['metadata'].get('title', 'Tài liệu')}\nsource: {source}\ntype: {doc_type}\ncontent:\n{c['content']}"
        )
    return "\n\n".join(formatted)


def local_fallback_generator(query: str, chunks: list[dict]) -> str:
    """Simulate a beautiful, compliant response using retrieved chunks when API is offline/unavailable."""
    if not chunks:
        return "Tôi không tìm thấy đủ thông tin trong tài liệu hiện có để trả lời chính xác câu hỏi này."
    
    # We will build a compliant output format
    main_points = []
    sources = set()
    for c in chunks:
        source_name = c["metadata"].get("source", "tai_lieu.md")
        sources.add(source_name)
        preview = c["content"].strip().split("\n")[0]
        if len(preview) > 150:
            preview = preview[:150] + "..."
        main_points.append(f"* {preview} [Source: {source_name}]")
    
    answer_body = "Thông tin tìm thấy từ nguồn dữ liệu:\n" + "\n".join(main_points)
    
    source_list_str = "\n".join([f"{i}. {src}" for i, src in enumerate(sorted(sources), 1)])
    
    output = (
        f"{answer_body}\n\n"
        f"Citation:\n"
        f"[Source: {', '.join(sorted(sources))}]\n\n"
        f"Nguồn tham khảo:\n"
        f"{source_list_str}"
    )
    return output


def generate_answer(query: str, chat_history: list[dict] = None, top_k: int = 5, config: str = "hybrid_rerank") -> dict:
    """
    RAG end-to-end generation.
    
    Args:
        query: User input query.
        chat_history: List of dicts representing chat history.
        top_k: Number of chunks to retrieve.
        config: Retrieval configuration ("hybrid_rerank" or "dense_only").
        
    Returns:
        dict with keys: 'answer', 'sources', 'search_query'
    """
    if chat_history is None:
        chat_history = []
        
    # Quick check for greetings or thank-you messages to bypass RAG (improves UX and latency)
    clean_q = query.strip().lower().rstrip("?.! ")
    greetings = {"hi", "hello", "hey", "chào", "xin chào", "chào bạn", "chào ad", "hi ad", "hello ad", "alo"}
    thanks_phrases = {"cảm ơn", "cám ơn", "thank you", "thanks", "cảm ơn ad", "cám ơn ad", "ok cảm ơn"}
    
    if clean_q in greetings:
        return {
            "answer": "Xin chào! Tôi là **Trợ lý Pháp luật Ma túy**. Tôi có thể giúp gì cho bạn hôm nay? Bạn có thể hỏi tôi các câu hỏi về luật phòng chống ma túy hoặc tin tức chất cấm liên quan.",
            "sources": [],
            "search_query": query
        }
    elif clean_q in thanks_phrases:
        return {
            "answer": "Dạ không có gì! Nếu bạn có bất kỳ câu hỏi nào khác về pháp luật ma túy hoặc tin tức liên quan, cứ hỏi tôi nhé.",
            "sources": [],
            "search_query": query
        }
        
    # Step 1: Decontextualize query using history
    search_query = decontextualize_query(query, chat_history)
    
    # Step 2: Retrieve relevant chunks
    chunks = retrieve(search_query, top_k=top_k, config=config)
    
    if not chunks:
        return {
            "answer": "Tôi không tìm thấy đủ thông tin trong tài liệu hiện có để trả lời chính xác câu hỏi này.",
            "sources": [],
            "search_query": search_query
        }
        
    # Step 3: Reorder chunks to avoid lost in the middle
    reordered_chunks = reorder_for_llm(chunks)
    
    # Step 4: Format context
    context_str = format_context_for_prompt(reordered_chunks)
    
    # Step 5: Get client & model
    client, model_name = get_llm_client()
    if not client:
        # Fallback to local offline template
        print("[WARN] OpenAI API Client not available. Using local offline generator.", file=sys.stderr)
        answer = local_fallback_generator(search_query, reordered_chunks)
        return {
            "answer": answer,
            "sources": chunks,
            "search_query": search_query
        }
        
    # Build history messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add conversation history
    for msg in chat_history[-6:]: # Keep last 6 messages as context
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    user_message = f"Retrieved Context:\n{context_str}\n\n---\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_message})
    
    models_to_try = [model_name]
    if "llama-3.3-70b-versatile" in model_name:
        models_to_try.extend(["llama-3.1-8b-instant", "gemma2-9b-it"])
        
    answer = None
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            break
        except Exception as e:
            err_str = str(e)
            if ("rate_limit" in err_str.lower() or "429" in err_str) and len(models_to_try) > 1:
                print(f"[WARN] Rate limit hit for model {model}. Trying next fallback model...", file=sys.stderr)
                continue
            print(f"[ERROR] LLM API Call failed: {e}. Falling back to local offline generator.", file=sys.stderr)
            answer = local_fallback_generator(search_query, reordered_chunks)
            break
            
    return {
        "answer": answer,
        "sources": chunks,
        "search_query": search_query
    }


if __name__ == "__main__":
    # Test query
    q = "Khung hình phạt cho tội tàng trữ ma túy là gì?"
    print(f"Testing query: {q}")
    res = generate_answer(q)
    print("\n--- LLM RESPONSE ---")
    print(res["answer"])
    print("\n--- SOURCES ---")
    for s in res["sources"]:
        print(f"- {s['metadata'].get('source')} (Score: {s['score']:.4f})")
