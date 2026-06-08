import streamlit as st
import time
import os
from pathlib import Path
from rag_backend import generate_answer, load_corpus, VECTORSTORE_PATH

# ---------------------------------------------------------
# Page Configurations & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Trợ Lý Pháp Luật Ma Túy - RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for glassmorphic cards, custom chat bubbles, and clean fonts
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* General Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Sidebar Card Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f1f5f9;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Title Styling */
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #3b82f6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Custom User Chat Bubble */
    .user-bubble-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff;
        padding: 0.8rem 1.2rem;
        border-radius: 20px 20px 0px 20px;
        max-width: 75%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Custom Assistant Chat Bubble */
    .assistant-bubble-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.2rem;
    }
    
    .assistant-bubble {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        padding: 1rem 1.4rem;
        border-radius: 20px 20px 20px 0px;
        max-width: 80%;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Glassmorphic Info Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(5px);
        margin-bottom: 1rem;
    }
    
    .glass-header {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    .glass-body {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .stat-val {
        font-size: 1.2rem;
        font-weight: 600;
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Panel (Configurations & Statistics)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;font-weight:600;font-size:1.6rem;margin-top:0;'>⚙️ Bảng Điều Khiển</h2>", unsafe_allow_html=True)
    st.write("Cấu hình tham số mô hình RAG.")
    st.write("")
    
    # Selection of RAG Config for A/B testing
    rag_config = st.selectbox(
        "Cấu hình RAG (Thử nghiệm A/B)",
        options=["hybrid_rerank", "dense_only"],
        format_func=lambda x: "Cấu hình A: Tìm kiếm Hybrid + Rerank" if x == "hybrid_rerank" else "Cấu hình B: Tìm kiếm Dense Semantic",
        help="Chọn cấu hình để thử nghiệm hệ thống RAG."
    )
    
    # Top K parameter
    top_k = st.slider(
        "Số lượng Chunks Truy Vấn (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Số lượng phân đoạn văn bản truyền vào prompt LLM."
    )
    
    st.write("---")
    st.markdown("<h3 style='color:#38bdf8;font-size:1.1rem;font-weight:600;'>📊 Trạng Thái Hệ Thống</h3>", unsafe_allow_html=True)
    
    # Load corpus details to show live stats
    corpus = load_corpus()
    total_chunks = len(corpus)
    
    st.markdown(f"""
        <div class="glass-card">
            <div class="glass-header">📍 Thống kê cơ sở dữ liệu</div>
            <div class="glass-body">
                <span class="stat-label">Kho lưu trữ vector:</span> <span class="stat-val">{total_chunks} phân đoạn (chunks)</span><br/>
                <span class="stat-label">Trạng thái:</span> <span style="color:#22c55e;font-weight:600;">Đang hoạt động</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display environment keys status
    openai_key_set = "Đã cấu hình" if (os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-or-v1-")) else "Chưa cấu hình"
    openrouter_key_set = "Đã cấu hình" if (os.getenv("OPENROUTER_API_KEY") or (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").startswith("sk-or-v1-"))) else "Chưa cấu hình"
    groq_key_set = "Đã cấu hình" if os.getenv("GROQ_API_KEY") else "Chưa cấu hình"
    
    openai_color = "#22c55e" if openai_key_set == "Đã cấu hình" else "#ef4444"
    openrouter_color = "#22c55e" if openrouter_key_set == "Đã cấu hình" else "#ef4444"
    groq_color = "#22c55e" if groq_key_set == "Đã cấu hình" else "#ef4444"
    
    st.markdown(f"""
        <div class="glass-card">
            <div class="glass-header">🔑 API Keys & Dịch vụ</div>
            <div class="glass-body">
                <span class="stat-label">OpenAI API:</span> <span style="color:{openai_color};font-weight:600;">{openai_key_set}</span><br/>
                <span class="stat-label">OpenRouter:</span> <span style="color:{openrouter_color};font-weight:600;">{openrouter_key_set}</span><br/>
                <span class="stat-label">Groq API:</span> <span style="color:{groq_color};font-weight:600;">{groq_key_set}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["sources_history"] = []
        st.rerun()


# ---------------------------------------------------------
# Main App Chat Interface
# ---------------------------------------------------------
st.markdown("<div class='app-title'>⚖️ Trợ Lý Pháp Luật Ma Túy</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Hệ thống RAG thông minh tư vấn pháp luật ma túy & tin tức chất cấm Việt Nam</div>", unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sources_history" not in st.session_state:
    st.session_state["sources_history"] = []

# Display Messages from Session State
for idx, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-bubble-container">
                <div class="user-bubble">
                    {msg["content"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-bubble-container">
                <div class="assistant-bubble">
                    {msg["content"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display Sources Expander if sources exist for this message
        if idx < len(st.session_state["sources_history"]) and st.session_state["sources_history"][idx]:
            with st.expander("🔍 Tài liệu nguồn chi tiết", expanded=False):
                sources = st.session_state["sources_history"][idx]
                cols = st.columns(min(len(sources), 3))
                for c_idx, s in enumerate(sources):
                    col = cols[c_idx % len(cols)]
                    with col:
                        st.markdown(f"""
                            <div class="glass-card">
                                <div style="font-weight:600;font-size:0.85rem;color:#38bdf8;">[{c_idx+1}] {s['metadata'].get('source', 'Unknown')}</div>
                                <div style="font-size:0.75rem;color:#64748b;">Độ tương đồng: {s.get('score', 0.0):.4f} | Loại: {s['metadata'].get('type', 'unknown')}</div>
                                <hr style="margin: 0.3rem 0; border-color: rgba(255,255,255,0.05);"/>
                                <div style="font-size:0.75rem;color:#cbd5e1;max-height:120px;overflow-y:auto;">
                                    {s['content']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# User Query Input
# ---------------------------------------------------------
query = st.chat_input("Hãy hỏi tôi về Luật ma túy hoặc tin tức người nổi tiếng...")

if query:
    # Display user's question immediately
    st.markdown(f"""
        <div class="user-bubble-container">
            <div class="user-bubble">
                {query}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Store user message
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["sources_history"].append(None) # Place holder for alignment
    
    # Process Assistant response
    with st.spinner("Đang tìm kiếm thông tin và phân tích dữ liệu..."):
        t0 = time.time()
        
        # Call RAG backend with conversation history
        # Prepare history for the backend
        chat_hist = []
        for m in st.session_state["messages"][:-1]: # exclude the current question
            chat_hist.append({"role": m["role"], "content": m["content"]})
            
        result = generate_answer(
            query=query,
            chat_history=chat_hist,
            top_k=top_k,
            config=rag_config
        )
        
        latency = time.time() - t0
        
    # Display assistant bubble
    st.markdown(f"""
        <div class="assistant-bubble-container">
            <div class="assistant-bubble">
                {result["answer"]}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Store assistant message & retrieved sources
    st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})
    st.session_state["sources_history"].append(result["sources"])
    
    # Show Search Stats in a small glass card
    st.markdown(f"""
        <div style="display:flex;gap:1.5rem;font-size:0.75rem;color:#64748b;margin-left: 1rem; margin-top:-0.5rem; margin-bottom:1.5rem;">
            <span>⏱️ Phản hồi: <b>{latency:.2f}s</b></span>
            <span>🔍 Standalone Query: <b>"{result['search_query']}"</b></span>
            <span>📂 Tổng nguồn: <b>{len(result['sources'])} phân đoạn</b></span>
        </div>
    """, unsafe_allow_html=True)
    
    st.rerun()
