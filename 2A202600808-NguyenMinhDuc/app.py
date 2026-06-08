"""Streamlit UI for the Day 08 RAG chatbot."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.task10_generation import generate_with_citation


SAMPLE_QUESTIONS = [
    "Luật Phòng, chống ma túy 2021 quy định những biện pháp cai nghiện nào?",
    "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam là gì?",
    "Nghệ sĩ nào từng bị bắt hoặc bị xử lý vì liên quan tới ma túy?",
    "Cai nghiện ma túy tự nguyện và bắt buộc khác nhau như thế nào?",
    "Các nguồn tin trong dữ liệu nói gì về những vụ việc nghệ sĩ liên quan ma túy?",
]


st.set_page_config(
    page_title="RAG Chatbot - Pháp luật ma túy",
    page_icon="⚖️",
    layout="wide",
)


def render_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        div[data-testid="stVerticalBlock"] > div:has(.source-card) {
            gap: 0.6rem;
        }
        .app-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .source-card {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 0.75rem;
            background: #ffffff;
        }
        .source-meta {
            color: #4b5563;
            font-size: 0.82rem;
            margin-bottom: 0.4rem;
        }
        .source-content {
            font-size: 0.9rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "question" not in st.session_state:
        st.session_state.question = SAMPLE_QUESTIONS[0]
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


def set_question(question: str) -> None:
    st.session_state.question = question


def source_name(source: dict[str, Any], index: int) -> str:
    metadata = source.get("metadata", {}) or {}
    return (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("title")
        or metadata.get("doc_id")
        or f"Source {index}"
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("Chưa có nguồn nào được truy xuất.")
        return

    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {}) or {}
        label = html.escape(str(source_name(source, index)))
        score = float(source.get("score", 0.0))
        retrieval_source = html.escape(str(source.get("source", "unknown")))
        doc_type = html.escape(str(metadata.get("type", "unknown")))
        content = html.escape(str(source.get("content", ""))[:900])

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-meta">
                    <strong>{index}. {label}</strong>
                    &nbsp;|&nbsp; score: {score:.3f}
                    &nbsp;|&nbsp; retrieval: {retrieval_source}
                    &nbsp;|&nbsp; type: {doc_type}
                </div>
                <div class="source-content">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_query(question: str, top_k: int) -> None:
    with st.spinner("Đang truy xuất tài liệu và sinh câu trả lời..."):
        st.session_state.last_result = generate_with_citation(question, top_k=top_k)
        st.session_state.last_question = question


render_css()
init_state()

st.markdown('<div class="app-title">RAG Chatbot - Pháp luật và tin tức về ma túy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Hỏi đáp dựa trên dữ liệu pháp luật và bài báo đã crawl. Câu trả lời hiển thị kèm nguồn truy xuất.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.62, 0.38], gap="large")

with left_col:
    question = st.text_area(
        "Câu hỏi",
        key="question",
        height=105,
        placeholder="Nhập câu hỏi về pháp luật ma túy hoặc các bài báo trong dữ liệu...",
    )

    controls_col, action_col = st.columns([0.35, 0.65])
    with controls_col:
        top_k = st.slider("Số nguồn", min_value=3, max_value=8, value=5, step=1)
    with action_col:
        st.write("")
        st.write("")
        ask_clicked = st.button("Hỏi bot", type="primary", use_container_width=True)

    st.caption("Câu hỏi mẫu")
    sample_cols = st.columns(2)
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        with sample_cols[idx % 2]:
            st.button(sample, key=f"sample_{idx}", on_click=set_question, args=(sample,), use_container_width=True)

    if ask_clicked:
        run_query(question.strip(), top_k)

    result = st.session_state.last_result
    if result:
        st.divider()
        st.subheader("Câu trả lời")
        st.write(result.get("answer", ""))
        st.caption(
            f"Retrieval source: {result.get('retrieval_source', 'unknown')} | "
            f"Số nguồn: {len(result.get('sources', []))}"
        )

with right_col:
    st.subheader("Nguồn truy xuất")
    if st.session_state.last_result:
        render_sources(st.session_state.last_result.get("sources", []))
    else:
        st.info("Chọn câu hỏi mẫu hoặc nhập câu hỏi rồi bấm Hỏi bot.")
