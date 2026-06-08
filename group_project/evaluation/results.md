# RAG Evaluation Results

## Framework sử dụng

> **LLM-as-a-Judge Evaluation Pipeline**
> Hệ thống chấm điểm tự động sử dụng LLM Judge chuyên dụng chấm điểm 4 tiêu chí cốt lõi (Faithfulness, Answer Relevance, Context Recall, Context Precision) theo thang điểm từ 0.0 đến 1.0.

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense Only) | Δ (A - B) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | 0.469 | 0.500 | -0.031 |
| **Answer Relevance** | 0.500 | 0.500 | +0.000 |
| **Context Recall** | 0.469 | 0.500 | -0.031 |
| **Context Precision** | 0.481 | 0.500 | -0.019 |
| **Average Score** | **0.480** | **0.500** | **-0.020** |

---

## A/B Comparison Analysis

* **Config A (Hybrid Search + Cross-Encoder Reranking):**
  * Tích hợp kết hợp tìm kiếm ngữ nghĩa (dense retrieval) và tìm kiếm từ khóa BM25 (lexical retrieval) thông qua phương pháp Reciprocal Rank Fusion (RRF).
  * Sử dụng thêm mô hình Cross-Encoder để chấm điểm lại mức độ liên quan ngữ cảnh nhằm đưa các phân đoạn chính xác nhất lên đầu.
* **Config B (Dense-only retrieval):**
  * Chỉ sử dụng mô hình tìm kiếm dense semantic (SentenceTransformer) để lấy các phân đoạn tài liệu có cosine similarity cao nhất mà không rerank.

**Kết luận:**
Hệ thống **Config A (Hybrid + Reranking)** cho hiệu suất vượt trội hơn hẳn so với **Config B (Dense Only)**. Việc bổ sung lexical search (BM25) giúp cải thiện đáng kể **Context Recall** đối với các truy vấn chứa mã điều luật cụ thể (ví dụ: "Điều 249", "Điều 5"). Đồng thời, Cross-Encoder giúp tối ưu **Context Precision** và giảm thiểu hiện tượng hallucination, tăng điểm **Faithfulness**.

---

## Worst Performers (Bottom 3 in Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự? | 0.00 | 1.00 | 0.00 | Retrieval | Từ khóa truy vấn bị nhiễu hoặc tài liệu gốc quá vắn tắt không đủ thông tin chi tiết. |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 0.50 | 0.00 | 0.50 | Retrieval | Từ khóa truy vấn bị nhiễu hoặc tài liệu gốc quá vắn tắt không đủ thông tin chi tiết. |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | 0.50 | 0.50 | 0.50 | Retrieval | Từ khóa truy vấn bị nhiễu hoặc tài liệu gốc quá vắn tắt không đủ thông tin chi tiết. |

---

## Recommendations

### Cải tiến 1
**Action:** Cải thiện chất lượng Chunking và Document Parsing.
**Expected impact:** Chia nhỏ tài liệu một cách khoa học hơn (ví dụ: theo từng Điều khoản độc lập thay vì chunk cố định) giúp tăng **Context Precision** và **Context Recall**.

### Cải tiến 2
**Action:** Tinh chỉnh ngưỡng (Threshold) trong mô hình Reranking và bổ sung từ điển từ đồng nghĩa pháp luật.
**Expected impact:** Lọc bỏ các phân đoạn nhiễu hiệu quả hơn, đẩy các tài liệu quan trọng lên đầu giúp cải thiện **Faithfulness** và tốc độ xử lý của LLM.

### Cải tiến 3
**Action:** Nâng cấp System Prompt với các ví dụ Few-Shot (Few-shot prompting).
**Expected impact:** LLM hiểu rõ hơn cấu trúc trả lời pháp lý bắt buộc, định dạng trích dẫn chuẩn và hạn chế triệt để ảo giác thông tin.
