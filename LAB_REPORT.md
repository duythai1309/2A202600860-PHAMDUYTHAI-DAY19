# BÁO CÁO LAB DAY 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS

- **Học viên**: PHẠM DUY THAI
- **Mã lớp**: Day 19 Lab
- **Công cụ sử dụng**: Python, Jupyter Notebook, NetworkX, Matplotlib, Gemini LLM (Google GenAI)

---

## 1. PHẦN 1: NGHIÊN CỨU VÀ CHUẨN BỊ (RESEARCH)

### 1.1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
- **Thực thể (Node/Entity)**: Đại diện cho các đối tượng độc lập, có định danh duy nhất trong ngữ cảnh thế giới thực (ví dụ: công ty như `OpenAI`, `Google`; con người như `Sam Altman`, `Elon Musk`; sản phẩm như `ChatGPT`, `Gemini`).
- **Thuộc tính (Property/Attribute)**: Là các đặc trưng, thông tin bổ sung mô tả cho một thực thể mà không đại diện cho một thực thể độc lập (ví dụ: `Founded Year: 2015`, `Version: 2.0`, `Revenue: 13B`).
- **Cơ chế phân biệt của LLM**: LLM dựa vào ngữ nghĩa của câu, cấu trúc ngữ pháp (chủ ngữ, vị ngữ, bổ ngữ) và các hướng dẫn rõ ràng trong System Prompt để phân định. System Prompt cung cấp danh sách định nghĩa và ví dụ về các loại Nodes cần trích xuất (ví dụ: Tên công ty, Con người, Sản phẩm) và hướng dẫn LLM chuyển hóa các mối liên kết có tính mô tả thành quan hệ có hướng thay vì thuộc tính tĩnh (ví dụ: Thay vì đặt `Founded Year` làm thuộc tính của node `OpenAI`, ta trích xuất nó thành bộ ba quan hệ `(OpenAI, FOUNDED_IN, 2015)` để dễ dàng liên kết đồ thị).

### 1.2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
- Trong văn bản tự nhiên, một thực thể thường được viết dưới nhiều dạng khác nhau do ngữ cảnh hoặc cách viết tắt (ví dụ: `OpenAI Inc.`, `OpenAI`, `công ty OpenAI`, hay `Alphabet Inc.`, `Alphabet`, `công ty mẹ của Google`).
- **Hậu quả của việc thiếu khử trùng lặp**: Nếu không khử trùng lặp, hệ thống sẽ tạo ra nhiều Node riêng biệt trên đồ thị cho cùng một thực thể thực tế. Điều này làm **đứt gãy các cạnh (Edges)** kết nối, khiến các thuật toán duyệt đồ thị (như BFS hay DFS) không thể đi qua để liên kết thông tin đa bước.
- **Vai trò của Deduplication**: Giúp gộp tất cả các biến thể về một Node đại diện duy nhất (Canonical Node). Khi đó, các mối quan hệ trích xuất từ các câu khác nhau sẽ cùng đổ dồn về một Node này, giúp đồ thị liên thông và cho phép thực hiện truy vấn đa bước (Multi-hop Querying) chính xác.

### 1.3. Query Answering: Sự khác biệt giữa duyệt đồ thị theo chiều rộng (BFS) và tìm kiếm vector thông thường là gì?
- **Tìm kiếm Vector thông thường (Flat RAG)**:
  - So sánh embedding của câu hỏi với embedding của từng chunk văn bản độc lập dựa trên độ tương đồng cosine.
  - Phù hợp với câu hỏi tra cứu thông tin cục bộ nằm gọn trong một đoạn văn (Single-hop).
  - Điểm yếu: Trả về các đoạn văn bản rời rạc, không biết kết nối các thông tin gián tiếp nằm ở các tài liệu khác nhau. Dễ dẫn đến việc LLM bỏ sót liên kết hoặc tự ý bịa đặt (ảo giác - hallucination) khi cố gắng trả lời các câu hỏi phức tạp.
- **Duyệt đồ thị (BFS Traversal)**:
  - Bước đầu tiên là tìm Node khởi đầu (Seed node) tương ứng với thực thể được nhắc đến trong câu hỏi.
  - Từ Node khởi đầu, hệ thống thực hiện thuật toán duyệt đồ thị theo chiều rộng (BFS) để đi qua các cạnh kề nhằm lấy các Node liên quan trong phạm vi 1-hop, 2-hop.
  - Phù hợp với các câu hỏi phức tạp cần kết nối nhiều mắt xích (Multi-hop).
  - Điểm mạnh: Lần theo các quan hệ rõ ràng có hướng trên đồ thị (ví dụ: `Mark Zuckerberg` -> `FOUNDED` -> `Meta` -> `DEVELOPED` -> `LLaMA` -> `RIVAL_OF` -> `ChatGPT`). Kết quả thu thập được là một ngữ cảnh cấu trúc logic, giúp LLM suy luận chính xác 100% dựa trên các đường đi thực tế trên đồ thị.

---

## 2. PHẦN 2: THIẾT LẬP MÔI TRƯỜNG VÀ TRIỂN KHAI PIPELINE

Bài lab được thực hiện đầy đủ trên Jupyter Notebook [lab19_graphrag.ipynb](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/lab19_graphrag.ipynb) với các bước:
1. **Environment Setup**: Cài đặt các thư viện `networkx`, `matplotlib`, `openai`, `google-genai`, `pandas`. Tích hợp khả năng tự động fallback sang Mock LLM khi không cấu hình API Key để đảm bảo notebook chạy thông suốt.
2. **Indexing**: Đọc "Tech Company Corpus" từ file [tech_company_corpus.txt](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/data/tech_company_corpus.txt). Dùng Gemini để chuyển đổi các câu văn thành các JSON Triples.
3. **Deduplication**: Xây dựng hàm `normalize_entity()` để chuẩn hóa các tên gọi trùng lặp (ví dụ: `twitter` -> `Twitter`, `alphabet` -> `Alphabet Inc.`, `openai inc.` -> `OpenAI`).
4. **Graph Construction**: Tạo đồ thị có hướng `DiGraph` trong NetworkX, lưu trữ thuộc tính `predicate` của cạnh và vẽ đồ thị trực quan bằng Matplotlib (lưu ảnh đồ thị tại [tech_company_graph.png](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/tech_company_graph.png)).
5. **Flat RAG Baseline**: Cài đặt tìm kiếm từ khóa trên tập câu thô và tạo prompt trả lời.
6. **GraphRAG Querying**: Thực thi trích xuất thực thể từ câu hỏi -> Duyệt đồ thị 2-hop bằng thuật toán BFS -> Chuyển đổi bộ ba kết quả thành văn bản tự nhiên (Textualization) -> Tạo prompt suy luận cho LLM.

---

## 3. PHẦN 3: SO SÁNH VÀ ĐÁNH GIÁ (EVALUATION)

Dưới đây là bảng so sánh chi tiết kết quả trả lời 5 câu hỏi phức tạp giữa hai hệ thống được kết xuất từ file [rag_comparison_results.csv](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/rag_comparison_results.csv):

| Câu hỏi (Question) | Kết quả của Flat RAG (Flat RAG Answer) | Kết quả của GraphRAG (GraphRAG Answer) | Đánh giá so sánh |
| :--- | :--- | :--- | :--- |
| **Câu 1**: Ai là người sáng lập ra công ty con của Alphabet Inc.? | Dựa trên ngữ cảnh: Google là công ty con của Alphabet Inc., được thành lập vào năm 2015. Không tìm thấy thông tin ai là người thành lập Alphabet Inc. | Google là công ty con của Alphabet Inc. Google được thành lập bởi Larry Page và Sergey Brin. Vậy người sáng lập công ty con của Alphabet Inc. là Larry Page và Sergey Brin. | **GraphRAG thắng**: Flat RAG lấy được câu Alphabet Inc. nhưng không liên kết được sang câu sáng lập Google nên trả lời không biết. GraphRAG đi từ `Alphabet Inc.` -> `Google` -> `Larry Page / Sergey Brin` thành công. |
| **Câu 2**: Ai là người thành lập công ty đã đầu tư vào OpenAI? | Không tìm thấy thông tin đủ để trả lời câu hỏi trong ngữ cảnh. | Microsoft đầu tư vào OpenAI. Microsoft được sáng lập bởi Bill Gates và Paul Allen. Vậy người sáng lập công ty đầu tư vào OpenAI là Bill Gates và Paul Allen. | **GraphRAG thắng**: Flat RAG không liên kết được việc Microsoft đầu tư và ai sáng lập Microsoft. GraphRAG duyệt đồ thị nối liền hai thông tin này một cách chính xác. |
| **Câu 3**: Mô hình AI của công ty do Mark Zuckerberg thành lập có đối thủ cạnh tranh trực tiếp là gì? | Ngữ cảnh chỉ đề cập đến LLaMA cạnh tranh với ChatGPT. Tôi không biết đối thủ trực tiếp của mô hình AI do công ty Mark Zuckerberg thành lập là gì. | Mark Zuckerberg sáng lập Meta. Meta phát triển LLaMA. LLaMA là đối thủ cạnh tranh của ChatGPT. Vậy đối thủ cạnh tranh là ChatGPT. | **GraphRAG thắng**: Đây là câu hỏi 3-hop (`Zuckerberg` -> `Meta` -> `LLaMA` -> `ChatGPT`). Flat RAG bị đứt chuỗi thông tin. GraphRAG đi theo các cạnh đồ thị dễ dàng tìm ra ChatGPT. |
| **Câu 4**: Công ty nào được thành lập vào năm 2015 có mối liên hệ với Elon Musk? | OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015. Alphabet Inc. cũng được thành lập vào năm 2015 nhưng không có thông tin Alphabet Inc. liên quan đến Elon Musk. | OpenAI được thành lập vào năm 2015 và được sáng lập bởi Elon Musk (cùng với Sam Altman). Đồ thị tri thức cho thấy Alphabet Inc. cũng thành lập năm 2015 nhưng không liên quan đến Elon Musk. | **GraphRAG thắng**: Flat RAG đưa thêm thông tin nhiễu về Alphabet Inc. mà không khẳng định rõ ràng mối liên kết chính xác. GraphRAG trả lời gãy gọn, chính xác OpenAI. |
| **Câu 5**: Có mối liên hệ nào giữa người sáng lập Siri và Sam Altman không? | Dựa trên ngữ cảnh cung cấp, không có thông tin nào liên kết giữa người sáng lập Siri và Sam Altman. | Siri được mua lại bởi Apple (sáng lập bởi Steve Jobs và Steve Wozniak). OpenAI được sáng lập bởi Sam Altman và Elon Musk. Đồ thị tri thức xác nhận không có mối liên hệ nào giữa người sáng lập Siri và Sam Altman. | **Cả hai đều đúng**: Tuy nhiên GraphRAG đưa ra lập luận chặt chẽ hơn nhờ chỉ ra rõ mối liên hệ gián tiếp (`Siri` -> `Apple` -> `Steve Jobs/Steve Wozniak` và `OpenAI` -> `Sam Altman`). |

### Phân tích nguyên nhân:
- **Flat RAG** chỉ tìm các câu chứa từ khóa tương tự câu hỏi. Trong các câu hỏi phức tạp (chứa mối quan hệ bắc cầu), các câu chứa thông tin bắc cầu không có điểm tương đồng cao về từ khóa với câu hỏi ban đầu nên không được xếp hạng cao và bị loại bỏ khỏi ngữ cảnh.
- **GraphRAG** sử dụng cấu trúc đồ thị để tìm điểm nối. Chỉ cần câu hỏi trích xuất được một thực thể (ví dụ: `Mark Zuckerberg`), thuật toán BFS sẽ tự động kéo theo các quan hệ của node này trong phạm vi 2 bước, thu thập đầy đủ toàn bộ chuỗi thông tin cần thiết trước khi gửi cho LLM.

---

## 4. PHẦN 4: ĐỀ XUẤT CÔNG CỤ (RECOMMENDATIONS)

Dựa trên kinh nghiệm thực tế từ bài lab, dưới đây là đề xuất lựa chọn công cụ quản lý đồ thị cho dự án thực tế:

1. **NodeRAG**:
   - *Phù hợp*: Các dự án RAG cần triển khai nhanh, tích hợp sẵn logic tìm kiếm đồ thị và tối ưu hóa prompt mà không cần cấu hình hạ tầng database phức tạp.
   - *Ưu điểm*: Đóng gói all-in-one, dễ sử dụng cho các ứng dụng cơ bản.
2. **Neo4j**:
   - *Phù hợp*: Các hệ thống lớn của doanh nghiệp, dữ liệu quan hệ phức tạp cần trực quan hóa tốt (thông qua Neo4j Bloom hoặc Browser), và cần truy vấn đồ thị thời gian thực bằng ngôn ngữ Cypher.
   - *Ưu điểm*: Chuẩn công nghiệp, hiệu năng cao, giao diện đồ họa xuất sắc.
3. **NetworkX**:
   - *Phù hợp*: Nghiên cứu học thuật, thử nghiệm thuật toán đồ thị (centrality, community detection, BFS/DFS custom), chạy offline trong môi trường notebook hoặc khi dữ liệu đồ thị nhỏ gọn nằm trên RAM.
   - *Ưu điểm*: Cực kỳ nhẹ, không cần cài đặt database server, linh hoạt tối đa.
