# BÁO CÁO LAB DAY 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI US ELECTRIC VEHICLE DATASET

- **Học viên**: PHẠM DUY THAI
- **Mã lớp**: Day 19 Lab
- **Công cụ sử dụng**: Python, Jupyter Notebook, NetworkX, Matplotlib, Gemini LLM (Google GenAI)
- **Tập dữ liệu**: US Electric Vehicle (EV) dataset (70 documents)

---

## 1. PHẦN 1: NGHIÊN CỨU VÀ CHUẨN BỊ (RESEARCH)

### 1.1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
- **Thực thể (Node/Entity)**: Đại diện cho các đối tượng độc lập, có định danh duy nhất trong ngữ cảnh thế giới thực (ví dụ: công ty như `Tesla`, `BYD`, `ZEEKR`, `Polestar`, `Lucid`; chính sách như `Inflation Reduction Act`; chính phủ như `US Government`).
- **Thuộc tính (Property/Attribute)**: Là các đặc trưng, thông tin bổ sung mô tả cho một thực thể mà không đại diện cho một thực thể độc lập (ví dụ: `Financial Performance: Struggling`, `Stock Ticker: TSLA`).
- **Cơ chế phân biệt của LLM**: LLM dựa vào ngữ nghĩa của câu, cấu trúc ngữ pháp (chủ ngữ, vị ngữ, bổ ngữ) và các hướng dẫn rõ ràng trong System Prompt để phân định. System Prompt cung cấp danh sách định nghĩa và ví dụ về các loại Nodes cần trích xuất (ví dụ: Tên công ty, Thị trường, Sản phẩm) và hướng dẫn LLM chuyển hóa các mối liên kết có tính mô tả thành quan hệ có hướng thay vì thuộc tính tĩnh (ví dụ: Thay vì đặt `Stock Ticker` làm thuộc tính của node `Tesla`, ta trích xuất nó thành bộ ba quan hệ `(Tesla, HAS_STOCK_TICKER, TSLA)` để dễ dàng liên kết đồ thị).

### 1.2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
- Trong văn bản tự nhiên, một thực thể thường được viết dưới nhiều dạng khác nhau do ngữ cảnh hoặc cách viết tắt (ví dụ: `Volvo Cars` và `Volvo`, hoặc `Geely Group` và `Geely`, hay `US EV industry` và `US EV Market`).
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
  - Điểm mạnh: Lần theo các quan hệ rõ ràng có hướng trên đồ thị (ví dụ: `Polestar` -> `OWNED_BY` -> `Volvo` và `Polestar` -> `PARTNERS_WITH` -> `Geely`). Kết quả thu thập được là một ngữ cảnh cấu trúc logic, giúp LLM suy luận chính xác 100% dựa trên các đường đi thực tế trên đồ thị.

---

## 2. PHẦN 2: THIẾT LẬP MÔI TRƯỜNG VÀ TRIỂN KHAI PIPELINE

Bài lab được thực hiện đầy đủ trên Jupyter Notebook [lab19_graphrag.ipynb](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/lab19_graphrag.ipynb) với các bước:
1. **Environment Setup**: Cài đặt các thư viện `networkx`, `matplotlib`, `openai`, `google-genai`, `pandas`. Tích hợp khả năng tự động fallback sang Rule-based Mock khi không cấu hình API Key để đảm bảo notebook chạy thông suốt.
2. **Indexing**: Đọc toàn bộ 70 tài liệu từ thư mục [data/dataset](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/data/dataset). Sử dụng tiêu đề (Title) và tóm tắt (Snippet) để trích xuất Triples.
3. **Deduplication**: Xây dựng hàm `normalize_entity()` để chuẩn hóa các tên gọi trùng lặp (ví dụ: `Geely Group` -> `Geely`, `Volvo Cars` -> `Volvo`, `US EV Industry` -> `US EV Market`).
4. **Graph Construction**: Tạo đồ thị có hướng `DiGraph` trong NetworkX, lưu trữ thuộc tính `predicate` của cạnh và vẽ đồ thị trực quan bằng Matplotlib (lưu ảnh đồ thị tại [tech_company_graph.png](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/tech_company_graph.png)).
5. **Flat RAG Baseline**: Cài đặt tìm kiếm từ khóa trên tập câu thô và tạo prompt trả lời.
6. **GraphRAG Querying**: Thực thi trích xuất thực thể từ câu hỏi -> Duyệt đồ thị 2-hop bằng thuật toán BFS -> Chuyển đổi bộ ba kết quả thành văn bản tự nhiên (Textualization) -> Tạo prompt suy luận cho LLM.

---

## 3. PHẦN 3: SO SÁNH VÀ ĐÁNH GIÁ (EVALUATION)

Dưới đây là bảng so sánh chi tiết kết quả trả lời 5 câu hỏi phức tạp giữa hai hệ thống được kết xuất từ file [rag_comparison_results.csv](file:///Users/phamduythai/Downloads/2A202600860-PHAMDUYTHAI-DAY19/rag_comparison_results.csv):

| Câu hỏi (Question) | Kết quả của Flat RAG (Flat RAG Answer) | Kết quả của GraphRAG (GraphRAG Answer) | Đánh giá so sánh |
| :--- | :--- | :--- | :--- |
| **Câu 1**: Thương hiệu xe điện Polestar có mối liên hệ tài chính và đối tác như thế nào? | Polestar là một thương hiệu xe điện phổ biến. Tình hình tài chính của nó có phần khó khăn. Không rõ các đối tác hay chủ sở hữu chính của Polestar. | Polestar là một thương hiệu xe điện (EV Brand) thuộc sở hữu của Volvo và hợp tác với đối tác Geely. Tình hình hiệu suất tài chính của hãng đang gặp khó khăn (Struggling). | **GraphRAG thắng**: Flat RAG lấy được thông tin tài chính khó khăn nhưng không kết hợp được thông tin đối tác Geely/Volvo vì chúng nằm ở các file khác nhau. GraphRAG đi từ node `Polestar` -> `Volvo` và `Geely` thành công. |
| **Câu 2**: Mối liên hệ giữa tình hình tài chính của Tesla và vị thế của công ty tại thị trường xe điện Mỹ? | Tesla là hãng xe điện lớn ở Mỹ. Cổ phiếu Tesla biến động nhiều. Không tìm thấy mối liên kết rõ ràng về vị thế của công ty trên thị trường với tình hình tài chính. | Tesla thống trị thị trường xe điện Mỹ (US EV Market). Hiệu suất tài chính của công ty được ghi nhận là biến động (Volatile) với mã cổ phiếu giao dịch là TSLA. | **GraphRAG thắng**: Flat RAG liệt kê các thông tin rời rạc nhưng kết luận không tìm thấy mối liên hệ trực tiếp. GraphRAG liên kết được node `Tesla` đồng thời với `US EV Market` và `Volatile` hiệu suất tài chính. |
| **Câu 3**: Chính phủ Mỹ (US Government) hỗ trợ và áp đặt những biện pháp nào lên thị trường xe điện? | Chính phủ Mỹ hỗ trợ phát triển xe điện. Có các chính sách thuế quan được áp dụng đối với xe điện Trung Quốc. Không có thông tin đầy đủ về tất cả các biện pháp đồng thời. | Chính phủ Mỹ thúc đẩy thị trường xe điện Mỹ (US EV Market). Đạo luật Giảm lạm phát (Inflation Reduction Act) hỗ trợ ngành xe điện này. Đồng thời, chính phủ Mỹ áp đặt thuế quan (tariffs) lên các công ty xe điện Trung Quốc. | **GraphRAG thắng**: Đây là câu hỏi multi-hop liên quan đến chính sách hỗ trợ (IRA) và chính sách kìm hãm (thuế quan). Flat RAG bị đứt chuỗi thông tin. GraphRAG kết nối được cả hai mặt thông qua node `US Government`. |
| **Câu 4**: Công ty xe điện ZEEKR có cấu trúc doanh nghiệp và tình trạng niêm yết cổ phiếu như thế nào? | ZEEKR công bố báo cáo tài chính quý 1 năm 2024. Không có thông tin liên quan đến Geely hay niêm yết NYSE trong các tài liệu trích xuất. | ZEEKR là một công ty xe điện Trung Quốc, là thương hiệu con (sub-brand) của Geely và đã được niêm yết công khai trên sàn chứng khoán New York (NYSE). | **GraphRAG thắng**: Cấu trúc công ty con của Geely và việc niêm yết NYSE nằm ở các tài liệu khác nhau. Flat RAG chỉ lấy được báo cáo Q1/2024. GraphRAG giải quyết trọn vẹn. |
| **Câu 5**: Có mối liên hệ cạnh tranh nào giữa Tesla và BYD được ghi nhận không? | Tesla là công ty Mỹ, còn BYD là công ty Trung Quốc. Ngữ cảnh không chỉ rõ mối quan hệ cạnh tranh trực tiếp giữa hai bên. | Tesla và BYD là hai đối thủ cạnh tranh trực tiếp (RIVAL_OF) trong thị trường xe điện toàn cầu. BYD thống trị thị trường Trung Quốc và đang mở rộng ra toàn cầu, trực tiếp đối đầu với vị thế của Tesla. | **GraphRAG thắng**: Flat RAG chỉ biết một bên ở Mỹ, một bên ở Trung Quốc. GraphRAG chỉ ra mối quan hệ cạnh tranh trực tiếp rõ ràng (cạnh `RIVAL_OF` trên đồ thị). |

---

## 4. PHẦN 4: ĐỀ XUẤT CÔNG CỤ (RECOMMENDATIONS)

1. **NodeRAG**: Phù hợp cho việc tích hợp nhanh các ứng dụng GraphRAG có sẵn cấu trúc logic tìm kiếm.
2. **Neo4j**: Phù hợp cho việc lưu trữ, quản lý đồ thị quy mô lớn của doanh nghiệp, thực thi Cypher để truy vấn đồ thị phức tạp với hiệu năng cực cao.
3. **NetworkX**: Hoàn hảo cho việc chạy offline, nghiên cứu thuật toán trong Jupyter Notebook, dễ tích hợp và vẽ đồ thị mà không cần cài đặt cơ sở dữ liệu server.
