# Context: Bài tập về nhà: phân tích dữ liệu khám phá trên cả 3 tập của MNIST (Digit, Fashion, Pneumonia), dữ liệu lưu trong thư mục 'data'

# BÁO CÁO PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA) ĐA MIỀN

**Tập dữ liệu:** Digit MNIST, Fashion MNIST, Pneumonia MNIST

### I. Khởi tạo & Tiền xử lý (Setup & Preprocessing)

* **Tập dữ liệu áp dụng:** **Cả 3 tập (Digit, Fashion, Pneumonia)**.
* **Yêu cầu cho Copilot:** * Viết hàm tải dữ liệu cho cả 3 tập (sử dụng `torchvision.datasets` cho Digit/Fashion và thư viện `medmnist` hoặc file numpy cho Pneumonia).
* Chuẩn hóa tất cả về định dạng numpy array, kích thước `(N, 28, 28)` và scale giá trị pixel về dải `[0, 1]`.


* **Kết quả đầu ra:** In ra màn hình một bảng text (DataFrame) gồm 4 cột: `Tên Dataset`, `Số lượng Train`, `Số lượng Test`, `Kích thước ảnh`, `Số lượng Class`.

### II. Khảo sát Phân phối Nhãn (Label Distribution)

* **Tập dữ liệu áp dụng:** **Cả 3 tập**, nhưng cách phân tích khác nhau.
* **Yêu cầu cho Copilot:** * Viết code vẽ 1 figure gồm 3 biểu đồ Bar Chart (1 hàng 3 cột) đếm số lượng ảnh của từng class cho 3 tập tương ứng.
* **Riêng tập Pneumonia:** Viết thêm code tính tỷ lệ **Imbalance Ratio (Normal / Pneumonia)** và in ra màn hình.


* **Kết quả đầu ra:** * Hình ảnh: 3 Bar Chart cạnh nhau. Digit/Fashion sẽ phẳng lỳ (cân bằng), Pneumonia sẽ chênh lệch.
* Thông số: Dòng chữ in ra terminal (VD: `Pneumonia Imbalance Ratio: 1:3 -> Cảnh báo: Cần cân nhắc dùng class_weight`).



### III. Phân tích Phân phối Điểm ảnh (Pixel Intensity Histogram)

* **Tập dữ liệu áp dụng:** So sánh đối chiếu giữa **Digit MNIST** và **Pneumonia MNIST**.
* **Yêu cầu cho Copilot:**
* Lấy ra toàn bộ pixel của class `0` và `1` trong tập Digit MNIST. Vẽ Histogram (màu xanh và cam đè lên nhau, độ trong suốt alpha=0.5).
* Lấy ra toàn bộ pixel của class `Normal` và `Pneumonia` trong tập Pneumonia MNIST. Vẽ Histogram tương tự.


* **Kết quả đầu ra:** 1 Figure có 2 biểu đồ. Insight kỳ vọng là Histogram của Digit sẽ có 2 đỉnh (0 và 1 rõ rệt, bimodal), trong khi Histogram của Pneumonia sẽ là 1 hình chuông (unimodal) và 2 class Normal/Pneumonia đè lên nhau gần như 100%, chứng tỏ không thể phân loại y khoa bằng độ sáng tối đơn thuần.

### IV. Phân tích Vùng Tín hiệu bằng Ảnh Trung Bình (Mean Image & Difference Heatmap)

* **Tập dữ liệu áp dụng:** **Cả 3 tập**, trọng tâm phân tích Heatmap nằm ở **Pneumonia**.
* **Yêu cầu cho Copilot:**
* **Thao tác 1:** Tính ảnh trung bình (Mean Image) cho từng class của cả 3 tập. Vẽ lưới ảnh (Digit: lưới 2x5, Fashion: lưới 2x5, Pneumonia: lưới 1x2).
* **Thao tác 2:** Lấy `Mean Image của Pneumonia` trừ đi `Mean Image của Normal`. Dùng `seaborn.heatmap` với dải màu `RdBu_r` để vẽ sự chênh lệch (Difference Heatmap).


* **Kết quả đầu ra:** * Hình ảnh: Các lưới ảnh Mean.
* Hình ảnh trọng tâm: 1 Heatmap của Pneumonia. Vùng giữa ngực (tim) sẽ có màu trung tính (không đổi), trong khi 2 bên thùy phổi sẽ có màu đỏ/xanh (chênh lệch mạnh), chỉ ra cho AI biết cần tập trung vào vùng không gian nào.



### V. Phân tích Hình thái Cạnh biên (Canny Edge Detection)

* **Tập dữ liệu áp dụng:** So sánh đối chiếu giữa **Fashion MNIST** và **Pneumonia MNIST**.
* **Yêu cầu cho Copilot:**
* Chọn ngẫu nhiên 5 ảnh áo/quần (Fashion) và 5 ảnh phổi (Pneumonia).
* Chạy thuật toán `cv2.Canny()` để trích xuất viền.
* Vẽ lưới ảnh gồm 2 hàng x 10 cột (Hàng trên là ảnh gốc 5 Fashion + 5 Pneumonia, hàng dưới là ảnh Canny tương ứng).


* **Kết quả đầu ra:** Hình ảnh lưới viền. Insight nhận được: Viền của Fashion rất gọn gàng, định hình được vật thể. Viền của Pneumonia là một mớ hỗn độn (nhiễu từ xương sườn), kết luận bài toán y tế phụ thuộc vào "kết cấu" (texture) thay vì "hình dáng" (shape/contour).

### VI. Đánh giá Khả năng Phân tách trong Không gian (t-SNE Dimensionality Reduction)

* **Tập dữ liệu áp dụng:** **Cả 3 tập** (chạy độc lập để so sánh độ khó).
* **Yêu cầu cho Copilot:**
* Flatten ảnh từ 2D thành 1D (784 features).
* Random sampling đúng **2000 ảnh** cho MỖI TẬP (để chạy cho nhanh).
* Fit thuật toán t-SNE (2 components) cho từng tập. Vẽ 3 biểu đồ Scatter Plot (1 hàng 3 cột) hiển thị các điểm dữ liệu được tô màu theo Label.
* Sử dụng `sklearn.metrics.silhouette_score` để tính điểm tách cụm cho 3 tập.


* **Kết quả đầu ra:** * Hình ảnh: 3 Scatter plot. Digit sẽ chia 10 cụm rõ ràng; Fashion sẽ có cụm nhưng hơi dính nhau; Pneumonia sẽ là 1 đám mây khổng lồ trộn lẫn 2 màu.
* Thông số: Bảng so sánh Silhouette Score. Digit cao nhất ($>0.5$), Pneumonia thấp nhất (gần $0$).



### VII. Đo lường Độ Tương Đồng Cấu Trúc (SSIM Matrix)

* **Tập dữ liệu áp dụng:** **Pneumonia MNIST**.
* **Yêu cầu cho Copilot:**
* Viết code tính SSIM (Structural Similarity Index) trung bình giữa: (1) 100 cặp ảnh Normal-Normal, (2) 100 cặp ảnh Pneumonia-Pneumonia, và (3) 100 cặp ảnh Normal-Pneumonia.
* In kết quả ra màn hình.


* **Kết quả đầu ra:** Dòng text in ra màn hình. Nếu SSIM giữa Normal-Pneumonia lên tới 0.85 hoặc 0.9, ta có luận điểm vững chắc rằng: Các tấm X-quang cực kỳ giống nhau, phải dùng Deep Learning để trích xuất đặc trưng vi mô.

### VIII. Truy tìm Ngoại lai và Ảnh Hỏng (Outlier Detection)

* **Tập dữ liệu áp dụng:** **Pneumonia MNIST** (mô phỏng rủi ro y khoa).
* **Yêu cầu cho Copilot:**
* Tính tổng cường độ pixel (Sum of intensity) cho toàn bộ ảnh trong tập Pneumonia.
* Sort mảng cường độ này. Lấy ra top 5 ảnh có tổng cường độ thấp nhất (under-exposed) và 5 ảnh cao nhất (over-exposed).
* Vẽ lưới 2x5 hiển thị 10 ảnh này.


* **Kết quả đầu ra:** 10 tấm X-quang hỏng (hoặc tối đen, hoặc trắng xóa bóng mờ). Đề xuất bước tiền xử lý như cân bằng histogram (CLAHE) ở giai đoạn huấn luyện.
