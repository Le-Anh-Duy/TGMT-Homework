Dưới đây là bản tổng hợp hoàn chỉnh phần **"Nhận xét bổ sung"**, được định dạng chuẩn Markdown. Bạn có thể chép trực tiếp phần này và đặt vào vị trí **Mục 5.3** (ngay sau bảng phân tích số liệu) trong sườn báo cáo của bạn. 

---

## 5.3 Nhận xét bổ sung và Biện luận học thuật

### 5.3.1 Phân tích chuyên sâu về sự biến động của các chỉ số hiệu suất

* **"Hiệu ứng trần" (Ceiling Effect) trên tập MNIST Handwritten:**
  Đối với tập dữ liệu cơ bản như MNIST chữ số viết tay, kiến trúc LeNet-5 cổ điển đã nhanh chóng đạt đến giới hạn trần (hiệu suất trên 99%). Các cải tiến về sau (Modernized, Wide) chủ yếu mang lại sự hội tụ nhanh, mượt mà và dòng gradient ổn định hơn chứ không tạo ra bước nhảy vọt lớn về Accuracy. Điều này minh chứng cho việc kiến trúc gốc của Yann LeCun đã được thiết kế cực kỳ tối ưu và hoàn hảo cho bài toán nhận dạng nét chữ đơn sắc.

* **Giải quyết "Nút thắt sức chứa" (Capacity Bottleneck) trên Fashion MNIST:**
  Sự kẹt lại ở ngưỡng ~89% trong hai mô hình Baseline cho thấy mạng LeNet gốc bị thiếu hụt nghiêm trọng về sức chứa (chỉ có 6 và 16 feature maps). Khi nâng cấp lên bản Wide (32-64-256 filters), mô hình có đủ không gian bộ nhớ (parameters) để trích xuất và học các đặc trưng vi mô như cổ áo, viền áo, nếp gấp. Chỉ số F1-Score đạt mức 0.93 chứng tỏ mô hình không chỉ đoán đúng các lớp dễ phân biệt (như quần, giày) mà đã giải quyết tốt sự chồng chéo đặc trưng ở các lớp khó (như Shirt và T-shirt).

* **Xử lý "Mất cân bằng dữ liệu" trên PneumoniaMNIST:**
  Trên tập dữ liệu y tế nhạy cảm, việc chỉ đổi hàm kích hoạt sang ReLU (ở bản Modernized) không giải quyết được gốc rễ vấn đề: sự mất cân bằng số lượng mẫu giữa nhãn Normal và Pneumonia. Sự đột phá lên mức ~90% ở phiên bản Advanced chủ yếu đến từ kỹ thuật **Class Weighting** được tích hợp vào hàm Loss. Mức Recall và Precision cân bằng (~0.90) chứng tỏ mô hình đã thực sự học được đặc trưng bệnh lý từ ảnh X-quang, thay vì dự đoán thiên lệch (bias) theo nhóm nhãn chiếm đa số.

* **Nghịch lý kích thước Kernel: Tại sao 5x5 lại nhỉnh hơn 3x3?**
  Mặc dù xu hướng Deep Learning hiện đại (như VGG, ResNet) ưa chuộng kernel 3x3 để tăng tính phi tuyến, kết quả thực nghiệm lại cho thấy bản Wide 5x5 có phần nhỉnh hơn nhẹ. Hiện tượng này có thể được lý giải bởi kích thước ảnh đầu vào rất nhỏ (chỉ 32x32 pixel). Một kernel 5x5 cung cấp vùng nhận cảm (receptive field) đủ lớn ngay từ lớp Conv đầu tiên để bắt trọn các cấu trúc vĩ mô (macro-structures) quan trọng – như hình dáng lồng ngực trong X-Quang hay phom dáng tổng thể của trang phục – thay vì tập trung quá sớm vào các nhiễu hạt cục bộ như kernel 3x3.

### 5.3.2 Đối chiếu các cải tiến với nguyên lý thiết kế gốc của LeNet-5 (1998)

Việc nâng cấp LeNet-5 không chỉ đơn thuần là thay thế các hàm thư viện mới, mà là quá trình điều chỉnh tư duy kiến trúc để phù hợp với độ phức tạp của dữ liệu và sức mạnh phần cứng hiện đại:

* **Hàm kích hoạt (Từ bỏ Tanh để chọn ReLU):**
  Năm 1998, `Tanh` được ưu chuộng vì có tâm đối xứng quanh trục 0 và đạo hàm trơn, giúp mạng nông hội tụ tốt trên dữ liệu giản giản. Tuy nhiên, với dữ liệu y tế và thời trang có độ dốc pixel phức tạp, mạng gặp hiện tượng bão hòa gradient (vanishing gradient). Việc chuyển sang `ReLU` giúp triệt tiêu hiện tượng này, giữ lại các tín hiệu kích hoạt dương mạnh mẽ, tăng tốc độ hội tụ và giảm đáng kể chi phí tính toán.

* **Cơ chế giảm chiều dữ liệu (Average Pooling sang Max Pooling):**
  LeNet-5 gốc dùng `Average Pooling` để làm "mờ" (blur) đặc trưng, với tư duy rằng một chữ số hơi méo mó thì trung bình pixel vẫn không đổi. Tuy nhiên, cách làm này vô tình san phẳng và xóa sổ các chi tiết cực kỳ quan trọng nhưng mờ nhạt (như nếp gấp áo hay tổn thương phổi nhỏ). `Max Pooling` được sử dụng để khắc phục điều này, giúp mạng bám sát vào các cấu trúc viền biên có cường độ tín hiệu cao nhất.

* **Kiến trúc kết nối (Sparse Connection sang Dense Connection):**
  Lớp C3 gốc sử dụng "bảng kết nối thưa" thủ công chủ yếu để giảm số lượng tham số, nhằm đưa mạng chạy vừa vặn trên bộ nhớ phần cứng hạn hẹp của thập niên 90. Trong bản cải tiến, việc gỡ bỏ bảng này và dùng `Conv2d` kết nối dày (Dense Connection) cho phép các kênh thông tin đối chiếu chéo (cross-channel interaction) toàn diện hơn, tối ưu hóa triệt để khả năng tính toán ma trận song song của GPU.

* **Kỹ thuật Điều chuẩn - Regularization (Sự bổ sung của Batch Norm và Dropout):**
  Thiết kế gốc hoàn toàn vắng bóng Regularization. Khi mở rộng sức chứa của mạng (LeNet-Wide), rủi ro lớn nhất là Overfitting. Việc bổ sung `Dropout(0.5)` đóng vai trò như một bộ phanh hãm chống học vẹt hiệu quả. Đặc biệt, `Batch Normalization` trở thành "cứu cánh" quyết định cho tập PneumoniaMNIST, giúp chuẩn hóa và san phẳng sự chênh lệch về độ sáng giữa các thiết bị chụp X-quang khác nhau mà nguyên bản LeNet-5 không thể xử lý được.

--- 

*Phần tổng hợp này sẽ giúp bài báo cáo của bạn thể hiện được tư duy nghiên cứu (Research Thinking) rất rõ nét, đóng vai trò chốt hạ hoàn hảo cho toàn bộ quá trình thực hành.*