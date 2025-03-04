## Yêu cầu hệ thống
- **Python**: 3.8+
- **Thư viện**: Xem `requirements.txt`
- **Phần cứng**: GPU khuyến nghị (CUDA hỗ trợ), nhưng chạy được trên CPU.

## Cài đặt
1. **Clone repository**:
   ```bash
   git clone https://github.com/TUANHM211224/deepfake-lip-detection.git
   cd deepfake-lip-detection
Cài đặt thư viện:
pip install -r requirements.txt
Tải dữ liệu:
Đặt file lip_data_checkpoint.npz vào thư mục data/.
Tải checkpoint (tùy chọn):
Đặt file cnn_gru_10_frames.pth vào thư mục models/.
Cách chạy
1. Huấn luyện mô hình
python src/train.py
2. Đánh giá mô hình
bash
python src/evaluate.py --checkpoint models/cnn_gru_10_frames.pth
Kết quả
CNN + GRU:
Train Loss: Giảm từ 0.6462 → 0.1732 (40 epoch).
Val Accuracy: Cao nhất 91.25% (Epoch 33 & 36), trung bình ~80%.
Liên hệ
GitHub: [username]