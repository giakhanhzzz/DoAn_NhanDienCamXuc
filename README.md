# Đồ Án: Nhận Diện Cảm Xúc Khuôn Mặt (Facial Emotion Recognition)

Dự án sử dụng AI (Deep Learning) để nhận diện 6 cảm xúc cơ bản của con người thông qua Camera.
Được phát triển bởi: **[Phạm Gia Khánh]** 

## 📂 Cấu trúc thư mục
- `app.py`: File code chính để chạy chương trình.
- `model.h5`: Model AI đã được train để nhận diện.
- `haarcascade_frontalface_default.xml`: Bộ lọc để phát hiện khuôn mặt.
- `requirements.txt`: Danh sách các thư viện cần thiết.

## ⚙️ Cài đặt môi trường
Để chạy được đồ án, bạn cần cài đặt Python và các thư viện sau:

1. **Cài đặt thư viện:**
Mở Terminal/CMD tại thư mục dự án và chạy lệnh:
```bash
pip install -r requirements.txt
