import cv2
import numpy as np
from tensorflow.keras.models import load_model
#keras.models là thư viện kiểu cũ, tensorflow.keras.models là thư viện mới hơn, nên dùng tensorflow.keras.models tich hợp thẳng
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' 

EMOTIONS = ['Gian du', 'Ghe tom', 'So hai', 'Vui ve', 'Buon', 'Ngac nhien']

# --- KHỞI TẠO ---
print("--- BAT DAU KHOI TAO ---")
try:
    if not os.path.exists(MODEL_PATH):
        print(f"LOI: Khong tim thay file '{MODEL_PATH}'!")
        exit()
    model = load_model(MODEL_PATH)
    print("-> Da load model thanh cong!")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("LOI: Khong bat duoc Webcam!")
        exit()

except Exception as e:
    print(f"CO LOI XAY RA: {e}")
    exit()

# --- VÒNG LẶP XỬ LÝ ---
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # MinNeighbors=3 để bắt mặt nhạy hơn
    # Tăng minNeighbors lên 5 để giảm nhiễu (ít bắt nhầm rèm cửa/bóng đèn là mặt)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        try:
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray.size == 0: continue

            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # 2. Dự đoán cảm xúc
            raw_prediction = model.predict(roi, verbose=0)[0]

            prediction_6 = np.delete(raw_prediction, 5)

            # Lấy kết quả dự đoán hợp lý nhất
            max_index = int(np.argmax(prediction_6))
            predicted_emotion = EMOTIONS[max_index]
            confidence = prediction_6[max_index] * 100 

            # Hiển thị
            color = (0, 255, 0) if confidence > 50 else (0, 0, 255)
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            
            cv2.putText(frame, predicted_emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        except Exception:
            pass

    cv2.imshow('Nhan dien cam xuc', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()