import cv2 
import numpy as np
import mediapipe as mp #Thư viện AI của Google để nhận diện bàn tay
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== MEDIAPIPE ==================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task") #Đường đến file mô hình
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1 # Số lượng bàn tay cần nhận diện
)
detector = vision.HandLandmarker.create_from_options(options) #Khởi tạo bộ nhận diện

# ================== CAMERA ==================
cap = cv2.VideoCapture(0) #Mở camera
canvas = None #Khởi tạo canvas trắng để vẽ
prev_x, prev_y = None, None #Tọa độ điểm vẽ trước đó
lost_frames = 0 #Số khung hình bị mất

draw_color = (0, 0, 0)
thickness = 6

# ================== Bang mau ==================
palette_height = 90 #Chieu cao bang mau
palette_colors = [
    (0, 0, 0),         # đen
    (0, 0, 255),       # đỏ
    (0, 255, 0),       # xanh lá
    (255, 0, 0),       # xanh dương
    (0, 255, 255),     # vàng
    (180, 105, 255)    # hồng
]

# ================== Xoa ==================
clear_counter = 0
CLEAR_FRAMES = 15

# ================== Vong lap ==================
while True:
    ret, frame = cap.read() #Đọc khung hình từ camera
    if not ret:
        break

    frame = cv2.flip(frame, 1) #Lật khung hình ngang
    h, w, _ = frame.shape 

    if canvas is None:
        canvas = np.ones_like(frame) * 255 #Tạo canvas trắng

    # ---------- Detect tay ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Định dạng RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        lost_frames = 0
        lm = result.hand_landmarks[0] #Lấy tọa độ các điểm mốc trên bàn tay

        x = int(lm[8].x * w) #Ngón trỏ x 
        y = int(lm[8].y * h) #Ngón trỏ y
        y = np.clip(y, 0, h - 1) 

        index_up  = lm[8].y  < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # ---------- VẼ (1 NGÓN) ----------
        if index_up and not middle_up:
            clear_counter = 0
            if prev_x is None:
                prev_x, prev_y = x, y #Điểm bắt đầu
            cv2.line(canvas, (prev_x, prev_y), (x, y),
                     draw_color, thickness, cv2.LINE_AA) #Vẽ nối
            prev_x, prev_y = x, y #Cập nhật điểm trước đó

        # ---------- XOÁ ÍT (TRỎ + GIỮA) ----------
        elif index_up and middle_up and not ring_up and not pinky_up:
            clear_counter = 0
            cv2.circle(canvas, (x, y), 30, (255, 255, 255), -1) #Xoá bằng hình tròn trắng

        # ---------- XOÁ HẾT ----------
        elif fingers_up >= 4:
            clear_counter += 1
            if clear_counter > CLEAR_FRAMES:
                canvas[:] = 255
                prev_x, prev_y = None, None
                clear_counter = 0
        else:
            prev_x, prev_y = None, None
            clear_counter = 0

    else:
        lost_frames += 1
        if lost_frames > 4:
            prev_x, prev_y = None, None

    # ---------- Canvas sang Mask ----------
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    # Kết hợp ảnh camera và nét vẽ
    frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
    frame = cv2.add(frame, cv2.bitwise_and(canvas, canvas, mask=mask))

    block_w = 100
    for i, color in enumerate(palette_colors):
        x1 = i * block_w
        x2 = x1 + block_w
        cv2.rectangle(frame, (x1, 0), (x2, palette_height), color, -1)
        cv2.rectangle(frame, (x1, 0), (x2, palette_height), (255, 255, 255), 2)

    # chọn màu bằng CHẠM 
    if result.hand_landmarks:
        if y < palette_height:
            idx = x // block_w
            if 0 <= idx < len(palette_colors):
                draw_color = palette_colors[idx]

    cv2.imshow("Air Drawing", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
