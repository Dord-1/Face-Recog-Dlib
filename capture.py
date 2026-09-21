import os
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog

import cv2
import face_recognition
import numpy as np

from config import (
    CHECK_EVERY_N,
    DETECT_DIR,
    DETECT_SCALE,
    FACE_CHIP,
    INV_SCALE,
    MAX_BRIGHTNESS,
    MAX_YAW_RATIO,
    MIN_BRIGHTNESS,
    MIN_FACE_WIDTH,
    MIN_SHARPNESS,
)
from Recognition import scale_locations


def face_gray_chip(frame, box):
    """Cắt vùng mặt (left, top, right, bottom) từ khung BGR, chuyển xám và đưa về FACE_CHIP x FACE_CHIP."""
    left, top, right, bottom = box
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FACE_CHIP, FACE_CHIP), interpolation=cv2.INTER_AREA)


# Sàn cho phương sai xám khi chuẩn hoá độ nét: vùng gần như phẳng (phương sai vài đơn vị) chỉ còn nhiễu
# lượng tử hoá uint8, nếu chia cho phương sai quá nhỏ sẽ bị chấm nhầm là "nét".
CONTRAST_VAR_FLOOR = 50.0


def sharpness(face_gray):
    """Độ nét: phương sai Laplacian chia cho phương sai xám (chuẩn hoá theo tương phản).

    Chia cho tương phản để ảnh tối/nhạt không bị chấm "mờ" (Laplacian thô giảm theo tương phản).
    """
    laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    return float(laplacian_var / max(float(face_gray.astype(np.float64).var()), CONTRAST_VAR_FLOOR))


def brightness(face_gray):
    """Độ sáng: trung bình mức xám (0-255) của vùng mặt."""
    return float(face_gray.mean())


def yaw_ratio(landmarks):
    """Tỉ lệ khoảng cách từ mũi tới hai mắt (>= 1): ~1 khi nhìn thẳng, tăng khi quay đầu sang một bên."""
    nose = np.mean(landmarks['nose_tip'], axis=0)
    left = np.linalg.norm(nose - np.mean(landmarks['left_eye'], axis=0))
    right = np.linalg.norm(nose - np.mean(landmarks['right_eye'], axis=0))
    return float(max(left, right) / max(min(left, right), 1e-6))


def check_frame(frame):
    """Trả về (hợp_lệ, thông_báo, khung_mặt) cho khung hình dùng để chụp ảnh.

    Kiểm tra lần lượt: đúng 1 mặt, đủ rộng, độ sáng, độ nét, mặt nhìn thẳng. Thông báo không dấu vì
    cv2.putText không vẽ được chữ tiếng Việt có dấu; khi không đạt có kèm số đo để chỉnh ngưỡng trong config.
    """
    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=0)

    if not locations:
        return False, "Khong thay khuon mat", None
    if len(locations) > 1:
        return False, "Chi duoc co 1 khuon mat", None

    full_location = scale_locations(locations, INV_SCALE, frame.shape)[0]  # (top, right, bottom, left)
    top, right, bottom, left = full_location
    box = (left, top, right, bottom)
    if right - left < MIN_FACE_WIDTH:
        return False, "Khuon mat qua nho, hay lai gan hon", box

    chip = face_gray_chip(frame, box)
    light = brightness(chip)
    if light < MIN_BRIGHTNESS:
        return False, f"Qua toi ({light:.0f})", box
    if light > MAX_BRIGHTNESS:
        return False, f"Qua sang ({light:.0f})", box
    focus = sharpness(chip)
    if focus < MIN_SHARPNESS:
        return False, f"Anh bi mo ({focus:.2f})", box

    landmarks = face_recognition.face_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [full_location])
    if landmarks:
        yaw = yaw_ratio(landmarks[0])
        if yaw > MAX_YAW_RATIO:
            return False, f"Hay nhin thang ({yaw:.2f})", box
    return True, "OK - nhan SPACE de chup", box


def next_free_index(file_name, detect_dir=DETECT_DIR):
    """Số thứ tự đầu tiên chưa dùng, để chụp lại cùng tên không ghi đè ảnh cũ."""
    index = 0
    while os.path.exists(os.path.join(detect_dir, f"{file_name}_{index}.jpg")):
        index += 1
    return index


def img_capture(parent):
    file_name = simpledialog.askstring(
        title="FR System",
        prompt="Tên bạn là gì:",
        parent=parent,
    )
    if not file_name or not file_name.strip():
        return

    file_name = file_name.strip()
    os.makedirs(DETECT_DIR, exist_ok=True)
    img_counter = next_free_index(file_name)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam.release()
        messagebox.showerror("FR System", "Không tìm thấy hoặc không mở được webcam", parent=parent)
        return

    window_name = "Face Capturing"
    frame_count = 0
    valid, message, box = False, "", None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        if frame_count % CHECK_EVERY_N == 0:
            valid, message, box = check_frame(frame)
        frame_count += 1

        # Vẽ trên bản sao để ảnh lưu ra không dính khung/chữ
        display = frame.copy()
        color = (0, 200, 0) if valid else (0, 0, 255)
        if box:
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(display, message, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.imshow(window_name, display)

        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Đã đóng cửa sổ")
            break
        elif k % 256 == 32:
            # SPACE pressed
            if not valid:
                print(f"Chưa chụp: {message}")
                continue
            img_name = f"{file_name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(DETECT_DIR, img_name), frame)
            print(f"{img_name} đã chụp!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
