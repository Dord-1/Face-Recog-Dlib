"""Đo chất lượng khung hình khi chụp ảnh đăng ký (thuần xử lý ảnh, không phụ thuộc giao diện)."""
import cv2
import face_recognition
import numpy as np

from face_recog.config import (
    DETECT_SCALE,
    FACE_CHIP,
    INV_SCALE,
    MAX_BRIGHTNESS,
    MAX_YAW_RATIO,
    MIN_BRIGHTNESS,
    MIN_FACE_WIDTH,
    MIN_SHARPNESS,
)
from face_recog.geometry import scale_locations


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
