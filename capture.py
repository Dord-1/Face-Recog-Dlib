import os
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog

import cv2
import face_recognition

from config import CHECK_EVERY_N, DETECT_DIR, DETECT_SCALE, MIN_FACE_WIDTH


def check_frame(frame):
    """Trả về (hợp_lệ, thông_báo, khung_mặt) cho khung hình dùng để chụp ảnh.

    Thông báo không dấu vì cv2.putText không vẽ được chữ tiếng Việt có dấu.
    """
    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=0)

    if not locations:
        return False, "Khong thay khuon mat", None
    if len(locations) > 1:
        return False, "Chi duoc co 1 khuon mat", None

    top, right, bottom, left = (int(v / DETECT_SCALE) for v in locations[0])
    box = (left, top, right, bottom)
    if right - left < MIN_FACE_WIDTH:
        return False, "Khuon mat qua nho, hay lai gan hon", box
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
