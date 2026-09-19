import os
import cv2
import face_recognition
from Recognition import FaceRecognition

import tkinter as tk
from tkinter import ttk
import tkinter.font as font
import tkinter.simpledialog as simpledialog

MIN_FACE_WIDTH = 100     # px trên khung hình gốc; nhỏ hơn thì ảnh quá xa/khó nhận diện
CHECK_EVERY_N = 3        # chỉ dò khuôn mặt mỗi N khung để không làm giật hình chụp
SCALE = 0.25             # dò trên khung thu nhỏ cho nhanh (cùng tỉ lệ với Recognition.py)


def check_frame(frame):
    """Trả về (hợp_lệ, thông_báo, khung_mặt) cho khung hình dùng để chụp ảnh."""
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=0)

    if not locations:
        return False, "Khong thay khuon mat", None
    if len(locations) > 1:
        return False, "Chi duoc co 1 khuon mat", None

    top, right, bottom, left = (int(v / SCALE) for v in locations[0])
    box = (left, top, right, bottom)
    if right - left < MIN_FACE_WIDTH:
        return False, "Khuon mat qua nho, hay lai gan hon", box
    return True, "OK - nhan SPACE de chup", box


def next_free_index(file_name):
    """Số thứ tự đầu tiên chưa dùng, để chụp lại cùng tên không ghi đè ảnh cũ."""
    index = 0
    while os.path.exists(os.path.join("detect", f"{file_name}_{index}.jpg")):
        index += 1
    return index


def img_capture():
    file_name = simpledialog.askstring(
        title="FR System",
        prompt="Tên bạn là gì:",
        parent=window,
    )
    if not file_name or not file_name.strip():
        return

    file_name = file_name.strip()
    os.makedirs("detect", exist_ok=True)
    img_counter = next_free_index(file_name)

    cam = cv2.VideoCapture(0)
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

        if k%256 == 27:
            # ESC pressed
            print("Đã đóng cửa sổ")
            break
        elif k%256 == 32:
            # SPACE pressed
            if not valid:
                print(f"Chưa chụp: {message}")
                continue
            img_name = f"{file_name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join("detect", img_name), frame)
            print(f"{img_name} đã chụp!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    window = tk.Tk()
    window.config(width=300, height=300, padx=20, pady=50)

    # Trên macOS, tk.Button dùng theme "aqua" gốc của hệ điều hành và BỎ QUA
    # hoàn toàn bg/fg tuỳ chỉnh (không liên quan dark/light mode, đây là giới
    # hạn cố định của Tkinter trên macOS). Dùng ttk.Button với theme "clam"
    # để màu nút hiển thị đúng như mong muốn trên mọi nền tảng.
    style = ttk.Style()
    style.theme_use("clam")

    button_font = font.Font(size=16)

    style.configure(
        "Red.TButton", background="red", foreground="white",
        font=button_font, padding=10,
    )
    style.map("Red.TButton", background=[("active", "#cc0000")])

    style.configure(
        "Blue.TButton", background="#0052cc", foreground="white",
        font=button_font, padding=10,
    )
    style.map("Blue.TButton", background=[("active", "#003d99")])

    label = tk.Label(
    window, text='Chào mừng bạn đến với FR System, mời bạn chọn các phím sau:\n',font=font.Font(size=16))
    label.pack()
    button = ttk.Button(window, text="Thêm ảnh vào hệ thống", command=img_capture, width=20, style="Red.TButton")
    button.pack()
    label = tk.Label(window, text='\n')
    label.pack()
    button = ttk.Button(window,
                    text="Nhận diện khuôn mặt",
                    command=lambda: FaceRecognition().run_recognition(), width=20, style="Blue.TButton"
            )
    button.pack()
    label = tk.Label(window,
                text="\nHướng dẫn\n1).Thêm ảnh vào bằng cách nhập tên và nhấn phím SPACEBAR để chụp ảnh. Sau khi xác nhận đã có ảnh, nhấn ESC để thoát.\n2).Khi muốn thoát khỏi nhận diện khuôn mặt, nhấn phím ESC để thoát.",
                font=font.Font(size=14))
    label.pack()
    window.mainloop()

