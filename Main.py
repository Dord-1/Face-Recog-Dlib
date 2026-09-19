import os
import cv2
from Recognition import FaceRecognition

import tkinter as tk
from tkinter import ttk
import tkinter.font as font
import tkinter.simpledialog as simpledialog

def img_capture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Training")
    img_counter = 0

    file_name = simpledialog.askstring(
        title="FR System",
        prompt="Tên bạn là gì:",
        parent=window,
    )
    if not file_name or not file_name.strip():
        cam.release()
        cv2.destroyAllWindows()
        return

    file_name = file_name.strip()
    os.makedirs("detect", exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Face Capturing", frame)

        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Đã đóng cửa sổ")
            break
        elif k%256 == 32:
            # SPACE pressed
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

