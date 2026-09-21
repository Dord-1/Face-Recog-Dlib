"""Cửa sổ chính (Tkinter): nút thêm ảnh và nhận diện."""
import tkinter as tk
import tkinter.font as font
import tkinter.messagebox as messagebox
from tkinter import ttk

from face_recog.camera import CameraError
from face_recog.capture import img_capture
from face_recog.recognizer import FaceRecognition


def start_recognition(parent):
    try:
        FaceRecognition().run_recognition()
    except CameraError as e:
        messagebox.showerror("FR System", str(e), parent=parent)


def main():
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
        window, text='Chào mừng bạn đến với FR System, mời bạn chọn các phím sau:\n', font=font.Font(size=16))
    label.pack()
    button = ttk.Button(window, text="Thêm ảnh vào hệ thống",
                        command=lambda: img_capture(window), width=20, style="Red.TButton")
    button.pack()
    label = tk.Label(window, text='\n')
    label.pack()
    button = ttk.Button(window, text="Nhận diện khuôn mặt",
                        command=lambda: start_recognition(window), width=20, style="Blue.TButton")
    button.pack()
    label = tk.Label(
        window,
        text="\nHướng dẫn\n1).Thêm ảnh vào bằng cách nhập tên và nhấn phím SPACEBAR để chụp ảnh. "
             "Sau khi xác nhận đã có ảnh, nhấn ESC để thoát.\n"
             "2).Khi muốn thoát khỏi nhận diện khuôn mặt, nhấn phím ESC để thoát.",
        font=font.Font(size=14))
    label.pack()
    window.mainloop()
