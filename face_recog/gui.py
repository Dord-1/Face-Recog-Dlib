"""Cửa sổ chính (Tkinter): nút thêm ảnh, nhận diện, nhật ký, quản lý người dùng và cài đặt."""
import tkinter as tk
import tkinter.font as font
import tkinter.messagebox as messagebox
from tkinter import ttk

from face_recog.activity_log import log_event as log_activity
from face_recog.activity_log import read_events
from face_recog.camera import CameraError
from face_recog.capture import img_capture
from face_recog.people import delete_person, list_people
from face_recog.recognizer import FaceRecognition
from face_recog.settings import load_settings, save_settings


def start_recognition(parent):
    try:
        FaceRecognition().run_recognition()
    except CameraError as e:
        messagebox.showerror("FR System", str(e), parent=parent)


def open_log_window(parent):
    window = tk.Toplevel(parent)
    window.title("Nhật ký")
    window.geometry("500x350")

    listbox = tk.Listbox(window, font=font.Font(size=11))
    listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

    def refresh():
        listbox.delete(0, tk.END)
        for line in reversed(read_events()):
            listbox.insert(tk.END, line)

    ttk.Button(window, text="Làm mới", command=refresh).pack(pady=10)
    refresh()


def open_people_window(parent):
    window = tk.Toplevel(parent)
    window.title("Quản lý người dùng")
    window.geometry("400x350")

    tree = ttk.Treeview(window, columns=("count",), show="headings")
    tree.heading("count", text="Tên (số ảnh)")
    tree.column("count", anchor="w")
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def refresh():
        tree.delete(*tree.get_children())
        for person in list_people():
            tree.insert("", tk.END, iid=person["name"],
                        values=(f'{person["name"]} ({person["count"]} ảnh)',))

    def delete_selected():
        selection = tree.selection()
        if not selection:
            return
        name = selection[0]
        if not messagebox.askyesno("FR System", f'Xoá toàn bộ ảnh của "{name}"?', parent=window):
            return
        count = delete_person(name)
        log_activity('DELETE_PERSON', f'{name} ({count} ảnh)')
        refresh()

    ttk.Button(window, text="Xoá người đã chọn", command=delete_selected).pack(pady=(0, 10))
    refresh()


def open_settings_window(parent):
    values = load_settings()
    window = tk.Toplevel(parent)
    window.title("Cài đặt")
    window.geometry("350x180")

    tk.Label(window, text="Ngưỡng nhận diện (0.0 - 1.0):").pack(pady=(15, 0))
    threshold_var = tk.StringVar(value=str(values["recognition_threshold"]))
    ttk.Spinbox(window, from_=0.05, to=1.0, increment=0.05, textvariable=threshold_var).pack()

    tk.Label(window, text="Số khung hình bỏ qua mỗi lần xử lý:").pack(pady=(15, 0))
    process_var = tk.StringVar(value=str(values["process_every_n"]))
    ttk.Spinbox(window, from_=1, to=30, increment=1, textvariable=process_var).pack()

    def save():
        try:
            new_values = {
                "recognition_threshold": float(threshold_var.get()),
                "process_every_n": int(process_var.get()),
            }
        except ValueError:
            messagebox.showerror("FR System", "Giá trị không hợp lệ", parent=window)
            return
        error = save_settings(new_values)
        if error:
            messagebox.showerror("FR System", error, parent=window)
            return
        messagebox.showinfo(
            "FR System", "Đã lưu. Áp dụng từ lần bấm \"Nhận diện khuôn mặt\" tiếp theo.", parent=window)

    ttk.Button(window, text="Lưu", command=save).pack(pady=15)


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

    style.configure(
        "Gray.TButton", background="#555555", foreground="white",
        font=font.Font(size=12), padding=6,
    )
    style.map("Gray.TButton", background=[("active", "#333333")])

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
    label = tk.Label(window, text='\n')
    label.pack()
    ttk.Button(window, text="Nhật ký", command=lambda: open_log_window(window),
              width=20, style="Gray.TButton").pack(pady=2)
    ttk.Button(window, text="Quản lý người dùng", command=lambda: open_people_window(window),
              width=20, style="Gray.TButton").pack(pady=2)
    ttk.Button(window, text="Cài đặt", command=lambda: open_settings_window(window),
              width=20, style="Gray.TButton").pack(pady=2)
    label = tk.Label(
        window,
        text="\nHướng dẫn\n1).Thêm ảnh vào bằng cách nhập tên và nhấn phím SPACEBAR để chụp ảnh. "
             "Sau khi xác nhận đã có ảnh, nhấn ESC để thoát.\n"
             "2).Khi muốn thoát khỏi nhận diện khuôn mặt, nhấn phím ESC để thoát.\n"
             "3).Nhật ký, quản lý người dùng và cài đặt mở cửa sổ riêng để xem lịch sử hoạt động, "
             "xoá người đã đăng ký hoặc chỉnh ngưỡng/tốc độ nhận diện.",
        font=font.Font(size=14))
    label.pack()
    window.mainloop()
