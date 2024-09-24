import os
import cv2
from Recognition import FaceRecognition

from tkinter import *
import tkinter.font as font
import tkinter.simpledialog as simpledialog

def img_capture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Training")
    img_counter = 0

    file_name = ''
    file_name= simpledialog.askstring(title="FR System",prompt="Tên bạn là gì:")
    window = Tk()
    window.withdraw()

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
            img_name = file_name + "_{}.jpg".format(img_counter)
            cv2.imwrite(os.path.join("detect/" + img_name), frame)
            print("{} đã chụp!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    window.destroy()
    
if __name__ == "__main__":
    window = Tk()
    window.config(width=300, height=300, padx=20, pady=50)
    label = Label(
    window, text='Chào mừng bạn đến với FR System, mời bạn chọn các phím sau:\n',font=font.Font(size=16))
    label.pack()
    button = Button(window, text="Thêm ảnh vào hệ thống", command=img_capture, width=20, bg="red", fg="white", pady=10)
    button['font']=font.Font(size=16)
    button.pack()
    label = Label(window, text='\n')
    label.pack()
    button = Button(window, 
                    text="Nhận diện khuôn mặt", 
                    command=lambda: FaceRecognition().run_recognition(), width=20, bg="#0052cc", fg="white", pady=10
            )
    button['font']=font.Font(size=16)
    button.pack()
    label=Label(window,
                text="\nHướng dẫn\n1).Thêm ảnh vào bằng cách nhập tên và nhấn phím SPACEBAR để chụp ảnh. Sau khi xác nhận đã có ảnh, nhấn ESC để thoát.\n2).Khi muốn thoát khỏi nhận diện khuôn mặt, nhấn phím ESC để thoát.",
                font=font.Font(size=14))
    label.pack()
    window.mainloop()

