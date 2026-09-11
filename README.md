# Face-Recog-Dlib

Ứng dụng nhận diện khuôn mặt qua webcam, xây dựng bằng Python với giao diện Tkinter đơn giản. Người dùng có thể thêm ảnh khuôn mặt mới vào hệ thống và sau đó nhận diện các khuôn mặt đó theo thời gian thực qua webcam.

## Tính năng

- **Thêm khuôn mặt vào hệ thống**: nhập tên, chụp ảnh khuôn mặt trực tiếp từ webcam và lưu lại để nhận diện sau này.
- **Nhận diện khuôn mặt theo thời gian thực**: mở webcam, phát hiện khuôn mặt trong khung hình, so khớp với dữ liệu đã lưu và hiển thị tên kèm độ tin cậy (%) ngay trên video.
- **Giao diện đồ họa (GUI)** đơn giản bằng Tkinter để thao tác mà không cần dùng dòng lệnh.

## Cấu trúc dự án

| File | Mô tả |
|---|---|
| [Main.py](Main.py) | Điểm khởi chạy chương trình. Hiển thị cửa sổ GUI với hai chức năng chính: thêm ảnh khuôn mặt và nhận diện khuôn mặt. |
| [Recognition.py](Recognition.py) | Lớp `FaceRecognition` — nạp ảnh khuôn mặt đã lưu trong thư mục `detect/`, mã hoá (encode) và so khớp khuôn mặt trực tiếp từ webcam bằng thư viện `face_recognition`. |
| [Simple Face Detection.py](Simple%20Face%20Detection.py) | Script độc lập, minh hoạ phát hiện khuôn mặt (không nhận diện danh tính) bằng Haar Cascade của OpenCV. |

## Yêu cầu cài đặt

- Python 3.x
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [face_recognition](https://pypi.org/project/face_recognition/) (dựa trên `dlib`)
- `numpy`
- `tkinter` (thường có sẵn trong bản cài Python chuẩn)

### Tạo môi trường ảo (khuyến nghị)

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

> **Lưu ý**: `face_recognition` phụ thuộc vào `dlib`, cần có `cmake` và trình biên dịch C++ để build. Trên Windows nên cài qua [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) hoặc dùng bản `dlib` wheel dựng sẵn.
>
> Trên Python 3.14, `setuptools` mới đã bỏ `pkg_resources` khiến `face_recognition_models` báo lỗi khi import — `requirements.txt` đã ghim `setuptools<81` để tránh lỗi này.

## Hướng dẫn sử dụng

### Trước khi chạy

Tạo thư mục `detect/` trong thư mục gốc của dự án — đây là nơi chương trình lưu ảnh khuôn mặt đã chụp:

```bash
mkdir detect
```

Nhận diện khuôn mặt có thể chạy chậm trên máy cấu hình yếu, thậm chí có thể bị lag hoặc không chạy được.

### Chạy chương trình

```bash
python Main.py
```

Cửa sổ GUI sẽ hiện ra với hai nút bấm:

1. **Thêm ảnh vào hệ thống**
   - Nhập tên khi được hỏi.
   - Nhấn phím `SPACEBAR` để chụp ảnh khuôn mặt (có thể chụp nhiều lần).
   - Nhấn `ESC` để đóng cửa sổ webcam khi đã chụp xong.

2. **Nhận diện khuôn mặt**
   - Mở webcam và nhận diện các khuôn mặt đã lưu trong `detect/`, hiển thị tên và độ tin cậy trên khung hình.
   - Nhấn `ESC` để thoát.

### Chạy thử phát hiện khuôn mặt đơn giản (tuỳ chọn)

```bash
python "Simple Face Detection.py"
```

Script này chỉ khoanh vùng khuôn mặt bằng khung chữ nhật (không nhận diện danh tính), dùng để kiểm tra nhanh webcam/OpenCV hoạt động tốt. Nhấn `q` để thoát.

## Ghi công

Chương trình có sử dụng code tham khảo từ:

- [Federico Azzurro](https://github.com/federicoazzu/webcam_face_recognition)
- [Dedeepya Yarlagadda](https://github.com/Dedepya/Face-Recognition-Using-SVM)
