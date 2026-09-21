# Face-Recog-Dlib

Ứng dụng nhận diện khuôn mặt qua webcam, xây dựng bằng Python với giao diện Tkinter đơn giản. Người dùng có thể thêm ảnh khuôn mặt mới vào hệ thống và sau đó nhận diện các khuôn mặt đó theo thời gian thực qua webcam.

## Tính năng

- **Thêm khuôn mặt vào hệ thống**: nhập tên, chụp ảnh khuôn mặt trực tiếp từ webcam và lưu lại để nhận diện sau này.
- **Nhận diện khuôn mặt theo thời gian thực**: mở webcam, phát hiện khuôn mặt trong khung hình, so khớp với dữ liệu đã lưu và hiển thị tên kèm độ tin cậy (%) ngay trên video.
- **Giao diện đồ họa (GUI)** đơn giản bằng Tkinter để thao tác mà không cần dùng dòng lệnh.

## Cấu trúc dự án

```
Main.py, Check_Detect.py, evaluate.py   # script mỏng ở gốc (giữ nguyên các lệnh chạy quen thuộc)
face_recog/                             # toàn bộ logic, mỗi module một trách nhiệm
├── config.py        # hằng số dùng chung (thư mục detect/, ngưỡng nhận diện, tỉ lệ thu nhỏ, ...)
├── geometry.py      # tiện ích khung mặt
├── camera.py        # VideoStream: đọc webcam ở thread riêng
├── known_faces.py   # ảnh đăng ký trong detect/, tên người, cache encoding
├── matching.py      # khoảng cách -> tên + độ tin cậy (%)
├── smoothing.py     # NameSmoother: bỏ phiếu để nhãn không nhấp nháy
├── recognizer.py    # điều phối nhận diện thời gian thực
├── quality.py       # kiểm tra chất lượng ảnh khi chụp (sáng, nét, nhìn thẳng)
├── capture.py       # chụp ảnh đăng ký từ webcam
├── gui.py           # cửa sổ chính (Tkinter)
└── tools/           # check_detect.py (dọn ảnh trùng/lỗi), evaluate.py (đo độ chính xác)
examples/            # simple_face_detection.py: demo Haar Cascade độc lập
tests/               # test pytest theo module (không cần webcam)
docs/                # tài liệu; bắt đầu từ docs/architecture.md
```

Bản đồ chi tiết, luật phụ thuộc giữa các module và "muốn sửa X thì vào đâu": [docs/architecture.md](docs/architecture.md).

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
python Main.py          # hoặc: python -m face_recog
```

Cửa sổ GUI sẽ hiện ra với hai nút bấm:

1. **Thêm ảnh vào hệ thống**
   - Nhập tên khi được hỏi.
   - Khung hình hiển thị viền **xanh** khi ảnh hợp lệ (đúng 1 khuôn mặt, đủ lớn, đủ sáng, nét, nhìn thẳng) và **đỏ** kèm lý do và số đo khi chưa hợp lệ (ngưỡng chỉnh trong [config.py](face_recog/config.py); chi tiết: [docs/capture.md](docs/capture.md)).
   - Nhấn phím `SPACEBAR` để chụp; ảnh chỉ được lưu khi viền xanh. Chụp lại cùng tên sẽ không ghi đè ảnh cũ.
   - Nhấn `ESC` để đóng cửa sổ webcam khi đã chụp xong.

2. **Nhận diện khuôn mặt**
   - Encoding khuôn mặt được cache trong `detect/.encodings.pkl` nên các lần mở sau khởi động nhanh; ảnh không có khuôn mặt bị bỏ qua thay vì gây lỗi.
   - Mở webcam và nhận diện các khuôn mặt đã lưu trong `detect/`, hiển thị tên và độ tin cậy trên khung hình. Nhãn hiện `...` vài phần giây đầu rồi mới hiện tên (bỏ phiếu qua nhiều lần nhận diện để không nhấp nháy).
   - Nhấn `ESC` để thoát.

### Chạy thử phát hiện khuôn mặt đơn giản (tuỳ chọn)

```bash
python examples/simple_face_detection.py
```

Script này chỉ khoanh vùng khuôn mặt bằng khung chữ nhật (không nhận diện danh tính), dùng để kiểm tra nhanh webcam/OpenCV hoạt động tốt. Nhấn `q` để thoát.

### Dọn dẹp thư mục `detect/` (tuỳ chọn)

Ảnh của cùng một người được nhận diện theo tên file (`Huy_0.jpg`, `Huy_1.jpg` → `Huy`); nhiều ảnh khác góc/ánh sáng giúp nhận diện tốt hơn. Nếu có ảnh trùng hệt nhau hoặc ảnh không chứa khuôn mặt, dùng script kiểm tra:

```bash
python Check_Detect.py            # chỉ báo cáo, không xoá gì
python Check_Detect.py --delete   # xoá sau khi xác nhận từng nhóm
```

Script gom các ảnh gần như giống hệt nhau, đề xuất giữ ảnh có khuôn mặt lớn nhất và chỉ xoá những ảnh bạn xác nhận. Chi tiết: [docs/check_detect.md](docs/check_detect.md).

### Đo độ chính xác nhận diện (tuỳ chọn)

Đặt ảnh test vào `eval/<tên>/*.jpg` (ảnh người lạ vào `eval/unknown/`) rồi chạy:

```bash
python evaluate.py --mode both
```

Công cụ báo tỉ lệ nhận đúng / nhận nhầm / bỏ sót theo từng ngưỡng và gợi ý `RECOGNITION_THRESHOLD`. Cách chuẩn bị dữ liệu và đọc kết quả: [docs/evaluate.md](docs/evaluate.md).

## Phát triển

Cài thêm công cụ dev (pytest, ruff) trong venv:

```bash
pip install -r requirements-dev.txt

pytest          # chạy test (không cần webcam)
ruff check .    # kiểm tra lint
```

Hằng số dùng chung (ngưỡng nhận diện, tỉ lệ thu nhỏ, số khung bỏ qua, ...) nằm trong [face_recog/config.py](face_recog/config.py) — sửa ở đó thay vì sửa rải rác nhiều file. Cấu trúc module và luật phụ thuộc (được `tests/test_architecture.py` kiểm tra) xem [docs/architecture.md](docs/architecture.md).

## Ghi công

Chương trình có sử dụng code tham khảo từ:

- [Federico Azzurro](https://github.com/federicoazzu/webcam_face_recognition)
- [Dedeepya Yarlagadda](https://github.com/Dedepya/Face-Recognition-Using-SVM)
