# Giải thích `Simple Face Detection.py`

File này là một script **độc lập**, minh hoạ cách **phát hiện khuôn mặt** (face *detection* — chỉ tìm vị trí khuôn mặt trong ảnh) bằng thuật toán **Haar Cascade** có sẵn trong OpenCV. Đây **không phải** nhận diện danh tính (face *recognition*) như [Recognition.py](Recognition.md) — nó chỉ vẽ khung quanh mọi khuôn mặt tìm được, không biết đó là ai.

Script này thường dùng để kiểm tra nhanh webcam/OpenCV hoạt động tốt trước khi chạy chương trình chính, hoặc để so sánh với cách tiếp cận dùng `face_recognition`/`dlib` trong `Recognition.py`.

## Toàn bộ code

```python
import pathlib
import cv2

# Load the cascade
cascade = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade))
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

## Giải thích từng phần

### 1. Nạp mô hình Haar Cascade

```python
cascade = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade))
```

- OpenCV đi kèm sẵn một bộ các file `.xml` mô tả mô hình **Haar Cascade** — một thuật toán phát hiện đối tượng cổ điển (Viola–Jones), được huấn luyện sẵn để nhận ra các đặc trưng của khuôn mặt người nhìn thẳng (frontal face).
- `cv2.__file__` trỏ tới file cài đặt của package `cv2` (ví dụ `.../site-packages/cv2/__init__.py`). Lấy thư mục cha của nó (`.parent`) rồi nối thêm `data/haarcascade_frontalface_default.xml` để tìm ra đường dẫn tới file mô hình đi kèm theo package.
- `cv2.CascadeClassifier(...)` nạp mô hình này thành một bộ phân loại (`clf`) có thể dùng để dò khuôn mặt trong ảnh xám.

> **Lưu ý tương thích**: hàm `cv2.CascadeClassifier` chỉ tồn tại ở OpenCV **4.x**. Bản **OpenCV 5.x** đã loại bỏ API này khỏi Python bindings, nên script sẽ báo lỗi `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'` nếu cài `opencv-python` mới nhất. Dự án này đã ghim `opencv-python<5` trong `requirements.txt` để tránh lỗi đó.

### 2. Mở webcam

```python
cam = cv2.VideoCapture(0)
```

Mở thiết bị camera mặc định (index `0`, thường là webcam tích hợp/gắn ngoài đầu tiên).

### 3. Vòng lặp xử lý từng khung hình

```python
while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

- `cam.read()` trả về `(ret, frame)`; ở đây bỏ qua `ret` (dùng `_`) — script **không kiểm tra** xem đọc khung hình có thành công hay không (khác với `Recognition.py` có kiểm tra `ret`). Nếu webcam lỗi, `frame` có thể là `None` và dòng tiếp theo sẽ crash.
- Chuyển ảnh sang **thang xám** (`grayscale`) vì Haar Cascade chỉ hoạt động trên ảnh xám (không cần thông tin màu để tìm đặc trưng sáng/tối của khuôn mặt).

### 4. Phát hiện khuôn mặt

```python
faces = clf.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

`detectMultiScale` quét ảnh ở nhiều tỉ lệ khác nhau để tìm mọi vùng "giống khuôn mặt". Các tham số:

| Tham số | Ý nghĩa |
|---|---|
| `scaleFactor=1.1` | Sau mỗi lần quét, thu nhỏ ảnh đi 10% để tìm khuôn mặt ở kích thước khác. Giá trị càng gần 1 càng chính xác nhưng càng chậm. |
| `minNeighbors=5` | Một vùng phải được ít nhất 5 "ứng viên" chồng lấn xác nhận thì mới được coi là khuôn mặt thật — tăng giá trị này giảm nhận diện nhầm (false positive) nhưng có thể bỏ sót khuôn mặt thật. |
| `minSize=(30, 30)` | Bỏ qua mọi vùng nhỏ hơn 30×30 pixel (loại các phát hiện nhiễu quá nhỏ). |
| `flags=cv2.CASCADE_SCALE_IMAGE` | Cờ tối ưu hoá cho thuật toán quét theo tỉ lệ ảnh (thay vì scale kernel), cách làm phổ biến/khuyến nghị hiện nay. |

Kết quả `faces` là danh sách các tuple `(x, y, w, h)`: toạ độ góc trên-trái và chiều rộng/cao của từng khuôn mặt phát hiện được.

### 5. Vẽ khung lên khuôn mặt

```python
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
```

Với mỗi khuôn mặt, vẽ một khung chữ nhật **màu vàng** (`(255, 255, 0)` theo BGR ≈ xanh dương + xanh lá = vàng), độ dày viền `2`, từ góc `(x, y)` tới `(x+w, y+h)`.

### 6. Hiển thị & thoát

```python
cv2.imshow('Face Recognition', frame)
if cv2.waitKey(1) == ord('q'):
    break
```

Hiển thị khung hình (kể cả khi không phát hiện khuôn mặt nào, `frame` gốc vẫn được show). Nhấn phím **`q`** để thoát vòng lặp.

> Khác với `Recognition.py` (dùng phím `ESC`), script này dùng phím `q` để thoát — cần lưu ý sự khác biệt khi hướng dẫn người dùng.

```python
cam.release()
cv2.destroyAllWindows()
```

Giải phóng webcam và đóng cửa sổ hiển thị.

## So sánh nhanh với `Recognition.py`

| | `Simple Face Detection.py` | `Recognition.py` |
|---|---|---|
| Mục đích | Chỉ **phát hiện** vị trí khuôn mặt | **Nhận diện danh tính** khuôn mặt |
| Thuật toán | Haar Cascade (OpenCV, cổ điển) | Deep learning embedding (`dlib`/`face_recognition`) |
| Biết tên người? | Không | Có (so khớp với dữ liệu đã lưu trong `detect/`) |
| Độ chính xác | Thấp hơn, dễ nhầm góc nghiêng/ánh sáng | Cao hơn đáng kể |
| Tốc độ | Nhanh, nhẹ | Chậm hơn (đã được tối ưu bằng resize 25% + xử lý cách khung) |
| Phím thoát | `q` | `ESC` |
