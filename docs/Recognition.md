# Giải thích `Recognition.py`

File này chứa lớp `FaceRecognition` — phần lõi xử lý nhận diện khuôn mặt của dự án, dựa trên thư viện [`face_recognition`](https://github.com/ageitgey/face_recognition) (xây trên `dlib`).

## Import

```python
import cv2
import face_recognition
import os, sys
import numpy as np
import math
```

- `cv2`: đọc khung hình từ webcam, vẽ khung/chữ lên ảnh, hiển thị cửa sổ.
- `face_recognition`: phát hiện vị trí khuôn mặt, mã hoá khuôn mặt thành vector đặc trưng, so khớp và tính khoảng cách.
- `numpy`: tìm chỉ số nhỏ nhất trong mảng khoảng cách (`np.argmin`).
- `math`: dùng trong công thức tính độ tin cậy.

## Hàm `face_confidence()`

```python
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"
```

Chuyển **face_distance** (khoảng cách Euclidean giữa hai vector 128 chiều đại diện khuôn mặt) thành **% độ tin cậy** để hiển thị cho người dùng.

### `face_distance` là gì?

`face_recognition` mã hoá mỗi khuôn mặt thành một vector 128 chiều (embedding). Khi so khớp, nó tính khoảng cách Euclidean giữa vector khuôn mặt đang xét và vector đã lưu:

- `face_distance` càng **nhỏ** → hai khuôn mặt càng **giống nhau**.
- `face_distance = 0` → giống hệt.
- Ngưỡng mặc định `face_match_threshold = 0.6`: nếu distance ≤ 0.6 thì coi là "match" (cùng một người).

Vấn đề: distance là một con số nghịch (nhỏ = tốt), không trực quan với người dùng. Hàm `face_confidence` quy đổi nó thành % thuận (lớn = tốt).

### Nhánh `else` (trường hợp match — distance ≤ 0.6, đây là trường hợp chính)

```python
range = (1.0 - face_match_threshold)                  # = 1.0 - 0.6 = 0.4
linear_val = (1.0 - face_distance) / (range * 2.0)     # = (1 - distance) / 0.8
```

**Bước 1 — `linear_val`**: quy đổi tuyến tính distance ∈ [0, 0.6] thành một giá trị:
- distance = 0 → `linear_val = 1/0.8 = 1.25`
- distance = 0.6 → `linear_val = 0.4/0.8 = 0.5`

Nghĩa là: distance = ngưỡng (0.6, "vừa đủ match") → `linear_val = 0.5` (50%). Distance = 0 (giống hệt) → `linear_val = 1.25` (>100%, sẽ được nén lại ở bước 2).

**Bước 2 — làm cong (nén) đường cong:**

```python
val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
```

Đây là một công thức "làm đẹp" (empirical, không có ý nghĩa toán học chuẩn) để:
- Kéo các giá trị `linear_val` gần 0.5 (match yếu, sát ngưỡng) lên cao hơn một chút, tránh hiển thị % quá thấp gây hiểu lầm "không giống".
- Nén các giá trị `linear_val` > 1 (match rất tốt) về gần 100% thay vì vượt quá 100%.

Nói cách khác: đây chỉ là một phép ánh xạ phi tuyến do tác giả gốc (`Dedeepya Yarlagadda`) tự chế để số % nhìn "hợp lý" hơn, **không phải xác suất thống kê thật sự**.

### Nhánh `if` (trường hợp không match — distance > 0.6)

```python
return str(round(linear_val * 100, 2)) + "%"
```

Chỉ dùng công thức tuyến tính đơn giản (không làm cong), vì trường hợp này là "Unknown" nên độ chính xác của con số % không quan trọng lắm — chỉ để tham khảo.

### Bảng tóm tắt

| Distance | Ý nghĩa | Confidence % |
|---|---|---|
| 0.0 | Giống hệt | ~100% |
| 0.6 | Ngưỡng match | ~50% |
| > 0.6 | Không match | < 50%, tính đơn giản hơn |

**Điểm mấu chốt**: đây **không phải xác suất do model học được**, mà là một công thức chuyển đổi thủ công (heuristic) từ khoảng cách vector sang phần trăm, mục đích chỉ để hiển thị UI cho dễ hiểu — con số càng cao thì hai khuôn mặt càng giống nhau theo cách đo của `face_recognition`, chứ không phải "AI chắc chắn X%" theo nghĩa thống kê.

## Lớp `FaceRecognition`

```python
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
```

Các thuộc tính này được khai báo ở **cấp lớp** (class-level), đóng vai trò state dùng chung xuyên suốt vòng lặp nhận diện:

| Thuộc tính | Ý nghĩa |
|---|---|
| `face_locations` | Toạ độ (top, right, bottom, left) của các khuôn mặt phát hiện được trong khung hình hiện tại |
| `face_encodings` | Vector đặc trưng 128 chiều của từng khuôn mặt trong khung hình hiện tại |
| `face_names` | Tên (+ độ tin cậy) tương ứng với từng khuôn mặt, để hiển thị lên ảnh |
| `known_face_encodings` | Danh sách vector đặc trưng của các khuôn mặt đã lưu (dữ liệu "đã biết") |
| `known_face_names` | Tên tương ứng với `known_face_encodings` (chính là tên file ảnh trong `detect/`) |
| `process_current_frame` | Cờ để chỉ xử lý nhận diện trên **1 trong 2** khung hình, giúp giảm tải CPU |

> **Lưu ý kỹ thuật**: đây là thuộc tính class chứ không phải `self.x` khởi tạo trong `__init__`, nên về lý thuyết chúng được **chia sẻ giữa mọi instance** của `FaceRecognition`. Với cách dùng hiện tại (chỉ tạo 1 instance) thì không gây vấn đề, nhưng đây không phải thực hành tốt — nên khởi tạo trong `__init__` bằng `self.face_locations = []`, v.v.

### `__init__(self)`

```python
def __init__(self):
    self.encode_faces()
```

Khi tạo `FaceRecognition()`, lập tức gọi `encode_faces()` để nạp toàn bộ dữ liệu khuôn mặt đã lưu.

### `encode_faces(self)`

```python
def encode_faces(self):
    for image in os.listdir('detect'):
        face_image = face_recognition.load_image_file(f'detect/{image}')
        face_encoding = face_recognition.face_encodings(face_image)[0]

        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(image)
    print(self.known_face_names)
```

- Duyệt qua mọi file ảnh trong thư mục `detect/` (nơi `Main.py` lưu ảnh khi người dùng bấm SPACEBAR).
- Với mỗi ảnh: nạp ảnh → tìm vector đặc trưng khuôn mặt đầu tiên trong ảnh (`face_encodings(...)[0]`) → lưu vào `known_face_encodings`.
- Tên hiển thị (`known_face_names`) chính là **tên file**, ví dụ `An_0.jpg`.

⚠️ **Rủi ro tiềm ẩn**:
- Nếu một ảnh trong `detect/` không có khuôn mặt nào, `face_encodings(face_image)` trả về list rỗng → `[0]` sẽ ném `IndexError` và làm crash chương trình.
- Nếu `detect/` chứa file không phải ảnh (ví dụ `.DS_Store` trên macOS), `load_image_file` cũng sẽ lỗi.

### `run_recognition(self)`

Vòng lặp chính, chạy webcam và nhận diện theo thời gian thực:

```python
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    sys.exit('Camera not found')
```

Mở webcam mặc định (index `0`). Nếu không mở được thì thoát chương trình.

```python
while True:
    ret, frame = video_capture.read()

    if self.process_current_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        ...
    self.process_current_frame = not self.process_current_frame
```

- **Tối ưu hiệu năng**: chỉ chạy nhận diện (bước tốn CPU nhất) trên **mỗi khung hình thứ 2** (`process_current_frame` đảo trạng thái mỗi vòng lặp). Khung hình còn lại chỉ hiển thị lại kết quả cũ.
- Ảnh được resize xuống còn **25%** kích thước gốc (`fx=0.25, fy=0.25`) trước khi nhận diện, giúp giảm đáng kể thời gian xử lý.
- OpenCV đọc ảnh theo thứ tự màu **BGR**, trong khi `face_recognition` cần **RGB**, nên phải chuyển đổi bằng `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.

```python
self.face_locations = face_recognition.face_locations(rgb_small_frame)
self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
```

Phát hiện vị trí tất cả khuôn mặt trong khung hình nhỏ, rồi mã hoá từng khuôn mặt thành vector đặc trưng.

```python
for face_encoding in self.face_encodings:
    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
    name = 'Unknown'
    confidence = 'Unknown'

    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = self.known_face_names[best_match_index]
        confidence = face_confidence(face_distances[best_match_index])
    self.face_names.append(f'{name} {confidence}')
```

Với mỗi khuôn mặt phát hiện được trong khung hình:
1. `compare_faces`: so khớp với toàn bộ `known_face_encodings`, trả về mảng `True/False`.
2. `face_distance`: tính khoảng cách tới từng khuôn mặt đã biết.
3. `np.argmin`: tìm khuôn mặt đã biết **gần nhất** (khoảng cách nhỏ nhất).
4. Nếu khuôn mặt gần nhất đó cũng nằm trong danh sách match (`matches[best_match_index] == True`) → gán tên và tính confidence. Ngược lại giữ `name = 'Unknown'`.

```python
for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
```

- Vì toạ độ được tính trên khung hình đã resize 25%, cần **nhân lại 4** (= 1/0.25) để quy đổi về toạ độ trên khung hình gốc.
- Vẽ khung chữ nhật đỏ quanh khuôn mặt, vẽ thêm nhãn tên phía dưới khung.

```python
cv2.imshow('Face Recognition', frame)
if cv2.waitKey(1) % 256 == 27:  # ESC pressed
    break
```

Hiển thị khung hình kết quả; nhấn `ESC` (mã 27) để thoát vòng lặp.

```python
video_capture.release()
cv2.destroyAllWindows()
```

Giải phóng webcam và đóng toàn bộ cửa sổ OpenCV khi kết thúc.

## Tóm tắt luồng xử lý

```
Khởi tạo FaceRecognition()
   → encode_faces(): nạp & mã hoá toàn bộ ảnh trong detect/
        ↓
run_recognition()
   → mở webcam
   → vòng lặp: đọc khung hình → (mỗi khung thứ 2) resize + phát hiện + mã hoá khuôn mặt
        → so khớp với known_face_encodings → chọn khuôn mặt gần nhất
        → nếu match: gán tên + % confidence, ngược lại: "Unknown"
   → vẽ khung + tên lên khung hình gốc → hiển thị
   → ESC để thoát
```
