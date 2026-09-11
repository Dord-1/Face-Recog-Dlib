# Giải thích `Recognition.py`

File này chứa lớp `FaceRecognition` — phần lõi xử lý nhận diện khuôn mặt của dự án, dựa trên thư viện [`face_recognition`](https://github.com/ageitgey/face_recognition) (xây trên `dlib`).

## Import

```python
import cv2
import face_recognition
import os, sys
import numpy as np
import math
import threading
import time
```

- `cv2`: đọc khung hình từ webcam, vẽ khung/chữ lên ảnh, hiển thị cửa sổ.
- `face_recognition`: phát hiện vị trí khuôn mặt, mã hoá khuôn mặt thành vector đặc trưng, so khớp và tính khoảng cách.
- `numpy`: tìm chỉ số nhỏ nhất trong mảng khoảng cách (`np.argmin`).
- `math`: dùng trong công thức tính độ tin cậy.
- `threading`: chạy việc đọc webcam trên 1 thread riêng (xem [`VideoStream`](#lớp-videostream-đọc-webcam-bất-đồng-bộ)) để tránh giật/lag.
- `time`: đo thời gian giữa các khung hình để tính FPS hiển thị trên màn hình.

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

## Lớp `VideoStream` (đọc webcam bất đồng bộ)

```python
class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.stream.isOpened():
            sys.exit('Camera not found')

        self.lock = threading.Lock()
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.stream.release()
```

### Vì sao cần class này?

Ban đầu code đọc webcam trực tiếp bằng `cv2.VideoCapture(0).read()` **ngay trong** vòng lặp xử lý nhận diện. Vấn đề: `cam.read()` và bước nhận diện (`face_locations`, `face_encodings`...) chạy **tuần tự trên cùng 1 thread**. Khi bước nhận diện chậm hơn tốc độ camera sinh khung hình, chương trình buộc phải "trả nợ" các khung hình cũ đang dồn trong buffer của driver camera trước khi đọc được khung hình mới nhất — gây cảm giác hình ảnh bị **trễ và giật cục** dù CPU vẫn đang chạy.

`VideoStream` giải quyết vấn đề này bằng cách tách việc đọc camera ra **1 thread nền riêng**:

- `__init__`: mở camera, **set độ phân giải tường minh** (`width=640, height=480` — mặc định) để giảm tải cho cả bước đọc lẫn resize sau này, rồi khởi động thread nền `_update`.
- `_update` (chạy trên thread nền): liên tục gọi `stream.read()` trong vòng lặp riêng, **luôn ghi đè** `self.frame` bằng khung hình mới nhất — **không dùng queue**, nên không có chuyện dồn (backlog) khung hình cũ.
- `read()` (gọi từ thread chính): lấy khung hình mới nhất hiện có tại thời điểm gọi. Dùng `threading.Lock` để tránh race condition khi thread nền đang ghi đè `self.frame` cùng lúc thread chính đang đọc; `.copy()` đảm bảo thread chính có bản sao riêng, không bị thread nền ghi đè giữa chừng khi đang xử lý.
- `stop()`: báo dừng thread nền, chờ nó kết thúc (`join`), rồi giải phóng camera — tránh thread bị treo hoặc lỗi khi đóng chương trình.

Kết quả: dù bước nhận diện AI chạy chậm, luồng hiển thị luôn lấy được khung hình **mới nhất** thay vì phải xử lý hết các khung cũ đã lỗi thời — hình ảnh mượt và ổn định hơn hẳn.

## Lớp `FaceRecognition`

```python
class FaceRecognition:
    PROCESS_EVERY_N = 3

    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_count = 0
        self.encode_faces()
```

| Thuộc tính | Ý nghĩa |
|---|---|
| `PROCESS_EVERY_N` | Hằng số cấp lớp: chỉ chạy nhận diện trên **1 trong mỗi N khung hình** (mặc định `3`). Tăng giá trị này để giảm tải CPU trên máy yếu, đổi lại độ trễ nhận diện tăng nhẹ. |
| `face_locations` | Toạ độ (top, right, bottom, left) của các khuôn mặt phát hiện được trong khung hình hiện tại |
| `face_encodings` | Vector đặc trưng 128 chiều của từng khuôn mặt trong khung hình hiện tại |
| `face_names` | Tên (+ độ tin cậy) tương ứng với từng khuôn mặt, để hiển thị lên ảnh |
| `known_face_encodings` | Danh sách vector đặc trưng của các khuôn mặt đã lưu (dữ liệu "đã biết") |
| `known_face_names` | Tên tương ứng với `known_face_encodings` (chính là tên file ảnh trong `detect/`) |
| `frame_count` | Bộ đếm số khung hình đã xử lý, dùng để quyết định khung nào chạy nhận diện (`frame_count % PROCESS_EVERY_N == 0`) |

> Các thuộc tính state (`face_locations`, `face_encodings`, ...) nay được khởi tạo trong `__init__` bằng `self.x = ...` thay vì khai báo ở cấp lớp như trước — mỗi instance của `FaceRecognition` có state riêng, tránh rủi ro chia sẻ dữ liệu ngoài ý muốn giữa các instance.

### `__init__(self)`

Khởi tạo toàn bộ state rỗng cho instance, rồi gọi `encode_faces()` để nạp dữ liệu khuôn mặt đã lưu.

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
video_stream = VideoStream(0)
```

Mở webcam qua `VideoStream` (thay vì `cv2.VideoCapture` trực tiếp như trước) để việc đọc camera chạy trên thread nền, tránh giật/lag như giải thích ở phần [`VideoStream`](#lớp-videostream-đọc-webcam-bất-đồng-bộ) phía trên.

```python
while True:
    ret, frame = video_stream.read()
    if not ret or frame is None:
        continue

    if self.frame_count % self.PROCESS_EVERY_N == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        ...
    self.frame_count += 1
```

- `video_stream.read()` luôn trả về khung hình mới nhất hiện có từ thread nền. Nếu chưa có khung hình hợp lệ (`ret` sai hoặc `frame` rỗng — có thể xảy ra ngay lúc mới khởi động camera), bỏ qua vòng lặp này (`continue`) thay vì crash.
- **Tối ưu hiệu năng**: chỉ chạy nhận diện (bước tốn CPU nhất) khi `frame_count % PROCESS_EVERY_N == 0`, tức **1 trong mỗi 3 khung hình** theo mặc định — thay cho cách cũ đảo bool xử lý 1/2 khung. Cách dùng bộ đếm chia hết cho phép tinh chỉnh linh hoạt hơn (chỉ cần đổi `PROCESS_EVERY_N`) để cân bằng giữa độ mượt và tải CPU tuỳ cấu hình máy.
- Ảnh được resize xuống còn **25%** kích thước gốc (`fx=0.25, fy=0.25`) trước khi nhận diện, giúp giảm đáng kể thời gian xử lý.
- OpenCV đọc ảnh theo thứ tự màu **BGR**, trong khi `face_recognition` cần **RGB**, nên phải chuyển đổi bằng `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.

```python
self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
```

Phát hiện vị trí tất cả khuôn mặt trong khung hình nhỏ, rồi mã hoá từng khuôn mặt thành vector đặc trưng.

- `number_of_times_to_upsample=0` (mặc định của thư viện là `1`): bỏ bước phóng to ảnh lên để tìm khuôn mặt nhỏ/xa camera, đổi lấy tốc độ nhanh hơn. Đánh đổi hợp lý cho use-case chính là người dùng đứng gần webcam.

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
now = time.time()
fps = 1.0 / (now - prev_time) if now > prev_time else fps
prev_time = now
cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
```

Tính FPS (khung hình/giây) dựa trên thời gian trôi qua giữa 2 vòng lặp liên tiếp, rồi vẽ số liệu lên góc trái phía trên màn hình. Đây là công cụ chẩn đoán trực quan: người dùng có thể tự quan sát tốc độ hiển thị thực tế, hữu ích khi tinh chỉnh `PROCESS_EVERY_N` hoặc độ phân giải camera cho phù hợp với cấu hình máy.

```python
cv2.imshow('Face Recognition', frame)
if cv2.waitKey(1) % 256 == 27:  # ESC pressed
    break
```

Hiển thị khung hình kết quả; nhấn `ESC` (mã 27) để thoát vòng lặp.

```python
video_stream.stop()
cv2.destroyAllWindows()
```

Dừng thread nền và giải phóng webcam (qua `VideoStream.stop()`), rồi đóng toàn bộ cửa sổ OpenCV khi kết thúc.

## Tóm tắt luồng xử lý

```
Khởi tạo FaceRecognition()
   → encode_faces(): nạp & mã hoá toàn bộ ảnh trong detect/
        ↓
run_recognition()
   → mở VideoStream (thread nền liên tục đọc khung hình mới nhất từ webcam)
   → vòng lặp (thread chính):
        → lấy khung hình mới nhất từ VideoStream
        → (mỗi PROCESS_EVERY_N khung) resize 25% + phát hiện + mã hoá khuôn mặt
             → so khớp với known_face_encodings → chọn khuôn mặt gần nhất
             → nếu match: gán tên + % confidence, ngược lại: "Unknown"
        → vẽ khung + tên + FPS lên khung hình gốc → hiển thị
   → ESC để thoát → dừng VideoStream, đóng cửa sổ
```

## Vì sao cách này giảm được lag/giật hình?

So với bản gốc (đọc + xử lý tuần tự trên 1 thread), các thay đổi trên nhắm vào 2 nguyên nhân chính gây giật/lag:

1. **Backlog khung hình do đọc/xử lý chung 1 thread** → tách `VideoStream` chạy nền, thread chính luôn lấy khung hình mới nhất thay vì phải xử lý hết hàng chờ khung cũ.
2. **Xử lý AI quá nặng trên mỗi khung** → giảm độ phân giải capture (640×480), tăng tỉ lệ bỏ khung có thể tinh chỉnh (`PROCESS_EVERY_N`), và bỏ bước upsample không cần thiết (`number_of_times_to_upsample=0`).

Nếu máy vẫn lag, có thể tăng `PROCESS_EVERY_N` (ví dụ lên `5`) hoặc giảm `width`/`height` khi khởi tạo `VideoStream` để giảm tải hơn nữa.
