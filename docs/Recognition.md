# Giải thích `Recognition.py`

File này chứa lớp `FaceRecognition` — phần lõi xử lý nhận diện khuôn mặt của dự án, dựa trên thư viện [`face_recognition`](https://github.com/ageitgey/face_recognition) (xây trên `dlib`).

## Import và cấu hình

```python
import math, os, pickle, threading, time
import cv2
import face_recognition
import numpy as np

from config import (CACHE_PATH, CACHE_VERSION, DETECT_DIR, DETECT_SCALE, IMAGE_EXTENSIONS,
                    INV_SCALE, MATCH_THRESHOLD, PROCESS_EVERY_N)
```

- `cv2`: đọc khung hình từ webcam, vẽ khung/chữ lên ảnh, hiển thị cửa sổ.
- `face_recognition`: phát hiện vị trí khuôn mặt, mã hoá khuôn mặt thành vector đặc trưng, tính khoảng cách.
- `numpy`: tìm chỉ số nhỏ nhất trong mảng khoảng cách (`np.argmin`).
- `math`: dùng trong công thức tính độ tin cậy.
- `threading`: chạy việc đọc webcam trên 1 thread riêng (xem [`VideoStream`](#lớp-videostream-đọc-webcam-bất-đồng-bộ)) để tránh giật/lag.
- `time`: đo thời gian giữa các khung hình để tính FPS hiển thị trên màn hình.
- `pickle`: lưu/đọc cache encoding khuôn mặt (xem [cache encoding](#load_known_faces-và-cache-encoding)).
- **`config.py`**: mọi hằng số dùng chung (thư mục `detect/`, ngưỡng `MATCH_THRESHOLD = 0.6`, tỉ lệ thu nhỏ `DETECT_SCALE = 0.5`, `PROCESS_EVERY_N`, ...) nằm ở một nơi, dùng chung với `Main.py`, `capture.py` và `Check_Detect.py` để giá trị không bị lệch giữa các file.

## Hàm `face_confidence()`

```python
def face_confidence(face_distance, face_match_threshold=MATCH_THRESHOLD):
    span = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (span * 2.0)

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
span = (1.0 - face_match_threshold)                   # = 1.0 - 0.6 = 0.4
linear_val = (1.0 - face_distance) / (span * 2.0)      # = (1 - distance) / 0.8
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
| 0.0 | Giống hệt | 97.89% |
| 0.2 | Rất giống | 100% (đỉnh) |
| 0.4 | Giống | 96.76% |
| 0.5 | Khá giống | 90.92% |
| 0.6 | Ngưỡng match | 50% |
| 0.7 | Không match | 37.5% (công thức tuyến tính đơn giản) |

> **Đặc điểm cần biết**: đây là công thức heuristic nên **không đơn điệu**. Độ tin cậy đạt đỉnh 100% ở distance ≈ 0.2 rồi *giảm nhẹ* khi khuôn mặt giống hơn (distance 0.0 chỉ hiện 97.89%), và có "bước nhảy" lớn từ ~91% (distance 0.5) xuống 50% (distance 0.6) ngay tại ngưỡng. Vì vậy chỉ nên coi con số này là chỉ báo tương đối. Hành vi này được ghi lại trong test `test_known_quirk_peak_is_at_0_2_not_0` ([tests/test_recognition.py](../tests/test_recognition.py)).

**Điểm mấu chốt**: đây **không phải xác suất do model học được**, mà là một công thức chuyển đổi thủ công (heuristic) từ khoảng cách vector sang phần trăm, mục đích chỉ để hiển thị UI cho dễ hiểu — con số càng cao thì hai khuôn mặt càng giống nhau theo cách đo của `face_recognition`, chứ không phải "AI chắc chắn X%" theo nghĩa thống kê.

## Lớp `VideoStream` (đọc webcam bất đồng bộ)

```python
class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.stream.isOpened():
            self.stream.release()
            raise CameraError('Không tìm thấy hoặc không mở được webcam')

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

- `__init__`: mở camera, **set độ phân giải tường minh** (`width=640, height=480` — mặc định) để giảm tải cho cả bước đọc lẫn resize sau này, rồi khởi động thread nền `_update`. Nếu không mở được camera thì ném `CameraError` (thay vì `sys.exit` như trước, vốn chỉ ném `SystemExit` khó hiểu khi chạy trong callback của Tkinter); `Main.py` bắt lỗi này và hiện hộp thoại báo lỗi.
- `_update` (chạy trên thread nền): liên tục gọi `stream.read()` trong vòng lặp riêng, **luôn ghi đè** `self.frame` bằng khung hình mới nhất — **không dùng queue**, nên không có chuyện dồn (backlog) khung hình cũ.
- `read()` (gọi từ thread chính): lấy khung hình mới nhất hiện có tại thời điểm gọi. Dùng `threading.Lock` để tránh race condition khi thread nền đang ghi đè `self.frame` cùng lúc thread chính đang đọc; `.copy()` đảm bảo thread chính có bản sao riêng, không bị thread nền ghi đè giữa chừng khi đang xử lý.
- `stop()`: báo dừng thread nền, chờ nó kết thúc (`join`), rồi giải phóng camera — tránh thread bị treo hoặc lỗi khi đóng chương trình.

Kết quả: dù bước nhận diện AI chạy chậm, luồng hiển thị luôn lấy được khung hình **mới nhất** thay vì phải xử lý hết các khung cũ đã lỗi thời — hình ảnh mượt và ổn định hơn hẳn.

## Lớp `FaceRecognition`

```python
class FaceRecognition:
    PROCESS_EVERY_N = PROCESS_EVERY_N  # lấy từ config.py (mặc định 3)

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
| `PROCESS_EVERY_N` | Chỉ chạy nhận diện trên **1 trong mỗi N khung hình** (mặc định `3`, cấu hình trong `config.py`). Tăng giá trị này để giảm tải CPU trên máy yếu, đổi lại độ trễ nhận diện tăng nhẹ. |
| `face_locations` | Toạ độ (top, right, bottom, left) của các khuôn mặt phát hiện được trong khung hình hiện tại |
| `face_encodings` | Vector đặc trưng 128 chiều của từng khuôn mặt trong khung hình hiện tại |
| `face_names` | Tên (+ độ tin cậy) tương ứng với từng khuôn mặt, để hiển thị lên ảnh |
| `known_face_encodings` | Danh sách vector đặc trưng của các khuôn mặt đã lưu (dữ liệu "đã biết") |
| `known_face_names` | Tên tương ứng với `known_face_encodings` (chính là tên file ảnh trong `detect/`) |
| `frame_count` | Bộ đếm số khung hình đã xử lý, dùng để quyết định khung nào chạy nhận diện (`frame_count % PROCESS_EVERY_N == 0`) |

> Các thuộc tính state (`face_locations`, `face_encodings`, ...) nay được khởi tạo trong `__init__` bằng `self.x = ...` thay vì khai báo ở cấp lớp như trước — mỗi instance của `FaceRecognition` có state riêng, tránh rủi ro chia sẻ dữ liệu ngoài ý muốn giữa các instance.

### `__init__(self)`

Khởi tạo toàn bộ state rỗng cho instance, rồi gọi `encode_faces()` để nạp dữ liệu khuôn mặt đã lưu.

### `load_known_faces()` và cache encoding

`encode_faces()` chỉ gọi hàm module `load_known_faces(detect_dir, cache_path, encoder)` (nhận đường dẫn và hàm mã hoá làm tham số nên test được mà không cần ảnh/webcam thật) rồi in tóm tắt. Hàm này nạp encoding của mọi ảnh trong `detect/`. Bước tốn kém nhất là `face_recognition.face_encodings()` (chạy model deep learning để biến ảnh thành vector 128 số, cỡ vài trăm ms mỗi ảnh), và kết quả không đổi nếu ảnh không đổi. Vì vậy encoding được **cache** trong file `detect/.encodings.pkl`:

```python
cached = cache.get(image)
if cached and cached['mtime'] == mtime:
    encoding = cached['encoding']   # ảnh không đổi -> dùng lại
else:
    encoding = encoder(path)        # ảnh mới/đã sửa -> mã hoá (encode_image)
```

- **Khoá cache là `mtime`** (thời điểm sửa file): thêm ảnh mới chỉ mã hoá ảnh đó; chụp lại/sửa ảnh cùng tên thì `mtime` đổi nên mã hoá lại; ảnh đã xoá tự biến mất khỏi cache vì cache được ghi lại theo danh sách ảnh hiện có.
- **Chỉ xét file `.jpg/.jpeg/.png`**, nên `.DS_Store` và chính file cache bị bỏ qua.
- **Ảnh lỗi không còn làm crash**: `encode_image()` trả về `None` cho ảnh không có khuôn mặt hoặc không đọc được (bắt `OSError`/`ValueError`, không nuốt mọi lỗi). Ảnh đó bị bỏ qua và được cảnh báo khi khởi động; kết quả cũng được cache nên không phải thử lại mỗi lần.
- **An toàn**: file cache hỏng, không có, hoặc khác `CACHE_VERSION` thì tự mã hoá lại từ đầu; ghi cache qua file tạm rồi `os.replace` để không bị hỏng nếu tắt giữa chừng. Khi đổi cách mã hoá, tăng `CACHE_VERSION` để bỏ cache cũ.
- File cache nằm trong `detect/` nên đã được `.gitignore` bỏ qua.

Khi khởi động, chương trình in tóm tắt, ví dụ `Đã nạp 12 ảnh (11 từ cache, 1 mã hoá mới)`.

### `match_face()` — so khớp một khuôn mặt

```python
def match_face(encoding, known_encodings, known_names, threshold=MATCH_THRESHOLD):
    if len(known_encodings) == 0:
        return 'Unknown', 'Unknown'

    distances = face_recognition.face_distance(known_encodings, encoding)
    best = int(np.argmin(distances))
    if distances[best] <= threshold:
        return known_names[best], face_confidence(distances[best], threshold)
    return 'Unknown', 'Unknown'
```

Hàm thuần (không đụng webcam/ảnh) nên dễ test:
1. Không có khuôn mặt đã lưu → `Unknown` (không crash như `np.argmin` trên mảng rỗng).
2. `face_distance` tính khoảng cách tới từng khuôn mặt đã biết, `np.argmin` chọn khuôn mặt **gần nhất**.
3. Nếu khoảng cách ≤ ngưỡng (bao gồm cả biên) → trả tên và độ tin cậy; ngược lại `Unknown`.

Cách này tương đương `compare_faces` (vốn chỉ so `distance <= 0.6`) nhưng bỏ được lượt so khớp thừa.

### `recognize(self, frame)`

Dò và nhận diện khuôn mặt trong một khung hình, cập nhật `face_locations` và `face_names`. **Dò trên khung thu nhỏ, mã hoá trên khung gốc**:

```python
small_frame = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
full_locations = scale_locations(self.face_locations, INV_SCALE, frame.shape)
self.face_encodings = face_recognition.face_encodings(rgb_frame, full_locations)

self.face_names = []
for face_encoding in self.face_encodings:
    name, confidence = match_face(face_encoding, self.known_face_encodings, self.known_face_names)
    self.face_names.append(f'{name} {confidence}')
```

- **Dò trên khung thu nhỏ** (`DETECT_SCALE = 0.5`) vì dò là bước tốn theo số điểm ảnh. `face_locations` giữ toạ độ của khung nhỏ để `draw_faces` dùng.
- **Mã hoá trên khung gốc**: `scale_locations()` nhân toạ độ lên `INV_SCALE` và kẹp trong biên khung, rồi `face_encodings` mã hoá đúng vùng mặt trên ảnh độ phân giải đầy đủ. Mã hoá từ khung thu nhỏ làm mặt nhỏ bị mờ và vector lệch xa vector đã lưu.
- OpenCV đọc ảnh theo thứ tự màu **BGR**, trong khi `face_recognition` cần **RGB**, nên phải chuyển đổi bằng `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.
- `number_of_times_to_upsample=0` (mặc định của thư viện là `1`): bỏ bước phóng to ảnh để tìm mặt nhỏ, đổi lấy tốc độ.

#### Vì sao `DETECT_SCALE = 0.5` chứ không phải `0.25`?

Đo trên ảnh thật, giả lập khung webcam 640 rộng (upsample = 0):

| Mặt rộng trong khung 640 | `DETECT_SCALE = 0.25` | `DETECT_SCALE = 0.5` |
|---|---|---|
| ≤ 200 px (≤ 31% khung) | không dò ra | dò ra từ ~120 px |
| ≥ 250 px (≥ 39% khung) | dò ra | dò ra |
| Thời gian dò | ~1 ms | ~4 ms |

Với `0.25`, phải ngồi rất sát camera mới được nhận diện (`MIN_FACE_WIDTH` khi chụp ảnh cũng vô nghĩa vì mặt < ~250 px không được dò ra). Với `0.5`, tổng thời gian `recognize()` ~8 ms cho mỗi lần chạy (mỗi `PROCESS_EVERY_N` khung).

Với mặt nhỏ (~120 px), mã hoá trên khung gốc đưa khoảng cách tới chính người đó xuống rõ rệt (ví dụ 0.084 → 0.024 và 0.138 → 0.072 trong hai ảnh thử) mà khoảng cách tới người khác không đổi (~0.75). Mặt ≥ 250 px thì hai cách gần như như nhau. Lưu ý phép đo này dùng cùng một ảnh gốc cho đăng ký và thử nên là ước lượng lạc quan; đo đầy đủ bằng ảnh khác nhau dùng `evaluate.py` (hạng mục 5).

### `draw_faces()` — vẽ kết quả

```python
def draw_faces(frame, locations, names):
    for (top, right, bottom, left), name in zip(locations, names, strict=True):
        top *= INV_SCALE
        right *= INV_SCALE
        bottom *= INV_SCALE
        left *= INV_SCALE
        ...
```

- Toạ độ được tính trên khung hình đã thu nhỏ, nên **nhân lại `INV_SCALE`** (= 2, suy ra từ `DETECT_SCALE` chứ không viết cứng) để quy đổi về khung hình gốc.
- Vẽ khung chữ nhật đỏ quanh khuôn mặt và nhãn tên phía dưới khung.
- `zip(..., strict=True)`: `locations` và `names` luôn cùng độ dài; nếu lệch thì báo lỗi thay vì cắt bớt im lặng.

### `run_recognition(self)`

Vòng lặp điều phối, chạy webcam và nhận diện theo thời gian thực:

```python
video_stream = VideoStream(0)

while True:
    ret, frame = video_stream.read()
    if not ret or frame is None:
        if cv2.waitKey(1) % 256 == 27:   # vẫn cho phép ESC khi chưa có khung hình
            break
        continue

    if self.frame_count % self.PROCESS_EVERY_N == 0:
        self.recognize(frame)
    self.frame_count += 1

    draw_faces(frame, self.face_locations, self.face_names)
    ...
```

- `VideoStream` đọc camera trên thread nền (xem [phần trên](#lớp-videostream-đọc-webcam-bất-đồng-bộ)); `video_stream.read()` luôn trả về khung hình mới nhất. Nếu chưa có khung hình hợp lệ (có thể xảy ra ngay lúc mới khởi động camera), bỏ qua vòng lặp này thay vì crash, nhưng vẫn kiểm tra `ESC` để không bị "kẹt" không thoát được.
- **Tối ưu hiệu năng**: chỉ chạy nhận diện (bước tốn CPU nhất) khi `frame_count % PROCESS_EVERY_N == 0`, tức **1 trong mỗi 3 khung hình** theo mặc định. Khung còn lại chỉ vẽ lại kết quả gần nhất. Chỉ cần đổi `PROCESS_EVERY_N` để cân bằng giữa độ mượt và tải CPU tuỳ cấu hình máy.

```python
now = time.time()
fps = 1.0 / (now - prev_time) if now > prev_time else fps
prev_time = now
cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
```

Tính FPS (khung hình/giây) dựa trên thời gian trôi qua giữa 2 vòng lặp liên tiếp, rồi vẽ số liệu lên góc trái phía trên màn hình. Đây là công cụ chẩn đoán trực quan, hữu ích khi tinh chỉnh `PROCESS_EVERY_N` hoặc độ phân giải camera.

```python
cv2.imshow('Face Recognition', frame)
if cv2.waitKey(1) % 256 == 27:  # ESC pressed
    break

video_stream.stop()
cv2.destroyAllWindows()
```

Hiển thị khung hình kết quả; nhấn `ESC` (mã 27) để thoát, sau đó dừng thread nền, giải phóng webcam và đóng các cửa sổ OpenCV.

## Tóm tắt luồng xử lý

```
Khởi tạo FaceRecognition()
   → encode_faces() → load_known_faces(): nạp encoding từ cache, chỉ mã hoá ảnh mới/đã đổi
        ↓
run_recognition()
   → mở VideoStream (thread nền liên tục đọc khung hình mới nhất từ webcam)
   → vòng lặp (thread chính):
        → lấy khung hình mới nhất từ VideoStream
        → (mỗi PROCESS_EVERY_N khung) recognize(): dò trên khung thu nhỏ, mã hoá trên khung gốc
             → match_face(): chọn khuôn mặt gần nhất; ≤ ngưỡng thì gán tên + % confidence, ngược lại "Unknown"
        → draw_faces() + FPS lên khung hình gốc → hiển thị
   → ESC để thoát → dừng VideoStream, đóng cửa sổ
```

## Vì sao cách này giảm được lag/giật hình?

So với bản gốc (đọc + xử lý tuần tự trên 1 thread), các thay đổi trên nhắm vào 2 nguyên nhân chính gây giật/lag:

1. **Backlog khung hình do đọc/xử lý chung 1 thread** → tách `VideoStream` chạy nền, thread chính luôn lấy khung hình mới nhất thay vì phải xử lý hết hàng chờ khung cũ.
2. **Xử lý AI quá nặng trên mỗi khung** → giảm độ phân giải capture (640×480), tăng tỉ lệ bỏ khung có thể tinh chỉnh (`PROCESS_EVERY_N`), và bỏ bước upsample không cần thiết (`number_of_times_to_upsample=0`).

Nếu máy vẫn lag, có thể tăng `PROCESS_EVERY_N` (ví dụ lên `5`) hoặc giảm `width`/`height` khi khởi tạo `VideoStream` để giảm tải hơn nữa.
