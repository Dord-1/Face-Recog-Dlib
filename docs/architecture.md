# Kiến trúc dự án

Tài liệu này là **bản đồ** để tìm đúng chỗ khi đọc hoặc sửa code. Chi tiết từng phần nằm ở các tài liệu chuyên đề (liên kết bên dưới).

## Cây thư mục

```
Face-Recog-Dlib/
├── Main.py                  # script mỏng: mở giao diện        (python Main.py)
├── Check_Detect.py          # script mỏng: dọn ảnh trùng/lỗi   (python Check_Detect.py)
├── evaluate.py              # script mỏng: đo độ chính xác     (python evaluate.py)
├── face_recog/              # toàn bộ logic
│   ├── config.py            # hằng số dùng chung
│   ├── geometry.py          # tiện ích khung mặt (scale_locations, face_area, largest_face)
│   ├── camera.py            # VideoStream (đọc webcam ở thread nền), CameraError
│   ├── known_faces.py       # ảnh đăng ký: person_name, cache encoding, load_known_faces
│   ├── matching.py          # face_confidence, match_face, format_label
│   ├── smoothing.py         # NameSmoother: bỏ phiếu qua nhiều lần nhận diện
│   ├── recognizer.py        # FaceRecognition, draw_faces: điều phối nhận diện thời gian thực
│   ├── quality.py           # đo độ sáng/độ nét/mặt nhìn thẳng, check_frame (không dùng Tkinter)
│   ├── settings.py          # cài đặt chỉnh từ GUI: ngưỡng nhận diện, tốc độ xử lý (detect/settings.json)
│   ├── activity_log.py      # nhật ký hoạt động: log_event/read_events (detect/activity.log)
│   ├── people.py            # quản lý người đăng ký: list_people, delete_person
│   ├── capture.py           # chụp ảnh đăng ký: hộp thoại + vòng lặp webcam (Tkinter)
│   ├── gui.py               # cửa sổ chính (Tkinter)
│   ├── __main__.py          # python -m face_recog
│   └── tools/
│       ├── check_detect.py  # logic của Check_Detect.py
│       └── evaluate.py      # logic của evaluate.py
├── examples/
│   └── simple_face_detection.py   # demo Haar Cascade độc lập
├── tests/                   # test theo module (không cần webcam/màn hình)
├── docs/
├── detect/                  # ảnh đăng ký + cache (không commit)
└── eval/                    # ảnh đo độ chính xác (không commit)
```

Ba file `.py` ở gốc chỉ có vài dòng gọi vào package, để các lệnh quen thuộc giữ nguyên. Logic thật nằm trong `face_recog/`.

## Mỗi module lo việc gì

| Module | Trách nhiệm | Không làm |
|---|---|---|
| `config` | Mọi hằng số (đường dẫn, ngưỡng, tỉ lệ, số khung) | Không có logic |
| `geometry` | Phép tính thuần trên khung mặt `(top, right, bottom, left)` | Không đọc ảnh/file |
| `camera` | Mở webcam, đọc khung mới nhất trên thread nền | Không xử lý ảnh |
| `known_faces` | Đọc ảnh đăng ký, suy ra tên người, mã hoá, cache theo `mtime` | Không so khớp |
| `matching` | Khoảng cách → tên + độ tin cậy; định dạng nhãn hiển thị | Không đọc ảnh/webcam |
| `smoothing` | Bỏ phiếu theo từng khuôn mặt qua các lần nhận diện | Không biết gì về ảnh |
| `recognizer` | Ghép các module trên thành vòng lặp nhận diện + vẽ kết quả | Không có giao diện Tk |
| `quality` | Đo chất lượng khung hình khi chụp; `check_frame` | Không dùng Tkinter |
| `settings` | Đọc/ghi `detect/settings.json` (ngưỡng nhận diện, tốc độ xử lý), validate | Không dùng Tkinter |
| `activity_log` | Ghi/đọc `detect/activity.log` | Không dùng Tkinter |
| `people` | Liệt kê/xoá người đăng ký trong `detect/` | Không dùng Tkinter |
| `capture` | Hộp thoại nhập tên, vòng lặp chụp, lưu ảnh, ghi log | Không tự đo chất lượng (gọi `quality`) |
| `gui` | Dựng cửa sổ, nối nút với `capture`/`recognizer`/`settings`/`activity_log`/`people` | Không chứa logic nhận diện |
| `tools.*` | Công cụ dòng lệnh dùng lại các module lõi | Không import `gui`/`capture` |

## Luật phụ thuộc

Phụ thuộc chỉ đi **một chiều** (mũi tên = "import"), không có vòng:

```
gui ──> capture ──> quality ──> geometry
 │         ├───────────────────> config
 │         └───────────────────> activity_log ──> config
 ├──────> recognizer ──> matching ──> smoothing ──> config
 │            ├────────> known_faces ─> config
 │            ├────────> camera
 │            ├────────> geometry
 │            ├────────> smoothing
 │            ├────────> settings ──> config
 │            └────────> activity_log
 ├──────> settings ──> config
 ├──────> activity_log ──> config
 └──────> people ──> known_faces, config
tools.check_detect, tools.evaluate ──> known_faces, geometry, config
```

Hai quy tắc chính:

1. **Chỉ `capture` và `gui` được import `tkinter`.** Lõi nhận diện và `quality` chạy được không cần giao diện, nên test được không cần màn hình/webcam.
2. **Lõi không import `gui`, `capture` hay `tools`.**

Luật này được kiểm tra tự động bởi [tests/test_architecture.py](../tests/test_architecture.py) (bảng `ALLOWED_DEPENDENCIES` là bản đồ chính xác, thêm module mới phải khai báo ở đó). Nếu test fail sau khi bạn thêm `import`, hãy xem phụ thuộc đó có hợp lý không; nếu có thì cập nhật bảng và sơ đồ trên, nếu không thì chuyển code sang chỗ phù hợp hơn.

## Luồng dữ liệu

**Nhận diện** (nút "Nhận diện khuôn mặt"):

```
gui.start_recognition
  └─> FaceRecognition()  ─ known_faces.load_known_faces ─ cache detect/.encodings.pkl
       └─> run_recognition
            ├─ camera.VideoStream.read           (khung mới nhất, thread nền)
            └─ mỗi PROCESS_EVERY_N khung: recognize
                 ├─ dò mặt trên khung thu nhỏ
                 ├─ geometry.scale_locations      (toạ độ lên khung gốc)
                 ├─ mã hoá trên khung gốc
                 ├─ matching.match_face           (người gần nhất + ngưỡng)
                 ├─ smoothing.NameSmoother.update (bỏ phiếu)
                 └─ matching.format_label
            └─ draw_faces → cv2.imshow
```

**Chụp ảnh** (nút "Thêm ảnh vào hệ thống"):

```
gui ─> capture.img_capture ─> quality.check_frame ─> (viền xanh/đỏ) ─> SPACE ─> detect/Ten_N.jpg
                                                                           └─> activity_log.log_event
```

**Quản lý người dùng** (nút "Quản lý người dùng"): `gui.open_people_window` gọi `people.list_people` để hiển thị, `people.delete_person` khi xoá (kèm `activity_log.log_event`). **Nhật ký** (nút "Nhật ký"): `gui.open_log_window` chỉ đọc bằng `activity_log.read_events`. **Cài đặt** (nút "Cài đặt"): `gui.open_settings_window` đọc/ghi bằng `settings.load_settings`/`settings.save_settings`; giá trị mới chỉ áp dụng từ lần `FaceRecognition()` tiếp theo (đọc settings trong `__init__`).

## Muốn sửa X thì vào đâu

| Muốn | Sửa ở |
|---|---|
| Đổi ngưỡng nhận diện, tỉ lệ thu nhỏ, số khung bỏ qua, ngưỡng độ nét... | `config.py` |
| Đổi cách tính độ tin cậy (%) hoặc quy tắc khớp | `matching.py` |
| Đổi cách gộp ảnh thành người (`Huy_0.jpg` → `Huy`) | `known_faces.person_name` |
| Đổi định dạng/cách cache encoding | `known_faces.py` (tăng `CACHE_VERSION` trong `config`) |
| Nhãn nhấp nháy, chậm hiện tên | `smoothing.py` + `VOTE_*` trong `config` |
| Thêm kiểm tra chất lượng khi chụp ảnh | `quality.check_frame` |
| Đổi giao diện, nút, chữ | `gui.py` (chụp ảnh: `capture.py`) |
| Đổi webcam, độ phân giải, cách đọc khung | `camera.py` |
| Thêm công cụ dòng lệnh mới | `tools/<ten>.py` + script mỏng ở gốc nếu cần |
| Thêm cài đặt chỉnh được từ GUI | `settings.py` (`DEFAULTS`, `validate_settings`) + đọc trong `recognizer.FaceRecognition.__init__` |
| Đổi việc gì được ghi vào nhật ký | `activity_log.log_event` tại nơi gọi (`recognizer.recognize`, `capture.img_capture`, `gui.open_people_window`) |
| Đổi cách liệt kê/xoá người trong quản lý người dùng | `people.py` |

## Quy ước

- **Tên file/module `snake_case`**; import trong package dùng đường dẫn tuyệt đối (`from face_recog.config import ...`).
- **Tách logic thuần khỏi I/O**: hàm nhận dữ liệu và trả kết quả (`match_face`, `check_frame`, `compute_metrics`...) để test không cần webcam; phần I/O (webcam, Tkinter, file) đặt ở lớp mỏng bên ngoài.
- **Hằng số chỉ khai báo ở `config.py`**, không viết cứng số ở nơi khác.
- Module có logic kiểm thử được đều có file test cùng tên trong `tests/` (`matching.py` ↔ `tests/test_matching.py`). Ngoại lệ: `gui.py` cần màn hình nên chưa có test tự động (chỉ kiểm tra dựng cửa sổ thủ công), `config.py` chỉ chứa hằng số.

## Phát triển

```bash
pip install -r requirements-dev.txt
pytest          # test (kể cả kiểm tra kiến trúc), không cần webcam
ruff check .    # lint
```

## Hạn chế đã biết

- `DETECT_DIR = 'detect'` là đường dẫn **tương đối theo thư mục hiện hành**: chạy lệnh từ thư mục khác sẽ dùng một `detect/` khác. Chạy từ thư mục gốc của dự án để tránh nhầm.

## Tài liệu chuyên đề

- [recognition.md](recognition.md): lõi nhận diện (`matching`, `known_faces`, `camera`, `smoothing`, `recognizer`)
- [capture.md](capture.md): chụp ảnh và kiểm tra chất lượng (`capture`, `quality`)
- [check_detect.md](check_detect.md): dọn ảnh trùng/lỗi
- [evaluate.md](evaluate.md): đo độ chính xác
- [simple_face_detection.md](simple_face_detection.md): demo Haar Cascade
