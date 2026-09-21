# Giải thích `capture.py`

Chụp ảnh khuôn mặt từ webcam để đăng ký người dùng. Ảnh xấu (mờ, tối, mặt nghiêng) là nguồn sai số ngay từ đầu vì nhận diện so khớp với chính những ảnh này, nên mỗi khung hình được kiểm tra trước khi cho phép lưu.

## Luồng chụp (`img_capture`)

1. Hỏi tên (hộp thoại Tkinter); bấm Cancel hoặc để trống thì thoát, chưa mở camera.
2. Mở webcam. Không mở được thì hiện hộp thoại báo lỗi.
3. Mỗi `CHECK_EVERY_N` khung chạy `check_frame()`; vẽ **viền xanh** khi hợp lệ, **viền đỏ** kèm lý do khi chưa hợp lệ. Vẽ trên bản sao nên ảnh lưu ra không dính khung hay chữ.
4. `SPACE` chỉ lưu khi hợp lệ (`Ten_0.jpg`, `Ten_1.jpg`, ...; dùng `next_free_index` nên chụp lại cùng tên không ghi đè ảnh cũ). `ESC` để thoát.

## `check_frame()` — các bước kiểm tra

Kiểm tra lần lượt, dừng ở lỗi đầu tiên (rẻ trước, đắt sau; landmarks chỉ tính khi các bước trước đã đạt):

| Bước | Điều kiện đạt | Thông báo khi lỗi |
|---|---|---|
| 1. Số khuôn mặt | đúng 1 | `Khong thay khuon mat` / `Chi duoc co 1 khuon mat` |
| 2. Kích thước | rộng ≥ `MIN_FACE_WIDTH` (100 px) | `Khuon mat qua nho, hay lai gan hon` |
| 3. Độ sáng | `MIN_BRIGHTNESS` ≤ trung bình xám ≤ `MAX_BRIGHTNESS` (60–215) | `Qua toi (số)` / `Qua sang (số)` |
| 4. Độ nét | `sharpness` ≥ `MIN_SHARPNESS` (0.10) | `Anh bi mo (số)` |
| 5. Nhìn thẳng | `yaw_ratio` ≤ `MAX_YAW_RATIO` (1.35) | `Hay nhin thang (số)` |

Thông báo không dấu vì `cv2.putText` không vẽ được chữ tiếng Việt có dấu. Khi lỗi có kèm **số đo** để bạn chỉnh ngưỡng trong [config.py](../config.py) cho hợp webcam của mình. Nếu không tính được landmarks thì không chặn việc chụp.

### Độ nét (`sharpness`)

Phương sai Laplacian của vùng mặt (cắt, chuyển xám, đưa về 100×100), **chia cho phương sai xám**.

- Không dùng Laplacian thô vì nó phụ thuộc mạnh vào tương phản của ảnh: trong phép đo, ảnh Huy nét (119) thấp hơn ảnh Elon đã làm mờ σ=2 (190), và ảnh tối đi bị chấm "mờ". Chia cho tương phản khử được điều đó (ảnh Huy tối đi 0.3 lần vẫn 0.21 so với 0.21 ban đầu).
- Phương sai xám có **sàn** `CONTRAST_VAR_FLOOR = 50`: vùng gần như phẳng chỉ còn nhiễu lượng tử hoá uint8, chia cho phương sai vài đơn vị sẽ chấm nhầm là "nét".

### Độ sáng và nhìn thẳng

- `brightness`: trung bình mức xám của vùng mặt (0–255).
- `yaw_ratio`: tỉ lệ khoảng cách từ đầu mũi tới tâm hai mắt (lấy từ `face_recognition.face_landmarks`), luôn ≥ 1. Nhìn thẳng ≈ 1; quay đầu sang một bên thì một mắt xa mũi hơn nên tỉ lệ tăng.

## Hiệu chỉnh ngưỡng: đã đo gì, chưa đo gì

Đo trên 2 ảnh thật (Huy, Elon) đặt vào khung 640×480 với mặt rộng 160 và 250 px, cùng các biến thể mờ (`GaussianBlur` σ 1–5), tối (×0.5, ×0.3) và sáng (×1.6, ×2.2):

| Chỉ số | Ảnh nét | Kết quả trên biến thể |
|---|---|---|
| Độ nét chuẩn hoá | 0.21–0.38 | mờ σ1: 0.08–0.23; σ2: 0.03–0.11; σ3: 0.02–0.06 |
| Độ sáng | 101–188 | tối ×0.5: 50–94; sáng ×2.2: 204–246 |
| `yaw_ratio` | 1.03–1.07 | — |

Ngưỡng mặc định chặn mờ σ≥3 (và σ=2 ở mặt nhỏ), tối ×0.5 của ảnh gốc tối, và sáng chói; mờ nhẹ σ=2 ở mặt lớn vẫn qua. **Cố ý đặt dễ chịu** vì sai theo hướng khó (không chụp được gì) khó chịu hơn nhiều so với để lọt một ảnh hơi mềm.

**Chưa kiểm chứng:**
- **Webcam thật**: phép đo dùng ảnh thu nhỏ từ ảnh độ phân giải cao nên nét hơn webcam thật (webcam mềm hơn do cảm biến và nén). Nếu bạn luôn gặp `Anh bi mo` dù ảnh ổn, hạ `MIN_SHARPNESS` theo số hiện trên thông báo.
- **Mặt nghiêng thật**: chưa có ảnh mặt nghiêng để đo. `MAX_YAW_RATIO = 1.35` là giá trị bảo thủ cách xa mức nhìn thẳng đã đo (≤ 1.07), chưa biết nó chặn ở góc quay bao nhiêu độ. Thử quay đầu và xem số hiện ra để chỉnh.
