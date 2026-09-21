# Giải thích `Check_Detect.py`

Logic nằm ở [face_recog/tools/check_detect.py](../face_recog/tools/check_detect.py); file `Check_Detect.py` ở gốc chỉ gọi vào đó nên lệnh `python Check_Detect.py` giữ nguyên. Script dùng lại `known_faces.list_image_files` và `geometry.largest_face`/`face_area` thay vì tự viết.

Script độc lập giúp kiểm tra thư mục `detect/` và dọn các ảnh **trùng hệt nhau** hoặc ảnh lỗi. Mỗi lần bấm SPACEBAR trong "Thêm ảnh vào hệ thống", một ảnh mới được lưu. Ảnh trùng hệt (chụp liên tiếp, gần như giống nhau) chỉ làm khởi động chậm hơn mà không giúp nhận diện; ngược lại **nhiều ảnh khác góc/ánh sáng của cùng một người là có ích** (nhận diện ghép theo người, xem [recognition.md](recognition.md)), nên script mặc định dùng ngưỡng chặt để không đề xuất xoá chúng.

## Cách dùng

```bash
# Chỉ báo cáo (mặc định, KHÔNG xoá gì)
python Check_Detect.py

# Cho phép xoá, hỏi xác nhận từng nhóm
python Check_Detect.py --delete

# Tuỳ chọn
python Check_Detect.py --dir detect --threshold 0.5
```

| Tham số | Ý nghĩa |
|---|---|
| `--dir` | Thư mục ảnh (mặc định `detect`) |
| `--threshold` | Ngưỡng khoảng cách coi là ảnh **trùng hệt** (mặc định `DUPLICATE_THRESHOLD = 0.3` trong [config.py](../face_recog/config.py), chặt hơn ngưỡng nhận diện). Tăng lên (vd `0.5`) nếu muốn gom cả ảnh khác góc/ánh sáng của cùng một người |
| `--delete` | Cho phép xoá ảnh sau khi bạn xác nhận. Không có cờ này script chỉ báo cáo |

## Cách hoạt động

1. **Quét** (`scan_folder`): chỉ xét file `.jpg/.jpeg/.png` (bỏ qua `.DS_Store`...). Mỗi ảnh được mã hoá bằng `face_recognition`.
   - Không có khuôn mặt, hoặc file hỏng / không phải ảnh thật → vào danh sách "ảnh lỗi" (không làm script crash).
   - Nhiều khuôn mặt → lấy khuôn mặt lớn nhất (theo diện tích) và cảnh báo.
2. **Gom nhóm** (`group_same_person`): so `face_distance` từng cặp ảnh; khoảng cách ≤ ngưỡng thì cùng nhóm (union-find, nên A~B và B~C sẽ gộp cả A, B, C). Nhóm theo khuôn mặt chứ không theo tiền tố tên file, vì có thể nhập tên khác nhau cho cùng một người.
3. **Đề xuất** giữ ảnh có khuôn mặt lớn nhất trong nhóm.
4. **Xác nhận** (`ask_keep`, chỉ khi có `--delete`): `Enter` = giữ ảnh đề xuất, nhập số = giữ ảnh khác, `s` = bỏ qua nhóm. Ảnh không có khuôn mặt được hỏi riêng `[y/N]`.
5. Chỉ những file đã xác nhận mới bị `os.remove`.

## Lưu ý

- **Ngưỡng 0.3 chưa được kiểm chứng trên ảnh thật khác góc.** Đã thử với các biến thể tổng hợp của cùng một tấm ảnh (thu nhỏ, làm tối, lật ngang, xoay 20°): khoảng cách tới ảnh gốc đều < 0.16 nên tất cả bị gom. Ảnh chụp khác góc/ánh sáng thật thường cách xa hơn, nhưng mức cụ thể cần đo bằng ảnh của bạn (dùng [evaluate.py](../face_recog/tools/evaluate.py) khi có, hoặc xem danh sách nhóm ở chế độ báo cáo trước khi `--delete`). Luôn chạy báo cáo (không `--delete`) và xem các nhóm trước khi xoá.

- "Diện tích khuôn mặt" chỉ là tiêu chí đề xuất (mặt to thường nhận diện tốt hơn), không đảm bảo ảnh nét hơn. Hai ảnh chụp liên tiếp có thể có diện tích bằng nhau; khi đó script đề xuất ảnh đầu tiên.
- Script phải chạy trong venv đã cài `face_recognition` (xem [README](../README.md)).
