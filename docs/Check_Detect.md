# Giải thích `Check_Detect.py`

Script độc lập giúp kiểm tra thư mục `detect/` và dọn các ảnh dư thừa. Mỗi lần bấm SPACEBAR trong "Thêm ảnh vào hệ thống", một ảnh mới được lưu; nhiều ảnh của cùng một người làm `Recognition.encode_faces()` mã hoá thừa (khởi động chậm hơn) và tên bị lặp. Ảnh không có khuôn mặt còn làm `encode_faces()` crash (`IndexError`).

## Cách dùng

```bash
# Chỉ báo cáo (mặc định, KHÔNG xoá gì)
python Check_Detect.py

# Cho phép xoá, hỏi xác nhận từng nhóm
python Check_Detect.py --delete

# Tuỳ chọn
python Check_Detect.py --dir detect --threshold 0.6
```

| Tham số | Ý nghĩa |
|---|---|
| `--dir` | Thư mục ảnh (mặc định `detect`) |
| `--threshold` | Ngưỡng khoảng cách coi là cùng một người (mặc định `MATCH_THRESHOLD = 0.6` trong [config.py](../config.py), cùng ngưỡng với nhận diện) |
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

- "Diện tích khuôn mặt" chỉ là tiêu chí đề xuất (mặt to thường nhận diện tốt hơn), không đảm bảo ảnh nét hơn. Hai ảnh chụp liên tiếp có thể có diện tích bằng nhau; khi đó script đề xuất ảnh đầu tiên.
- Script phải chạy trong venv đã cài `face_recognition` (xem [README](../README.md)).
