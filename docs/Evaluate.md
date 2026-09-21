# Giải thích `evaluate.py`

Công cụ **đo độ chính xác nhận diện** bằng ảnh có nhãn, để chọn ngưỡng và kiểm tra thay đổi dựa trên số liệu thay vì cảm giác. Nó mô phỏng đúng pipeline chạy trực tiếp ([Recognition.md](Recognition.md)): đưa ảnh về khung 640 rộng như webcam, dò trên khung thu nhỏ (`DETECT_SCALE`), rồi mã hoá.

## Chuẩn bị dữ liệu

```
detect/                  <- ảnh đăng ký (như khi chạy thật): Huy_0.jpg, Elon_0.jpg, ...
eval/
├── Huy/                 <- ảnh TEST của người đã đăng ký (tên thư mục = tên người, không phân biệt hoa thường)
│   ├── 1.jpg
│   └── ...
├── Elon/
└── unknown/             <- ảnh NGƯỜI LẠ (thư mục tên khác với mọi người đã đăng ký)
    └── ...
```

Để số liệu có ý nghĩa:

- Ảnh test phải là **ảnh chụp khác** ảnh đăng ký (khác lần chụp, góc, ánh sáng, có/không đeo kính, ngồi xa/gần). Dùng lại chính ảnh đăng ký hoặc biến thể của nó sẽ cho kết quả quá lạc quan.
- Mỗi loại (đúng người / người lạ) nên có **≥ 10 ảnh**; nên có vài người lạ khác nhau, kể cả người trông hơi giống. Công cụ cảnh báo "mẫu nhỏ" khi ít hơn.
- Chụp bằng chính webcam bạn dùng thật, kể cả ngược sáng và hơi nghiêng đầu.
- `eval/` (và `detect/`) đã nằm trong `.gitignore` vì là dữ liệu sinh trắc học.

## Cách dùng

```bash
python evaluate.py                       # đo theo cách hiện tại (mã hoá trên khung gốc)
python evaluate.py --mode both           # so sánh cách cũ (khung thu nhỏ) và cách hiện tại
python evaluate.py --thresholds 0.4,0.45,0.5,0.55,0.6
python evaluate.py --enroll detect --eval eval
```

| Tham số | Ý nghĩa |
|---|---|
| `--enroll` | Thư mục ảnh đăng ký (mặc định `detect`; dùng cache encoding như khi chạy thật) |
| `--eval` | Thư mục ảnh test (mặc định `eval`) |
| `--mode` | `small` = mã hoá trên khung thu nhỏ (cách cũ), `full` = khung gốc (mặc định, hiện tại), `both` = cả hai |
| `--thresholds` | Các ngưỡng khoảng cách cần thử, cách nhau bằng dấu phẩy (mặc định 0.30–0.70, bước 0.05) |

## Đọc kết quả

Với mỗi ảnh test, công cụ tính **một lần** khoảng cách tới người gần nhất; các con số theo từng ngưỡng suy ra từ đó (không mã hoá lại). Mỗi hàng của bảng ứng với một ngưỡng:

| Cột | Ý nghĩa | Mong muốn |
|---|---|---|
| nhận đúng | ảnh đúng người được nhận đúng tên | cao |
| nhận nhầm người | ảnh của A bị nhận thành B | **0** |
| người lạ bị nhận | ảnh người lạ bị nhận thành một người đã đăng ký | **0** |
| bỏ sót | ảnh đúng người nhưng xa hơn ngưỡng nên thành `Unknown` | thấp |
| không thấy mặt | không dò ra mặt (trước cả khi so khớp) | thấp |

Phần đầu báo cáo còn cho **khoảng cách nhỏ nhất / trung bình / lớn nhất** của nhóm đúng người và nhóm người lạ. Hai nhóm tách xa nhau thì có ngưỡng tốt; chồng lấn thì không ngưỡng nào tránh được lỗi.

**Gợi ý ngưỡng**: ngưỡng ít lỗi nhận nhầm nhất (nhận nhầm người + người lạ bị nhận), sau đó nhận đúng nhiều nhất, sau đó nhỏ nhất. Chỉ hiển thị khi có **cả** ảnh đúng người lẫn người lạ (đã dò ra mặt); nếu không sẽ báo "Chưa đủ dữ liệu" thay vì in con số vô nghĩa. Ngưỡng gợi ý áp dụng bằng cách sửa `RECOGNITION_THRESHOLD` trong [config.py](../config.py).

Ảnh không dò ra mặt được đếm riêng (cột "không thấy mặt") chứ không tính là "bỏ sót", vì đó là giới hạn của bước dò chứ không phải của ngưỡng.

## Lưu ý

- Với `--mode both`, so sánh khoảng cách trung bình nhóm đúng người giữa `small` và `full` để biết mã hoá trên khung gốc có thật sự giúp trên dữ liệu của bạn không.
- Ngưỡng tốt phụ thuộc vào ảnh đăng ký, webcam và ánh sáng của bạn; hãy đo lại khi đổi các yếu tố đó.
- Kiểm thử công cụ mới chỉ chạy trên bộ dữ liệu nhỏ dựng từ biến thể của 1 ảnh (tối, lật, thu nhỏ, mờ) và 1 người lạ, nên **chưa có kết luận nào về độ chính xác thật**; số liệu thật cần ảnh của bạn.
