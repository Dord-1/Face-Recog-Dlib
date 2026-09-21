"""Hằng số dùng chung cho toàn dự án (một nguồn duy nhất, tránh lệch giá trị giữa các file)."""
import os

# Thư mục ảnh khuôn mặt và file cache encoding
DETECT_DIR = 'detect'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
CACHE_PATH = os.path.join(DETECT_DIR, '.encodings.pkl')
CACHE_VERSION = 1  # tăng số này khi đổi cách mã hoá để bỏ cache cũ

# Ngưỡng khoảng cách khuôn mặt coi là "cùng một người" (mặc định của face_recognition)
MATCH_THRESHOLD = 0.6

# Ngưỡng khoảng cách để nhận diện chấp nhận một khuôn mặt là "đã biết". Chặt hơn MATCH_THRESHOLD
# (mặc định 0.6 của thư viện, dễ nhận nhầm người lạ). Điểm khởi đầu, chỉnh bằng số liệu từ evaluate.py.
# Lưu ý: chỉ ảnh hưởng quyết định khớp; thang % của face_confidence vẫn dùng MATCH_THRESHOLD.
RECOGNITION_THRESHOLD = 0.5

# Bỏ phiếu qua nhiều lần nhận diện để nhãn không nhấp nháy: chỉ hiện tên khi thắng >= VOTE_MIN trong
# VOTE_WINDOW lần gần nhất. Track (một khuôn mặt qua thời gian) bị xoá sau TRACK_MAX_MISSES lần vắng;
# ghép mặt với track nếu tâm lệch không quá TRACK_MAX_DIST_RATIO lần bề rộng mặt.
VOTE_WINDOW = 5
VOTE_MIN = 3
TRACK_MAX_MISSES = 3
TRACK_MAX_DIST_RATIO = 0.75

# Ngưỡng Check_Detect.py gom ảnh "trùng hệt" của một người (chặt hơn MATCH_THRESHOLD để không
# đề xuất xoá ảnh khác góc/ánh sáng — loại ảnh đa dạng này có ích cho nhận diện)
DUPLICATE_THRESHOLD = 0.3

# Dò khuôn mặt trên khung hình thu nhỏ cho nhanh; toạ độ nhân lại INV_SCALE để vẽ/mã hoá.
# Đo thực tế (khung webcam 640 rộng, upsample=0): 0.25 chỉ thấy mặt rộng >= ~250 px (phải ngồi rất sát),
# 0.5 thấy từ ~120 px với chi phí ~4 ms thay vì ~1 ms.
DETECT_SCALE = 0.5
INV_SCALE = round(1 / DETECT_SCALE)

# Chỉ chạy nhận diện trên 1 trong mỗi N khung hình. Tăng nếu máy yếu.
PROCESS_EVERY_N = 3

# Chụp ảnh: khuôn mặt hẹp hơn giá trị này (px, khung gốc) bị coi là quá xa; kiểm tra mỗi N khung
MIN_FACE_WIDTH = 100
CHECK_EVERY_N = 3
