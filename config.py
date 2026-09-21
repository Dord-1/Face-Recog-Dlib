"""Hằng số dùng chung cho toàn dự án (một nguồn duy nhất, tránh lệch giá trị giữa các file)."""
import os

# Thư mục ảnh khuôn mặt và file cache encoding
DETECT_DIR = 'detect'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
CACHE_PATH = os.path.join(DETECT_DIR, '.encodings.pkl')
CACHE_VERSION = 1  # tăng số này khi đổi cách mã hoá để bỏ cache cũ

# Ngưỡng khoảng cách khuôn mặt coi là "cùng một người" (mặc định của face_recognition)
MATCH_THRESHOLD = 0.6

# Nhận diện/dò khuôn mặt trên khung hình thu nhỏ cho nhanh; toạ độ nhân lại INV_SCALE để vẽ
DETECT_SCALE = 0.25
INV_SCALE = round(1 / DETECT_SCALE)

# Chỉ chạy nhận diện trên 1 trong mỗi N khung hình. Tăng nếu máy yếu.
PROCESS_EVERY_N = 3

# Chụp ảnh: khuôn mặt hẹp hơn giá trị này (px, khung gốc) bị coi là quá xa; kiểm tra mỗi N khung
MIN_FACE_WIDTH = 100
CHECK_EVERY_N = 3
