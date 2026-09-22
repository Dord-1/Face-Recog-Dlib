"""Nhận diện khuôn mặt qua webcam (dlib / face_recognition + OpenCV). Xem docs/architecture.md."""
import warnings

# face_recognition_models dùng pkg_resources (đang bị deprecate, dự kiến gỡ bỏ ~cuối 2025) để tìm
# file model; vô hại với bản setuptools đã ghim trong requirements.txt, chỉ ẩn cảnh báo lặp mỗi lần import.
warnings.filterwarnings(
    'ignore', message=r'pkg_resources is deprecated', category=UserWarning, module='face_recognition_models')
