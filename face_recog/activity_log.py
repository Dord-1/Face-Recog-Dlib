"""Nhật ký hoạt động (nhận diện, thêm/xoá ảnh) lưu vào detect/activity.log."""
import datetime
import os

from face_recog.config import DETECT_DIR

LOG_PATH = os.path.join(DETECT_DIR, 'activity.log')


def log_event(kind, detail, log_path=LOG_PATH):
    """Ghi thêm một dòng nhật ký. Không raise nếu ghi thất bại (log không được làm crash tính năng chính)."""
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    line = f'{timestamp}\t{kind}\t{detail}\n'
    try:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line)
    except OSError as e:
        print(f'Không ghi được nhật ký: {e}')


def read_events(limit=200, log_path=LOG_PATH):
    """Trả về tối đa `limit` dòng nhật ký gần nhất, cũ -> mới. Rỗng nếu chưa có file."""
    try:
        with open(log_path, encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
    except OSError:
        return []
    return lines[-limit:]
