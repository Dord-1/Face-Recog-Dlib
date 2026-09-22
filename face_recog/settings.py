"""Cài đặt người dùng chỉnh được từ GUI (ngưỡng nhận diện, tốc độ xử lý), lưu vào detect/settings.json."""
import json
import os

from face_recog.config import DETECT_DIR, PROCESS_EVERY_N, RECOGNITION_THRESHOLD

SETTINGS_PATH = os.path.join(DETECT_DIR, 'settings.json')

DEFAULTS = {
    'recognition_threshold': RECOGNITION_THRESHOLD,
    'process_every_n': PROCESS_EVERY_N,
}


def load_settings(settings_path=SETTINGS_PATH):
    """Đọc cài đặt đã lưu; thiếu/hỏng file hoặc key sai kiểu thì dùng giá trị mặc định cho key đó."""
    try:
        with open(settings_path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}

    values = dict(DEFAULTS)
    threshold = data.get('recognition_threshold')
    if isinstance(threshold, (int, float)) and 0 < threshold <= 1:
        values['recognition_threshold'] = float(threshold)

    process_every_n = data.get('process_every_n')
    if isinstance(process_every_n, int) and not isinstance(process_every_n, bool) and process_every_n >= 1:
        values['process_every_n'] = process_every_n

    return values


def validate_settings(values):
    """Trả về thông báo lỗi (str) nếu `values` không hợp lệ, hoặc None nếu hợp lệ."""
    threshold = values.get('recognition_threshold')
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not (0 < threshold <= 1):
        return 'Ngưỡng nhận diện phải là số trong khoảng (0, 1]'

    process_every_n = values.get('process_every_n')
    if not isinstance(process_every_n, int) or isinstance(process_every_n, bool) or process_every_n < 1:
        return 'Tốc độ xử lý (số khung bỏ qua) phải là số nguyên >= 1'

    return None


def save_settings(values, settings_path=SETTINGS_PATH):
    """Ghi cài đặt (atomic). Trả về thông báo lỗi nếu `values` không hợp lệ và không ghi gì."""
    error = validate_settings(values)
    if error:
        return error

    os.makedirs(os.path.dirname(settings_path) or '.', exist_ok=True)
    tmp_path = settings_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(values, f)
    os.replace(tmp_path, settings_path)
    return None
