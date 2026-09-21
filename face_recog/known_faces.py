"""Ảnh khuôn mặt đã đăng ký trong detect/: tên người, mã hoá và cache encoding."""
import os
import pickle
import re

import face_recognition

from face_recog.config import CACHE_PATH, CACHE_VERSION, DETECT_DIR, IMAGE_EXTENSIONS


def person_name(filename):
    """Tên người từ tên file ảnh: 'Huy_0.jpg' -> 'Huy' (bỏ đuôi và hậu tố _số cuối)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'_\d+$', '', stem) or stem


def list_image_files(folder):
    """Tên các file ảnh (theo IMAGE_EXTENSIONS) trong `folder`, sắp xếp theo tên; bỏ qua file khác."""
    return sorted(name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS))


def encode_image(path):
    """Encoding của khuôn mặt đầu tiên trong ảnh, hoặc None nếu ảnh không có mặt/không đọc được."""
    try:
        found = face_recognition.face_encodings(face_recognition.load_image_file(path))
    except (OSError, ValueError):
        return None
    return found[0] if found else None


def load_cache(cache_path=CACHE_PATH):
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if data.get('version') == CACHE_VERSION:
            return data['entries']
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, AttributeError, ValueError):
        pass  # không có cache, hỏng hoặc khác phiên bản -> mã hoá lại từ đầu
    return {}


def save_cache(entries, cache_path=CACHE_PATH):
    tmp_path = cache_path + '.tmp'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump({'version': CACHE_VERSION, 'entries': entries}, f)
        os.replace(tmp_path, cache_path)
    except OSError as e:
        print(f'Không lưu được cache encoding: {e}')


def load_known_faces(detect_dir=DETECT_DIR, cache_path=CACHE_PATH, encoder=encode_image):
    """Nạp encoding của ảnh trong detect_dir, dùng cache theo mtime để khỏi mã hoá lại.

    Ảnh không đọc được hoặc không có khuôn mặt bị bỏ qua (thay vì làm crash).
    Trả về (encodings, names, skipped, reused).
    """
    os.makedirs(detect_dir, exist_ok=True)
    cache = load_cache(cache_path)
    entries, encodings, names, skipped = {}, [], [], []
    reused = 0

    for image in list_image_files(detect_dir):
        path = os.path.join(detect_dir, image)
        mtime = os.path.getmtime(path)
        cached = cache.get(image)

        if cached and cached['mtime'] == mtime:
            encoding = cached['encoding']
            reused += 1
        else:
            encoding = encoder(path)

        entries[image] = {'mtime': mtime, 'encoding': encoding}  # None = ảnh lỗi, cũng được cache

        if encoding is None:
            skipped.append(image)
            continue
        encodings.append(encoding)
        names.append(image)

    save_cache(entries, cache_path)
    return encodings, names, skipped, reused
