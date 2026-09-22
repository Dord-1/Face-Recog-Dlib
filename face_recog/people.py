"""Quản lý người đã đăng ký trong detect/: liệt kê và xoá theo tên."""
import os

from face_recog.config import DETECT_DIR
from face_recog.known_faces import CACHE_PATH, list_image_files, load_cache, person_name, save_cache


def list_people(detect_dir=DETECT_DIR):
    """Danh sách người đã đăng ký, sắp xếp theo tên: [{'name', 'count', 'files'}]."""
    people = {}
    for image in list_image_files(detect_dir):
        name = person_name(image)
        people.setdefault(name, []).append(image)

    return [
        {'name': name, 'count': len(files), 'files': files}
        for name, files in sorted(people.items())
    ]


def delete_person(name, detect_dir=DETECT_DIR, cache_path=CACHE_PATH):
    """Xoá toàn bộ ảnh của `name` (và entry cache tương ứng). Trả về số ảnh đã xoá."""
    files = next((p['files'] for p in list_people(detect_dir) if p['name'] == name), [])

    cache = load_cache(cache_path)
    for image in files:
        os.remove(os.path.join(detect_dir, image))
        cache.pop(image, None)
    if files:
        save_cache(cache, cache_path)

    return len(files)
