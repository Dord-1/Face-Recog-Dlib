import math
import os
import pickle
import re
import threading
import time

import cv2
import face_recognition
import numpy as np

from config import (
    CACHE_PATH,
    CACHE_VERSION,
    DETECT_DIR,
    DETECT_SCALE,
    IMAGE_EXTENSIONS,
    INV_SCALE,
    MATCH_THRESHOLD,
    PROCESS_EVERY_N,
)


class CameraError(RuntimeError):
    """Không mở được webcam."""


def face_confidence(face_distance, face_match_threshold=MATCH_THRESHOLD):
    span = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (span * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"


def person_name(filename):
    """Tên người từ tên file ảnh: 'Huy_0.jpg' -> 'Huy' (bỏ đuôi và hậu tố _số cuối)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'_\d+$', '', stem) or stem


def match_face(encoding, known_encodings, known_names, threshold=MATCH_THRESHOLD):
    """So khớp 1 khuôn mặt với danh sách đã biết. Trả về (tên, độ tin cậy); 'Unknown' nếu không khớp."""
    if len(known_encodings) == 0:
        return 'Unknown', 'Unknown'

    distances = face_recognition.face_distance(known_encodings, encoding)
    best = int(np.argmin(distances))
    if distances[best] <= threshold:
        return known_names[best], face_confidence(distances[best], threshold)
    return 'Unknown', 'Unknown'


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

    for image in sorted(os.listdir(detect_dir)):
        if not image.lower().endswith(IMAGE_EXTENSIONS):
            continue

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


def scale_locations(locations, factor, shape):
    """Nhân toạ độ (top, right, bottom, left) lên `factor` và kẹp trong khung hình có `shape` (h, w, ...)."""
    height, width = shape[:2]
    return [
        (
            min(max(int(round(top * factor)), 0), height),
            min(max(int(round(right * factor)), 0), width),
            min(max(int(round(bottom * factor)), 0), height),
            min(max(int(round(left * factor)), 0), width),
        )
        for top, right, bottom, left in locations
    ]


def draw_faces(frame, locations, names):
    """Vẽ khung + nhãn lên frame. `locations` tính trên khung thu nhỏ nên nhân lại INV_SCALE."""
    for (top, right, bottom, left), name in zip(locations, names, strict=True):
        top *= INV_SCALE
        right *= INV_SCALE
        bottom *= INV_SCALE
        left *= INV_SCALE

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


class VideoStream:
    """Đọc webcam trên 1 thread nền riêng, luôn giữ khung hình MỚI NHẤT.

    Không dùng queue để tránh dồn (backlog) khung hình cũ: nếu thread xử lý
    chính chậm hơn tốc độ camera, các khung hình cũ bị bỏ qua thay vì xếp
    hàng chờ, giúp hình hiển thị luôn "theo kịp" thời gian thực thay vì bị
    trễ dần và giật cục.
    """

    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.stream.isOpened():
            self.stream.release()
            raise CameraError('Không tìm thấy hoặc không mở được webcam')

        self.lock = threading.Lock()
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.stream.release()


class FaceRecognition:
    # Số khung hình bỏ qua giữa 2 lần xử lý nhận diện (chỉ xử lý 1/PROCESS_EVERY_N khung).
    PROCESS_EVERY_N = PROCESS_EVERY_N

    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_count = 0
        self.encode_faces()

    def encode_faces(self):
        encodings, names, skipped, reused = load_known_faces()
        self.known_face_encodings = encodings
        self.known_face_names = [person_name(name) for name in names]  # nhiều ảnh -> cùng một người

        print(f'Đã nạp {len(names)} ảnh của {len(set(self.known_face_names))} người '
              f'({reused} từ cache, {len(names) + len(skipped) - reused} mã hoá mới)')
        if skipped:
            print(f'Bỏ qua {len(skipped)} ảnh không có khuôn mặt hoặc không đọc được: {skipped}')

    def recognize(self, frame):
        """Dò và nhận diện khuôn mặt trong frame, cập nhật face_locations/face_names.

        Dò trên khung thu nhỏ (nhanh) nhưng mã hoá trên khung gốc: mặt nhỏ mã hoá từ khung thu nhỏ
        cho vector kém chính xác hơn (khoảng cách tới chính người đó cao hơn hẳn).
        """
        small_frame = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
        full_locations = scale_locations(self.face_locations, INV_SCALE, frame.shape)
        self.face_encodings = face_recognition.face_encodings(rgb_frame, full_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            name, confidence = match_face(face_encoding, self.known_face_encodings, self.known_face_names)
            self.face_names.append(f'{name} {confidence}')

    def run_recognition(self):
        video_stream = VideoStream(0)

        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = video_stream.read()
            if not ret or frame is None:
                if cv2.waitKey(1) % 256 == 27:  # vẫn cho phép ESC khi chưa có khung hình
                    break
                continue

            if self.frame_count % self.PROCESS_EVERY_N == 0:
                self.recognize(frame)
            self.frame_count += 1

            draw_faces(frame, self.face_locations, self.face_names)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else fps
            prev_time = now
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) % 256 == 27:  # ESC pressed
                break

        video_stream.stop()
        cv2.destroyAllWindows()
