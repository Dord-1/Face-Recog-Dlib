"""Nhận diện khuôn mặt theo thời gian thực từ webcam (điều phối các module lõi)."""
import time

import cv2
import face_recognition

from face_recog.camera import VideoStream
from face_recog.config import DETECT_SCALE, INV_SCALE, PROCESS_EVERY_N
from face_recog.geometry import scale_locations
from face_recog.known_faces import load_known_faces, person_name
from face_recog.matching import format_label, match_face
from face_recog.smoothing import NameSmoother


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


class FaceRecognition:
    # Số khung hình bỏ qua giữa 2 lần xử lý nhận diện (chỉ xử lý 1/PROCESS_EVERY_N khung).
    PROCESS_EVERY_N = PROCESS_EVERY_N

    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.smoother = NameSmoother()
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

        raw_results = [
            match_face(face_encoding, self.known_face_encodings, self.known_face_names)
            for face_encoding in self.face_encodings
        ]
        smoothed = self.smoother.update(self.face_locations, raw_results)
        self.face_names = [format_label(name, confidence) for name, confidence in smoothed]

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
