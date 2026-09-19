import cv2
import face_recognition
import os, sys
import numpy as np
import math
import threading
import time

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"


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
            sys.exit('Camera not found')

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
    # Tăng giá trị này nếu máy yếu để đổi lấy tốc độ hiển thị mượt hơn.
    PROCESS_EVERY_N = 3

    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_count = 0
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('detect'):
            face_image = face_recognition.load_image_file(f'detect/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_stream = VideoStream(0)

        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = video_stream.read()
            if not ret or frame is None:
                continue

            if self.frame_count % self.PROCESS_EVERY_N == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding) #RetinaFace.verify
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    self.face_names.append(f'{name} {confidence}')

            self.frame_count += 1

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else fps
            prev_time = now
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) % 256 == 27: #ESC pressed
                break

        video_stream.stop()
        cv2.destroyAllWindows()
