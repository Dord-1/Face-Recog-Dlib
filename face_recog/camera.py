"""Webcam: mở camera và đọc khung hình mới nhất trên thread nền."""
import threading

import cv2


class CameraError(RuntimeError):
    """Không mở được webcam."""


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
