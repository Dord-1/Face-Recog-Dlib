import threading
import time

import numpy as np
import pytest

from face_recog import camera
from face_recog.camera import CameraError, VideoStream


class FakeCapture:
    """Thay cv2.VideoCapture: khung hình có giá trị tăng dần để phân biệt khung mới/cũ."""

    def __init__(self, opened=True):
        self.opened = opened
        self.settings = []
        self.released = threading.Event()
        self.reads = 0

    def set(self, prop, value):
        self.settings.append((prop, value))

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        time.sleep(0.001)
        return True, np.full((4, 4, 3), self.reads % 250, dtype=np.uint8)

    def release(self):
        self.released.set()


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def factory(src):
        holder['cap'] = FakeCapture(opened=holder.get('opened', True))
        holder['src'] = src
        return holder['cap']

    monkeypatch.setattr(camera.cv2, 'VideoCapture', factory)
    return holder


def test_unopenable_camera_raises_camera_error_and_releases(fake):
    fake['opened'] = False
    with pytest.raises(CameraError):
        VideoStream(3)
    assert fake['src'] == 3
    assert fake['cap'].released.is_set()


def test_requested_resolution_is_applied(fake):
    stream = VideoStream(0, width=320, height=240)
    try:
        assert (camera.cv2.CAP_PROP_FRAME_WIDTH, 320) in fake['cap'].settings
        assert (camera.cv2.CAP_PROP_FRAME_HEIGHT, 240) in fake['cap'].settings
    finally:
        stream.stop()


def test_read_returns_the_latest_frame_not_a_backlog(fake):
    stream = VideoStream(0)
    try:
        ok, first = stream.read()
        assert ok
        deadline = time.time() + 2
        latest = first
        while time.time() < deadline and int(latest[0, 0, 0]) == int(first[0, 0, 0]):
            latest = stream.read()[1]
        assert int(latest[0, 0, 0]) != int(first[0, 0, 0])   # thread nền liên tục ghi đè khung mới
    finally:
        stream.stop()


def test_read_returns_a_copy(monkeypatch):
    gate = threading.Event()

    class FrozenCapture(FakeCapture):
        def read(self):
            if self.reads >= 1:          # sau khung đầu tiên, thread nền chờ -> khung nội bộ đứng yên
                gate.wait(timeout=5)
            return super().read()

    monkeypatch.setattr(camera.cv2, 'VideoCapture', lambda src: FrozenCapture())
    stream = VideoStream(0)
    try:
        _, first = stream.read()
        first[:] = 255                    # sửa bản trả về không được làm hỏng khung nội bộ
        _, second = stream.read()
        assert not np.shares_memory(first, second)
        assert not np.all(second == 255)
    finally:
        gate.set()
        stream.stop()


def test_stop_ends_thread_and_releases_camera(fake):
    stream = VideoStream(0)
    stream.stop()
    assert fake['cap'].released.is_set()
    assert not stream.thread.is_alive()
    reads_after_stop = fake['cap'].reads
    time.sleep(0.05)
    assert fake['cap'].reads == reads_after_stop            # không còn đọc sau khi dừng
