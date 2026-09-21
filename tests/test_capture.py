import numpy as np

import capture
from config import DETECT_SCALE, MIN_FACE_WIDTH


def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def location_with_width(width_px):
    """Toạ độ (top, right, bottom, left) trên khung thu nhỏ cho khuôn mặt rộng width_px ở khung gốc."""
    w = int(width_px * DETECT_SCALE)
    return (10, 10 + w, 10 + w, 10)


def patch_locations(monkeypatch, locations):
    monkeypatch.setattr(capture.face_recognition, 'face_locations', lambda *a, **k: locations)


class TestCheckFrame:
    def test_no_face(self, monkeypatch):
        patch_locations(monkeypatch, [])
        valid, _, box = capture.check_frame(frame())
        assert not valid and box is None

    def test_multiple_faces(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200), location_with_width(200)])
        valid, message, box = capture.check_frame(frame())
        assert not valid and box is None and '1' in message

    def test_face_too_small(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(MIN_FACE_WIDTH - 20)])
        valid, _, box = capture.check_frame(frame())
        assert not valid and box is not None

    def test_valid_face(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(MIN_FACE_WIDTH + 100)])
        valid, _, box = capture.check_frame(frame())
        assert valid
        assert box is not None
        left, top, right, bottom = box
        assert right - left >= MIN_FACE_WIDTH


class TestNextFreeIndex:
    def test_new_name_starts_at_zero(self, tmp_path):
        assert capture.next_free_index('An', str(tmp_path)) == 0

    def test_skips_existing_files(self, tmp_path):
        (tmp_path / 'Huy_0.jpg').write_bytes(b'x')
        (tmp_path / 'Huy_1.jpg').write_bytes(b'x')
        assert capture.next_free_index('Huy', str(tmp_path)) == 2

    def test_other_names_do_not_interfere(self, tmp_path):
        (tmp_path / 'Huy_0.jpg').write_bytes(b'x')
        assert capture.next_free_index('An', str(tmp_path)) == 0
