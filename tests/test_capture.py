import numpy as np

from face_recog import capture


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


class FakeCam:
    def __init__(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def release(self):
        pass


class TestImgCapture:
    """Mô phỏng webcam + hộp thoại để kiểm tra ảnh chỉ lưu khi hợp lệ và có ghi nhật ký."""

    def prepare(self, monkeypatch, tmp_path, name='Huy', valid=True, keys=(32, 27)):
        monkeypatch.setattr(capture, 'DETECT_DIR', str(tmp_path))
        monkeypatch.setattr(capture.simpledialog, 'askstring', lambda **kw: name)
        monkeypatch.setattr(capture.cv2, 'VideoCapture', lambda src: FakeCam())
        monkeypatch.setattr(capture.cv2, 'imshow', lambda *a: None)
        monkeypatch.setattr(capture.cv2, 'rectangle', lambda *a, **k: None)
        monkeypatch.setattr(capture.cv2, 'putText', lambda *a, **k: None)
        monkeypatch.setattr(capture.cv2, 'destroyAllWindows', lambda: None)
        monkeypatch.setattr(capture, 'check_frame',
                            lambda frame: (valid, 'ok' if valid else 'no', (0, 10, 10, 0)))

        key_iter = iter(keys)
        monkeypatch.setattr(capture.cv2, 'waitKey', lambda ms: next(key_iter))

        logged = []
        monkeypatch.setattr(capture, 'log_event', lambda kind, detail: logged.append((kind, detail)))
        saved = []
        monkeypatch.setattr(capture.cv2, 'imwrite', lambda path, frame: saved.append(path))
        return logged, saved

    def test_space_on_valid_frame_saves_and_logs(self, monkeypatch, tmp_path):
        logged, saved = self.prepare(monkeypatch, tmp_path, valid=True)
        capture.img_capture(parent=None)
        assert len(saved) == 1
        assert logged and logged[0][0] == 'CAPTURE'

    def test_space_on_invalid_frame_does_not_save_or_log(self, monkeypatch, tmp_path):
        logged, saved = self.prepare(monkeypatch, tmp_path, valid=False)
        capture.img_capture(parent=None)
        assert saved == [] and logged == []

    def test_no_name_entered_does_nothing(self, monkeypatch, tmp_path):
        logged, saved = self.prepare(monkeypatch, tmp_path, name=None)
        capture.img_capture(parent=None)
        assert saved == [] and logged == []
