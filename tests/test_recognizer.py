import numpy as np

from face_recog import recognizer
from face_recog.smoothing import PENDING, NameSmoother
from tests.helpers import unit_at_distance, vec


def test_recognize_detects_on_small_frame_but_encodes_on_full_frame(monkeypatch):
    calls = {}
    monkeypatch.setattr(recognizer.face_recognition, 'face_locations',
                        lambda img, **kw: calls.update(detect_shape=img.shape) or [(10, 50, 40, 20)])
    monkeypatch.setattr(recognizer.face_recognition, 'face_encodings',
                        lambda img, locs: calls.update(encode_shape=img.shape, encode_locs=locs) or [vec(0)])

    fr = recognizer.FaceRecognition.__new__(recognizer.FaceRecognition)  # bỏ qua nạp ảnh từ detect/
    fr.known_face_encodings, fr.known_face_names = [], []
    fr.smoother = NameSmoother(min_votes=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fr.recognize(frame)

    assert calls['encode_shape'][:2] == (480, 640)                       # mã hoá trên khung gốc
    assert calls['detect_shape'][0] < 480 and calls['detect_shape'][1] < 640  # dò trên khung nhỏ
    inv = recognizer.INV_SCALE
    assert calls['encode_locs'] == [(10 * inv, 50 * inv, 40 * inv, 20 * inv)]
    assert fr.face_locations == [(10, 50, 40, 20)]   # toạ độ khung nhỏ giữ nguyên cho draw_faces
    assert fr.face_names == ['Unknown']


def test_encode_faces_maps_files_to_people(monkeypatch):
    enc = [vec(0), vec(1), vec(2)]
    monkeypatch.setattr(recognizer, 'load_known_faces',
                        lambda: (enc, ['Elon.png', 'Huy_0.jpg', 'Huy_1.jpg'], [], 3))
    fr = recognizer.FaceRecognition()
    assert fr.known_face_names == ['Elon', 'Huy', 'Huy']
    assert len(fr.known_face_encodings) == 3


def test_recognize_smooths_labels_across_calls(monkeypatch):
    """Nhãn chỉ hiện sau đủ phiếu, và một lần nhận diện lạc không làm nhấp nháy nhãn."""
    monkeypatch.setattr(recognizer.face_recognition, 'face_locations', lambda img, **kw: [(10, 50, 40, 20)])
    monkeypatch.setattr(recognizer.face_recognition, 'face_encodings', lambda img, locs: [vec(0)])
    fr = recognizer.FaceRecognition.__new__(recognizer.FaceRecognition)
    fr.known_face_encodings, fr.known_face_names = [unit_at_distance(0.1)], ['Huy']
    fr.smoother = NameSmoother(window=5, min_votes=2)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fr.recognize(frame)
    assert fr.face_names == [PENDING]                       # mới 1 phiếu
    fr.recognize(frame)
    assert fr.face_names[0].startswith('Huy ')              # đủ 2 phiếu

    fr.known_face_encodings = []                            # một lần nhận diện lạc: nhìn ra 'Unknown'
    fr.recognize(frame)
    assert fr.face_names[0].startswith('Huy ')              # nhãn vẫn giữ nguyên
