import numpy as np
import pytest

import Recognition
from Recognition import face_confidence, format_label, match_face, scale_locations
from smoothing import PENDING, NameSmoother


def vec(value):
    """Vector 128 chiều giả, mọi phần tử bằng `value` (khoảng cách tới vec(0) = value * sqrt(128))."""
    return np.full(128, value, dtype=float)


def unit_at_distance(distance):
    """Vector cách vec(0) đúng `distance` (đặt toàn bộ độ lệch vào 1 chiều)."""
    v = np.zeros(128)
    v[0] = distance
    return v


def percent(text):
    assert text.endswith('%')
    return float(text[:-1])


class TestFaceConfidence:
    def test_identical_faces_high_confidence(self):
        assert percent(face_confidence(0.0)) == pytest.approx(97.89, abs=0.01)

    def test_at_threshold_is_50(self):
        assert percent(face_confidence(0.6)) == pytest.approx(50.0)

    def test_beyond_threshold_uses_linear_branch_below_50(self):
        assert percent(face_confidence(0.8)) == pytest.approx(25.0)

    def test_decreasing_in_distance_from_peak(self):
        values = [percent(face_confidence(d)) for d in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)]
        assert values == sorted(values, reverse=True)

    def test_known_quirk_peak_is_at_0_2_not_0(self):
        # Đặc trưng hoá hành vi của công thức gốc (heuristic, không phải xác suất): đỉnh 100% ở
        # distance ~0.2, và khuôn mặt giống hơn (0.0) lại hiển thị thấp hơn một chút (97.89%).
        assert percent(face_confidence(0.2)) == pytest.approx(100.0)
        assert percent(face_confidence(0.0)) < percent(face_confidence(0.2))

    def test_does_not_shadow_builtin_range(self):
        # Hồi quy: hàm từng đặt biến tên `range`; đảm bảo built-in vẫn dùng được sau khi gọi.
        face_confidence(0.3)
        assert list(range(3)) == [0, 1, 2]


class TestMatchFace:
    def test_no_known_faces_is_unknown(self):
        assert match_face(vec(0), [], []) == ('Unknown', 'Unknown')

    def test_matches_closest_within_threshold(self):
        known = [unit_at_distance(0.9), unit_at_distance(0.2)]
        name, confidence = match_face(vec(0), known, ['far.jpg', 'near.jpg'])
        assert name == 'near.jpg'
        assert confidence.endswith('%')

    def test_unknown_when_all_beyond_threshold(self):
        known = [unit_at_distance(0.9), unit_at_distance(0.7)]
        assert match_face(vec(0), known, ['a.jpg', 'b.jpg']) == ('Unknown', 'Unknown')

    def test_default_threshold_is_recognition_threshold_inclusive(self):
        from config import RECOGNITION_THRESHOLD

        edge = [unit_at_distance(RECOGNITION_THRESHOLD)]
        assert match_face(vec(0), edge, ['edge.jpg'])[0] == 'edge.jpg'
        beyond = [unit_at_distance(RECOGNITION_THRESHOLD + 0.01)]
        assert match_face(vec(0), beyond, ['edge.jpg']) == ('Unknown', 'Unknown')

    def test_recognition_threshold_stricter_than_library_default(self):
        from config import MATCH_THRESHOLD, RECOGNITION_THRESHOLD

        assert RECOGNITION_THRESHOLD < MATCH_THRESHOLD
        # 0.55: trước đây (ngưỡng 0.6) được nhận, giờ là người lạ
        assert match_face(vec(0), [unit_at_distance(0.55)], ['a.jpg']) == ('Unknown', 'Unknown')

    def test_confidence_scale_does_not_depend_on_threshold(self):
        known = [unit_at_distance(0.4)]
        _, conf_default = match_face(vec(0), known, ['a.jpg'])
        _, conf_loose = match_face(vec(0), known, ['a.jpg'], threshold=0.6)
        _, conf_strict = match_face(vec(0), known, ['a.jpg'], threshold=0.45)
        assert conf_default == conf_loose == conf_strict == face_confidence(0.4)

    def test_custom_threshold(self):
        known = [unit_at_distance(0.4)]
        assert match_face(vec(0), known, ['a.jpg'], threshold=0.3) == ('Unknown', 'Unknown')


class TestScaleLocations:
    def test_scales_all_coordinates(self):
        assert scale_locations([(10, 50, 40, 20)], 2, (480, 640, 3)) == [(20, 100, 80, 40)]

    def test_clamps_to_frame_bounds(self):
        # bên phải/dưới vượt biên khung, bên trái/trên âm -> kẹp về [0, kích thước]
        assert scale_locations([(-5, 400, 300, -3)], 2, (480, 640, 3)) == [(0, 640, 480, 0)]

    def test_empty(self):
        assert scale_locations([], 2, (480, 640, 3)) == []

    def test_multiple_faces_keep_order(self):
        out = scale_locations([(1, 2, 3, 4), (5, 6, 7, 8)], 3, (100, 100, 3))
        assert out == [(3, 6, 9, 12), (15, 18, 21, 24)]


def test_recognize_detects_on_small_frame_but_encodes_on_full_frame(monkeypatch):
    calls = {}
    monkeypatch.setattr(Recognition.face_recognition, 'face_locations',
                        lambda img, **kw: calls.update(detect_shape=img.shape) or [(10, 50, 40, 20)])
    monkeypatch.setattr(Recognition.face_recognition, 'face_encodings',
                        lambda img, locs: calls.update(encode_shape=img.shape, encode_locs=locs) or [vec(0)])

    fr = Recognition.FaceRecognition.__new__(Recognition.FaceRecognition)  # bỏ qua nạp ảnh từ detect/
    fr.known_face_encodings, fr.known_face_names = [], []
    fr.smoother = NameSmoother(min_votes=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fr.recognize(frame)

    assert calls['encode_shape'][:2] == (480, 640)                       # mã hoá trên khung gốc
    assert calls['detect_shape'][0] < 480 and calls['detect_shape'][1] < 640  # dò trên khung nhỏ
    inv = Recognition.INV_SCALE
    assert calls['encode_locs'] == [(10 * inv, 50 * inv, 40 * inv, 20 * inv)]
    assert fr.face_locations == [(10, 50, 40, 20)]   # toạ độ khung nhỏ giữ nguyên cho draw_faces
    assert fr.face_names == ['Unknown']


class TestPersonName:
    def test_strips_extension_and_numeric_suffix(self):
        assert Recognition.person_name('Huy_0.jpg') == 'Huy'
        assert Recognition.person_name('Huy_12.png') == 'Huy'

    def test_only_last_numeric_suffix_removed(self):
        assert Recognition.person_name('Nguyen_Van_A_3.jpg') == 'Nguyen_Van_A'
        assert Recognition.person_name('Room_101_2.jpg') == 'Room_101'

    def test_name_without_suffix_kept(self):
        assert Recognition.person_name('Elon.png') == 'Elon'

    def test_case_and_path_handled(self):
        assert Recognition.person_name('detect/An_1.JPG') == 'An'

    def test_purely_numeric_stem_not_emptied(self):
        assert Recognition.person_name('_5.jpg') == '_5'


def test_encode_faces_maps_files_to_people(monkeypatch):
    enc = [vec(0), vec(1), vec(2)]
    monkeypatch.setattr(Recognition, 'load_known_faces',
                        lambda: (enc, ['Elon.png', 'Huy_0.jpg', 'Huy_1.jpg'], [], 3))
    fr = Recognition.FaceRecognition()
    assert fr.known_face_names == ['Elon', 'Huy', 'Huy']
    assert len(fr.known_face_encodings) == 3


def test_match_face_returns_person_when_many_images_per_person():
    known = [unit_at_distance(0.45), unit_at_distance(0.1), unit_at_distance(0.9)]
    name, _ = match_face(vec(0), known, ['Huy', 'Huy', 'Elon'])
    assert name == 'Huy'


class TestFormatLabel:
    def test_known_person_shows_name_and_confidence(self):
        assert format_label('Huy', '98.3%') == 'Huy 98.3%'

    def test_unknown_and_pending_have_no_confidence_suffix(self):
        assert format_label('Unknown', 'Unknown') == 'Unknown'
        assert format_label(PENDING, '') == PENDING


def test_recognize_smooths_labels_across_calls(monkeypatch):
    """Nhãn chỉ hiện sau đủ phiếu, và một lần nhận diện lạc không làm nhấp nháy nhãn."""
    monkeypatch.setattr(Recognition.face_recognition, 'face_locations', lambda img, **kw: [(10, 50, 40, 20)])
    monkeypatch.setattr(Recognition.face_recognition, 'face_encodings', lambda img, locs: [vec(0)])
    fr = Recognition.FaceRecognition.__new__(Recognition.FaceRecognition)
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
