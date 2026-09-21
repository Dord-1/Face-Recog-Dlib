import pytest

from face_recog.matching import face_confidence, format_label, match_face
from face_recog.smoothing import PENDING
from tests.helpers import unit_at_distance, vec


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
        from face_recog.config import RECOGNITION_THRESHOLD

        edge = [unit_at_distance(RECOGNITION_THRESHOLD)]
        assert match_face(vec(0), edge, ['edge.jpg'])[0] == 'edge.jpg'
        beyond = [unit_at_distance(RECOGNITION_THRESHOLD + 0.01)]
        assert match_face(vec(0), beyond, ['edge.jpg']) == ('Unknown', 'Unknown')

    def test_recognition_threshold_stricter_than_library_default(self):
        from face_recog.config import MATCH_THRESHOLD, RECOGNITION_THRESHOLD

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
