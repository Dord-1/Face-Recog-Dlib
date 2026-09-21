import numpy as np
import pytest

from Recognition import face_confidence, match_face


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

    def test_threshold_is_inclusive(self):
        known = [unit_at_distance(0.6)]
        name, _ = match_face(vec(0), known, ['edge.jpg'])
        assert name == 'edge.jpg'

    def test_custom_threshold(self):
        known = [unit_at_distance(0.4)]
        assert match_face(vec(0), known, ['a.jpg'], threshold=0.3) == ('Unknown', 'Unknown')
