import cv2
import numpy as np
import pytest

from face_recog import quality
from face_recog.config import DETECT_SCALE, MAX_YAW_RATIO, MIN_FACE_WIDTH


def frame():
    """Khung có vân ngẫu nhiên, sáng vừa phải: qua được kiểm tra độ sáng và độ nét."""
    rng = np.random.default_rng(0)
    return rng.integers(70, 190, size=(480, 640, 3), dtype=np.uint8)


def structured_frame():
    """Khung có cấu trúc lớn + chi tiết mịn (giống ảnh thật hơn nhiễu trắng): còn tương phản khi bị làm mờ."""
    rng = np.random.default_rng(3)
    big = cv2.resize(rng.integers(40, 220, size=(12, 16), dtype=np.uint8), (640, 480),
                     interpolation=cv2.INTER_CUBIC)
    fine = np.clip(big.astype(int) + rng.integers(-25, 25, size=(480, 640)), 0, 255).astype(np.uint8)
    return cv2.merge([fine] * 3)


def location_with_width(width_px):
    """Toạ độ (top, right, bottom, left) trên khung thu nhỏ cho khuôn mặt rộng width_px ở khung gốc."""
    w = int(width_px * DETECT_SCALE)
    return (10, 10 + w, 10 + w, 10)


def patch_locations(monkeypatch, locations):
    monkeypatch.setattr(quality.face_recognition, 'face_locations', lambda *a, **k: locations)


def frontal_landmarks():
    return [{'nose_tip': [(50, 60)], 'left_eye': [(40, 40), (42, 40)], 'right_eye': [(58, 40), (60, 40)]}]


def turned_landmarks():
    # mũi lệch hẳn về phía mắt phải: khoảng cách tới mắt phải ngắn hơn nhiều
    return [{'nose_tip': [(56, 60)], 'left_eye': [(30, 40), (32, 40)], 'right_eye': [(58, 40), (60, 40)]}]


def patch_landmarks(monkeypatch, landmarks):
    monkeypatch.setattr(quality.face_recognition, 'face_landmarks', lambda *a, **k: landmarks)


class TestCheckFrame:
    def test_no_face(self, monkeypatch):
        patch_locations(monkeypatch, [])
        valid, _, box = quality.check_frame(frame())
        assert not valid and box is None

    def test_multiple_faces(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200), location_with_width(200)])
        valid, message, box = quality.check_frame(frame())
        assert not valid and box is None and '1' in message

    def test_face_too_small(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(MIN_FACE_WIDTH - 20)])
        valid, _, box = quality.check_frame(frame())
        assert not valid and box is not None

    def test_valid_face(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(MIN_FACE_WIDTH + 100)])
        patch_landmarks(monkeypatch, frontal_landmarks())
        valid, _, box = quality.check_frame(frame())
        assert valid
        left, top, right, bottom = box
        assert right - left >= MIN_FACE_WIDTH

    def test_too_dark(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        patch_landmarks(monkeypatch, frontal_landmarks())
        valid, message, box = quality.check_frame((frame() * 0.2).astype(np.uint8))
        assert not valid and 'toi' in message and box is not None

    def test_too_bright(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        patch_landmarks(monkeypatch, frontal_landmarks())
        valid, message, _ = quality.check_frame(np.full((480, 640, 3), 245, dtype=np.uint8))
        assert not valid and 'sang' in message

    def test_blurry(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        patch_landmarks(monkeypatch, frontal_landmarks())
        sharp_frame = structured_frame()
        assert quality.check_frame(sharp_frame)[0]                       # bản nét thì hợp lệ
        valid, message, _ = quality.check_frame(cv2.GaussianBlur(sharp_frame, (0, 0), 4))
        assert not valid and 'mo' in message

    def test_face_not_frontal(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        patch_landmarks(monkeypatch, turned_landmarks())
        valid, message, _ = quality.check_frame(frame())
        assert not valid and 'nhin thang' in message

    def test_missing_landmarks_does_not_block_capture(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        patch_landmarks(monkeypatch, [])
        valid, _, _ = quality.check_frame(frame())
        assert valid

    def test_landmarks_not_computed_when_earlier_check_fails(self, monkeypatch):
        patch_locations(monkeypatch, [location_with_width(200)])
        monkeypatch.setattr(quality.face_recognition, 'face_landmarks',
                            lambda *a, **k: pytest.fail('không được tính landmarks khi đã thất bại'))
        valid, _, _ = quality.check_frame((frame() * 0.2).astype(np.uint8))
        assert not valid


class TestQualityMetrics:
    def textured(self):
        return np.random.default_rng(1).integers(40, 220, size=(100, 100), dtype=np.uint8)

    def test_sharpness_drops_with_blur(self):
        sharp = self.textured()
        assert quality.sharpness(cv2.GaussianBlur(sharp, (0, 0), 3)) < quality.sharpness(sharp) / 3

    def test_sharpness_is_invariant_to_darkening(self):
        img = self.textured()
        darker = (img * 0.5).astype(np.uint8)
        assert quality.sharpness(darker) == pytest.approx(quality.sharpness(img), rel=0.15)

    def test_sharpness_of_flat_image_is_zero(self):
        assert quality.sharpness(np.full((100, 100), 120, dtype=np.uint8)) == 0.0

    def test_nearly_flat_image_is_not_scored_sharp_from_quantization_noise(self):
        # nhiễu trắng làm mờ mạnh: phương sai xám chỉ vài đơn vị, chỉ còn nhiễu lượng tử hoá
        rng = np.random.default_rng(0)
        noise = rng.integers(70, 190, size=(300, 300), dtype=np.uint8)
        nearly_flat = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 6)[100:200, 100:200], (100, 100))
        assert quality.sharpness(nearly_flat) < 0.1

    def test_brightness_is_mean_gray(self):
        assert quality.brightness(np.full((100, 100), 77, dtype=np.uint8)) == pytest.approx(77.0)

    def test_yaw_ratio_frontal_is_about_one(self):
        assert quality.yaw_ratio(frontal_landmarks()[0]) == pytest.approx(1.0, abs=0.1)

    def test_yaw_ratio_grows_when_turned(self):
        assert quality.yaw_ratio(turned_landmarks()[0]) > MAX_YAW_RATIO

    def test_yaw_ratio_is_symmetric_and_at_least_one(self):
        lm = turned_landmarks()[0]
        mirrored = dict(lm, left_eye=lm['right_eye'], right_eye=lm['left_eye'])
        assert quality.yaw_ratio(lm) == pytest.approx(quality.yaw_ratio(mirrored))
        assert quality.yaw_ratio(lm) >= 1.0
