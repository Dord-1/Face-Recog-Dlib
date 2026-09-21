import argparse

import cv2
import numpy as np
import pytest

from face_recog.tools import evaluate
from face_recog.tools.evaluate import (
    compute_metrics,
    format_report,
    has_enough_data,
    list_eval_images,
    parse_thresholds,
    simulate_webcam,
    small_sample_warning,
    suggest_threshold,
)

ENROLLED = {'huy', 'elon'}


def rec(label, nearest, distance):
    return (label, nearest, distance)


class TestComputeMetrics:
    def test_genuine_correct_wrong_and_missed(self):
        records = [
            rec('huy', 'huy', 0.3),      # đúng
            rec('huy', 'elon', 0.4),     # nhận thành người khác
            rec('huy', 'huy', 0.7),      # xa quá ngưỡng: bỏ sót
        ]
        m = compute_metrics(records, ENROLLED, 0.5)
        assert (m['correct'], m['wrong_person'], m['missed']) == (1, 1, 1)
        assert m['genuine_total'] == 3 and m['impostor_total'] == 0

    def test_impostor_accepted_or_rejected(self):
        records = [rec('unknown', 'huy', 0.45), rec('unknown', 'huy', 0.8)]
        m = compute_metrics(records, ENROLLED, 0.5)
        assert (m['false_accept'], m['correct_reject']) == (1, 1)
        assert m['impostor_total'] == 2

    def test_no_face_counted_separately_for_each_class(self):
        records = [rec('huy', None, None), rec('unknown', None, None)]
        m = compute_metrics(records, ENROLLED, 0.5)
        assert m['no_face_genuine'] == 1 and m['no_face_impostor'] == 1
        assert m['correct'] == m['false_accept'] == 0

    def test_threshold_is_inclusive(self):
        assert compute_metrics([rec('huy', 'huy', 0.5)], ENROLLED, 0.5)['correct'] == 1
        assert compute_metrics([rec('huy', 'huy', 0.5)], ENROLLED, 0.49)['missed'] == 1

    def test_higher_threshold_never_reduces_acceptance(self):
        records = [rec('huy', 'huy', d) for d in (0.2, 0.4, 0.55)] + [rec('unknown', 'elon', 0.62)]
        accepted = [
            m['correct'] + m['wrong_person'] + m['false_accept']
            for m in (compute_metrics(records, ENROLLED, t) for t in (0.3, 0.5, 0.6, 0.7))
        ]
        assert accepted == sorted(accepted)


class TestSuggestThreshold:
    def sweep(self, records, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
        return {t: compute_metrics(records, ENROLLED, t) for t in thresholds}

    def test_prefers_zero_false_accepts_then_most_correct(self):
        records = [rec('huy', 'huy', 0.25), rec('huy', 'huy', 0.45), rec('huy', 'huy', 0.55),
                   rec('unknown', 'huy', 0.58), rec('unknown', 'elon', 0.7)]
        # 0.5: 2 đúng, 0 nhận nhầm; 0.6: 3 đúng nhưng người lạ 0.58 bị nhận -> chọn 0.5
        assert suggest_threshold(self.sweep(records)) == 0.5

    def test_ties_choose_smallest_threshold(self):
        records = [rec('huy', 'huy', 0.2), rec('unknown', 'huy', 0.95)]
        assert suggest_threshold(self.sweep(records)) == 0.3

    def test_when_every_threshold_has_errors_picks_fewest(self):
        records = [rec('unknown', 'huy', 0.1), rec('unknown', 'huy', 0.35), rec('huy', 'huy', 0.05)]
        sweep = self.sweep(records)
        best = suggest_threshold(sweep)
        def errors(t):
            return sweep[t]['false_accept'] + sweep[t]['wrong_person']

        assert errors(best) == min(errors(t) for t in sweep)

    def test_empty(self):
        assert suggest_threshold({}) is None


class TestDataSufficiency:
    def test_needs_both_classes(self):
        genuine = [rec('huy', 'huy', 0.3)]
        impostor = [rec('unknown', 'huy', 0.8)]
        assert has_enough_data(genuine + impostor, ENROLLED)
        assert not has_enough_data(genuine, ENROLLED)
        assert not has_enough_data(impostor, ENROLLED)
        assert not has_enough_data([], ENROLLED)

    def test_images_without_a_face_do_not_count(self):
        assert not has_enough_data([rec('huy', None, None), rec('unknown', 'huy', 0.8)], ENROLLED)

    def test_small_sample_warning(self):
        few = [rec('huy', 'huy', 0.3)] * 3 + [rec('unknown', 'huy', 0.8)] * 3
        assert 'mẫu nhỏ' in small_sample_warning(few, ENROLLED)
        enough = [rec('huy', 'huy', 0.3)] * 10 + [rec('unknown', 'huy', 0.8)] * 10
        assert small_sample_warning(enough, ENROLLED) is None


def test_report_says_not_enough_data_instead_of_a_recommendation():
    text = format_report([rec('huy', 'huy', 0.3)], ENROLLED, (0.4, 0.5), 'full')
    assert 'Chưa đủ dữ liệu' in text and 'Gợi ý ngưỡng' not in text


def test_report_recommends_when_data_is_sufficient():
    records = [rec('huy', 'huy', 0.3), rec('unknown', 'huy', 0.8)]
    text = format_report(records, ENROLLED, (0.4, 0.5), 'full')
    assert 'Gợi ý ngưỡng: 0.40' in text and 'Chưa đủ dữ liệu' not in text


class TestFilesystem:
    def test_list_eval_images_groups_by_folder_and_ignores_others(self, tmp_path):
        (tmp_path / 'Huy').mkdir()
        (tmp_path / 'Huy' / 'a.jpg').write_bytes(b'x')
        (tmp_path / 'Huy' / 'b.PNG').write_bytes(b'x')
        (tmp_path / 'Huy' / 'notes.txt').write_bytes(b'x')
        (tmp_path / 'empty').mkdir()
        (tmp_path / 'stray.jpg').write_bytes(b'x')
        found = list_eval_images(str(tmp_path))
        assert list(found) == ['Huy']
        assert [p.split('/')[-1] for p in found['Huy']] == ['a.jpg', 'b.PNG']

    def test_simulate_webcam_resizes_to_width_keeping_aspect(self):
        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert simulate_webcam(big).shape == (360, 640, 3)
        small = np.zeros((416, 416, 3), dtype=np.uint8)
        assert simulate_webcam(small).shape == (640, 640, 3)

    def test_parse_thresholds_sorts_and_dedups(self):
        assert parse_thresholds('0.5, 0.3,0.5') == (0.3, 0.5)
        with pytest.raises(argparse.ArgumentTypeError):
            parse_thresholds(' , ')


class TestMeasure:
    def make_images(self, tmp_path):
        """Ảnh phẳng có màu khác nhau; encode_test_image được thay bằng bảng tra theo màu."""
        for label, name, value in (('Huy', 'a.jpg', 10), ('unknown', 'b.jpg', 90), ('unknown', 'c.jpg', 30)):
            (tmp_path / label).mkdir(exist_ok=True)
            cv2.imwrite(str(tmp_path / label / name), np.full((50, 50, 3), value, dtype=np.uint8))
        return list_eval_images(str(tmp_path))

    def test_records_nearest_person_distance_and_no_face(self, tmp_path, monkeypatch):
        eval_images = self.make_images(tmp_path)
        # màu -> encoding (None = không thấy mặt)
        by_color = {10: np.zeros(128), 90: np.full(128, 0.9), 30: None}
        monkeypatch.setattr(evaluate, 'encode_test_image',
                            lambda frame, mode: by_color[int(frame[0, 0, 0])])

        enrolled = [np.zeros(128), np.full(128, 1.0)]                      # Huy, Elon
        records = evaluate.measure(enrolled, ['Huy', 'Elon'], eval_images, 'full')

        assert ('huy', 'huy', 0.0) in records                              # trùng Huy đã đăng ký
        elon_hit = [r for r in records if r[0] == 'unknown' and r[1] == 'elon']
        assert len(elon_hit) == 1
        assert elon_hit[0][2] == pytest.approx(0.1 * np.sqrt(128))         # người lạ gần Elon hơn Huy
        assert ('unknown', None, None) in records                          # không thấy mặt
        assert len(records) == 3

    def test_unreadable_image_counts_as_no_face(self, tmp_path, monkeypatch):
        (tmp_path / 'Huy').mkdir()
        (tmp_path / 'Huy' / 'broken.jpg').write_bytes(b'not an image')
        monkeypatch.setattr(evaluate, 'encode_test_image',
                            lambda frame, mode: pytest.fail('không được mã hoá ảnh không đọc được'))
        records = evaluate.measure([np.zeros(128)], ['Huy'], list_eval_images(str(tmp_path)), 'full')
        assert records == [('huy', None, None)]

    def test_mode_is_passed_through(self, tmp_path, monkeypatch):
        eval_images = self.make_images(tmp_path)
        modes = []
        monkeypatch.setattr(evaluate, 'encode_test_image', lambda frame, mode: modes.append(mode))
        evaluate.measure([np.zeros(128)], ['Huy'], eval_images, 'small')
        assert set(modes) == {'small'}
