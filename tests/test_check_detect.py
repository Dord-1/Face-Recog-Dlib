import numpy as np

from Check_Detect import face_area, group_same_person


def entry(name, x, area=100):
    """Ảnh giả có encoding nằm ở toạ độ x trên trục đầu (khoảng cách = chênh lệch x)."""
    v = np.zeros(128)
    v[0] = x
    return {'name': name, 'encoding': v, 'area': area}


def names(groups):
    return sorted(sorted(e['name'] for e in g) for g in groups)


def test_face_area():
    assert face_area((10, 60, 50, 20)) == (50 - 10) * (60 - 20)


def test_groups_close_faces_and_ignores_singletons():
    entries = [entry('a', 0.0), entry('b', 0.1), entry('c', 5.0)]
    assert names(group_same_person(entries, 0.6)) == [['a', 'b']]


def test_transitive_grouping():
    # a~b (0.5) và b~c (0.5) nhưng a-c = 1.0 > ngưỡng: vẫn phải gộp cả ba
    entries = [entry('a', 0.0), entry('b', 0.5), entry('c', 1.0)]
    assert names(group_same_person(entries, 0.6)) == [['a', 'b', 'c']]


def test_threshold_boundary_is_inclusive():
    entries = [entry('a', 0.0), entry('b', 0.5)]
    assert names(group_same_person(entries, 0.5)) == [['a', 'b']]
    assert group_same_person(entries, 0.49) == []


def test_two_separate_people():
    entries = [entry('a', 0.0), entry('b', 0.1), entry('c', 9.0), entry('d', 9.2)]
    assert names(group_same_person(entries, 0.6)) == [['a', 'b'], ['c', 'd']]


def test_empty_and_single():
    assert group_same_person([], 0.6) == []
    assert group_same_person([entry('a', 0.0)], 0.6) == []


def test_scan_folder_reports_unreadable_image_instead_of_crashing(tmp_path):
    from Check_Detect import scan_folder

    (tmp_path / 'broken.png').write_bytes(b'not an image')
    (tmp_path / 'notes.txt').write_text('ignored')
    entries, no_face, multi_face = scan_folder(str(tmp_path))
    assert entries == [] and no_face == ['broken.png'] and multi_face == []


def test_default_threshold_is_stricter_than_recognition_match():
    from config import DUPLICATE_THRESHOLD, MATCH_THRESHOLD

    assert DUPLICATE_THRESHOLD < MATCH_THRESHOLD


def test_same_person_different_shots_not_grouped_at_default_threshold():
    from config import DUPLICATE_THRESHOLD

    # cùng người nhưng khác góc/ánh sáng (khoảng cách 0.45): không phải ảnh trùng -> giữ cả hai
    entries = [entry('a', 0.0), entry('b', 0.45)]
    assert group_same_person(entries, DUPLICATE_THRESHOLD) == []
    assert names(group_same_person(entries, 0.6)) == [['a', 'b']]
