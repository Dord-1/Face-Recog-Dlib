from face_recog.geometry import face_area, largest_face, scale_locations


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


def test_face_area():
    assert face_area((10, 60, 50, 20)) == (50 - 10) * (60 - 20)


def test_largest_face_picks_biggest_area():
    small, big = (10, 50, 40, 20), (0, 200, 150, 0)
    assert largest_face([small, big]) == big
    assert largest_face([big, small]) == big


def test_largest_face_single():
    assert largest_face([(1, 2, 3, 0)]) == (1, 2, 3, 0)
