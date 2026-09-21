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
