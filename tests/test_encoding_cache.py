import os

import numpy as np

from Recognition import load_cache, load_known_faces


class FakeEncoder:
    """Thay encode_image: đếm số lần gọi, trả None cho ảnh có tên bắt đầu bằng 'bad'."""

    def __init__(self):
        self.calls = []

    def __call__(self, path):
        self.calls.append(os.path.basename(path))
        if os.path.basename(path).startswith('bad'):
            return None
        return np.full(128, float(len(self.calls)))


def make_images(folder, *names):
    for name in names:
        (folder / name).write_bytes(b'x')


def run(folder, encoder):
    return load_known_faces(str(folder), str(folder / '.encodings.pkl'), encoder)


def test_first_run_encodes_everything(tmp_path):
    make_images(tmp_path, 'a.jpg', 'b.png')
    enc = FakeEncoder()
    encodings, names, skipped, reused = run(tmp_path, enc)
    assert names == ['a.jpg', 'b.png']
    assert len(encodings) == 2 and skipped == [] and reused == 0
    assert enc.calls == ['a.jpg', 'b.png']


def test_second_run_uses_cache(tmp_path):
    make_images(tmp_path, 'a.jpg', 'b.jpg')
    run(tmp_path, FakeEncoder())
    enc = FakeEncoder()
    _, names, _, reused = run(tmp_path, enc)
    assert names == ['a.jpg', 'b.jpg']
    assert reused == 2
    assert enc.calls == []


def test_cached_encoding_values_are_reused(tmp_path):
    make_images(tmp_path, 'a.jpg')
    first, *_ = run(tmp_path, FakeEncoder())
    second, *_ = run(tmp_path, FakeEncoder())
    assert np.array_equal(first[0], second[0])


def test_changed_mtime_reencodes_only_that_image(tmp_path):
    make_images(tmp_path, 'a.jpg', 'b.jpg')
    run(tmp_path, FakeEncoder())
    stat = (tmp_path / 'b.jpg').stat()
    os.utime(tmp_path / 'b.jpg', (stat.st_atime, stat.st_mtime + 10))

    enc = FakeEncoder()
    _, _, _, reused = run(tmp_path, enc)
    assert enc.calls == ['b.jpg']
    assert reused == 1


def test_new_image_encodes_only_new(tmp_path):
    make_images(tmp_path, 'a.jpg')
    run(tmp_path, FakeEncoder())
    make_images(tmp_path, 'c.jpg')
    enc = FakeEncoder()
    _, names, _, _ = run(tmp_path, enc)
    assert names == ['a.jpg', 'c.jpg']
    assert enc.calls == ['c.jpg']


def test_bad_images_skipped_and_cached(tmp_path):
    make_images(tmp_path, 'a.jpg', 'bad.jpg')
    _, names, skipped, _ = run(tmp_path, FakeEncoder())
    assert names == ['a.jpg'] and skipped == ['bad.jpg']

    enc = FakeEncoder()  # lần 2: ảnh lỗi cũng đã được cache, không thử lại
    _, _, skipped, reused = run(tmp_path, enc)
    assert skipped == ['bad.jpg'] and reused == 2
    assert enc.calls == []


def test_removed_image_dropped_from_cache(tmp_path):
    make_images(tmp_path, 'a.jpg', 'b.jpg')
    run(tmp_path, FakeEncoder())
    (tmp_path / 'b.jpg').unlink()
    run(tmp_path, FakeEncoder())
    assert set(load_cache(str(tmp_path / '.encodings.pkl'))) == {'a.jpg'}


def test_non_image_files_ignored(tmp_path):
    make_images(tmp_path, 'a.jpg', '.DS_Store', 'notes.txt')
    enc = FakeEncoder()
    _, names, _, _ = run(tmp_path, enc)
    assert names == ['a.jpg'] and enc.calls == ['a.jpg']


def test_corrupt_cache_is_ignored(tmp_path):
    make_images(tmp_path, 'a.jpg')
    (tmp_path / '.encodings.pkl').write_bytes(b'junk')
    enc = FakeEncoder()
    _, names, _, reused = run(tmp_path, enc)
    assert names == ['a.jpg'] and reused == 0 and enc.calls == ['a.jpg']


def test_version_mismatch_discards_cache(tmp_path, monkeypatch):
    import Recognition

    make_images(tmp_path, 'a.jpg')
    run(tmp_path, FakeEncoder())
    monkeypatch.setattr(Recognition, 'CACHE_VERSION', Recognition.CACHE_VERSION + 1)
    enc = FakeEncoder()
    _, _, _, reused = run(tmp_path, enc)
    assert reused == 0 and enc.calls == ['a.jpg']


def test_missing_folder_is_created(tmp_path):
    folder = tmp_path / 'detect'
    encodings, names, skipped, reused = run(folder, FakeEncoder())
    assert folder.is_dir()
    assert (encodings, names, skipped, reused) == ([], [], [], 0)
