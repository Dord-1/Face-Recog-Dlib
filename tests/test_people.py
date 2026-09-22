from face_recog.known_faces import load_cache, save_cache
from face_recog.people import delete_person, list_people


def make_images(folder, *names):
    for name in names:
        (folder / name).write_bytes(b'x')


def test_list_people_groups_by_name(tmp_path):
    make_images(tmp_path, 'Huy_0.jpg', 'Huy_1.jpg', 'Elon.png')
    people = list_people(str(tmp_path))
    assert people == [
        {'name': 'Elon', 'count': 1, 'files': ['Elon.png']},
        {'name': 'Huy', 'count': 2, 'files': ['Huy_0.jpg', 'Huy_1.jpg']},
    ]


def test_list_people_empty_folder(tmp_path):
    assert list_people(str(tmp_path)) == []


def test_delete_person_removes_files_and_returns_count(tmp_path):
    make_images(tmp_path, 'Huy_0.jpg', 'Huy_1.jpg', 'Elon.png')
    cache_path = str(tmp_path / '.encodings.pkl')

    count = delete_person('Huy', str(tmp_path), cache_path)

    assert count == 2
    remaining = {p.name for p in tmp_path.iterdir()} - {'.encodings.pkl'}
    assert remaining == {'Elon.png'}


def test_delete_person_removes_matching_cache_entries(tmp_path):
    make_images(tmp_path, 'Huy_0.jpg', 'Elon.png')
    cache_path = str(tmp_path / '.encodings.pkl')
    save_cache(
        {'Huy_0.jpg': {'mtime': 1, 'encoding': None}, 'Elon.png': {'mtime': 1, 'encoding': None}},
        cache_path,
    )

    delete_person('Huy', str(tmp_path), cache_path)

    assert set(load_cache(cache_path)) == {'Elon.png'}


def test_delete_unknown_person_removes_nothing(tmp_path):
    make_images(tmp_path, 'Elon.png')
    assert delete_person('Nobody', str(tmp_path), str(tmp_path / '.encodings.pkl')) == 0
    assert (tmp_path / 'Elon.png').exists()
