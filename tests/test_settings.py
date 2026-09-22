import json

from face_recog.settings import DEFAULTS, load_settings, save_settings


def path(tmp_path):
    return str(tmp_path / 'settings.json')


def test_missing_file_returns_defaults(tmp_path):
    assert load_settings(path(tmp_path)) == DEFAULTS


def test_corrupt_file_returns_defaults(tmp_path):
    (tmp_path / 'settings.json').write_text('not json', encoding='utf-8')
    assert load_settings(path(tmp_path)) == DEFAULTS


def test_save_then_load_round_trips(tmp_path):
    values = {'recognition_threshold': 0.4, 'process_every_n': 5}
    error = save_settings(values, path(tmp_path))
    assert error is None
    assert load_settings(path(tmp_path)) == values


def test_out_of_range_threshold_rejected(tmp_path):
    error = save_settings({'recognition_threshold': 1.5, 'process_every_n': 3}, path(tmp_path))
    assert error is not None
    assert load_settings(path(tmp_path)) == DEFAULTS  # không ghi gì


def test_non_positive_process_every_n_rejected(tmp_path):
    error = save_settings({'recognition_threshold': 0.5, 'process_every_n': 0}, path(tmp_path))
    assert error is not None


def test_partial_bad_key_falls_back_to_default_for_that_key(tmp_path):
    p = path(tmp_path)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'recognition_threshold': 'bad', 'process_every_n': 4}, f)
    values = load_settings(p)
    assert values['recognition_threshold'] == DEFAULTS['recognition_threshold']
    assert values['process_every_n'] == 4
