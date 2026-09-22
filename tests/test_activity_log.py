import os

from face_recog.activity_log import log_event, read_events


def path(tmp_path):
    return str(tmp_path / 'activity.log')


def test_read_events_empty_when_no_file(tmp_path):
    assert read_events(log_path=path(tmp_path)) == []


def test_log_event_then_read_back(tmp_path):
    p = path(tmp_path)
    log_event('CAPTURE', 'Huy_0.jpg (Huy)', p)
    events = read_events(log_path=p)
    assert len(events) == 1
    assert 'CAPTURE' in events[0] and 'Huy_0.jpg (Huy)' in events[0]


def test_events_kept_in_write_order(tmp_path):
    p = path(tmp_path)
    log_event('CAPTURE', 'first', p)
    log_event('RECOGNIZE', 'second', p)
    events = read_events(log_path=p)
    assert 'first' in events[0]
    assert 'second' in events[1]


def test_limit_returns_only_most_recent(tmp_path):
    p = path(tmp_path)
    for i in range(5):
        log_event('CAPTURE', f'item{i}', p)
    events = read_events(limit=2, log_path=p)
    assert len(events) == 2
    assert 'item3' in events[0] and 'item4' in events[1]


def test_creates_parent_directory(tmp_path):
    p = str(tmp_path / 'nested' / 'activity.log')
    log_event('CAPTURE', 'x', p)
    assert os.path.exists(p)


def test_log_event_does_not_raise_on_unwritable_path(tmp_path):
    # đường dẫn nằm dưới một file (không phải thư mục) -> makedirs/open sẽ lỗi
    blocker = tmp_path / 'blocker'
    blocker.write_bytes(b'x')
    log_event('CAPTURE', 'x', str(blocker / 'activity.log'))  # không raise
