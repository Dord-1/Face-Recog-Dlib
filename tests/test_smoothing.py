import pytest

from smoothing import PENDING, NameSmoother

FACE = (100, 200, 200, 100)           # (top, right, bottom, left): mặt rộng 100
FAR_FACE = (100, 700, 200, 600)       # cách xa FACE hơn nhiều bề rộng mặt


def raw(name, confidence='90%'):
    return (name, confidence)


def run(smoother, sequence, location=FACE):
    """Chạy chuỗi kết quả thô cho 1 khuôn mặt đứng yên; trả về danh sách đầu ra sau mỗi lần."""
    return [smoother.update([location], [r])[0] for r in sequence]


class TestVoting:
    def test_pending_until_min_votes(self):
        out = run(NameSmoother(window=5, min_votes=3), [raw('Huy')] * 3)
        assert out[0] == (PENDING, '') and out[1] == (PENDING, '')
        assert out[2][0] == 'Huy'

    def test_single_glitch_does_not_flicker(self):
        out = run(NameSmoother(window=5, min_votes=3),
                  [raw('Huy'), raw('Huy'), raw('Huy'), raw('Unknown', 'Unknown'), raw('Huy')])
        assert [o[0] for o in out[2:]] == ['Huy', 'Huy', 'Huy']

    def test_switches_after_majority_changes(self):
        out = run(NameSmoother(window=5, min_votes=3), [raw('Huy')] * 3 + [raw('Elon')] * 4)
        assert out[2][0] == 'Huy'
        assert out[-1][0] == 'Elon'

    def test_window_limits_history(self):
        s = NameSmoother(window=3, min_votes=2)
        out = run(s, [raw('Huy')] * 3 + [raw('Elon')] * 2)
        assert out[-1][0] == 'Elon'                 # 3 phiếu Huy cũ đã bị đẩy khỏi cửa sổ 3

    def test_confidence_comes_from_latest_vote_of_winner(self):
        out = run(NameSmoother(window=5, min_votes=2),
                  [raw('Huy', '80%'), raw('Huy', '95%'), raw('Unknown', 'Unknown')])
        assert out[-1] == ('Huy', '95%')

    def test_tie_prefers_most_recent_name(self):
        out = run(NameSmoother(window=4, min_votes=2),
                  [raw('Huy'), raw('Huy'), raw('Elon'), raw('Elon')])
        assert out[-1][0] == 'Elon'

    def test_unknown_can_win_a_vote(self):
        out = run(NameSmoother(window=5, min_votes=3), [raw('Unknown', 'Unknown')] * 3)
        assert out[-1][0] == 'Unknown'


class TestTracking:
    def test_two_faces_keep_separate_votes_even_if_order_swaps(self):
        s = NameSmoother(window=5, min_votes=2)
        for _ in range(3):
            s.update([FACE, FAR_FACE], [raw('Huy'), raw('Elon')])
        out = s.update([FAR_FACE, FACE], [raw('Elon'), raw('Huy')])   # thứ tự đảo
        assert out[0][0] == 'Elon' and out[1][0] == 'Huy'

    def test_face_moving_a_little_stays_on_same_track(self):
        s = NameSmoother(window=5, min_votes=3)
        out = []
        for shift in (0, 10, 20, 30):
            loc = (100, 200 + shift, 200, 100 + shift)
            out.append(s.update([loc], [raw('Huy')])[0])
        assert out[2][0] == 'Huy'

    def test_far_new_face_starts_new_track(self):
        s = NameSmoother(window=5, min_votes=2)
        run(s, [raw('Huy')] * 3)
        out = s.update([FAR_FACE], [raw('Elon')])[0]
        assert out == (PENDING, '')                 # không thừa hưởng phiếu của Huy

    def test_short_absence_keeps_votes(self):
        s = NameSmoother(window=5, min_votes=3, max_misses=2)
        run(s, [raw('Huy')] * 3)
        s.update([], [])
        assert s.update([FACE], [raw('Huy')])[0][0] == 'Huy'

    def test_long_absence_expires_track(self):
        s = NameSmoother(window=5, min_votes=3, max_misses=2)
        run(s, [raw('Huy')] * 3)
        for _ in range(3):
            s.update([], [])
        assert s.tracks == []
        assert s.update([FACE], [raw('Elon')])[0] == (PENDING, '')

    def test_no_faces_is_fine(self):
        assert NameSmoother().update([], []) == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            NameSmoother().update([FACE], [])
