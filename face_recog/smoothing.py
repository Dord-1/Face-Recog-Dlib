"""Làm mượt nhãn nhận diện qua nhiều lần chạy bằng bỏ phiếu theo từng khuôn mặt (track)."""
from collections import Counter, deque

from face_recog.config import TRACK_MAX_DIST_RATIO, TRACK_MAX_MISSES, VOTE_MIN, VOTE_WINDOW

PENDING = '...'  # chưa đủ phiếu để kết luận


def _center_and_width(location):
    top, right, bottom, left = location
    return ((left + right) / 2, (top + bottom) / 2), max(right - left, 1)


class NameSmoother:
    """Ghép mỗi khuôn mặt với track trước đó (theo khoảng cách tâm hộp) và bỏ phiếu tên qua các lần nhận diện.

    `update(locations, results)`: `locations` là toạ độ (top, right, bottom, left) cùng hệ với các lần trước,
    `results` là danh sách (tên, độ_tin_cậy) thô tương ứng. Trả về danh sách (tên, độ_tin_cậy) đã làm mượt:
    tên thắng phiếu kèm độ tin cậy của lần gần nhất mang tên đó, hoặc (PENDING, '') nếu chưa đủ `min_votes`.
    """

    def __init__(self, window=VOTE_WINDOW, min_votes=VOTE_MIN, max_misses=TRACK_MAX_MISSES,
                 max_dist_ratio=TRACK_MAX_DIST_RATIO):
        self.window = window
        self.min_votes = min_votes
        self.max_misses = max_misses
        self.max_dist_ratio = max_dist_ratio
        self.tracks = []

    def update(self, locations, results):
        faces = [(_center_and_width(loc), res) for loc, res in zip(locations, results, strict=True)]
        assignment = self._match(faces)

        outputs = []
        matched_tracks = set()
        for index, ((center, width), result) in enumerate(faces):
            track = self.tracks[assignment[index]] if index in assignment else None
            if track is None:
                track = {'votes': deque(maxlen=self.window)}
                self.tracks.append(track)
                assignment[index] = len(self.tracks) - 1
            track.update(center=center, width=width, misses=0)
            track['votes'].append(result)
            matched_tracks.add(id(track))
            outputs.append(self._decide(track['votes']))

        for track in self.tracks:
            if id(track) not in matched_tracks:
                track['misses'] += 1
        self.tracks = [t for t in self.tracks if t['misses'] <= self.max_misses]
        return outputs

    def _match(self, faces):
        """Ghép tham lam theo khoảng cách nhỏ nhất: trả về {chỉ số mặt: chỉ số track}."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for fi, ((center, width), _) in enumerate(faces):
                dist = ((center[0] - track['center'][0]) ** 2 + (center[1] - track['center'][1]) ** 2) ** 0.5
                if dist <= self.max_dist_ratio * max(width, track['width']):
                    pairs.append((dist, ti, fi))

        assignment, used_tracks = {}, set()
        for _, ti, fi in sorted(pairs):
            if fi not in assignment and ti not in used_tracks:
                assignment[fi] = ti
                used_tracks.add(ti)
        return assignment

    def _decide(self, votes):
        counts = Counter(name for name, _ in votes)
        best = max(counts.values())
        if best < self.min_votes:
            return PENDING, ''
        tied = {name for name, count in counts.items() if count == best}
        for name, confidence in reversed(votes):  # hoà phiếu -> lấy tên xuất hiện gần đây nhất
            if name in tied:
                return name, confidence
        raise AssertionError('unreachable')  # pragma: no cover
