"""Đo độ chính xác nhận diện bằng ảnh có nhãn, mô phỏng đúng pipeline trực tiếp (xem docs/Evaluate.md).

    python evaluate.py [--enroll detect] [--eval eval] [--mode small|full|both] [--thresholds 0.3,0.4,...]

Đăng ký người từ `--enroll` (mặc định detect/). Ảnh test nằm ở `--eval/<tên>/*.jpg`: thư mục trùng tên
người đã đăng ký là ảnh của người đó; thư mục tên khác (vd `unknown/`) là người lạ.
"""
import argparse
import os

import cv2
import face_recognition
import numpy as np

from config import DETECT_DIR, DETECT_SCALE, IMAGE_EXTENSIONS, INV_SCALE
from Recognition import load_known_faces, person_name, scale_locations

EVAL_DIR = 'eval'
WEBCAM_WIDTH = 640
DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))  # 0.30 ... 0.70
MODES = {'small': 'mã hoá trên khung thu nhỏ (cách cũ)', 'full': 'mã hoá trên khung gốc (cách hiện tại)'}
MIN_RECOMMENDED_PER_CLASS = 10


def _norm(name):
    return name.strip().lower()


def list_eval_images(eval_dir):
    """{tên thư mục: [đường dẫn ảnh]} cho mỗi thư mục con của eval_dir có ảnh."""
    found = {}
    for label in sorted(os.listdir(eval_dir)):
        folder = os.path.join(eval_dir, label)
        if not os.path.isdir(folder):
            continue
        images = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.lower().endswith(IMAGE_EXTENSIONS)]
        if images:
            found[label] = images
    return found


def simulate_webcam(bgr, width=WEBCAM_WIDTH):
    """Đưa ảnh về khung có bề rộng `width` (giữ tỉ lệ) như khung webcam, để đo đúng điều kiện chạy thật."""
    height = max(1, round(bgr.shape[0] * width / bgr.shape[1]))
    interpolation = cv2.INTER_AREA if bgr.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(bgr, (width, height), interpolation=interpolation)


def _face_area(location):
    top, right, bottom, left = location
    return (bottom - top) * (right - left)


def encode_test_image(frame, mode):
    """Encoding khuôn mặt lớn nhất theo đúng cách Recognition.recognize(), hoặc None nếu không thấy mặt."""
    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    small_locations = face_recognition.face_locations(cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                                                      number_of_times_to_upsample=0)
    if not small_locations:
        return None
    largest = max(small_locations, key=_face_area)

    if mode == 'small':
        image, locations = cv2.cvtColor(small, cv2.COLOR_BGR2RGB), [largest]
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = scale_locations([largest], INV_SCALE, frame.shape)
    return face_recognition.face_encodings(image, locations)[0]


def measure(enrolled_encodings, enrolled_people, eval_images, mode):
    """Với mỗi ảnh test: (nhãn thật, người gần nhất, khoảng cách); (nhãn, None, None) nếu không thấy mặt.

    Khoảng cách tính một lần; chỉ số theo từng ngưỡng suy ra từ đây (compute_metrics) mà không mã hoá lại.
    """
    records = []
    for label, paths in eval_images.items():
        for path in paths:
            bgr = cv2.imread(path)
            encoding = encode_test_image(simulate_webcam(bgr), mode) if bgr is not None else None
            if encoding is None:
                records.append((_norm(label), None, None))
                continue
            distances = face_recognition.face_distance(enrolled_encodings, encoding)
            best = int(np.argmin(distances))
            records.append((_norm(label), _norm(enrolled_people[best]), float(distances[best])))
    return records


def compute_metrics(records, enrolled, threshold):
    """Đếm kết quả theo một ngưỡng. `enrolled`: tập tên người đã đăng ký (đã chuẩn hoá chữ thường).

    Ảnh của người đã đăng ký: correct (đúng tên), wrong_person (nhận thành người khác), missed (bỏ sót).
    Ảnh người lạ: false_accept (bị nhận nhầm là ai đó), correct_reject (bị từ chối đúng).
    """
    m = dict.fromkeys(('correct', 'wrong_person', 'missed', 'false_accept', 'correct_reject',
                       'no_face_genuine', 'no_face_impostor'), 0)
    for label, nearest, distance in records:
        genuine = label in enrolled
        if distance is None:
            m['no_face_genuine' if genuine else 'no_face_impostor'] += 1
            continue
        accepted = distance <= threshold
        if genuine:
            if not accepted:
                m['missed'] += 1
            elif nearest == label:
                m['correct'] += 1
            else:
                m['wrong_person'] += 1
        elif accepted:
            m['false_accept'] += 1
        else:
            m['correct_reject'] += 1
    m['genuine_total'] = m['correct'] + m['wrong_person'] + m['missed'] + m['no_face_genuine']
    m['impostor_total'] = m['false_accept'] + m['correct_reject'] + m['no_face_impostor']
    return m


def has_enough_data(records, enrolled):
    """Cần cả ảnh đúng người và ảnh người lạ (đã thấy mặt) mới ước lượng được ngưỡng."""
    seen = [(label in enrolled) for label, _, distance in records if distance is not None]
    return any(seen) and not all(seen)


def suggest_threshold(metrics_by_threshold):
    """Ngưỡng ít nhận nhầm nhất (nhầm người + người lạ bị nhận), rồi nhận đúng nhiều nhất, rồi nhỏ nhất."""
    if not metrics_by_threshold:
        return None

    def rank(t):
        m = metrics_by_threshold[t]
        return (m['false_accept'] + m['wrong_person'], -m['correct'], t)

    return min(metrics_by_threshold, key=rank)


def distance_summary(records, enrolled):
    genuine = [d for label, _, d in records if d is not None and label in enrolled]
    impostor = [d for label, _, d in records if d is not None and label not in enrolled]
    return genuine, impostor


def _range(values):
    if not values:
        return '—'
    return f'{min(values):.3f} / {np.mean(values):.3f} / {max(values):.3f} (n={len(values)})'


def format_report(records, enrolled, thresholds, mode):
    lines = [f'=== Chế độ {mode}: {MODES[mode]} ===']
    genuine, impostor = distance_summary(records, enrolled)
    lines.append('Khoảng cách tới người gần nhất (nhỏ nhất / trung bình / lớn nhất):')
    lines.append(f'  đúng người : {_range(genuine)}')
    lines.append(f'  người lạ   : {_range(impostor)}')
    lines.append('')
    lines.append(f'{"ngưỡng":>7} | {"nhận đúng":>9} | {"nhận nhầm người":>15} | '
                 f'{"người lạ bị nhận":>16} | {"bỏ sót":>6} | {"không thấy mặt":>14}')
    by_threshold = {}
    for t in thresholds:
        m = compute_metrics(records, enrolled, t)
        by_threshold[t] = m
        no_face = m['no_face_genuine'] + m['no_face_impostor']
        lines.append(f'{t:7.2f} | {m["correct"]:9d} | {m["wrong_person"]:15d} | '
                     f'{m["false_accept"]:16d} | {m["missed"]:6d} | {no_face:14d}')
    if has_enough_data(records, enrolled):
        best = suggest_threshold(by_threshold)
        m = by_threshold[best]
        lines.append('')
        lines.append(f'Gợi ý ngưỡng: {best:.2f} (nhận nhầm {m["false_accept"] + m["wrong_person"]}, '
                     f'nhận đúng {m["correct"]}/{m["genuine_total"]} ảnh đúng người)')
    else:
        lines.append('')
        lines.append('Chưa đủ dữ liệu để gợi ý ngưỡng: cần cả ảnh đúng người (thư mục trùng tên người '
                     'đã đăng ký) lẫn ảnh người lạ (thư mục tên khác, vd unknown/).')
    return '\n'.join(lines)


def small_sample_warning(records, enrolled):
    genuine, impostor = distance_summary(records, enrolled)
    if min(len(genuine), len(impostor)) < MIN_RECOMMENDED_PER_CLASS:
        return (f'Lưu ý: mẫu nhỏ (đúng người: {len(genuine)}, người lạ: {len(impostor)}); nên có '
                f'>= {MIN_RECOMMENDED_PER_CLASS} ảnh mỗi loại, kết quả chỉ mang tính tham khảo.')
    return None


def parse_thresholds(text):
    values = sorted({round(float(part), 4) for part in text.split(',') if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError('cần ít nhất một ngưỡng')
    return tuple(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Đo độ chính xác nhận diện bằng ảnh có nhãn.')
    parser.add_argument('--enroll', default=DETECT_DIR, help=f'thư mục ảnh đăng ký (mặc định: {DETECT_DIR})')
    parser.add_argument('--eval', default=EVAL_DIR, dest='eval_dir',
                        help=f'thư mục ảnh test, mỗi thư mục con là một người (mặc định: {EVAL_DIR})')
    parser.add_argument('--mode', choices=('small', 'full', 'both'), default='full',
                        help='small = mã hoá khung thu nhỏ (cách cũ), full = khung gốc (hiện tại), '
                             'both = so sánh')
    parser.add_argument('--thresholds', type=parse_thresholds, default=DEFAULT_THRESHOLDS,
                        help='danh sách ngưỡng cách nhau bằng dấu phẩy (mặc định 0.30..0.70)')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.enroll):
        raise SystemExit(f'Không tìm thấy thư mục đăng ký "{args.enroll}"')
    if not os.path.isdir(args.eval_dir):
        raise SystemExit(f'Không tìm thấy thư mục ảnh test "{args.eval_dir}". '
                         f'Tạo {args.eval_dir}/<tên>/*.jpg (xem docs/Evaluate.md).')

    encodings, files, skipped, _ = load_known_faces(args.enroll, os.path.join(args.enroll, '.encodings.pkl'))
    people = [person_name(name) for name in files]
    if not encodings:
        raise SystemExit(f'Không có ảnh đăng ký hợp lệ trong "{args.enroll}"')
    eval_images = list_eval_images(args.eval_dir)
    if not eval_images:
        raise SystemExit(f'Không có ảnh test trong "{args.eval_dir}" (cần {args.eval_dir}/<tên>/*.jpg)')

    enrolled = {_norm(p) for p in people}
    total_images = sum(len(v) for v in eval_images.values())
    print(f'Đăng ký: {len(enrolled)} người, {len(files)} ảnh'
          + (f' (bỏ qua {len(skipped)} ảnh lỗi)' if skipped else ''))
    print(f'Ảnh test: {total_images} ảnh trong {len(eval_images)} thư mục; '
          f'người lạ = thư mục không trùng tên người đã đăng ký\n')

    modes = ('small', 'full') if args.mode == 'both' else (args.mode,)
    for mode in modes:
        records = measure(encodings, people, eval_images, mode)
        print(format_report(records, enrolled, args.thresholds, mode))
        warning = small_sample_warning(records, enrolled)
        if warning:
            print(warning)
        print()


if __name__ == '__main__':
    main()
