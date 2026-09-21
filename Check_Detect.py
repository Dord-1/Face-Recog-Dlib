import argparse
import os

import face_recognition

from config import DETECT_DIR, IMAGE_EXTENSIONS, MATCH_THRESHOLD


def face_area(location):
    top, right, bottom, left = location
    return (bottom - top) * (right - left)


def scan_folder(folder):
    """Trả về (entries, no_face, multi_face).

    entries: list dict {name, encoding, area} cho ảnh có khuôn mặt.
    Nếu ảnh có nhiều khuôn mặt thì lấy khuôn mặt lớn nhất.
    """
    entries, no_face, multi_face = [], [], []

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        try:
            image = face_recognition.load_image_file(os.path.join(folder, name))
        except (OSError, ValueError):  # file hỏng / không phải ảnh thật
            no_face.append(name)
            continue
        locations = face_recognition.face_locations(image)
        if not locations:
            no_face.append(name)
            continue

        if len(locations) > 1:
            multi_face.append(name)
        largest = max(locations, key=face_area)
        encoding = face_recognition.face_encodings(image, [largest])[0]
        entries.append({'name': name, 'encoding': encoding, 'area': face_area(largest)})

    return entries, no_face, multi_face


def group_same_person(entries, threshold):
    """Gom các ảnh có khoảng cách khuôn mặt <= threshold thành cùng nhóm (union-find)."""
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entries)):
        others = [e['encoding'] for e in entries[i + 1:]]
        if not others:
            continue
        distances = face_recognition.face_distance(others, entries[i]['encoding'])
        for offset, distance in enumerate(distances):
            if distance <= threshold:
                parent[find(i)] = find(i + 1 + offset)

    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(find(i), []).append(entry)
    return [g for g in groups.values() if len(g) > 1]


def ask_keep(group):
    """Hỏi người dùng giữ ảnh nào. Trả về danh sách tên ảnh cần xoá (rỗng nếu bỏ qua)."""
    suggested = max(range(len(group)), key=lambda i: group[i]['area'])
    answer = input(
        f'  Enter = giữ [{suggested + 1}] (đề xuất), số = giữ ảnh khác, s = bỏ qua nhóm: '
    ).strip().lower()

    if answer == 's':
        return []
    if answer == '':
        keep = suggested
    elif answer.isdigit() and 1 <= int(answer) <= len(group):
        keep = int(answer) - 1
    else:
        print('  Lựa chọn không hợp lệ, bỏ qua nhóm này.')
        return []
    return [e['name'] for i, e in enumerate(group) if i != keep]


def main():
    parser = argparse.ArgumentParser(description='Kiểm tra và dọn ảnh trùng trong thư mục detect.')
    parser.add_argument('--dir', default=DETECT_DIR, help=f'thư mục ảnh (mặc định: {DETECT_DIR})')
    parser.add_argument('--threshold', type=float, default=MATCH_THRESHOLD,
                        help=f'ngưỡng khoảng cách coi là cùng 1 người (mặc định: {MATCH_THRESHOLD})')
    parser.add_argument('--delete', action='store_true',
                        help='cho phép xoá ảnh (sau khi xác nhận từng nhóm). Mặc định chỉ báo cáo.')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f'Không tìm thấy thư mục "{args.dir}"')

    print(f'Đang quét "{args.dir}"...')
    entries, no_face, multi_face = scan_folder(args.dir)
    groups = group_same_person(entries, args.threshold)

    print(f'\nẢnh hợp lệ: {len(entries)} | Không có khuôn mặt: {len(no_face)} | '
          f'Nhóm trùng: {len(groups)}')

    if no_face:
        print('\nẢnh KHÔNG có khuôn mặt hoặc không đọc được (Recognition sẽ bỏ qua):')
        for name in no_face:
            print(f'  - {name}')
    if multi_face:
        print('\nẢnh có NHIỀU khuôn mặt (đã lấy khuôn mặt lớn nhất):')
        for name in multi_face:
            print(f'  - {name}')

    to_delete = []
    for number, group in enumerate(groups, start=1):
        print(f'\nNhóm {number}: cùng 1 người')
        suggested = max(range(len(group)), key=lambda i: group[i]['area'])
        for i, entry in enumerate(group):
            mark = '  <- đề xuất giữ' if i == suggested else ''
            print(f'  [{i + 1}] {entry["name"]} (diện tích khuôn mặt: {entry["area"]} px){mark}')
        if args.delete:
            to_delete += ask_keep(group)

    if not args.delete:
        if groups or no_face:
            print('\nChế độ báo cáo (dry-run): chưa xoá gì. Thêm --delete để xoá sau khi xác nhận.')
        return

    for name in no_face:
        if input(f'\nXoá ảnh không có khuôn mặt "{name}"? [y/N]: ').strip().lower() == 'y':
            to_delete.append(name)

    for name in to_delete:
        os.remove(os.path.join(args.dir, name))
        print(f'Đã xoá {name}')
    print(f'\nHoàn tất, đã xoá {len(to_delete)} ảnh.')


if __name__ == '__main__':
    main()
