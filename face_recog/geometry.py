"""Tiện ích hình học cho khung mặt (top, right, bottom, left), dùng chung nhiều module."""


def scale_locations(locations, factor, shape):
    """Nhân toạ độ (top, right, bottom, left) lên `factor` và kẹp trong khung hình có `shape` (h, w, ...)."""
    height, width = shape[:2]
    return [
        (
            min(max(int(round(top * factor)), 0), height),
            min(max(int(round(right * factor)), 0), width),
            min(max(int(round(bottom * factor)), 0), height),
            min(max(int(round(left * factor)), 0), width),
        )
        for top, right, bottom, left in locations
    ]


def face_area(location):
    top, right, bottom, left = location
    return (bottom - top) * (right - left)


def largest_face(locations):
    """Khung mặt có diện tích lớn nhất trong danh sách."""
    return max(locations, key=face_area)
