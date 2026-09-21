import numpy as np


def vec(value):
    """Vector 128 chiều giả, mọi phần tử bằng `value` (khoảng cách tới vec(0) = value * sqrt(128))."""
    return np.full(128, value, dtype=float)


def unit_at_distance(distance):
    """Vector cách vec(0) đúng `distance` (đặt toàn bộ độ lệch vào 1 chiều)."""
    v = np.zeros(128)
    v[0] = distance
    return v
