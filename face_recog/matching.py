"""So khớp một khuôn mặt với danh sách đã biết và định dạng nhãn hiển thị."""
import math

import face_recognition
import numpy as np

from face_recog.config import MATCH_THRESHOLD, RECOGNITION_THRESHOLD
from face_recog.smoothing import PENDING


def face_confidence(face_distance, face_match_threshold=MATCH_THRESHOLD):
    span = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (span * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"


def match_face(encoding, known_encodings, known_names, threshold=RECOGNITION_THRESHOLD):
    """So khớp 1 khuôn mặt với danh sách đã biết. Trả về (tên, độ tin cậy); 'Unknown' nếu không khớp.

    `threshold` chỉ quyết định khớp hay không; thang % của face_confidence không đổi theo nó.
    """
    if len(known_encodings) == 0:
        return 'Unknown', 'Unknown'

    distances = face_recognition.face_distance(known_encodings, encoding)
    best = int(np.argmin(distances))
    if distances[best] <= threshold:
        return known_names[best], face_confidence(distances[best])
    return 'Unknown', 'Unknown'


def format_label(name, confidence):
    """Nhãn hiển thị trên khung hình: 'Huy 98%', 'Unknown', hoặc '...' khi chưa đủ phiếu."""
    if name == PENDING or name == 'Unknown':
        return name
    return f'{name} {confidence}'
