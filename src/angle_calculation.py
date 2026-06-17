import math
import numpy as np


def get_pivot(frame_shape):
    h, w = frame_shape[:2]
    return w // 2, int(h * 0.75)


def detect_pivot_from_lines(lines, frame_shape, fallback=None):
    if fallback is None:
        fallback = get_pivot(frame_shape)

    if len(lines) < 10:
        return fallback

    A = []
    C = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = (y2 - y1)
        b = -(x2 - x1)
        c = a * x1 + b * y1
        norm = math.hypot(a, b)
        if norm < 1e-6:
            continue
        A.append([a / norm, b / norm])
        C.append(c / norm)

    if len(A) < 10:
        return fallback

    A = np.array(A)
    C = np.array(C)

    try:
        result, _, _, _ = np.linalg.lstsq(A, C, rcond=None)
        px, py = result[0], result[1]
    except np.linalg.LinAlgError:
        return fallback

    h, w = frame_shape[:2]
    if not (-0.2 * w <= px <= 1.2 * w and -0.2 * h <= py <= 1.2 * h):
        return fallback

    return (px, py)


def calculate_tip_angle(line, frame_shape):
    x1, y1, x2, y2 = line[0]
    cx, cy = get_pivot(frame_shape)

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tx - cx
    dy = cy - ty

    angle = math.degrees(math.atan2(dy, dx))

    if angle < 0:
        angle = -angle

    return float(angle)