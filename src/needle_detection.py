import cv2
import numpy as np
import math
from angle_calculation import get_pivot


def _hough_candidate(frame, px, py, radius):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_gray = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges_gray, 1, np.pi / 180,
        threshold=40, minLineLength=int(radius * 0.3), maxLineGap=10
    )
    if lines is None:
        return None, None

    best_line = None
    best_score = None

    for l in lines:
        x1, y1, x2, y2 = l[0]

        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue

        dist = abs((px - x1) * dy - (py - y1) * dx) / seg_len

        if dist > radius * 0.20:
            continue

        score = (dist, -seg_len)
        if best_score is None or score < best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    x1, y1, x2, y2 = best_line
    d1 = (x1 - px) ** 2 + (y1 - py) ** 2
    d2 = (x2 - px) ** 2 + (y2 - py) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx, dy = tx - px, ty - py
    ang = math.degrees(math.atan2(-dy, dx)) % 360

    return int(ang), [[px, py, tx, ty]]


def detect_needle(edges, frame=None):
    h, w = edges.shape
    px, py = get_pivot((h, w))
    radius = 240

    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(edges, (radius, 360), (int(px), int(py)), radius, flags)

    start_col = int(radius * 0.15)
    end_col = int(radius * 0.92)
    row_sums = np.sum(unrolled[:, start_col:end_col], axis=1).astype(np.float32)
    row_sums = cv2.GaussianBlur(row_sums, (1, 11), 0).flatten()

    mean_density = np.mean(row_sums)
    max_density = np.max(row_sums)

    sat_ok = not (max_density < (mean_density * 2.5) or max_density < 100)
    sat_angle = int(np.argmax(row_sums)) if sat_ok else None

    hough_angle, hough_line = (None, None)
    if frame is not None:
        hough_angle, hough_line = _hough_candidate(frame, px, py, radius)

    if not sat_ok and hough_angle is None:
        return None, "MISSING_OR_OBSTRUCTED", None

    if sat_ok:
        target_angle = sat_angle
    else:
        return hough_line, "HEALTHY", hough_angle

    significant_peaks = np.where(row_sums > (max_density * 0.75))[0]
    if len(significant_peaks) > 0:
        peak_spread = np.max(significant_peaks) - np.min(significant_peaks)
        if 350 in significant_peaks and 0 in significant_peaks:
            wrapped_peaks = [(p if p < 180 else p - 360) for p in significant_peaks]
            peak_spread = np.max(wrapped_peaks) - np.min(wrapped_peaks)
        if peak_spread > 40:
            return None, "MULTIPLE_PEAKS_ANOMALY", target_angle

    angle_rad = np.radians(target_angle)
    length = 220
    x1, y1 = int(px), int(py)
    x2 = int(px + length * np.cos(angle_rad))
    y2 = int(py + length * np.sin(angle_rad))

    return [[x1, y1, x2, y2]], "HEALTHY", target_angle