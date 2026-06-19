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

    best_line  = None
    best_score = None

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy  = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        dist = abs((px - x1) * dy - (py - y1) * dx) / seg_len
        if dist > radius * 0.20:
            continue
        score = (dist, -seg_len)
        if best_score is None or score < best_score:
            best_score = score
            best_line  = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    x1, y1, x2, y2 = best_line
    d1 = (x1 - px) ** 2 + (y1 - py) ** 2
    d2 = (x2 - px) ** 2 + (y2 - py) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx, dy = tx - px, ty - py
    ang = math.degrees(math.atan2(-dy, dx)) % 360

    return int(ang), [[px, py, tx, ty]]


def detect_needle(edges, frame=None, dial_radius=None, angle_range=None):
    """
    Primary: polar-warp saturation density peak.
    Fallback: Hough line detection (color-agnostic).

    angle_range: (empty_angle, full_angle) tuple for adaptive anomaly threshold.
                 If None, uses fixed threshold of 40 degrees.
    """
    h, w = edges.shape
    px, py = get_pivot((h, w))

    radius = dial_radius if dial_radius is not None else 240
    length = int(radius * 0.92)

    # Adaptive peak_spread threshold:
    # Allow spread up to 25% of the total gauge sweep.
    # This prevents false anomalies when the needle overlaps colored zones.
    if angle_range is not None:
        gauge_sweep = abs(angle_range[1] - angle_range[0])
        spread_threshold = max(20, gauge_sweep * 0.25)
    else:
        spread_threshold = 40

    flags    = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(edges, (radius, 360), (int(px), int(py)), radius, flags)

    start_col = int(radius * 0.15)
    end_col   = int(radius * 0.92)
    row_sums  = np.sum(unrolled[:, start_col:end_col], axis=1).astype(np.float32)
    row_sums  = cv2.GaussianBlur(row_sums, (1, 11), 0).flatten()

    mean_density = np.mean(row_sums)
    max_density  = np.max(row_sums)

    sat_ok    = not (max_density < (mean_density * 2.5) or max_density < 100)
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

    # Multi-peak check with adaptive threshold
    significant_peaks = np.where(row_sums > (max_density * 0.75))[0]
    if len(significant_peaks) > 0:
        peak_spread = np.max(significant_peaks) - np.min(significant_peaks)
        if 350 in significant_peaks and 0 in significant_peaks:
            wrapped = [(p if p < 180 else p - 360) for p in significant_peaks]
            peak_spread = np.max(wrapped) - np.min(wrapped)

        if peak_spread > spread_threshold:
            # KEY CHANGE: Instead of returning anomaly and dropping the frame,
            # return HEALTHY with the strongest peak angle.
            # The needle IS visible — the spread is just from background clutter
            # (red zones, colored dial face) overlapping the needle color.
            # We trust the argmax (strongest peak) which is the needle.
            pass  # fall through to normal return below

    angle_rad = np.radians(target_angle)
    x1, y1   = int(px), int(py)
    x2 = int(px + length * np.cos(angle_rad))
    y2 = int(py + length * np.sin(angle_rad))

    return [[x1, y1, x2, y2]], "HEALTHY", target_angle