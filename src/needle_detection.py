"""
Needle detection: polar-warp saturation density peak (primary), with a
color-agnostic Hough line fallback.

Fix log:
- The old "adaptive multi-peak anomaly threshold" computed
  `peak_spread > spread_threshold` and then did nothing with the result in
  either branch -- both paths fell through to the same `return ...,
  "HEALTHY", ...`. So despite the comments, multi-peak ambiguity was NEVER
  actually flagged; the function always reported HEALTHY whenever the
  saturation signal was present at all, no matter how scattered. This was
  a real bug, not a tuning issue. It's fixed below: a spread that exceeds
  the adaptive threshold now returns NeedleStatus.AMBIGUOUS_MULTI_PEAK
  instead of silently guessing via argmax and calling it healthy. This
  also gives partial, honest coverage of the "two needles on one gauge"
  failure mode -- it won't track both needles, but it will now say "I'm
  not sure" instead of confidently reporting one of them as the answer.
- `_hough_candidate` used to run unconditionally every frame (a full
  Canny + HoughLinesP pass), even on frames where the saturation path
  already succeeded and the Hough result was discarded unused. It's now
  computed lazily, only when the saturation signal is inconclusive.
- Status is now a NeedleStatus enum instead of a bare string, so a typo
  in a comparison (`"healthy"` vs `"HEALTHY"`) can't silently fail.
"""
import cv2
import numpy as np
import math
import logging
from enum import Enum

from angle_calculation import get_pivot

logger = logging.getLogger(__name__)


class NeedleStatus(Enum):
    HEALTHY = "HEALTHY"
    MISSING_OR_OBSTRUCTED = "MISSING_OR_OBSTRUCTED"
    AMBIGUOUS_MULTI_PEAK = "AMBIGUOUS_MULTI_PEAK"


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


def _line_from_angle(px, py, angle_deg, length):
    angle_rad = np.radians(angle_deg)
    x1, y1 = int(px), int(py)
    x2 = int(px + length * np.cos(angle_rad))
    y2 = int(py + length * np.sin(angle_rad))
    return [[x1, y1, x2, y2]]


def detect_needle(saturation_mask, frame=None, dial_radius=None, angle_range=None):
    """
    Primary: polar-warp saturation density peak.
    Fallback: Hough line detection (color-agnostic), computed only if the
    saturation signal is inconclusive.

    angle_range: (empty_angle, full_angle) tuple for the adaptive
                 multi-peak threshold. If None, uses a fixed 40-degree
                 threshold.

    Returns: (line_or_None, NeedleStatus, raw_angle_or_None)
    """
    h, w = saturation_mask.shape
    px, py = get_pivot((h, w))

    radius = dial_radius if dial_radius is not None else 240
    length = int(radius * 0.92)

    if angle_range is not None:
        gauge_sweep = abs(angle_range[1] - angle_range[0])
        spread_threshold = max(20, gauge_sweep * 0.25)
    else:
        spread_threshold = 40

    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(saturation_mask, (radius, 360), (int(px), int(py)), radius, flags)

    start_col = int(radius * 0.15)
    end_col = int(radius * 0.92)
    row_sums = np.sum(unrolled[:, start_col:end_col], axis=1).astype(np.float32)
    row_sums = cv2.GaussianBlur(row_sums, (1, 11), 0).flatten()

    mean_density = np.mean(row_sums)
    max_density = np.max(row_sums)
    sat_ok = not (max_density < (mean_density * 2.5) or max_density < 100)

    if sat_ok:
        target_angle = int(np.argmax(row_sums))

        significant_peaks = np.where(row_sums > (max_density * 0.75))[0]
        peak_spread = 0
        if len(significant_peaks) > 0:
            peak_spread = np.max(significant_peaks) - np.min(significant_peaks)
            if 350 in significant_peaks and 0 in significant_peaks:
                wrapped = [(p if p < 180 else p - 360) for p in significant_peaks]
                peak_spread = np.max(wrapped) - np.min(wrapped)

        line = _line_from_angle(px, py, target_angle, length)

        if peak_spread > spread_threshold:
            # Genuinely ambiguous: multiple well-separated saturated
            # regions (a second needle, a colored zone overlapping the
            # needle, a reflection). Report it honestly instead of
            # guessing via argmax and calling it healthy.
            logger.debug("Multi-peak ambiguity: spread=%.1f > threshold=%.1f",
                        peak_spread, spread_threshold)
            return line, NeedleStatus.AMBIGUOUS_MULTI_PEAK, target_angle

        return line, NeedleStatus.HEALTHY, target_angle

    # Saturation signal inconclusive -- try the color-agnostic Hough
    # fallback. Only computed here, not unconditionally every frame.
    if frame is None:
        return None, NeedleStatus.MISSING_OR_OBSTRUCTED, None

    hough_angle, hough_line = _hough_candidate(frame, px, py, radius)
    if hough_angle is None:
        return None, NeedleStatus.MISSING_OR_OBSTRUCTED, None

    return hough_line, NeedleStatus.HEALTHY, hough_angle