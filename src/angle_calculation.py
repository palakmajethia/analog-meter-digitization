"""
Geometry: pivot detection, dial radius detection, tip-angle calculation.

Fix log:
- `calculate_tip_angle` used to reflect negative angles into 0-180
  (`if angle < 0: angle = -angle`), which silently aliased e.g. -10 deg
  and +10 deg to the same reading. That's only correct for gauges where
  the needle is guaranteed to live in the upper semicircle with the
  pivot at the bottom -- it actively mis-reads 270-degree and 360-degree
  gauges. The fold is removed; this now returns the raw, unwrapped-at-
  calibration-time angle (0-360, standard math convention). Wraparound
  across the 0/360 boundary is handled once, properly, via np.unwrap in
  gauge_mapping.calibrate_from_angles, instead of being silently baked
  into every single per-frame angle.
- `detect_dial_radius` and `detect_pivot_from_lines` used to fall back to
  guessed defaults with only a print() and no signal the caller could act
  on. They now return an explicit `is_fallback` boolean alongside the
  value, and log a clear warning, so a failed calibration is visible
  instead of silently producing plausible-looking-but-wrong geometry.
"""
import cv2
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_pivot(frame_shape):
    """
    Bottom-center heuristic. Used only as a last-resort fallback when
    neither circle detection nor needle-line-intersection succeeds.
    Assumes the gauge is horizontally centered with its pivot near the
    bottom of frame -- true for most semicircular dial gauges, NOT
    guaranteed for 360-degree gauges, off-center framing, or gauges shot
    at an angle.
    """
    h, w = frame_shape[:2]
    return w // 2, int(h * 0.75)


def detect_dial_radius(frame, pivot=None):
    """
    Detects the dial radius via Hough circle detection on a grayscale
    image. Returns (radius_px, pivot_xy, is_fallback).

    Known limitation: picks the largest detected circle, with no check
    that it's actually the gauge rim rather than some other circular
    object in frame (bezel, background clutter). See README.
    """
    h, w = frame.shape[:2]
    default_radius = int(min(h, w) * 0.40)
    default_pivot = get_pivot(frame.shape) if pivot is None else pivot

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.3),
        param1=80,
        param2=40,
        minRadius=int(min(h, w) * 0.15),
        maxRadius=int(min(h, w) * 0.55),
    )

    if circles is None:
        logger.warning(
            "Dial radius detection FAILED -- falling back to an UNVERIFIED "
            "%dpx default at %s. Readings may be inaccurate until this is "
            "fixed (clearer rim contrast, or pass a known radius).",
            default_radius, default_pivot,
        )
        return default_radius, default_pivot, True

    circles = np.round(circles[0, :]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])

    logger.info("Dial radius detected: %dpx at center (%d, %d) "
                "[fallback would have been %dpx at %s]",
                r, cx, cy, default_radius, default_pivot)

    return int(r), (int(cx), int(cy)), False


def detect_pivot_from_lines(lines, frame_shape, fallback=None, min_lines=10):
    """
    Estimates the pivot as the least-squares intersection of needle line
    segments collected during calibration. Returns (pivot_xy, is_fallback).
    """
    if fallback is None:
        fallback = get_pivot(frame_shape)

    if len(lines) < min_lines:
        logger.warning(
            "Only %d needle line(s) collected (need >= %d) -- pivot "
            "fallback to %s is an UNVERIFIED guess.",
            len(lines), min_lines, fallback,
        )
        return fallback, True

    A, C = [], []
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

    if len(A) < min_lines:
        logger.warning("Pivot fit degenerate after filtering degenerate "
                        "lines -- using fallback %s.", fallback)
        return fallback, True

    A = np.array(A)
    C = np.array(C)

    try:
        result, _, _, _ = np.linalg.lstsq(A, C, rcond=None)
        px, py = float(result[0]), float(result[1])
    except np.linalg.LinAlgError:
        logger.warning("Pivot least-squares solve failed -- using fallback %s.", fallback)
        return fallback, True

    h, w = frame_shape[:2]
    if not (-0.2 * w <= px <= 1.2 * w and -0.2 * h <= py <= 1.2 * h):
        logger.warning(
            "Computed pivot (%.1f, %.1f) is implausibly far outside the "
            "frame -- using fallback %s instead.", px, py, fallback,
        )
        return fallback, True

    return (px, py), False


def calculate_tip_angle(line, frame_shape, pivot=None):
    """
    Returns the needle tip angle in standard math convention: degrees,
    0-360, counter-clockwise from the positive x-axis, with image Y
    flipped so it behaves like a normal Cartesian plane.

    Does NOT fold/reflect into a fixed semicircle -- see module docstring.
    Calibration (gauge_mapping.calibrate_from_angles) is responsible for
    unwrapping the 0/360 boundary using the actual sequence of readings.
    """
    x1, y1, x2, y2 = line[0]
    cx, cy = get_pivot(frame_shape) if pivot is None else pivot

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tx - cx
    dy = cy - ty  # invert Y -> standard Cartesian

    return math.degrees(math.atan2(dy, dx)) % 360