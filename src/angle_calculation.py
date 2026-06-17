import cv2
import math
import numpy as np


def get_pivot(frame_shape):
    h, w = frame_shape[:2]
    return w // 2, int(h * 0.75)


def detect_dial_radius(frame, pivot=None):
    """
    Detects the dial radius by finding the largest circle in the frame
    using Hough circle detection on a grayscale image.

    Falls back to a default radius (240 for 800x600) if detection fails.

    Returns: (radius_px, pivot_xy)
    - radius_px: detected radius in pixels
    - pivot_xy: centre of the detected circle (overrides get_pivot if found)
    """
    h, w = frame.shape[:2]
    default_radius = int(min(h, w) * 0.40)  # ~40% of smaller dimension
    default_pivot  = get_pivot(frame.shape) if pivot is None else pivot

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
        maxRadius=int(min(h, w) * 0.55)
    )

    if circles is None:
        print(f"[CAL] Dial radius: detection failed, using fallback {default_radius}px")
        return default_radius, default_pivot

    circles = np.round(circles[0, :]).astype(int)
    # Pick the largest detected circle
    cx, cy, r = max(circles, key=lambda c: c[2])

    print(f"[CAL] Dial radius: detected {r}px at centre ({cx},{cy})  "
          f"[fallback was {default_radius}px at {default_pivot}]")

    return int(r), (int(cx), int(cy))


def detect_pivot_from_lines(lines, frame_shape, fallback=None):
    """
    Estimates the pivot point as the least-squares intersection of all
    needle line segments collected during calibration.
    """
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