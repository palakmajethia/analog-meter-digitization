import math
import cv2
import numpy as np

def get_pivot(frame_shape):
    """
    Returns (cx, cy) pivot point.
    For a centered gauge (phone screen, mounted gauge):  frame center.
    The 0.75 offset was only correct for gauges mounted at the bottom
    of the frame — now uses true center which works universally.
    """
    h, w = frame_shape[:2]
    return w // 2, h // 2


def calculate_tip_angle(line, frame_shape):
    """
    Returns needle tip angle in degrees (0-360).
    Picks the endpoint FURTHEST from the pivot as the tip.
    """
    x1, y1, x2, y2 = line[0]
    cx, cy = get_pivot(frame_shape)

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tx - cx
    dy = cy - ty      # invert Y → standard Cartesian (up = positive)

    angle = math.degrees(math.atan2(dy, dx)) % 360
    return float(angle)