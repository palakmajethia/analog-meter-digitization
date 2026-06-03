import cv2
import numpy as np
import math

def detect_needle(edges):

    h, w = edges.shape
    min_line_length = int(min(h, w) * 0.2)
    max_line_gap = int(min(h, w) * 0.05)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return None

    h, w = edges.shape
    cx, cy = w // 2, h // 2

    best_line = None
    best_score = 0

    for line in lines:

        x1, y1, x2, y2 = line[0]

        length = math.hypot(x2 - x1, y2 - y1)

        if length < 60:
            continue

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        dist = math.hypot(mid_x - cx, mid_y - cy)

        score = length - dist

        if score > best_score:
            best_score = score
            best_line = line

    return best_line