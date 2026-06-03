import cv2
import numpy as np
import math

def detect_needle(edges):

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=120,
        maxLineGap=15
    )

    if lines is None:
        return None

    height, width = edges.shape

    center_x = width // 2
    center_y = height // 2

    best_line = None
    best_score = 0

    for line in lines:

        x1, y1, x2, y2 = line[0]

        # Length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Ignore tiny lines
        if length < 100:
            continue

        # Ignore almost horizontal lines
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

        if angle < 10:
            continue

        # Midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Distance from image center
        dist = math.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)

        # Prefer long lines near center
        score = length - dist

        if score > best_score:
            best_score = score
            best_line = line

    return best_line