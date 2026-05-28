import cv2
import numpy as np

def detect_needle(edges):

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        50,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is not None:
        return None
    longest_line = max(
        lines,
        key = lambda l:
        np.sqrt(
            (l[0][2]-l[0][0])**2 +
            (l[0][3]-l[0][1])**2
        )
    )
    return longest_line