import cv2
import numpy as np


def preprocess_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1]
    _, thresh = cv2.threshold(saturation_channel, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return clean_mask