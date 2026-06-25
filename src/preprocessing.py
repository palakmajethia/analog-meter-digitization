"""
Frame preprocessing.

Fix log:
- `preprocess_image` returned a saturation binary mask but was called
  `edges` at every call site across the codebase (confusingly overlapping
  with the *actual* Canny edge maps computed separately and independently
  in needle_detection.py and gauge_mapping.py). Renamed to
  `compute_saturation_mask` so the name says what it is.
- Added `resize_with_letterbox`: the old code did a hard
  `cv2.resize(frame, (800, 600))` everywhere, which silently squashes any
  non-4:3 source video, turning a circular dial into an ellipse before it
  ever reaches Hough circle detection (which is specifically looking for
  circles). This resizes while preserving aspect ratio and pads with
  black bars instead.
- Added `normalize_lighting` (CLAHE on the V channel) as an optional step
  to reduce -- not eliminate -- sensitivity to moving shadows / uneven
  lighting before saturation thresholding. This is a mitigation, not a
  fix; see README "Known limitations" for what it doesn't solve.
"""
import cv2
import numpy as np

# Tuned for a colored (non-gray/black) needle against a duller dial face.
# See README "Known limitations" -- this is the root assumption behind the
# whole saturation-based detection strategy, not just a tunable number.
SATURATION_THRESHOLD = 80


def resize_with_letterbox(frame, target_size=(800, 600)):
    """
    Resizes `frame` to fit within target_size (width, height) while
    preserving aspect ratio, padding with black bars rather than
    stretching. Returns (letterboxed_frame, scale, (pad_x, pad_y)) so
    callers can map coordinates back to the original frame if needed.
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def normalize_lighting(frame):
    """
    CLAHE-based local contrast normalization on the HSV value channel.
    Reduces (does not eliminate) sensitivity to moving shadows / glare
    before saturation thresholding. Call this before
    compute_saturation_mask if your footage has uneven lighting.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_saturation_mask(frame, threshold=SATURATION_THRESHOLD):
    """
    Binary mask of high-saturation pixels -- the primary needle-detection
    signal. Returns a saturation mask, NOT an edge map (see module
    docstring); needle_detection.py and gauge_mapping.py compute their
    own separate Canny edge maps where they actually need edges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1]
    _, thresh = cv2.threshold(saturation_channel, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return clean_mask