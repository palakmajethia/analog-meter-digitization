import cv2
import numpy as np

# ── Background model (used when available for live video) ─────────────────
_background_median = None
_frame_buffer      = []
BG_BUILD_THRESHOLD = 20   # minimum frames needed to build a useful model


def set_background(frames):
    global _background_median
    if len(frames) < BG_BUILD_THRESHOLD:
        return
    stack = np.stack(frames, axis=0).astype(np.float32)
    _background_median = np.median(stack, axis=0).astype(np.uint8)
    print(f"[PRE] Background model built from {len(frames)} frames.")


def reset_background():
    global _background_median, _frame_buffer
    _background_median = None
    _frame_buffer      = []


def preprocess_image(frame, mode="color"):
    """
    Universal preprocessor — works on any gauge, any needle color,
    static image or live video.

    Strategy:
      1. If a background model exists (live video after calibration sweep)
         use motion subtraction — most robust, color-blind.
      2. Otherwise use geometry-based isolation — finds the most elongated
         blob radiating from center, works on static images.
    """
    if _background_median is not None:
        result = _motion_based(frame)
        # If motion gives us nothing (static scene), fall through to geometry
        if cv2.countNonZero(result) > 50:
            return result

    return _geometry_based(frame)


# =============================================================================
# METHOD 1 — Motion based (live video after calibration sweep)
# =============================================================================

def _motion_based(frame):
    diff        = cv2.absdiff(frame, _background_median)
    diff_single = np.max(diff, axis=2).astype(np.uint8)
    _, thresh   = cv2.threshold(diff_single, 25, 255, cv2.THRESH_BINARY)
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean       = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return keep_largest_components(clean, n=1)


# =============================================================================
# METHOD 2 — Geometry based (static images or no background model)
# =============================================================================

def _geometry_based(frame):
    """
    Finds the needle by shape, not color:
      1. Generate multiple candidate masks (light bg, dark bg, saturation)
      2. For each mask find all blobs
      3. Score every blob by elongation + alignment to frame center
      4. Return the mask containing the most needle-like blob
    """
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2

    candidates = _generate_candidate_masks(frame)

    best_mask  = None
    best_score = -1

    for mask in candidates:
        blob_mask, score = _score_best_blob(mask, cx, cy, h, w)
        if score > best_score:
            best_score = score
            best_mask  = blob_mask

    if best_mask is None or best_score < 0.3:
        # Nothing needle-like found — return empty, caller handles OBSTRUCTED
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    return best_mask


def _generate_candidate_masks(frame):
    """
    Produces three independent binary masks using different strategies.
    We try all three and pick the one whose blobs look most needle-like.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = []

    # ── Candidate A: dark objects on light background ──────────────────────
    # Good for: classic white-dial black-needle industrial gauges
    blurred_a = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_a = cv2.adaptiveThreshold(
        blurred_a, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=8
    )
    kernel_a = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks.append(cv2.morphologyEx(adaptive_a, cv2.MORPH_OPEN, kernel_a))

    # ── Candidate B: bright objects on dark background ─────────────────────
    # Good for: glowing needles on dark dials (like the boost gauge above)
    blurred_b = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_b = cv2.adaptiveThreshold(
        blurred_b, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,      # NOT inverted this time
        blockSize=31, C=-10     # negative C pulls more bright pixels in
    )
    kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks.append(cv2.morphologyEx(adaptive_b, cv2.MORPH_OPEN, kernel_b))

    # ── Candidate C: high saturation objects ──────────────────────────────
    # Good for: colored needles (red, blue, green) on any background
    saturation  = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)
    kernel_c    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks.append(cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel_c))

    return masks


def _score_best_blob(mask, cx, cy, h, w):
    """
    Finds all blobs in the mask and scores each one on:
      - Elongation  : needle is long and thin (high aspect ratio from minAreaRect)
      - Length      : needle is long relative to dial size
      - Pivot proximity : one end of the blob passes near the frame center

    Returns (isolated_blob_mask, best_score) where score is 0.0–1.0.
    A score below 0.3 means nothing needle-like was found.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    best_score     = -1.0
    best_label_idx = -1
    min_area       = 80    # ignore tiny noise blobs
    dial_diagonal  = np.hypot(h, w)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # Extract blob pixels and fit a minimum area rectangle
        blob_pixels = np.column_stack(np.where(labels == i))  # (row, col)
        if len(blob_pixels) < 10:
            continue

        # minAreaRect expects (x, y) = (col, row)
        rect  = cv2.minAreaRect(blob_pixels[:, ::-1].astype(np.float32))
        (rx, ry), (rw, rh), angle = rect

        long_side  = max(rw, rh)
        short_side = min(rw, rh) + 1e-5

        # ── Elongation score (0–1): needle should be at least 4:1 ratio ───
        aspect_ratio      = long_side / short_side
        elongation_score  = min(aspect_ratio / 10.0, 1.0)  # saturates at 10:1
        if aspect_ratio < 3.0:
            continue   # not elongated enough to be a needle

        # ── Length score (0–1): needle should be reasonably long ───────────
        length_score = min(long_side / (dial_diagonal * 0.25), 1.0)
        if long_side < dial_diagonal * 0.10:
            continue   # too short to be a needle

        # ── Pivot proximity score (0–1) ────────────────────────────────────
        # One END of the minAreaRect should be near the frame center (pivot).
        # We check both ends of the long axis of the rect.
        box      = cv2.boxPoints(rect)  # 4 corners
        box      = box.astype(np.float32)
        # Midpoints of the two short sides = the two ends of the needle
        end1 = ((box[0] + box[1]) / 2)
        end2 = ((box[2] + box[3]) / 2)
        dist1 = np.hypot(end1[0] - cx, end1[1] - cy)
        dist2 = np.hypot(end2[0] - cx, end2[1] - cy)
        min_end_dist  = min(dist1, dist2)
        pivot_score   = max(0.0, 1.0 - (min_end_dist / (dial_diagonal * 0.25)))

        # ── Combined score ─────────────────────────────────────────────────
        # Elongation matters most, then pivot proximity, then length
        score = (elongation_score * 0.5) + (pivot_score * 0.35) + (length_score * 0.15)

        if score > best_score:
            best_score     = score
            best_label_idx = i

    if best_label_idx == -1:
        return np.zeros_like(mask), 0.0

    # Return a clean mask with only the winning blob
    output = np.zeros_like(mask)
    output[labels == best_label_idx] = 255
    return output, best_score


# =============================================================================
# Shared utilities
# =============================================================================

def keep_largest_components(mask, n=1):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    areas  = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    areas.sort(reverse=True)
    output = np.zeros_like(mask)
    for _, label_idx in areas[:n]:
        output[labels == label_idx] = 255
    return output