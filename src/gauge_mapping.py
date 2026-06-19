import cv2
import numpy as np
import math
import re

# Try to import pytesseract — gracefully degrade if not installed
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except (ImportError, Exception):
    TESSERACT_AVAILABLE = False
    print("[OCR] pytesseract not available — using linear interpolation fallback")


# =========================
# LINEAR INTERPOLATION (fallback, same as before)
# =========================

def angle_to_percent(angle, empty_angle, full_angle):
    span = full_angle - empty_angle
    if span == 0:
        return 0.0
    percent = (angle - empty_angle) / span * 100
    return max(0.0, min(100.0, percent))


def calibrate_from_angles(all_angles):
    if len(all_angles) < 10:
        print("[CAL] WARNING: Not enough detections — using fallback 30 / 150.")
        return 30.0, 150.0

    angles_rad   = np.radians(all_angles)
    unwrapped    = np.unwrap(angles_rad)
    unwrapped_deg = np.degrees(unwrapped)

    p5  = float(np.percentile(unwrapped_deg, 5))
    p95 = float(np.percentile(unwrapped_deg, 95))

    if abs(p95 - p5) < 10:
        print(f"[CAL] WARNING: Range too small ({p5:.1f} -> {p95:.1f}).")

    return p5, p95


# =========================
# ITEM 6: TICK-MARK OCR
# =========================

def _extract_roi_near_angle(frame, pivot, radius, angle_deg, roi_width=60, roi_height=40):
    """
    Extracts a small image patch just outside the dial rim at a given angle.
    This is where numeric labels typically appear next to tick marks.
    """
    px, py = int(pivot[0]), int(pivot[1])
    rad    = math.radians(angle_deg)

    # Sample at 110% of radius (just outside the rim where labels are)
    label_r = radius * 1.15
    cx = int(px + label_r * math.cos(rad))
    cy = int(py - label_r * math.sin(rad))  # Y flipped

    x1 = max(0, cx - roi_width  // 2)
    y1 = max(0, cy - roi_height // 2)
    x2 = min(frame.shape[1], cx + roi_width  // 2)
    y2 = min(frame.shape[0], cy + roi_height // 2)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _ocr_number_from_roi(roi):
    """
    Runs Tesseract on a small ROI to extract a numeric label.
    Returns float or None.
    """
    if roi is None or not TESSERACT_AVAILABLE:
        return None

    # Preprocess: grayscale, upscale, threshold for better OCR
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    big   = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3),
                       interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789.-'
    text   = pytesseract.image_to_string(th, config=config).strip()

    # Extract first valid number
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return None
    return None


def detect_tick_marks(frame, pivot, radius, dial_radius_px):
    """
    Finds short radial line segments (tick marks) around the dial perimeter
    using Canny + HoughLinesP, then samples just outside each tick for a label.

    Returns list of (angle_deg, label_value) pairs — sorted by angle.
    Only returns entries where OCR successfully read a number.
    """
    if not TESSERACT_AVAILABLE:
        return []

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=int(dial_radius_px * 0.05),
        maxLineGap=5
    )

    if lines is None:
        return []

    px, py  = int(pivot[0]), int(pivot[1])
    results = []
    seen_angles = set()

    for l in lines:
        x1, y1, x2, y2 = l[0]

        # Midpoint of the line segment
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Distance from pivot to midpoint
        dist = math.hypot(mx - px, my - py)

        # Must be near the rim: between 70% and 95% of radius
        if not (dial_radius_px * 0.70 <= dist <= dial_radius_px * 0.95):
            continue

        # Line must be roughly radial (perpendicular to the rim tangent)
        dx, dy     = x2 - x1, y2 - y1
        seg_len    = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue

        # Direction from pivot to midpoint
        to_mid_x = mx - px
        to_mid_y = my - py
        to_mid_len = math.hypot(to_mid_x, to_mid_y)
        if to_mid_len < 1e-6:
            continue

        # Dot product between segment and radial direction
        radial_alignment = abs(
            (dx * to_mid_x + dy * to_mid_y) / (seg_len * to_mid_len)
        )

        # Must be at least 60% aligned with radial direction = tick mark
        if radial_alignment < 0.6:
            continue

        # Angle of this tick mark
        angle_deg = math.degrees(math.atan2(-(my - py), mx - px)) % 360

        # Deduplicate: skip if we already have a tick within 3 degrees
        angle_rounded = round(angle_deg / 3) * 3
        if angle_rounded in seen_angles:
            continue
        seen_angles.add(angle_rounded)

        # OCR the label near this tick
        roi   = _extract_roi_near_angle(frame, pivot, dial_radius_px, angle_deg)
        value = _ocr_number_from_roi(roi)

        if value is not None:
            results.append((angle_deg, value))

    results.sort(key=lambda x: x[0])
    print(f"[OCR] Found {len(results)} labeled tick marks: {results}")
    return results


# =========================
# SCALE MAPPING (linear or tabulated)
# =========================

class GaugeScale:
    """
    Converts needle angle to a physical reading.

    If tick_map has >= 2 entries: uses piecewise linear interpolation
    between the detected tick marks (handles non-linear scales).

    Otherwise: falls back to linear percent mapping between empty/full.
    """

    def __init__(self, empty_angle, full_angle, tick_map=None):
        self.empty_angle = empty_angle
        self.full_angle  = full_angle
        self.tick_map    = []

        if tick_map and len(tick_map) >= 2:
            # Sort by angle
            self.tick_map = sorted(tick_map, key=lambda x: x[0])
            angles = [t[0] for t in self.tick_map]
            values = [t[1] for t in self.tick_map]
            print(f"[OCR] Scale built from {len(self.tick_map)} ticks: "
                  f"angle {angles[0]:.1f}°→{values[0]} .. "
                  f"angle {angles[-1]:.1f}°→{values[-1]}")
        else:
            print("[OCR] Insufficient tick data — using linear percent fallback")

    def angle_to_value(self, angle):
        """Returns (value, unit_type) where unit_type is 'percent' or 'physical'."""
        if len(self.tick_map) >= 2:
            angles = [t[0] for t in self.tick_map]
            values = [t[1] for t in self.tick_map]
            # Clamp to known range
            if angle <= angles[0]:
                return values[0], 'physical'
            if angle >= angles[-1]:
                return values[-1], 'physical'
            # Piecewise linear interpolation
            for i in range(len(angles) - 1):
                if angles[i] <= angle <= angles[i+1]:
                    t = (angle - angles[i]) / (angles[i+1] - angles[i])
                    v = values[i] + t * (values[i+1] - values[i])
                    return round(v, 2), 'physical'

        # Fallback: linear percent
        pct = angle_to_percent(angle, self.empty_angle, self.full_angle)
        return pct, 'percent'

    def to_percent(self, angle):
        """Always returns 0-100% regardless of scale type (for dashboard arc)."""
        return angle_to_percent(angle, self.empty_angle, self.full_angle)