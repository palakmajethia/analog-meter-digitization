"""
Gauge scale mapping: calibration, OCR tick-mark reading, angle→value
conversion.

Fix log:
- calibrate_from_angles used to fall back to hardcoded 30 / 150 degree
  values when not enough angle data was collected, silently continuing with
  wrong numbers. It now raises CalibrationError so the caller is forced to
  handle the failure rather than unknowingly publishing garbage readings.
- calibrate() renamed calibrate_from_sweep_video() to make the fundamental
  assumption explicit: it expects a dedicated calibration clip where the
  needle sweeps from one extreme to the other. It is NOT suitable for
  pointing at arbitrary monitoring footage. See docstring.
- Tesseract path was hardcoded to C:/Program Files/... -- a Windows-only
  path that caused a crash on Linux/Mac because pytesseract.image_to_string
  had no try/except around it. Now uses shutil.which to find the binary
  portably, and _ocr_number_from_roi wraps the OCR call in try/except so a
  missing Tesseract binary fails gracefully per-ROI rather than crashing
  the whole calibration step.
- GaugeScale.angle_to_value now logs a warning (not silently falls back)
  when the OCR tick map is insufficient.
"""
import cv2
import math
import re
import logging
import shutil
import numpy as np

from config import CalibrationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract / OCR availability
# ---------------------------------------------------------------------------

def _find_tesseract():
    """
    Locate the Tesseract binary portably: checks PATH first (Linux / Mac /
    properly installed Windows), then falls back to the common Windows
    installer path, then gives up. Returns the path or None.
    """
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path
    windows_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    import os
    if os.path.isfile(windows_default):
        return windows_default
    return None


TESSERACT_PATH = _find_tesseract()

try:
    import pytesseract
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        TESSERACT_AVAILABLE = True
        logger.info("Tesseract found at: %s", TESSERACT_PATH)
    else:
        TESSERACT_AVAILABLE = False
        logger.warning("pytesseract imported but tesseract binary not found -- "
                       "OCR tick-mark reading disabled, using linear fallback.")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.info("pytesseract not installed -- using linear interpolation fallback.")


# ---------------------------------------------------------------------------
# Linear helpers
# ---------------------------------------------------------------------------

def angle_to_percent(angle, empty_angle, full_angle):
    span = full_angle - empty_angle
    if span == 0:
        return 0.0
    return max(0.0, min(100.0, (angle - empty_angle) / span * 100))


def calibrate_from_angles(all_angles):
    """
    Given a sequence of angles collected from a full calibration sweep,
    returns (p5_angle, p95_angle) as the estimated empty/full range.

    Raises CalibrationError instead of silently substituting hardcoded
    fallback values -- because a silent wrong calibration produces
    confidently wrong readings, which is worse than a visible failure.

    Requires: the input must be a temporal sequence (not a random set) so
    that np.unwrap can handle the 0/360 wrap-around correctly.
    """
    if len(all_angles) < 10:
        raise CalibrationError(
            f"Calibration requires at least 10 angle readings; only "
            f"{len(all_angles)} were collected. Check that the video "
            f"contains a visible needle sweep and that needle detection is "
            f"working (run with DEBUG logging for per-frame status)."
        )

    angles_rad = np.radians(all_angles)
    unwrapped = np.degrees(np.unwrap(angles_rad))

    p5 = float(np.percentile(unwrapped, 5))
    p95 = float(np.percentile(unwrapped, 95))

    if abs(p95 - p5) < 10:
        raise CalibrationError(
            f"Calibration sweep range is too small ({p5:.1f}° to {p95:.1f}°, "
            f"spread {abs(p95-p5):.1f}°). The needle may not be moving, or "
            f"detection is locking onto a static dial feature. Check the "
            f"video contains a genuine needle sweep."
        )

    return p5, p95


# ---------------------------------------------------------------------------
# OCR tick-mark detection
# ---------------------------------------------------------------------------

def _extract_roi_near_angle(frame, pivot, radius, angle_deg, roi_width=60, roi_height=40):
    """
    Extracts a small image patch just outside the dial rim at a given angle.
    Samples at 1.15× the dial radius, where numeric labels typically appear.
    """
    px, py = int(pivot[0]), int(pivot[1])
    rad = math.radians(angle_deg)
    label_r = radius * 1.15  # just outside the rim where labels sit
    cx = int(px + label_r * math.cos(rad))
    cy = int(py - label_r * math.sin(rad))  # Y flipped

    x1 = max(0, cx - roi_width // 2)
    y1 = max(0, cy - roi_height // 2)
    x2 = min(frame.shape[1], cx + roi_width // 2)
    y2 = min(frame.shape[0], cy + roi_height // 2)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _ocr_number_from_roi(roi):
    """
    Runs Tesseract on a small ROI to extract a numeric label.
    Returns float or None. Wraps the OCR call in try/except so a missing
    or misbehaving Tesseract binary fails gracefully per-ROI rather than
    crashing the whole calibration step.
    """
    if roi is None or not TESSERACT_AVAILABLE:
        return None
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3),
                         interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789.-"
        text = pytesseract.image_to_string(th, config=config).strip()
        nums = re.findall(r"-?\d+\.?\d*", text)
        if nums:
            return float(nums[0])
    except Exception as exc:
        logger.debug("OCR failed on ROI: %s", exc)
    return None


def detect_tick_marks(frame, pivot, radius, dial_radius_px):
    """
    Finds short radial line segments (tick marks) around the dial perimeter
    using Canny + HoughLinesP, then samples just outside each tick for a label.
    Returns list of (angle_deg, label_value) sorted by angle.
    Only includes entries where OCR successfully read a number.
    """
    if not TESSERACT_AVAILABLE:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=int(dial_radius_px * 0.05),
        maxLineGap=5,
    )
    if lines is None:
        return []

    px, py = int(pivot[0]), int(pivot[1])
    results = []
    seen_angles = set()

    for l in lines:
        x1, y1, x2, y2 = l[0]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.hypot(mx - px, my - py)

        if not (dial_radius_px * 0.70 <= dist <= dial_radius_px * 0.95):
            continue

        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue

        to_mid_x, to_mid_y = mx - px, my - py
        to_mid_len = math.hypot(to_mid_x, to_mid_y)
        if to_mid_len < 1e-6:
            continue

        radial_alignment = abs(
            (dx * to_mid_x + dy * to_mid_y) / (seg_len * to_mid_len)
        )
        if radial_alignment < 0.6:
            continue

        angle_deg = math.degrees(math.atan2(-(my - py), mx - px)) % 360
        angle_rounded = round(angle_deg / 3) * 3
        if angle_rounded in seen_angles:
            continue
        seen_angles.add(angle_rounded)

        roi = _extract_roi_near_angle(frame, pivot, dial_radius_px, angle_deg)
        value = _ocr_number_from_roi(roi)
        if value is not None:
            results.append((angle_deg, value))

    results.sort(key=lambda x: x[0])
    logger.info("OCR found %d labeled tick marks: %s", len(results), results)
    return results


# ---------------------------------------------------------------------------
# Scale mapping
# ---------------------------------------------------------------------------

class GaugeScale:
    """
    Converts needle angle to a physical reading.

    With >= 2 OCR tick entries: piecewise-linear interpolation between
    detected labels (handles non-linear scales).
    Otherwise: falls back to linear 0-100% between empty/full.
    """

    def __init__(self, cfg):
        self._empty_angle = cfg.empty_angle
        self._full_angle = cfg.full_angle
        self._tick_map = []

        if cfg.tick_map and len(cfg.tick_map) >= 2:
            self._tick_map = sorted(cfg.tick_map, key=lambda x: x[0])
            angles = [t[0] for t in self._tick_map]
            values = [t[1] for t in self._tick_map]
            logger.info(
                "GaugeScale built from %d OCR ticks: %.1f°→%s .. %.1f°→%s",
                len(self._tick_map),
                angles[0], values[0],
                angles[-1], values[-1],
            )
        else:
            if cfg.tick_map:  # some ticks but fewer than 2
                logger.warning(
                    "Only %d OCR tick(s) found -- need >= 2 for interpolation; "
                    "falling back to linear percent.", len(cfg.tick_map)
                )
            else:
                logger.info("No OCR ticks -- using linear percent scale.")

    def angle_to_value(self, angle):
        """Returns (value, unit_type) where unit_type is 'percent' or 'physical'."""
        if len(self._tick_map) >= 2:
            angles = [t[0] for t in self._tick_map]
            values = [t[1] for t in self._tick_map]
            if angle <= angles[0]:
                return values[0], "physical"
            if angle >= angles[-1]:
                return values[-1], "physical"
            for i in range(len(angles) - 1):
                if angles[i] <= angle <= angles[i + 1]:
                    t = (angle - angles[i]) / (angles[i + 1] - angles[i])
                    v = values[i] + t * (values[i + 1] - values[i])
                    return round(v, 2), "physical"

        pct = angle_to_percent(angle, self._empty_angle, self._full_angle)
        return pct, "percent"

    def to_percent(self, angle):
        """Always returns 0-100% (for dashboard arc), regardless of scale type."""
        return angle_to_percent(angle, self._empty_angle, self._full_angle)