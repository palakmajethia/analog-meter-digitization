"""
Dashboard rendering, extracted from main.py.

Fix log:
- Alert tier colors used to be defined inline with *different* thresholds
  than classify_range: arc_color used <20/<40, classify_range used <25.
  The visual dashboard was lying about the actual alert state. Both now
  import LOW_ALERT_PERCENT / WARNING_PERCENT from alert_logic -- one source
  of truth.
- Accepts AlertStatus enum instead of a bare string to match needle_detection
  and alert_logic.
"""
import cv2
import math
import numpy as np

from alert_logic import AlertStatus, LOW_ALERT_PERCENT, WARNING_PERCENT

DASH_W, DASH_H = 500, 320

_C_BG        = (30,  30,  30)
_C_ARC_TRACK = (60,  60,  60)
_C_ARC_FILL  = (0,   200, 120)   # green  -- NORMAL
_C_ARC_WARN  = (0,   165, 255)   # orange -- WARNING
_C_ARC_ALERT = (0,   0,   220)   # red    -- LOW ALERT
_C_NEEDLE    = (255, 255, 255)
_C_TEXT      = (220, 220, 220)
_C_DIM       = (120, 120, 120)


def _arc_color(percent):
    if percent < LOW_ALERT_PERCENT:
        return _C_ARC_ALERT
    if percent < WARNING_PERCENT:
        return _C_ARC_WARN
    return _C_ARC_FILL


def _status_badge_color(status: AlertStatus):
    if status == AlertStatus.LOW_ALERT:
        return _C_ARC_ALERT
    if status == AlertStatus.WARNING:
        return _C_ARC_WARN
    return _C_ARC_FILL


def draw_dashboard(percent, smooth_angle, status: AlertStatus, cfg,
                   physical_value=None, unit_type="percent"):
    """
    Renders the dashboard panel as a (DASH_H x DASH_W x 3) numpy array.

    Parameters
    ----------
    percent       : float  0-100 (always linear sweep %)
    smooth_angle  : float  smoothed needle angle (degrees)
    status        : AlertStatus
    cfg           : GaugeConfig
    physical_value: float | None  OCR-derived physical reading
    unit_type     : 'percent' | 'physical'
    """
    dash = np.full((DASH_H, DASH_W, 3), _C_BG, dtype=np.uint8)
    cx, cy = DASH_W // 2, 235
    radius, thickness = 150, 18
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Arc track (background)
    cv_empty = -cfg.empty_angle
    cv_full  = -cfg.full_angle
    arc_start = min(cv_empty, cv_full)
    arc_end   = max(cv_empty, cv_full)
    cv2.ellipse(dash, (cx, cy), (radius, radius), 0, arc_start, arc_end,
                _C_ARC_TRACK, thickness)

    # Arc fill
    sweep = arc_end - arc_start
    fill  = sweep * (percent / 100.0)
    if fill > 0.5:
        cv2.ellipse(dash, (cx, cy), (radius, radius), 0,
                    arc_start, arc_start + fill, _arc_color(percent), thickness)

    # Needle
    rad = math.radians(smooth_angle)
    needle_len = radius - thickness - 8
    nx = int(cx + needle_len * math.cos(rad))
    ny = int(cy - needle_len * math.sin(rad))
    cv2.line(dash, (cx, cy), (nx, ny), _C_NEEDLE, 3, cv2.LINE_AA)
    cv2.circle(dash, (cx, cy), 7, _C_NEEDLE, -1)

    # E / F labels
    def _label_pos(gauge_angle, offset=22):
        r = math.radians(gauge_angle)
        return (int(cx + (radius + offset) * math.cos(r)),
                int(cy - (radius + offset) * math.sin(r)))

    ex, ey = _label_pos(cfg.empty_angle)
    fx, fy = _label_pos(cfg.full_angle)
    cv2.putText(dash, "E", (ex - 8, ey + 6), font, 0.75, _C_ARC_FILL,  2, cv2.LINE_AA)
    cv2.putText(dash, "F", (fx - 8, fy + 6), font, 0.75, _C_ARC_ALERT, 2, cv2.LINE_AA)

    # Central value display
    if unit_type == "physical" and physical_value is not None:
        val_str = f"{physical_value}"
    else:
        val_str = f"{percent:.1f}%"

    (pw, _), _ = cv2.getTextSize(val_str, font, 1.8, 3)
    cv2.putText(dash, val_str, (cx - pw // 2, cy - 18),
                font, 1.8, _C_TEXT, 3, cv2.LINE_AA)

    # Status badge
    badge_col = _status_badge_color(status)
    status_label = status.value
    (bw, bh), _ = cv2.getTextSize(status_label, font, 0.75, 2)
    pad = 7
    bx, by = cx - bw // 2, cy + 32
    cv2.rectangle(dash, (bx - pad, by - bh - pad), (bx + bw + pad, by + pad),
                  badge_col, -1)
    cv2.putText(dash, status_label, (bx, by), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Header / footer
    cv2.putText(dash, "GAUGE MONITOR", (18, 26), font, 0.65, _C_DIM, 1, cv2.LINE_AA)

    # Fallback warning
    if cfg.any_fallback:
        cv2.putText(dash, "! CALIBRATION FALLBACK ACTIVE",
                    (14, DASH_H - 28), font, 0.38, _C_ARC_ALERT, 1, cv2.LINE_AA)

    scale_type = "OCR" if unit_type == "physical" else "LINEAR"
    info = (f"Angle:{smooth_angle:.1f}  "
            f"E={cfg.empty_angle:.0f} F={cfg.full_angle:.0f}  [{scale_type}]")
    cv2.putText(dash, info, (14, DASH_H - 10), font, 0.42, _C_DIM, 1, cv2.LINE_AA)

    return dash