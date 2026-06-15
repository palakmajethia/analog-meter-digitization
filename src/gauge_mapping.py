import cv2
import numpy as np
from angle_calculation import get_pivot

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

    # FUTURE-PROOF UPGRADE: Convert to radians and unwrap phase jumps
    # This removes the sudden 360 -> 0 degree drops automatically!
    angles_rad = np.radians(all_angles)
    unwrapped_rad = np.unwrap(angles_rad)
    unwrapped_deg = np.degrees(unwrapped_rad)

    # Now percentiles can be safely calculated on a continuous scale
    empty_angle = float(np.percentile(unwrapped_deg, 5))
    full_angle  = float(np.percentile(unwrapped_deg, 95))

    if abs(full_angle - empty_angle) < 10:
        print(f"[CAL] WARNING: Range too small ({empty_angle:.1f} -> {full_angle:.1f}).")

    return empty_angle, full_angle
# In gauge_mapping.py — add this function



def auto_detect_scale_from_frame(frame):
    """
    Detects EMPTY and FULL angles from the tick mark arc on a static gauge.
    Works by finding the angular extent of all tick-like blobs arranged
    in a circle around the pivot — the outermost angles of that arc
    define the gauge scale limits.
    
    Returns (empty_angle, full_angle) in degrees, or None if detection fails.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 1. Get all edges in the frame ─────────────────────────────────────
    edges = cv2.Canny(blurred, 30, 100)

    # ── 2. Polar warp — unroll the dial into a flat strip ─────────────────
    radius = int(min(h, w) * 0.45)
    flags  = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(
        edges, (radius, 360), (cx, cy), radius, flags
    )

    # ── 3. Look in the tick mark zone only (60%-90% of radius) ────────────
    # Tick marks live in a specific radial band — inside the needle tip,
    # outside the center text/logo area
    start_col = int(radius * 0.60)
    end_col   = int(radius * 0.90)
    strip     = unrolled[:, start_col:end_col]

    # ── 4. Sum each angle row — angles with tick marks have high density ──
    row_sums = np.sum(strip, axis=1).astype(np.float32)
    row_sums = cv2.GaussianBlur(row_sums, (1, 5), 0).flatten()

    # ── 5. Find the threshold — angles that have significant tick activity ─
    mean_val  = np.mean(row_sums)
    std_val   = np.std(row_sums)
    threshold = mean_val + (std_val * 0.5)

    active_angles = np.where(row_sums > threshold)[0]

    if len(active_angles) < 10:
        return None   # not enough tick structure found

    # ── 6. Find the angular extent of the tick arc ─────────────────────────
    # Handle wrap-around: if ticks span across 0°/360° boundary
    # (common for gauges where scale goes from ~210° to ~330°)
    sorted_angles = np.sort(active_angles)
    gaps          = np.diff(sorted_angles)
    
    # If there's a big gap it means the arc wraps around 0°
    if len(gaps) > 0 and np.max(gaps) > 60:
        # Wrap case: the gap IS the empty region at the bottom of the gauge
        # The scale ends just before the gap and starts just after
        gap_idx     = np.argmax(gaps)
        full_angle  = float(sorted_angles[gap_idx])       # end of arc
        empty_angle = float(sorted_angles[gap_idx + 1])   # start of arc
    else:
        # No wrap: simple min/max of active angles
        empty_angle = float(sorted_angles[0])
        full_angle  = float(sorted_angles[-1])

    print(f"[CAL] Scale auto-detected from dial: EMPTY={empty_angle:.1f}° FULL={full_angle:.1f}°")
    return empty_angle, full_angle