def angle_to_percent(angle, empty_angle, full_angle):
    """
    Linearly maps `angle` from [empty_angle, full_angle] onto [0, 100].

    Works whether the sweep goes left-to-right or right-to-left —
    the sign of (full_angle - empty_angle) handles direction automatically.

    Returns a value clamped to [0.0, 100.0].
    """
    span = full_angle - empty_angle
    if span == 0:
        return 0.0
    percent = (angle - empty_angle) / span * 100
    return max(0.0, min(100.0, percent))


def calibrate_from_angles(all_angles):
    """
    Given a list of all needle angles observed across a video,
    returns (empty_angle, full_angle) using robust percentiles
    so a few bad detections don't skew the calibration.

    Uses p5 as EMPTY and p95 as FULL.
    Falls back to (30.0, 150.0) if fewer than 10 angles are provided.
    """
    import numpy as np

    if len(all_angles) < 10:
        print("[CAL] WARNING: Not enough detections — using fallback 30 / 150.")
        return 30.0, 150.0

    empty_angle = float(np.percentile(all_angles, 5))
    full_angle  = float(np.percentile(all_angles, 95))

    if abs(full_angle - empty_angle) < 10:
        print(f"[CAL] WARNING: Range too small "
              f"({empty_angle:.1f} -> {full_angle:.1f}). "
              f"Needle may not move much in this video.")

    return empty_angle, full_angle