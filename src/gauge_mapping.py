import numpy as np


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

    angles_rad = np.radians(all_angles)
    unwrapped_rad = np.unwrap(angles_rad)
    unwrapped_deg = np.degrees(unwrapped_rad)

    empty_angle = float(np.percentile(unwrapped_deg, 5))
    full_angle  = float(np.percentile(unwrapped_deg, 95))

    if abs(full_angle - empty_angle) < 10:
        print(f"[CAL] WARNING: Range too small ({empty_angle:.1f} -> {full_angle:.1f}).")

    return empty_angle, full_angle