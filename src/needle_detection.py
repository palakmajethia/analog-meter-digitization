import cv2
import numpy as np
from angle_calculation import get_pivot

def detect_needle(edges):
    h, w = edges.shape
    px, py = get_pivot((h, w))

    # Dynamic radius — scales with zoom level
    radius = int(min(h, w) * 0.45)

    # Anomaly: frame too bright
    white_ratio = np.count_nonzero(edges) / (h * w)
    if white_ratio > 0.50:
        return None, "MISSING_OR_OBSTRUCTED", None

    # Polar warp
    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(
        edges, (radius, 360), (int(px), int(py)), radius, flags
    )

    start_col = int(radius * 0.25)
    end_col   = int(radius * 0.85)
    strip     = unrolled[:, start_col:end_col].astype(np.float32)

    # ── RADIAL WEIGHT RAMP ─────────────────────────────────────────────────
    # Each column gets a weight from 0.0 (inner, near pivot/fat base)
    # to 1.0 (outer, thin tip only). Multiplied across all 360 rows at once.
    num_cols = end_col - start_col
    ramp = np.linspace(0.0, 1.0, num_cols, dtype=np.float32)  # shape: (num_cols,)
    strip = strip * ramp[np.newaxis, :]   # broadcast: (360, num_cols) * (1, num_cols)

    row_sums = np.sum(strip, axis=1)

    # Smooth to find structural center of mass
    row_sums = cv2.GaussianBlur(row_sums, (1, 11), 0).flatten()

    # Anomaly: no signal
    mean_density = np.mean(row_sums)
    max_density  = np.max(row_sums)
    if max_density < (mean_density * 2.0) or max_density < 30:
        return None, "MISSING_OR_OBSTRUCTED", None

    # Primary peak = needle angle
    target_angle = int(np.argmax(row_sums))

    # Multi-peak check
    significant = np.where(row_sums > (max_density * 0.75))[0]
    if len(significant) > 0:
        spread = int(significant[-1]) - int(significant[0])
        if spread > 180:
            adjusted = [(p if p < 180 else p - 360) for p in significant]
            spread = int(max(adjusted)) - int(min(adjusted))
        if spread > 20:
            return None, "MULTIPLE_PEAKS_ANOMALY", target_angle

    # Reconstruct line
    angle_rad = np.radians(target_angle)
    length    = int(radius * 0.85)
    x2 = int(px + length * np.cos(angle_rad))
    y2 = int(py + length * np.sin(angle_rad))

    return [[int(px), int(py), x2, y2]], "HEALTHY", target_angle