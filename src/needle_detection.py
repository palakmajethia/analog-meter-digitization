import cv2
import numpy as np
from angle_calculation import get_pivot

def detect_needle(edges):
    """
    Warps the circular dial image into a flat linear ribbon using Polar Coordinates.
    Locates the needle angle via density peak detection and screens for anomalies
    such as missing needles, severe glare, or physical damage.
    
    Returns:
        tuple: (line_coordinates, status_string, target_angle)
               line_coordinates: [[x1, y1, x2, y2]] or None if compromised.
               status_string: "HEALTHY", "MISSING_OR_OBSTRUCTED", or "MULTIPLE_PEAKS_ANOMALY"
               target_angle: Integer angle (0-359) or None.
    """
    h, w = edges.shape
    
    # 1. Fetch the center pivot coordinates from the angle_calculation module
    px, py = get_pivot((h, w))
    
    # 2. Set search radius footprint for the 800x600 gauge display canvas
    radius = 240 
    
    # 3. Warp the circular image into an unrolled rectangular matrix (360 rows x 240 cols)
    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC
    unrolled = cv2.warpPolar(edges, (radius, 360), (int(px), int(py)), radius, flags)
    
    # 4. Slice columns to look only between 15% and 92% of the radius distance.
    # Blinds detector to central pivot assembly and outer background text elements/rim borders.
    start_col = int(radius * 0.15)
    end_col = int(radius * 0.92)
    row_sums = np.sum(unrolled[:, start_col:end_col], axis=1).astype(np.float32)
    
    # 5. Apply 1D Gaussian Smoothing to eliminate high-frequency edge jitter
    # This ensures we find the true structural "center of mass" of the needle.
    row_sums = cv2.GaussianBlur(row_sums, (1, 11), 0).flatten()
    
    # 6. ANOMALY DETECTION: Signal-to-Noise Ratio (SNR) Check
    mean_density = np.mean(row_sums)
    max_density = np.max(row_sums)
    
    # Threshold: If peak is flat or buried in noise, needle is likely missing or covered
    if max_density < (mean_density * 2.5) or max_density < 100:
        return None, "MISSING_OR_OBSTRUCTED", None

    # 7. Extract target angle peak
    target_angle = int(np.argmax(row_sums))
    
    # 8. ANOMALY DETECTION: Multi-Peak / Conflict Detection (e.g., Glass Crack or Glare)
    # Find all angles holding at least 75% of the primary peak's mass
    significant_peaks = np.where(row_sums > (max_density * 0.75))[0]
    
    if len(significant_peaks) > 0:
        peak_spread = np.max(significant_peaks) - np.min(significant_peaks)
        # Handle 360-degree wrap-around boundaries cleanly
        if 350 in significant_peaks and 0 in significant_peaks:
            wrapped_peaks = [(p if p < 180 else p - 360) for p in significant_peaks]
            peak_spread = np.max(wrapped_peaks) - np.min(wrapped_peaks)
            
        # If wide spread, there are competing linear shapes (like a crack or distinct shadow)
        if peak_spread > 15:
            return None, "MULTIPLE_PEAKS_ANOMALY", target_angle

    # 9. Reconstruct absolute Cartesian endpoints originating from pivot hub
    angle_rad = np.radians(target_angle)
    length = 220
    x1, y1 = int(px), int(py)
    x2 = int(px + length * np.cos(angle_rad))
    y2 = int(py + length * np.sin(angle_rad))
    
    return [[x1, y1, x2, y2]], "HEALTHY", target_angle
