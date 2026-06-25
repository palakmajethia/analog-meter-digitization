"""
Calibration: scans a dedicated sweep video and builds a GaugeConfig.

The key assumption here (documented explicitly, because it was previously
hidden): this function requires a DEDICATED CALIBRATION CLIP where the
needle sweeps from one extreme to the other. It is NOT suitable for
pointing at arbitrary monitoring footage of a gauge that sits at a fixed
reading, because p5/p95 of a flat signal will be a tight noise band, not
the true empty/full range.

Fix log:
- calibrate() in old main.py silently produced a GaugeConfig with wrong
  fallback values (30 / 150 degrees) when not enough data was collected.
  CalibrationError is now raised explicitly so the caller is forced to
  handle it, rather than unknowingly continuing with garbage geometry.
- Geometry quality (pivot_is_fallback, radius_is_fallback) is captured
  in GaugeConfig so the dashboard and logs can surface it to the user.
"""
import cv2
import logging
import numpy as np

from config import GaugeConfig, CalibrationError
from preprocessing import resize_with_letterbox, compute_saturation_mask
from needle_detection import detect_needle, NeedleStatus
from angle_calculation import (
    calculate_tip_angle,
    get_pivot,
    detect_pivot_from_lines,
    detect_dial_radius,
)
from gauge_mapping import calibrate_from_angles, detect_tick_marks

logger = logging.getLogger(__name__)

TARGET_SIZE = (800, 600)   # internal processing resolution
CALIBRATION_FRAME_STEP = 3  # sample every Nth frame for speed


def calibrate_from_sweep_video(cap) -> GaugeConfig:
    """
    Scans `cap` (an opened cv2.VideoCapture) and returns a GaugeConfig.

    Raises
    ------
    CalibrationError
        If the video contains no usable needle readings, the sweep range
        is too small to calibrate from, or geometry detection falls back
        on every frame and the result is untrustworthy.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Starting calibration scan (%d total frames, every %dth sampled)...",
                total_frames, CALIBRATION_FRAME_STEP)

    # --- Step 1: detect dial geometry from the first readable frame -------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, raw_first = cap.read()
    if not ret:
        raise CalibrationError("Could not read even the first frame from the video.")

    first_frame, _, _ = resize_with_letterbox(raw_first, TARGET_SIZE)
    dial_radius, circle_pivot, radius_is_fallback = detect_dial_radius(first_frame)
    frame_shape = first_frame.shape

    # --- Step 2: collect angles across the full video ---------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_angles = []
    all_lines  = []
    frame_idx  = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        if frame_idx % CALIBRATION_FRAME_STEP == 0:
            frame, _, _ = resize_with_letterbox(raw, TARGET_SIZE)
            sat_mask = compute_saturation_mask(frame)
            line, status, _ = detect_needle(
                sat_mask, frame,
                dial_radius=dial_radius,
                angle_range=None,
            )
            if status == NeedleStatus.HEALTHY and line is not None:
                all_lines.append(line)
                # Temporary angle using heuristic pivot (real pivot not yet known)
                all_angles.append(calculate_tip_angle(line, frame_shape))
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not all_angles:
        raise CalibrationError(
            "Calibration failed: zero frames with a healthy needle reading. "
            "Check that the video has a visible colored needle and that "
            "preprocessing thresholds suit this gauge."
        )

    # --- Step 3: refine pivot via least-squares line intersection ---------
    fallback_pivot = get_pivot(frame_shape)
    pivot, pivot_is_fallback = detect_pivot_from_lines(
        all_lines, frame_shape, fallback=fallback_pivot
    )

    # --- Step 4: recompute angles with the real pivot ---------------------
    all_angles_real = [
        calculate_tip_angle(l, frame_shape, pivot=pivot) for l in all_lines
    ]

    # --- Step 5: calibrate sweep range (raises CalibrationError on failure)
    p5_angle, p95_angle = calibrate_from_angles(all_angles_real)

    # --- Step 6: auto-detect sweep direction ------------------------------
    n_edge = max(1, len(all_angles_real) // 10)
    start_median = float(np.median(all_angles_real[:n_edge]))
    end_median   = float(np.median(all_angles_real[-n_edge:]))

    dist_start_p5  = abs(start_median - p5_angle)
    dist_start_p95 = abs(start_median - p95_angle)

    if dist_start_p5 < dist_start_p95:
        empty_angle, full_angle = p95_angle, p5_angle
        direction = "descending (start≈p5=FULL, end≈p95=EMPTY)"
    else:
        empty_angle, full_angle = p5_angle, p95_angle
        direction = "ascending (start≈p5=EMPTY, end≈p95=FULL)"

    logger.info("Sweep direction: %s", direction)
    logger.info("EMPTY=%.2f°  FULL=%.2f°  Radius=%dpx", empty_angle, full_angle, dial_radius)
    logger.info("Sampled %d readings from %d frames.", len(all_angles_real), total_frames)

    # --- Step 7: OCR tick marks from first frame --------------------------
    tick_map = []
    logger.info("Scanning first frame for tick-mark labels (OCR)...")
    tick_map = detect_tick_marks(first_frame, pivot, dial_radius, dial_radius)

    cfg = GaugeConfig(
        pivot=pivot,
        dial_radius=dial_radius,
        empty_angle=empty_angle,
        full_angle=full_angle,
        tick_map=tick_map,
        pivot_is_fallback=pivot_is_fallback,
        radius_is_fallback=radius_is_fallback,
        angles_are_fallback=False,
    )

    if cfg.any_fallback:
        logger.warning(
            "Calibration completed with one or more FALLBACK values -- "
            "geometry may be inaccurate:\n%s", cfg.describe()
        )
    else:
        logger.info("Calibration complete:\n%s", cfg.describe())

    return cfg