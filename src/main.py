"""
Analog Gauge Digitization -- main entry point.

Usage
-----
  python src/main.py [--video PATH] [--no-gui] [--normalize-lighting]

If --video is omitted, defaults to videos/gauge.mp4 relative to the
working directory. Use --no-gui to disable cv2.imshow windows (e.g. when
running headless / in CI).

Architecture after refactor
---------------------------
main.py   -- this file: argument parsing, main loop, I/O only
calibrator.py -- full-video calibration scan -> GaugeConfig
gauge_mapping.py -- GaugeScale (angle -> value)
dashboard.py    -- draw_dashboard() rendering
alert_logic.py  -- AlertStateMachine, AlertStatus, thresholds
needle_detection.py -- detect_needle(), NeedleStatus
preprocessing.py    -- compute_saturation_mask(), resize_with_letterbox()
angle_calculation.py -- calculate_tip_angle(), detect_dial_radius(), etc.
config.py       -- GaugeConfig dataclass, CalibrationError
"""
import sys
import logging
import argparse
import cv2
import numpy as np

from config import CalibrationError
from calibrator import calibrate_from_sweep_video, TARGET_SIZE
from preprocessing import resize_with_letterbox, compute_saturation_mask, normalize_lighting
from needle_detection import detect_needle, NeedleStatus
from angle_calculation import calculate_tip_angle
from gauge_mapping import GaugeScale
from alert_logic import AlertStateMachine, AlertStatus
from dashboard import draw_dashboard

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analog gauge digitization")
    p.add_argument("--video", default="videos/gauge.mp4",
                   help="Path to gauge video (default: videos/gauge.mp4)")
    p.add_argument("--no-gui", action="store_true",
                   help="Disable cv2.imshow windows (headless mode)")
    p.add_argument("--normalize-lighting", action="store_true",
                   help="Apply CLAHE lighting normalization before detection "
                        "(helps with shadows / glare; small CPU cost)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Open video -------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Could not open video: %s", args.video)
        sys.exit(1)

    # --- Calibrate --------------------------------------------------------
    logger.info("Starting calibration on: %s", args.video)
    try:
        cfg = calibrate_from_sweep_video(cap)
    except CalibrationError as exc:
        logger.error("CALIBRATION FAILED: %s", exc)
        logger.error("Cannot proceed without a valid calibration. "
                     "Ensure the video shows a full needle sweep with a "
                     "clearly visible colored needle.")
        cap.release()
        sys.exit(1)

    # --- Build scale from calibration result ------------------------------
    scale = GaugeScale(cfg)

    # --- Processing state -------------------------------------------------
    angle_history  = []      # rolling window for temporal smoothing
    SMOOTH_WINDOW  = 10
    last_good_line = None    # last healthy detected line, for stale-data rendering
    alert_sm       = AlertStateMachine(streak_needed=5)
    frame_count    = 0

    logger.info("Entering main loop. Press Q in any window to quit.")

    while True:
        ret, raw = cap.read()
        if not ret:
            logger.info("End of video. Exiting.")
            break

        frame, _, _ = resize_with_letterbox(raw, TARGET_SIZE)

        if args.normalize_lighting:
            frame = normalize_lighting(frame)

        sat_mask = compute_saturation_mask(frame)
        line, status, _ = detect_needle(
            sat_mask, frame,
            dial_radius=cfg.dial_radius,
            angle_range=(cfg.empty_angle, cfg.full_angle),
        )

        is_fresh = status == NeedleStatus.HEALTHY and line is not None

        if is_fresh:
            last_good_line = line
            angle = calculate_tip_angle(line, frame.shape, pivot=cfg.pivot)
            angle_history.append(angle)
            if len(angle_history) > SMOOTH_WINDOW:
                angle_history.pop(0)

        if not angle_history:
            # No readings at all yet -- skip rendering this frame
            frame_count += 1
            continue

        smooth_angle = float(np.mean(angle_history))
        percent      = scale.to_percent(smooth_angle)
        phys_val, unit_type = scale.angle_to_value(smooth_angle)

        alert_status = alert_sm.update(percent)
        display_status = alert_status if is_fresh else AlertStatus.NORMAL

        lower, upper = _get_interval(percent)
        freshness    = "[FRESH]" if is_fresh else f"[STALE-{status.value}]"

        if unit_type == "physical":
            logger.debug("%s Angle:%.2f | Value:%s [OCR] | Pct:%.1f%% | %s",
                         freshness, smooth_angle, phys_val, percent, alert_status.value)
        else:
            logger.debug("%s Angle:%.2f | Pct:%.1f%% | Interval:%s-%s | %s",
                         freshness, smooth_angle, percent, lower, upper, alert_status.value)

        # --- Overlay on camera feed ---------------------------------------
        render_line = last_good_line if last_good_line is not None else line
        if render_line is not None:
            px, py = int(cfg.pivot[0]), int(cfg.pivot[1])
            x1, y1, x2, y2 = render_line[0]
            d1 = (x1 - px) ** 2 + (y1 - py) ** 2
            d2 = (x2 - px) ** 2 + (y2 - py) ** 2
            tx, ty = (x1, y1) if d1 > d2 else (x2, y2)
            cv2.line(frame, (px, py), (tx, ty), (0, 255, 0), 3)
            cv2.circle(frame, (px, py), 6, (0, 200, 255), -1)

        # Status anomaly overlay
        if status == NeedleStatus.AMBIGUOUS_MULTI_PEAK:
            cv2.putText(frame, "AMBIGUOUS (multi-peak)", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        elif status == NeedleStatus.MISSING_OR_OBSTRUCTED:
            cv2.putText(frame, "NEEDLE NOT DETECTED", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        pct_color = (0, 0, 220) if alert_status == AlertStatus.LOW_ALERT else (0, 200, 120)
        cv2.putText(frame, f"{display_status.value} | {percent:.1f}%",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pct_color, 2)
        cv2.putText(frame, f"Angle: {smooth_angle:.1f}",
                    (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if cfg.any_fallback:
            cv2.putText(frame, "! CALIBRATION FALLBACK", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        # --- Dashboard and display ----------------------------------------
        if not args.no_gui:
            dashboard = draw_dashboard(
                percent, smooth_angle, display_status, cfg,
                physical_value=phys_val,
                unit_type=unit_type,
            )
            cv2.imshow("Gauge Feed", frame)
            cv2.imshow("Saturation Mask", sat_mask)
            cv2.imshow("Gauge Dashboard", dashboard)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User pressed Q -- stopping.")
                break

        frame_count += 1

    # --- Cleanup ----------------------------------------------------------
    cap.release()

    if not args.no_gui:
        logger.info("Processing finished. Press any key in a window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info("Done. Processed %d frames.", frame_count)


def _get_interval(percent, major_step=20):
    minor_step = major_step / 10.0
    lower = (percent // minor_step) * minor_step
    return round(lower, 2), round(lower + minor_step, 2)


if __name__ == "__main__":
    main()