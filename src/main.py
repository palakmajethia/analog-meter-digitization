import cv2
import numpy as np
import math

from preprocessing     import preprocess_image
from needle_detection  import detect_needle
from angle_calculation import (calculate_tip_angle, get_pivot,
                                detect_pivot_from_lines, detect_dial_radius)
from gauge_mapping     import (angle_to_percent, calibrate_from_angles,
                                detect_tick_marks, GaugeScale)
from alert_logic       import classify_range, get_interval


# =========================
# AUTO-CALIBRATION
# =========================

def calibrate(cap):
    print("[CAL] Scanning full video to find needle range...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_angles   = []
    all_lines    = []
    frame_idx    = 0

    # Step 1: detect dial radius from first frame
    ret, first_frame = cap.read()
    if ret:
        first_frame = cv2.resize(first_frame, (800, 600))
        DIAL_RADIUS, circle_pivot = detect_dial_radius(first_frame)
    else:
        DIAL_RADIUS  = 240
        circle_pivot = get_pivot((600, 800))
        first_frame  = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Step 2: scan every 3rd frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 3 == 0:
            frame = cv2.resize(frame, (800, 600))
            edges = preprocess_image(frame)
            line, status, _ = detect_needle(edges, frame,
                                            dial_radius=DIAL_RADIUS,
                                            angle_range=None)
            if status == "HEALTHY" and line is not None:
                all_angles.append(calculate_tip_angle(line, frame.shape))
                all_lines.append(line)
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not all_angles:
        print("[ERROR] Calibration failed: No healthy needle signatures found.")
        return 0, 360, DIAL_RADIUS, get_pivot((600, 800)), None

    # Step 3: real pivot from line intersections
    frame_shape = (600, 800)
    PIVOT = detect_pivot_from_lines(all_lines, frame_shape,
                                    fallback=get_pivot(frame_shape))
    print(f"[CAL] Pivot from line intersection: {PIVOT}")
    print(f"[CAL] Pivot from circle detection:  {circle_pivot}")

    # Step 4: recompute angles with real pivot
    all_angles_real = [calculate_tip_angle(l, frame_shape, pivot=PIVOT)
                       for l in all_lines]

    # Step 5: calibrate range
    p5_angle, p95_angle = calibrate_from_angles(all_angles_real)

    # Step 6: auto direction detection
    n_start      = max(1, len(all_angles_real) // 10)
    n_end        = max(1, len(all_angles_real) // 10)
    start_median = float(np.median(all_angles_real[:n_start]))
    end_median   = float(np.median(all_angles_real[-n_end:]))

    print(f"[CAL] Start median: {start_median:.1f}°  End median: {end_median:.1f}°")
    print(f"[CAL] p5={p5_angle:.1f}°  p95={p95_angle:.1f}°")

    dist_start_p5  = abs(start_median - p5_angle)
    dist_start_p95 = abs(start_median - p95_angle)

    if dist_start_p5 < dist_start_p95:
        empty_angle, full_angle = p95_angle, p5_angle
        direction = "descending (start≈p5=FULL, end≈p95=EMPTY)"
    else:
        empty_angle, full_angle = p5_angle, p95_angle
        direction = "ascending (start≈p95=EMPTY, end≈p5=FULL)"

    print(f"[CAL] Auto direction: {direction}")
    print(f"[CAL] Scanned {len(all_angles_real)} readings from {total_frames} frames.")
    print(f"[CAL] EMPTY = {empty_angle:.2f}°  |  FULL = {full_angle:.2f}°")
    print(f"[CAL] Dial radius: {DIAL_RADIUS}px")

    # Step 7: ITEM 6 — OCR tick marks from first frame
    tick_map = []
    if first_frame is not None:
        print("[OCR] Scanning first frame for tick mark labels...")
        tick_map = detect_tick_marks(first_frame, PIVOT, DIAL_RADIUS, DIAL_RADIUS)

    return empty_angle, full_angle, DIAL_RADIUS, PIVOT, tick_map


# =========================
# DASHBOARD
# =========================

DASH_W, DASH_H = 500, 320

C_BG        = (30,  30,  30)
C_ARC_TRACK = (60,  60,  60)
C_ARC_FILL  = (0,   200, 120)
C_ARC_WARN  = (0,   165, 255)
C_ARC_ALERT = (0,   0,   220)
C_NEEDLE    = (255, 255, 255)
C_TEXT      = (220, 220, 220)
C_DIM       = (120, 120, 120)


def arc_color(percent):
    if percent < 20:
        return C_ARC_ALERT
    elif percent < 40:
        return C_ARC_WARN
    return C_ARC_FILL


def draw_dashboard(percent, smooth_angle, status, empty_angle, full_angle,
                   physical_value=None, unit_type='percent'):
    dash = np.full((DASH_H, DASH_W, 3), C_BG, dtype=np.uint8)

    cx, cy    = DASH_W // 2, 235
    radius    = 150
    thickness = 18
    font      = cv2.FONT_HERSHEY_SIMPLEX

    def gauge_to_cv(a):
        return -a

    cv_empty  = gauge_to_cv(empty_angle)
    cv_full   = gauge_to_cv(full_angle)
    arc_start = min(cv_empty, cv_full)
    arc_end   = max(cv_empty, cv_full)

    cv2.ellipse(dash, (cx, cy), (radius, radius),
                0, arc_start, arc_end, C_ARC_TRACK, thickness)

    sweep = arc_end - arc_start
    fill  = sweep * (percent / 100.0)
    if fill > 0.5:
        cv2.ellipse(dash, (cx, cy), (radius, radius),
                    0, arc_start, arc_start + fill,
                    arc_color(percent), thickness)

    rad        = math.radians(smooth_angle)
    needle_len = radius - thickness - 8
    nx = int(cx + needle_len * math.cos(rad))
    ny = int(cy - needle_len * math.sin(rad))
    cv2.line(dash, (cx, cy), (nx, ny), C_NEEDLE, 3, cv2.LINE_AA)
    cv2.circle(dash, (cx, cy), 7, C_NEEDLE, -1)

    def label_pos(gauge_angle, offset=22):
        r  = math.radians(gauge_angle)
        lx = int(cx + (radius + offset) * math.cos(r))
        ly = int(cy - (radius + offset) * math.sin(r))
        return lx, ly

    ex, ey = label_pos(empty_angle)
    fx, fy = label_pos(full_angle)
    cv2.putText(dash, "F", (ex - 8, ey + 6), font, 0.75, C_ARC_FILL,  2, cv2.LINE_AA)
    cv2.putText(dash, "E", (fx - 8, fy + 6), font, 0.75, C_ARC_ALERT, 2, cv2.LINE_AA)

    # Show physical value if OCR found a scale, else show percent
    if unit_type == 'physical' and physical_value is not None:
        val_str = f"{physical_value}"
    else:
        val_str = f"{percent:.1f}%"

    (pw, _), _ = cv2.getTextSize(val_str, font, 1.8, 3)
    cv2.putText(dash, val_str,
                (cx - pw // 2, cy - 18),
                font, 1.8, C_TEXT, 3, cv2.LINE_AA)

    badge_col = C_ARC_ALERT if status == "LOW ALERT" else C_ARC_FILL
    (bw, bh), _ = cv2.getTextSize(status, font, 0.75, 2)
    pad = 7
    bx  = cx - bw // 2
    by  = cy + 32
    cv2.rectangle(dash,
                  (bx - pad, by - bh - pad),
                  (bx + bw + pad, by + pad),
                  badge_col, -1)
    cv2.putText(dash, status, (bx, by),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(dash, "GAUGE MONITOR",
                (18, 26), font, 0.65, C_DIM, 1, cv2.LINE_AA)

    scale_type = "OCR" if unit_type == 'physical' else "LINEAR"
    info = (f"Angle: {smooth_angle:.1f}  |  "
            f"E={empty_angle:.0f} F={full_angle:.0f}  [{scale_type}]")
    cv2.putText(dash, info,
                (14, DASH_H - 10), font, 0.42, C_DIM, 1, cv2.LINE_AA)

    return dash


# =========================
# MAIN EXECUTION
# =========================

cap = cv2.VideoCapture("videos/gauge.mp4")

if not cap.isOpened():
    print("Error: could not open video.")
    exit()

print("[CAL] Starting auto-calibration (full video scan)...")
EMPTY_ANGLE, FULL_ANGLE, DIAL_RADIUS, PIVOT, TICK_MAP = calibrate(cap)
print(f"[CAL] Done.  EMPTY={EMPTY_ANGLE:.2f}  FULL={FULL_ANGLE:.2f}  "
      f"RADIUS={DIAL_RADIUS}px\n")

# Build gauge scale (uses OCR tick map if available, else linear)
SCALE = GaugeScale(EMPTY_ANGLE, FULL_ANGLE, tick_map=TICK_MAP)

last_line         = None
angle_history     = []
low_streak        = 0
LOW_STREAK_NEEDED = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n[INFO] No more frames. Exiting stream loop...")
        break

    frame = cv2.resize(frame, (800, 600))
    edges = preprocess_image(frame)
    line, anomaly_status, _ = detect_needle(
        edges, frame,
        dial_radius=DIAL_RADIUS,
        angle_range=(EMPTY_ANGLE, FULL_ANGLE)
    )

    if anomaly_status == "HEALTHY" and line is not None:
        last_line    = line
        angle        = calculate_tip_angle(line, frame.shape, pivot=PIVOT)
        angle_history.append(angle)
        if len(angle_history) > 10:
            angle_history.pop(0)
        smooth_angle  = float(np.mean(angle_history))
        percent       = SCALE.to_percent(smooth_angle)
        phys_val, unit_type = SCALE.angle_to_value(smooth_angle)
        is_stale_data = False
    else:
        is_stale_data = True
        if len(angle_history) > 0:
            smooth_angle = float(np.mean(angle_history))
            percent      = SCALE.to_percent(smooth_angle)
            phys_val, unit_type = SCALE.angle_to_value(smooth_angle)
        else:
            smooth_angle = EMPTY_ANGLE
            percent      = 0.0
            phys_val, unit_type = 0.0, 'percent'

    active_render_line = (line if (anomaly_status == "HEALTHY" and line is not None)
                          else last_line)

    if active_render_line is None:
        cv2.putText(frame, f"ANOMALY: {anomaly_status}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Gauge Feed", frame)
        cv2.imshow("Edges",      edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    x1, y1, x2, y2 = active_render_line[0]
    px, py = int(PIVOT[0]), int(PIVOT[1])
    d1 = (x1 - px) ** 2 + (y1 - py) ** 2
    d2 = (x2 - px) ** 2 + (y2 - py) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    cv2.line(frame, (px, py), (tx, ty), (0, 255, 0), 3)
    cv2.circle(frame, (px, py), 6, (0, 200, 255), -1)

    raw_status = classify_range(percent)
    if raw_status == "LOW ALERT":
        low_streak += 1
    else:
        low_streak = 0

    status         = "LOW ALERT" if low_streak >= LOW_STREAK_NEEDED else "NORMAL"
    display_status = anomaly_status if is_stale_data else status

    lower, upper = get_interval(percent)
    prefix_flag  = f"[STALE - {anomaly_status}]" if is_stale_data else "[FRESH]"

    if unit_type == 'physical':
        print(f"{prefix_flag} Angle: {smooth_angle:.2f} | "
              f"Value: {phys_val} [OCR scale] | "
              f"Percent: {percent:.1f}% | Status: {status}")
    else:
        print(f"{prefix_flag} Angle: {smooth_angle:.2f} | "
              f"Percent: {percent:.1f}% | "
              f"Interval: {lower}-{upper} | Status: {status}")

    cv2.putText(frame, f"{display_status} | {percent:.1f}%",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Angle: {smooth_angle:.1f}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    dashboard = draw_dashboard(
        percent, smooth_angle, status,
        EMPTY_ANGLE, FULL_ANGLE,
        physical_value=phys_val,
        unit_type=unit_type
    )

    cv2.imshow("Gauge Feed",      frame)
    cv2.imshow("Edges",           edges)
    cv2.imshow("Gauge Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

print("\n" + "="*50)
print("[HOLD] Video processing finished successfully.")
print("[HOLD] Windows are now locked. Click inside any window and press ANY KEY to quit.")
print("="*50)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Program Ended Cleanly.")