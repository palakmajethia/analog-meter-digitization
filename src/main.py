import cv2
import numpy as np

from preprocessing     import preprocess_image
from needle_detection  import detect_needle
from angle_calculation import calculate_tip_angle, get_pivot
from gauge_mapping     import angle_to_percent, calibrate_from_angles
from alert_logic       import classify_range, get_interval
import math


# =========================
# AUTO-CALIBRATION
# =========================

def calibrate(cap):
    """
    Scans every 3rd frame of the video, collects all valid needle angles,
    then delegates to gauge_mapping.calibrate_from_angles() for the
    p5/p95 robust range. Resets video to frame 0 when done.
    """
    print("[CAL] Scanning full video to find needle range...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_angles   = []
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 3 == 0:
            frame = cv2.resize(frame, (800, 600))
            _, _, edges = preprocess_image(frame)
            line = detect_needle(edges)
            if line is not None:
                all_angles.append(calculate_tip_angle(line, frame.shape))
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    empty_angle, full_angle = calibrate_from_angles(all_angles)

    print(f"[CAL] Scanned {len(all_angles)} readings from {total_frames} frames.")
    print(f"[CAL] EMPTY (p5)  = {empty_angle:.2f} deg")
    print(f"[CAL] FULL  (p95) = {full_angle:.2f} deg")
    return empty_angle, full_angle


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


def draw_dashboard(percent, smooth_angle, status, empty_angle, full_angle):
    dash = np.full((DASH_H, DASH_W, 3), C_BG, dtype=np.uint8)

    cx, cy    = DASH_W // 2, 235
    radius    = 150
    thickness = 18
    font      = cv2.FONT_HERSHEY_SIMPLEX

    # Our angle system: 0=left(west), 90=up(north), 180=right(east)
    # OpenCV ellipse:   0=right,      90=down,       180=left
    # Conversion: oc = (180 + our_angle) % 360
    def to_oc(a):
        return (180 + a) % 360

    oc_empty = to_oc(empty_angle)
    oc_full  = to_oc(full_angle)
    arc_start = min(oc_empty, oc_full)
    arc_end   = max(oc_empty, oc_full)

    # Background track
    cv2.ellipse(dash, (cx, cy), (radius, radius),
                0, arc_start, arc_end, C_ARC_TRACK, thickness)

    # Filled arc — grows from EMPTY side
    sweep = arc_end - arc_start
    fill  = sweep * (percent / 100.0)
    if fill > 0.5:
        if oc_empty <= oc_full:
            cv2.ellipse(dash, (cx, cy), (radius, radius),
                        0, arc_start, arc_start + fill,
                        arc_color(percent), thickness)
        else:
            cv2.ellipse(dash, (cx, cy), (radius, radius),
                        0, arc_end - fill, arc_end,
                        arc_color(percent), thickness)

    # Needle
    rad        = math.radians(smooth_angle)
    needle_len = radius - thickness - 8
    nx = int(cx - needle_len * math.cos(rad))
    ny = int(cy - needle_len * math.sin(rad))
    cv2.line(dash, (cx, cy), (nx, ny), C_NEEDLE, 3, cv2.LINE_AA)
    cv2.circle(dash, (cx, cy), 7, C_NEEDLE, -1)

    # E / F labels at arc endpoints
    def label_pos(our_angle, offset=22):
        r  = math.radians(our_angle)
        lx = int(cx - (radius + offset) * math.cos(r))
        ly = int(cy - (radius + offset) * math.sin(r))
        return lx, ly

    ex, ey = label_pos(empty_angle)
    fx, fy = label_pos(full_angle)
    cv2.putText(dash, "E", (ex - 8, ey + 6), font, 0.75, C_ARC_ALERT, 2, cv2.LINE_AA)
    cv2.putText(dash, "F", (fx - 8, fy + 6), font, 0.75, C_ARC_FILL,  2, cv2.LINE_AA)

    # Percent label
    pct_str = f"{percent:.1f}%"
    (pw, _), _ = cv2.getTextSize(pct_str, font, 1.8, 3)
    cv2.putText(dash, pct_str,
                (cx - pw // 2, cy - 18),
                font, 1.8, C_TEXT, 3, cv2.LINE_AA)

    # Status badge
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

    # Title + footer
    cv2.putText(dash, "GAUGE MONITOR",
                (18, 26), font, 0.65, C_DIM, 1, cv2.LINE_AA)
    info = f"Angle: {smooth_angle:.1f}  |  CAL  E={empty_angle:.0f}  F={full_angle:.0f}"
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
EMPTY_ANGLE, FULL_ANGLE = calibrate(cap)
print(f"[CAL] Done.  EMPTY={EMPTY_ANGLE:.2f}  FULL={FULL_ANGLE:.2f}\n")

last_line         = None
angle_history     = []
last_percent      = None
low_streak        = 0
LOW_STREAK_NEEDED = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n[INFO] No more frames. Exiting stream loop...")
        break

    frame = cv2.resize(frame, (800, 600))
    _, _, edges = preprocess_image(frame)
    
    # Try to grab a fresh line signature
    line = detect_needle(edges)

    if line is not None:
        # Fresh frame data matched perfectly! Update historical moving queue
        last_line = line
        angle = calculate_tip_angle(line, frame.shape)
        
        angle_history.append(angle)
        if len(angle_history) > 10:
            angle_history.pop(0)
            
        smooth_angle = float(np.mean(angle_history))
        percent = angle_to_percent(smooth_angle, EMPTY_ANGLE, FULL_ANGLE)
        last_percent = percent
        is_stale_data = False
    else:
        # Blind zone hit. Rely on stable historical queue without copying stale positions
        is_stale_data = True
        if len(angle_history) > 0:
            smooth_angle = float(np.mean(angle_history))
            percent = angle_to_percent(smooth_angle, EMPTY_ANGLE, FULL_ANGLE)
        else:
            smooth_angle = EMPTY_ANGLE
            percent = 0.0

    # Ensure visualization fallback exists if data goes missing early
    active_render_line = line if line is not None else last_line

    if active_render_line is None:
        cv2.putText(frame, "NO NEEDLE DETECTED",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Gauge Feed",      frame)
        cv2.imshow("Edges",           edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # UI Rendering Vector Math
    x1, y1, x2, y2 = active_render_line[0]
    px, py = get_pivot(frame.shape)
    d1 = (x1 - px) ** 2 + (y1 - py) ** 2
    d2 = (x2 - px) ** 2 + (y2 - py) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)
    
    cv2.line(frame, (px, py), (tx, ty), (0, 255, 0), 3)
    cv2.circle(frame, (px, py), 6, (0, 200, 255), -1)

    # Dynamic Recovery Alert Logic
    raw_status = classify_range(percent)
    if raw_status == "LOW ALERT":
        low_streak += 1
    else:
        low_streak = 0

    status = "LOW ALERT" if low_streak >= LOW_STREAK_NEEDED else "NORMAL"

    lower, upper = get_interval(percent)
    prefix_flag = "[STALE]" if is_stale_data else "[FRESH]"
    print(f"{prefix_flag} Angle: {smooth_angle:.2f} | Percent: {percent:.1f}% | "
          f"Interval: {lower}-{upper} | Status: {status}")

    cv2.putText(frame, f"{status} | {percent:.1f}%",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Angle: {smooth_angle:.1f}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    dashboard = draw_dashboard(
        percent, smooth_angle, status, EMPTY_ANGLE, FULL_ANGLE
    )

    cv2.imshow("Gauge Feed",      frame)
    cv2.imshow("Edges",           edges)
    cv2.imshow("Gauge Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP & INDEFINITE HOLD
# =========================
cap.release()

print("\n" + "="*50)
print("[HOLD] Video processing finished successfully.")
print("[HOLD] Windows are now locked. Click inside any window and press ANY KEY to quit.")
print("="*50)

# The zero parameter freezes OpenCV windows indefinitely until a keyboard strike happens
cv2.waitKey(0) 
cv2.destroyAllWindows()
print("Program Ended Cleanly.")