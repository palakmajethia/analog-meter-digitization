import cv2
import numpy as np
import math

from preprocessing import preprocess_image, set_background
from needle_detection  import detect_needle
from angle_calculation import calculate_tip_angle, get_pivot
from gauge_mapping import angle_to_percent, calibrate_from_angles, auto_detect_scale_from_frame
from alert_logic       import classify_range, get_interval

# =========================
# SYSTEM CONFIGURATION
# =========================
# Toggle this depending on the physical gauge you are deploying the camera on.
# "color" = Red, Blue, or Green needles (Uses Saturation logic)
# "black" = Classic industrial black needles on white dials (Uses Luminance logic)
# =========================
# SYSTEM CONFIGURATION
# =========================
print("\n" + "="*40)
print("  INDUSTRIAL GAUGE DIGITIZER INITIALIZATION  ")
print("="*40)
print("Select target gauge profile:")
print("  [1] Colored Needle (Red/Blue/Green)")
print("  [2] Industrial Black Needle")

while True:
    choice = input("\nEnter 1 or 2: ")
    if choice == '1':
        GAUGE_MODE = "color"
        break
    elif choice == '2':
        GAUGE_MODE = "black"
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")
print(f"-> System booting in {GAUGE_MODE.upper()} mode...\n")



# =========================
# AUTO-CALIBRATION
# =========================

def calibrate(cap, target_mode):
    print(f"[CAL] Scanning live webcam. SWEEP NEEDLE NOW if possible!")

    all_angles = []
    raw_frames = []
    frame_idx  = 0
    max_cal_frames = 150

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
            frame = cv2.resize(frame, (800, 600))
            raw_frames.append(frame.copy())

            edges = preprocess_image(frame, mode=target_mode)
            line, status, _ = detect_needle(edges)

            if status == "HEALTHY" and line is not None:
                all_angles.append(calculate_tip_angle(line, frame.shape))

            cv2.putText(edges,
                f"CALIBRATING... Sweep needle! ({frame_idx}/{max_cal_frames})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            cv2.imshow("Calibration Preview", edges)
            cv2.waitKey(1)

        frame_idx += 1
        if frame_idx >= max_cal_frames:
            break

    cv2.destroyWindow("Calibration Preview")

    if len(raw_frames) >= 20:
        set_background(raw_frames)

    # ── Decide which calibration path to take ─────────────────────────────
    needle_range = 0.0
    empty_angle  = None
    full_angle   = None

    if len(all_angles) >= 10:
        e, f        = calibrate_from_angles(all_angles)
        needle_range = abs(f - e)

        if needle_range >= 20:
            # ── PATH A: Needle moved enough — use motion calibration ───────
            print(f"[CAL] PATH A: Motion calibration succeeded. Range={needle_range:.1f}°")
            empty_angle, full_angle = e, f
        else:
            print(f"[CAL] Needle range too small ({needle_range:.1f}°) — trying dial detection.")

    if empty_angle is None:
        # ── PATH B: Static gauge — read scale from tick mark arc ──────────
        print("[CAL] PATH B: Attempting static dial scale detection...")
        
        # Use the clearest frame (middle of calibration window)
        best_frame = raw_frames[len(raw_frames) // 2]
        result     = auto_detect_scale_from_frame(best_frame)

        if result is not None:
            empty_angle, full_angle = result
            print(f"[CAL] PATH B succeeded. EMPTY={empty_angle:.1f}° FULL={full_angle:.1f}°")
        else:
            # ── PATH C: Nothing worked — use safe defaults ─────────────────
            print("[CAL] PATH B failed. Using geometric defaults (210° → 330°).")
            print("[CAL] These defaults assume a standard semicircular gauge.")
            empty_angle, full_angle = 210.0, 330.0

    print(f"[CAL] Final: EMPTY={empty_angle:.2f}  FULL={full_angle:.2f}")
    return empty_angle, full_angle


# =========================
# DASHBOARD (UI Code Remains the Same)
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

    def to_oc(a):
        return (180 + a) % 360

    oc_empty = to_oc(empty_angle)
    oc_full  = to_oc(full_angle)
    arc_start = min(oc_empty, oc_full)
    arc_end   = max(oc_empty, oc_full)

    cv2.ellipse(dash, (cx, cy), (radius, radius),
                0, arc_start, arc_end, C_ARC_TRACK, thickness)

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

    rad        = math.radians(smooth_angle)
    needle_len = radius - thickness - 8
    nx = int(cx - needle_len * math.cos(rad))
    ny = int(cy - needle_len * math.sin(rad))
    cv2.line(dash, (cx, cy), (nx, ny), C_NEEDLE, 3, cv2.LINE_AA)
    cv2.circle(dash, (cx, cy), 7, C_NEEDLE, -1)

    def label_pos(our_angle, offset=22):
        r  = math.radians(our_angle)
        lx = int(cx - (radius + offset) * math.cos(r))
        ly = int(cy - (radius + offset) * math.sin(r))
        return lx, ly

    ex, ey = label_pos(empty_angle)
    fx, fy = label_pos(full_angle)
    cv2.putText(dash, "E", (ex - 8, ey + 6), font, 0.75, C_ARC_ALERT, 2, cv2.LINE_AA)
    cv2.putText(dash, "F", (fx - 8, fy + 6), font, 0.75, C_ARC_FILL,  2, cv2.LINE_AA)

    pct_str = f"{percent:.1f}%"
    (pw, _), _ = cv2.getTextSize(pct_str, font, 1.8, 3)
    cv2.putText(dash, pct_str,
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
    info = f"Angle: {smooth_angle:.1f}  |  CAL  E={empty_angle:.0f}  F={full_angle:.0f}"
    cv2.putText(dash, info,
                (14, DASH_H - 10), font, 0.42, C_DIM, 1, cv2.LINE_AA)

    return dash


# =========================
# MAIN EXECUTION
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: could not open video or webcam.")
    exit()

print("[CAL] Starting auto-calibration (webcam scan)...")
# Pass the Master Toggle into the calibration loop
EMPTY_ANGLE, FULL_ANGLE = calibrate(cap, GAUGE_MODE)
print(f"[CAL] Done.  EMPTY={EMPTY_ANGLE:.2f}  FULL={FULL_ANGLE:.2f}\n")

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
    
    # PASSING THE MODE TO PREPROCESSING
    edges = preprocess_image(frame, mode=GAUGE_MODE)
    line, anomaly_status, _ = detect_needle(edges)

    if anomaly_status == "HEALTHY" and line is not None:
        last_line = line
        angle = calculate_tip_angle(line, frame.shape)
        
        angle_history.append(angle)
        if len(angle_history) > 10:
            angle_history.pop(0)
            
        smooth_angle = float(np.mean(angle_history))
        percent = angle_to_percent(smooth_angle, EMPTY_ANGLE, FULL_ANGLE)
        is_stale_data = False
    else:
        is_stale_data = True
        if len(angle_history) > 0:
            smooth_angle = float(np.mean(angle_history))
            percent = angle_to_percent(smooth_angle, EMPTY_ANGLE, FULL_ANGLE)
        else:
            smooth_angle = EMPTY_ANGLE
            percent = 0.0

    active_render_line = line if (anomaly_status == "HEALTHY" and line is not None) else last_line

    if active_render_line is None:
        cv2.putText(frame, f"ANOMALY: {anomaly_status}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Gauge Feed",      frame)
        cv2.imshow("Edges",           edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    x1, y1, x2, y2 = active_render_line[0]
    px, py = get_pivot(frame.shape)
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

    status = "LOW ALERT" if low_streak >= LOW_STREAK_NEEDED else "NORMAL"

    display_status = anomaly_status if is_stale_data else status

    lower, upper = get_interval(percent)
    prefix_flag = f"[STALE - {anomaly_status}]" if is_stale_data else "[FRESH]"
    print(f"{prefix_flag} Angle: {smooth_angle:.2f} | Percent: {percent:.1f}% | "
          f"Interval: {lower}-{upper} | Status: {status}")

    cv2.putText(frame, f"{display_status} | {percent:.1f}%",
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

cap.release()
cv2.destroyAllWindows()
print("Program Ended Cleanly.")