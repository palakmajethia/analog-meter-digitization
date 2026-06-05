import cv2
import numpy as np
import math

from preprocessing import preprocess_image
from needle_detection import detect_needle
from alert_logic import classify_range, get_interval


# =========================
# CALIBRATION
# =========================
EMPTY_ANGLE = 23
FULL_ANGLE = 148


def calculate_tip_angle(line, frame_shape):

    x1, y1, x2, y2 = line[0]

    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2

    if d1 > d2:
        tx, ty = x1, y1
    else:
        tx, ty = x2, y2

    angle = math.degrees(math.atan2(cy - ty, tx - cx))

    if angle < 0:
        angle += 180

    return angle


def angle_to_percent(angle):
    percent = (angle - EMPTY_ANGLE) / (FULL_ANGLE - EMPTY_ANGLE) * 100
    return max(0, min(100, percent))


# =========================
# VIDEO
# =========================

cap = cv2.VideoCapture("videos/gauge.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

last_line = None
angle_history = []
last_percent = None
low_streak = 0
LOW_STREAK_NEEDED = 5
seen_low = False

while True:

    ret, frame = cap.read()
    if not ret:
        print("No more frames. Exiting...")
        break

    frame = cv2.resize(frame, (800, 600))

    _, _, edges = preprocess_image(frame)

    line = detect_needle(edges)

    if line is not None:
        last_line = line
    else:
        line = last_line

    if line is None:
        cv2.putText(frame, "NO NEEDLE DETECTED",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv2.imshow("Gauge Monitor", frame)
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    x1, y1, x2, y2 = line[0]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    angle = calculate_tip_angle(line, frame.shape)

    angle_history.append(angle)

    if len(angle_history) > 10:
        angle_history.pop(0)

    smooth_angle = np.mean(angle_history)

    percent = angle_to_percent(smooth_angle)

    if last_percent is not None and abs(percent - last_percent) > 20:
        cv2.imshow("Gauge Monitor", frame)
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    last_percent = percent
    raw_status = classify_range(percent)

    if raw_status == "LOW ALERT":
        low_streak += 1
    else:
        low_streak = 0

    if low_streak >= LOW_STREAK_NEEDED:
        status = "LOW ALERT"
        seen_low = True
    elif seen_low:
        status = "LOW ALERT"
    else:
        status = "NORMAL"

    lower, upper = get_interval(percent)

    print(f"Angle: {smooth_angle:.2f} | Percent: {percent:.1f} | Interval: {lower}-{upper} | Status: {status}")

    cv2.putText(frame,
                f"{status} | {percent:.1f}%",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.putText(frame,
                f"Angle: {smooth_angle:.1f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.imshow("Gauge Monitor", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Program Ended")