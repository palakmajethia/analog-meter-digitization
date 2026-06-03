import cv2

from preprocessing import preprocess_image
from needle_detection import detect_needle
from angle_calculation import calculate_angle
from alert_logic import classify_range


# VIDEO PATH
cap = cv2.VideoCapture("videos/gauge.mp4")

# Check video opened
if not cap.isOpened():
    print("Error opening video")
    exit()

frame_count = 0

while True:

    ret, frame = cap.read()

    # Video ended
    if not ret:
        print("No more frames. Exiting...")
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % 5 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Preprocessing
    gray, blur, edges = preprocess_image(frame)

    # Detect needle line
    line = detect_needle(edges)

    print("Detected Line:", line)

    if line is not None:

        x1, y1, x2, y2 = line[0]

        # Draw detected line
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Calculate angle
        angle = calculate_angle(line)

        print("Angle:", angle)

        # Normalize angle
        normalized_angle = angle + 90

        # Classify status
        status = classify_range(normalized_angle)

        print("Status:", status)

        # Put status text
        cv2.putText(
            frame,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:
        cv2.putText(
            frame,
            "NO NEEDLE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # SHOW WINDOWS
    cv2.imshow("Gauge Monitor", frame)
    cv2.imshow("Edges", edges)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Program Ended.")