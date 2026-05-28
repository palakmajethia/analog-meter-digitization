import cv2

from preprocessing import preprocess_image
from needle_detection import detect_needle
from angle_calculation import calculate_angle
from alert_logic import classify_range

cap = cv2.VideoCapture("videos/gauge.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray, blur, edges = preprocess_image(frame)

    line = detect_needle(edges)

    if line is not None:

        angle = calculate_angle(line)

        normalized_angle = angle + 90

        status = classify_range(normalized_angle)

        x1, y1, x2, y2 = line[0]

        cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            status,
            (30,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Gauge Monitor", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()