import cv2
import numpy as np
import math

# Read image
img = cv2.imread("images/test.jpg")

# Show original image
cv2.imshow("1. Original Image", img)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("2. Grayscale Image", gray)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

cv2.imshow("3. Blurred Image", blur)

# Edge Detection
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("4. Edge Detection", edges)

# Copy image for line drawing
line_img = img.copy()

# Detect lines
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi/180,
    50,
    minLineLength=50,
    maxLineGap=10
)

if lines is not None:

    # Draw all detected lines
    for line in lines:

        x1, y1, x2, y2 = line[0]

        cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.imshow("5. All Detected Lines", line_img)

    # Find longest line
    longest_line = max(lines, key=lambda l:
        np.sqrt((l[0][2]-l[0][0])**2 +
                (l[0][3]-l[0][1])**2)
    )

    x1, y1, x2, y2 = longest_line[0]

    # Draw longest line separately
    final_img = img.copy()

    cv2.line(final_img, (x1,y1), (x2,y2), (0,255,0), 3)

    # Calculate angle
    angle = math.degrees(math.atan2(y2-y1, x2-x1))

    print("Needle Angle:", angle)

    # Meter calibration
    min_angle = 30
    max_angle = 150

    min_value = 0
    max_value = 100

    # Convert angle to reading
    value = np.interp(
        angle,
        [min_angle, max_angle],
        [min_value, max_value]
    )

    print("Meter Reading:", round(value,2))

    cv2.imshow("6. Final Needle Detection", final_img)

# Wait for key press
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()