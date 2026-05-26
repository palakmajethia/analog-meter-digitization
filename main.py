import cv2
import numpy as np
import math

img = cv2.imread("images/test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi/180,
    50,
    minLineLength=50,
    maxLineGap=10
)

if lines is not None:

    longest_line = max(lines, key=lambda l:
        np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2)
    )

    x1, y1, x2, y2 = longest_line[0]

    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

    angle = math.degrees(math.atan2(y2-y1, x2-x1))

    print("Needle Angle:", angle)

    min_angle = -90
    max_angle = 90

    min_value = 0
    max_value = 100

    value = np.interp(
        angle,
        [min_angle, max_angle],
        [min_value, max_value]
    )

    print("Meter Reading:", round(value,2))

cv2.imshow("Final Output", img)

cv2.waitKey(0)
cv2.destroyAllWindows()