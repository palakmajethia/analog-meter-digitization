import cv2
import numpy as np

img = cv2.imread("images/test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5),0)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    1
    100
    param1=50
    param2=30
    minRadius=50
    maxRadius=300
)

if circles is not None:
    circles= np.uint16(np.around(circles))
    for i in circles[0, :]:

        cv2.circle(img, (i[0],i[1]), i[2], (0,255,0) , 2)

         cv2.circle(img, (i[0],i[1]), 2, (0,255,0) , 3)
        
cv2.imshow("Circle Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()