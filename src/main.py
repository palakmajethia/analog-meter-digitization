import cv2

from preprocessing import preprocessing_image
from needle_detection import detect_needle
from angle_calculation import calculate_angle
from alert_logic import classify_range

img = cv2.imread("images/test.jpg")
gray, blur, edges = preprocessing_image(img)
line = detect_needle(edges)

if lines is not None:

    angle = calculate_angle(line)
    status = classify_range(angle)

    x1, y1, x2, y2 = line[0]

    cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 2)

    print ("Needle Angle:" , angle)

    print ("Status:" , status)

cv2.imwrite("outputs/final_output.jpg", img)

cv2.imshow("Final Output", img)

cv2.waitKey(0)

cv2.destroyAllWindows()