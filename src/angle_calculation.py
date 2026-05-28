import math 
def calculate_angle(line):
    x1,y1,x2,y2 = line[0]
    angle = math.degrees(
        math.atan2(y2-y1, x2-x1)
    )
    return angle