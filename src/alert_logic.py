def classify_range(angle):
    if angle < 20:
        return "LOW ALERT"
    elif angle > 70:
        return "HIGH ALERT"
    else:
        return "NORMAL"
        