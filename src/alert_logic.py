def get_interval(percent, major_step=20):
    minor_step = major_step / 10.0
    lower = (percent // minor_step) * minor_step
    upper = lower + minor_step
    return round(lower, 2), round(upper, 2)


def classify_range(percent):
    if percent < 20:
        return "LOW ALERT"
    return "NORMAL"