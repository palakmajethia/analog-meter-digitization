def classify_range(percent):
    """
    Returns 'LOW ALERT' when percent is below 25, otherwise 'NORMAL'.
    """
    if percent < 25:
        return "LOW ALERT"
    return "NORMAL"


def get_interval(percent, major_step=20):
    """
    Returns the (lower, upper) minor-interval bucket for a given percent.
    Default: major_step=20 gives minor buckets of 2.0 units each.
    """
    minor_step = major_step / 10.0
    lower = (percent // minor_step) * minor_step
    upper = lower + minor_step
    return round(lower, 2), round(upper, 2)