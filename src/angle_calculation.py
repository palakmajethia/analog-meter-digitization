import math


def get_pivot(frame_shape):
    """
    Returns the (cx, cy) pivot point — bottom-centre of the frame.
    This is where the needle base sits on a typical semicircular gauge.
    """
    h, w = frame_shape[:2]
    return w // 2, int(h * 0.75)


def calculate_tip_angle(line, frame_shape):
    """
    Returns the needle tip angle on a 0-180 scale using the actual
    pivot position (bottom-centre of frame, not frame centre).

      0   = pointing left  (west  = E / Empty side)
      90  = pointing straight up  (north = middle of gauge)
      180 = pointing right (east  = F / Full side)

    Picks the endpoint FURTHEST from the pivot as the tip.
    Reflects any downward angle into 0-180 (upper semicircle).
    """
    x1, y1, x2, y2 = line[0]
    cx, cy = get_pivot(frame_shape)

    # Endpoint furthest from pivot = tip
    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    tx, ty = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tx - cx
    dy = cy - ty       # invert Y → standard Cartesian (up = positive)

    angle = math.degrees(math.atan2(dy, dx))

    # Reflect lower half into upper half so result is always 0-180
    if angle < 0:
        angle = -angle

    return float(angle)