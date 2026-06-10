import cv2
import numpy as np

def detect_needle(edges):
    """
    Detects the gauge needle by filtering out large background arcs
    and running a linear regression strictly through the remaining needle pixels.
    """
    # 1. Find all distinct red shapes (contours) in our color mask
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # 2. Create a blank black canvas of the exact same size
    clean_needle_mask = np.zeros_like(edges)

    # 3. Filter shapes by area size
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Gating loop: The red warning block is massive (> 3000 pixels). 
        # The needle is thin and fits cleanly in this area window.
        if 80 < area < 2500:
            cv2.drawContours(clean_needle_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 4. Now extract pixels ONLY from our cleaned needle canvas
    pts = np.argwhere(clean_needle_mask == 255)
    
    # Guard clause: If no needle pixels survived the gate, exit
    if len(pts) < 30: 
        return None
        
    # 5. Flip coordinates from [y, x] to standard Cartesian [x, y]
    points = np.fliplr(pts).astype(np.float32)
    
    # 6. Fit the line through the isolated needle points
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Extract native Python scalars safely using .item()
    vx = float(vx.item())
    vy = float(vy.item())
    x0 = float(x0.item())
    y0 = float(y0.item())
    
    # 7. Extrapolate out the line segment length for main.py to render
    length = 250
    x1 = int(x0 - vx * length)
    y1 = int(y0 - vy * length)
    x2 = int(x0 + vx * length)
    y2 = int(y0 + vy * length)
    
    # Return in nested format to match main.py structure requirements
    return [[x1, y1, x2, y2]]