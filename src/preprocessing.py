import cv2
import numpy as np

def preprocess_image(frame):
    """
    Universal Color-Agnostic Filter.
    Extracts high-saturation elements (the needle) while ignoring
    monochromatic elements (white ticks, black background, gray shadows).
    """
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Extract ONLY the Saturation channel (Index 1)
    # Hue (0) is the specific color. Value (2) is the brightness.
    # Saturation (1) is simply "How colorful is this pixel?"
    saturation_channel = hsv[:, :, 1]
    
    # 3. Apply a simple binary threshold.
    # Any pixel with a saturation score above 80 (out of 255) becomes pure white.
    # Monochromatic ticks/backgrounds (score ~0) become pure black.
    _, thresh = cv2.threshold(saturation_channel, 80, 255, cv2.THRESH_BINARY)
    
    # 4. Clean up stray noise using Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Return the clean 2D matrix for the polar detector
    return clean_mask