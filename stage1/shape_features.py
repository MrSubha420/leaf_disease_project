import cv2
import numpy as np


def extract_shape_features(leaf_mask):

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "Leaf_Area_px": 0,
            "Perimeter_px": 0,
            "Circularity": 0,
            "Solidity": 0,
            "Aspect_Ratio": 0
        }

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    # Circularity
    if perimeter > 0:
        circularity = (
            4 * np.pi * area /
            (perimeter ** 2)
        )
    else:
        circularity = 0

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0

    # Aspect ratio
    x, y, w, h = cv2.boundingRect(contour)

    if h > 0:
        aspect_ratio = w / h
    else:
        aspect_ratio = 0

    return {
        "Leaf_Area_px": float(area),
        "Perimeter_px": float(perimeter),
        "Circularity": float(circularity),
        "Solidity": float(solidity),
        "Aspect_Ratio": float(aspect_ratio)
    }