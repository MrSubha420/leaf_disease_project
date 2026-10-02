import cv2
import numpy as np


def remove_background(image):
    """
    Remove background using HSV thresholding.

    Assumption:
    The leaf is primarily green and the background
    is sufficiently different from the leaf.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 30, 20])
    upper_green = np.array([100, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return result, mask