import cv2
import numpy as np


def extract_color_features(image, mask):

    pixels = image[mask > 0]

    if len(pixels) == 0:
        return {
            "Mean_R": 0,
            "Mean_G": 0,
            "Mean_B": 0,
            "SD_R": 0,
            "SD_G": 0,
            "SD_B": 0
        }

    # OpenCV uses BGR
    blue = pixels[:, 0]
    green = pixels[:, 1]
    red = pixels[:, 2]

    return {
        "Mean_R": float(np.mean(red)),
        "Mean_G": float(np.mean(green)),
        "Mean_B": float(np.mean(blue)),

        "SD_R": float(np.std(red)),
        "SD_G": float(np.std(green)),
        "SD_B": float(np.std(blue))
    }