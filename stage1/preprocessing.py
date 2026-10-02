import cv2
import numpy as np


def resize_image(image, width=512, height=512):
    """
    Resize image to a fixed size.
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def remove_noise(image):
    """
    Remove image noise while preserving important edges.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def preprocess_image(image_path, width=512, height=512):
    """
    Complete basic preprocessing:
    1. Load image
    2. Resize
    3. Noise removal
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = resize_image(image, width, height)
    image = remove_noise(image)

    return image