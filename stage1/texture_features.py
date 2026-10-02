import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_texture_features(image, mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Keep only leaf/disease region
    gray_values = gray[mask > 0]

    if len(gray_values) < 10:
        return {
            "Contrast": 0,
            "Correlation": 0,
            "Energy": 0,
            "Homogeneity": 0,
            "Entropy": 0
        }

    # Create masked grayscale image
    masked_gray = np.zeros_like(gray)
    masked_gray[mask > 0] = gray[mask > 0]

    # Quantize to 8 levels
    quantized = (
        masked_gray.astype(np.float32) / 32
    ).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    )[0, 0]

    correlation = graycoprops(
        glcm,
        "correlation"
    )[0, 0]

    energy = graycoprops(
        glcm,
        "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    )[0, 0]

    # Entropy
    probabilities = glcm.flatten()
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities *
        np.log2(probabilities)
    )

    return {
        "Contrast": float(contrast),
        "Correlation": float(correlation),
        "Energy": float(energy),
        "Homogeneity": float(homogeneity),
        "Entropy": float(entropy)
    }