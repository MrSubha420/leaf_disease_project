import cv2
import numpy as np


def extract_disease_features(
    image,
    leaf_mask,
    disease_mask
):

    leaf_area = np.sum(
        leaf_mask > 0
    )

    disease_area = np.sum(
        disease_mask > 0
    )

    if leaf_area == 0:
        disease_percentage = 0
        lesion_density = 0
    else:
        disease_percentage = (
            disease_area /
            leaf_area
        ) * 100

        lesion_density = (
            disease_area /
            leaf_area
        )

    # Connected components = lesions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        disease_mask,
        connectivity=8
    )

    lesion_areas = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # Ignore tiny noise
        if area >= 10:
            lesion_areas.append(area)

    lesion_count = len(lesion_areas)

    if lesion_count > 0:
        average_lesion_area = np.mean(
            lesion_areas
        )
    else:
        average_lesion_area = 0

    # Edge density
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    leaf_edges = edges[
        leaf_mask > 0
    ]

    if len(leaf_edges) > 0:
        edge_density = (
            np.sum(leaf_edges > 0) /
            len(leaf_edges)
        )
    else:
        edge_density = 0

    # HSV features
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv_pixels = hsv[
        leaf_mask > 0
    ]

    if len(hsv_pixels) > 0:
        mean_h = np.mean(
            hsv_pixels[:, 0]
        )

        mean_s = np.mean(
            hsv_pixels[:, 1]
        )
    else:
        mean_h = 0
        mean_s = 0

    return {
        "Lesion_Count": int(lesion_count),

        "Lesion_Density": float(
            lesion_density
        ),

        "Average_Lesion_Area_px": float(
            average_lesion_area
        ),

        "Disease_Percentage": float(
            disease_percentage
        ),

        "Edge_Density": float(
            edge_density
        ),

        "Mean_H": float(mean_h),

        "Mean_S": float(mean_s)
    }