import cv2
import numpy as np


# ============================================================
# Agglomerative Fuzzy K-Means Clustering (AFKMC)
#
# Eq. (1) : Objective P(U,Z)
# Eq. (2) : Sum of memberships = 1
# Eq. (3) : Cluster-center update
# Eq. (7) : Membership update
# ============================================================


def initialize_centers(data, k, random_state=42):
    """Randomly choose initial centers Z^(0)."""

    if k < 1:
        raise ValueError("k must be >= 1.")

    if k > data.shape[0]:
        raise ValueError(
            "k cannot be greater than the number of data points."
        )

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        data.shape[0],
        size=k,
        replace=False
    )

    return data[indices].copy()


def calculate_distances(data, centers):
    """
    Squared Euclidean distance:

        D_ij = sum_l (z_jl - x_il)^2
    """

    difference = (
        data[:, np.newaxis, :]
        - centers[np.newaxis, :, :]
    )

    distances = np.sum(
        difference ** 2,
        axis=2
    )

    return distances


def update_memberships(distances, lambda_value):
    """
    Fuzzy membership using Equation (7):

        u_ij = exp(-D_ij / lambda) / sum_l exp(-D_il / lambda)
    """

    if lambda_value <= 0:
        raise ValueError(
            "lambda_value must be greater than 0."
        )

    exponent = -distances / lambda_value

    maximum = np.max(
        exponent,
        axis=1,
        keepdims=True
    )

    exponent = exponent - maximum

    exp_values = np.exp(exponent)

    denominator = np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )

    denominator = np.maximum(denominator, 1e-12)

    return exp_values / denominator


def update_centers(data, memberships):
    """
    Cluster centers using Equation (3):

        z_jl = sum_i(u_ij * x_il) / sum_i(u_ij)
    """

    numerator = np.sum(
        memberships[:, :, np.newaxis]
        * data[:, np.newaxis, :],
        axis=0
    )

    denominator = np.sum(
        memberships,
        axis=0
    )[:, np.newaxis]

    denominator = np.maximum(denominator, 1e-12)

    return numerator / denominator


def calculate_objective(distances, memberships, lambda_value):
    """
    Equation (1):

        P(U,Z) = sum_j sum_i u_ij D_ij
                 + lambda * sum_i sum_j u_ij log(u_ij)
    """

    safe_memberships = np.clip(
        memberships,
        1e-12,
        1.0
    )

    distance_term = np.sum(memberships * distances)

    entropy_term = lambda_value * np.sum(
        safe_memberships * np.log(safe_memberships)
    )

    return float(distance_term + entropy_term)


def afkmc(
    data,
    k,
    lambda_value=1.0,
    max_iterations=100,
    tolerance=1e-6,
    random_state=42
):
    """
    Alternating AFKMC optimization.

    Returns:
        centers, memberships, labels, objective_history
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("data must be a 2-D array.")

    n_samples = data.shape[0]

    if k < 1:
        raise ValueError("k must be >= 1.")

    if k > n_samples:
        raise ValueError(
            "k cannot be greater than number of samples."
        )

    if lambda_value <= 0:
        raise ValueError(
            "lambda_value must be greater than 0."
        )

    centers = initialize_centers(
        data,
        k=k,
        random_state=random_state
    )

    objective_history = []
    previous_objective = None
    memberships = None

    for _ in range(max_iterations):

        distances = calculate_distances(data, centers)

        memberships = update_memberships(
            distances,
            lambda_value
        )

        new_centers = update_centers(data, memberships)

        new_distances = calculate_distances(
            data,
            new_centers
        )

        objective = calculate_objective(
            new_distances,
            memberships,
            lambda_value
        )

        objective_history.append(objective)

        if previous_objective is not None:

            change = abs(previous_objective - objective)

            if change < tolerance:
                centers = new_centers
                break

        centers = new_centers
        previous_objective = objective

    final_distances = calculate_distances(data, centers)

    memberships = update_memberships(
        final_distances,
        lambda_value
    )

    labels = np.argmax(memberships, axis=1)

    return (
        centers,
        memberships,
        labels,
        objective_history
    )


def estimate_k_from_distinct_colors(
    color_pixels,
    color_bin=32,
    min_k=2,
    max_k=16
):
    """
    Set k from the number of distinct leaf-pixel colors.

    Exact 8-bit unique colors are usually thousands (sensor
    noise), which is not a usable AFKMC cluster count.
    Nearby colors are therefore grouped by dividing each
    LAB channel into bins, then unique bins are counted.

    k = number of distinct color groups, kept in
    [min_k, max_k] only so clustering stays tractable.
    """

    if color_pixels.shape[0] == 0:
        return 1, 0

    quantized = (
        color_pixels.astype(np.int32) // color_bin
    )

    n_distinct = int(
        np.unique(quantized, axis=0).shape[0]
    )

    if n_distinct <= 1:
        return 1, n_distinct

    k = n_distinct
    k = max(min_k, min(k, max_k, color_pixels.shape[0]))

    return k, n_distinct


def afkmc_segmentation(
    image,
    leaf_mask,
    k=None,
    lambda_value=1.0,
    max_iterations=100,
    tolerance=1e-6,
    random_state=42,
    color_bin=32,
    min_k=2,
    max_k=16
):
    """
    Apply AFKMC to leaf pixels in LAB color space.

    If k is None, k is calculated from distinct quantized
    leaf-pixel colors.

    The dominant (largest) cluster is treated as healthy
    tissue. All remaining clusters form the disease mask.

    Returns:
        disease_mask
        segmented_image
        k
    """

    if image is None:
        raise ValueError("Input image is None.")

    if leaf_mask.ndim == 3:
        leaf_mask = cv2.cvtColor(
            leaf_mask,
            cv2.COLOR_BGR2GRAY
        )

    leaf_mask_bin = (leaf_mask > 0).astype(np.uint8)

    if image.shape[:2] != leaf_mask_bin.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} "
            f"does not match leaf mask shape "
            f"{leaf_mask_bin.shape}"
        )

    empty_mask = np.zeros(
        leaf_mask_bin.shape,
        dtype=np.uint8
    )

    positions = np.where(leaf_mask_bin > 0)

    if positions[0].size < 10:
        return empty_mask, image, 1

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_pixels = lab_image[positions]
    pixels = lab_pixels.astype(np.float64) / 255.0

    if k is None:
        k, n_distinct = estimate_k_from_distinct_colors(
            lab_pixels,
            color_bin=color_bin,
            min_k=min_k,
            max_k=max_k
        )
        print(
            f"    Distinct LAB color groups: {n_distinct} "
            f"-> AFKMC k = {k}"
        )
    else:
        k = int(k)
        k = max(1, min(k, pixels.shape[0]))
        n_distinct = k

    (
        _centers,
        _memberships,
        labels,
        _objective_history
    ) = afkmc(
        data=pixels,
        k=k,
        lambda_value=lambda_value,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_state=random_state
    )

    disease_mask = np.zeros(
        leaf_mask_bin.shape,
        dtype=np.uint8
    )

    if k == 1:
        segmented_image = cv2.bitwise_and(
            image,
            image,
            mask=empty_mask
        )
        return empty_mask, segmented_image, k

    cluster_sizes = np.array([
        np.sum(labels == i)
        for i in range(k)
    ])

    healthy_cluster = int(np.argmax(cluster_sizes))

    disease_pixels = labels != healthy_cluster

    disease_mask[
        positions[0][disease_pixels],
        positions[1][disease_pixels]
    ] = 255

    kernel = np.ones((3, 3), np.uint8)

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    segmented_image = cv2.bitwise_and(
        image,
        image,
        mask=disease_mask
    )

    return disease_mask, segmented_image, k
