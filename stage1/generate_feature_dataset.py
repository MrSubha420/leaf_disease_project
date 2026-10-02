import os
import cv2
import pandas as pd

from preprocessing import preprocess_image
from background_removal import remove_background
from afkmc_segmentation import afkmc_segmentation

from color_features import extract_color_features
from texture_features import extract_texture_features
from shape_features import extract_shape_features
from disease_features import extract_disease_features


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "../dataset"

OUTPUT_DIR = "../output"

PREPROCESSED_DIR = os.path.join(
    OUTPUT_DIR,
    "preprocessed_images"
)

SEGMENTED_DIR = os.path.join(
    OUTPUT_DIR,
    "segmented_images"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "final_feature_vector_dataset_for_all_image.csv"
)


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_LABELS = {
    "class_0_no_disease": 0,
    "class_1_downy_mildew": 1,
    "class_2_anthracnose": 2,
    "class_3_white_rust": 3,
    "class_4_cladosporium": 4,
    "class_5_spinach_blight": 5
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    PREPROCESSED_DIR,
    exist_ok=True
)

os.makedirs(
    SEGMENTED_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS DATASET
# ============================================================

all_features = []

total_images = 0
successful_images = 0
failed_images = 0


for class_folder, label in CLASS_LABELS.items():

    class_path = os.path.join(
        DATASET_DIR,
        class_folder
    )

    if not os.path.exists(class_path):

        print(
            f"WARNING: Folder not found: "
            f"{class_path}"
        )

        continue

    print("\n===================================")
    print(f"Processing: {class_folder}")
    print(f"Label: {label}")
    print("===================================")

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        )
    ]

    for filename in image_files:

        total_images += 1

        image_path = os.path.join(
            class_path,
            filename
        )

        try:

            print(
                f"[{total_images}] "
                f"Processing {filename}"
            )

            # ------------------------------------------------
            # STEP 1: PREPROCESS
            # ------------------------------------------------

            image = preprocess_image(
                image_path
            )

            # ------------------------------------------------
            # STEP 2: BACKGROUND REMOVAL
            # ------------------------------------------------

            background_removed, leaf_mask = (
                remove_background(image)
            )

            # ------------------------------------------------
            # SAVE PREPROCESSED IMAGE
            # ------------------------------------------------

            output_name = (
                f"{label}_{filename}"
            )

            cv2.imwrite(
                os.path.join(
                    PREPROCESSED_DIR,
                    output_name
                ),
                background_removed
            )

            # ------------------------------------------------
            # STEP 3: AFKMC SEGMENTATION
            # k is calculated from distinct leaf-pixel colors.
            # Largest cluster = healthy; remaining = disease.
            # ------------------------------------------------

            disease_mask, segmented_image, cluster_k = (
                afkmc_segmentation(
                    background_removed,
                    leaf_mask,
                    k=None,
                    lambda_value=1.0
                )
            )

            # ------------------------------------------------
            # SAVE SEGMENTED IMAGE
            # ------------------------------------------------

            cv2.imwrite(
                os.path.join(
                    SEGMENTED_DIR,
                    output_name
                ),
                segmented_image
            )

            # ------------------------------------------------
            # STEP 4: FEATURE EXTRACTION
            # ------------------------------------------------

            color_features = extract_color_features(
                background_removed,
                leaf_mask
            )

            texture_features = extract_texture_features(
                background_removed,
                leaf_mask
            )

            shape_features = extract_shape_features(
                leaf_mask
            )

            disease_features = extract_disease_features(
                background_removed,
                leaf_mask,
                disease_mask
            )

            # ------------------------------------------------
            # COMBINE FEATURES
            # ------------------------------------------------

            feature_row = {
                "Image_Name": filename,

                **color_features,
                **texture_features,
                **shape_features,
                **disease_features,

                "Cluster_K": cluster_k,
                "Label": label
            }

            all_features.append(
                feature_row
            )

            successful_images += 1

        except Exception as e:

            failed_images += 1

            print(
                f"ERROR processing "
                f"{filename}: {e}"
            )


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_features
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1 COMPLETED")
print("=" * 60)

print(
    f"Total images found: {total_images}"
)

print(
    f"Successfully processed: "
    f"{successful_images}"
)

print(
    f"Failed images: "
    f"{failed_images}"
)

print(
    f"Feature dataset shape: "
    f"{df.shape}"
)

print(
    f"CSV saved to:\n{CSV_PATH}"
)

print("\nClass distribution:")

if not df.empty:

    print(
        df["Label"].value_counts()
        .sort_index()
    )