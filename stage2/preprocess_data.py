import pandas as pd
import numpy as np


CSV_PATH = "../output/spinach_leaf_image_analysis_1080.csv"


def load_dataset():
    print("Loading feature dataset...")

    df = pd.read_csv(CSV_PATH)

    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["Label"].value_counts().sort_index())

    return df


def preprocess_dataset(df):
    # Remove identifiers and text labels (numeric Label is the target)
    drop_cols = [c for c in ["Image_Name", "Disease_Type"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Separate features and target
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinity values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values using column median
    X = X.fillna(X.median())

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


if __name__ == "__main__":

    df = load_dataset()

    X, y = preprocess_dataset(df)

    print("\nPreprocessing completed successfully.")