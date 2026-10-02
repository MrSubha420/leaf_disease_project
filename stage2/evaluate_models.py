import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_data import load_dataset, preprocess_dataset
from train_test_split import split_dataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_DIR = "../models"
RESULT_DIR = "../results"

CONFUSION_DIR = os.path.join(
    RESULT_DIR,
    "confusion_matrices"
)

REPORT_DIR = os.path.join(
    RESULT_DIR,
    "classification_reports"
)

os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = load_dataset()

X, y = preprocess_dataset(df)

X_train, X_test, y_train, y_test = split_dataset(X, y)


# --------------------------------------------------
# Models
# --------------------------------------------------

model_names = [
    "logistic_regression",
    "decision_tree",
    "svm",
    "random_forest",
    "mlp",
    "knn"
]


results = []


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\n======================================")
print("MODEL EVALUATION")
print("======================================")


for name in model_names:

    print(f"\nEvaluating: {name}")

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.pkl"
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(
        REPORT_DIR,
        f"{name}_classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)


    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Disease",
            "Downy Mildew",
            "Anthracnose",
            "White Rust",
            "Cladosporium",
            "Spinach Blight"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    cm_path = os.path.join(
        CONFUSION_DIR,
        f"{name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # Store Results
    # --------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

comparison_path = os.path.join(
    RESULT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(results_df.to_string(index=False))

print("\nSaved:", comparison_path)

print("\n======================================")
print("EVALUATION COMPLETED")
print("======================================")