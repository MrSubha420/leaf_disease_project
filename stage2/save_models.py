import os


MODEL_DIR = "../models"

models = [
    "logistic_regression.pkl",
    "decision_tree.pkl",
    "svm.pkl",
    "random_forest.pkl",
    "mlp.pkl",
    "knn.pkl"
]


print("Checking saved models...\n")


all_found = True

for model in models:

    path = os.path.join(
        MODEL_DIR,
        model
    )

    if os.path.exists(path):

        size = os.path.getsize(path)

        print(
            f"[OK] {model} "
            f"({size / 1024:.2f} KB)"
        )

    else:

        print(
            f"[MISSING] {model}"
        )

        all_found = False


print()

if all_found:
    print("All six models are available.")
else:
    print("Some models are missing.")