import os
import sys
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Allow importing files from stage2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_data import load_dataset, preprocess_dataset
from train_test_split import split_dataset, augment_train_with_noise


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = load_dataset()

X, y = preprocess_dataset(df)


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = split_dataset(X, y)

X_train, y_train = augment_train_with_noise(
    X_train,
    y_train,
    n_copies=2,
    noise_frac=0.03,
    random_state=42
)


# --------------------------------------------------
# Define Models
# --------------------------------------------------

models = {

    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=10.0,
            max_iter=4000,
            random_state=42
        ))
    ]),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    ),

    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=800,
            random_state=42
        ))
    ]),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


# --------------------------------------------------
# Train Models
# --------------------------------------------------

trained_models = {}

print("\n======================================")
print("STARTING MODEL TRAINING")
print("======================================")

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    trained_models[name] = model

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.pkl"
    )

    joblib.dump(model, model_path)

    print(f"Saved: {model_path}")


print("\n======================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("======================================")