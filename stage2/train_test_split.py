import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def augment_train_with_noise(X_train, y_train, n_copies=2, noise_frac=0.03, random_state=42):
    """Add Gaussian feature jitter to training samples only."""
    rng = np.random.default_rng(random_state)

    X_base = X_train.to_numpy(dtype=float) if isinstance(X_train, pd.DataFrame) else np.asarray(X_train, dtype=float)
    y_base = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)

    col_std = np.nanstd(X_base, axis=0)
    col_std = np.where(col_std == 0, 1.0, col_std)

    parts_x = [X_base]
    parts_y = [y_base]

    for i in range(n_copies):
        noise = rng.normal(0.0, noise_frac, size=X_base.shape) * col_std
        parts_x.append(X_base + noise)
        parts_y.append(y_base)

    X_aug = np.vstack(parts_x)
    y_aug = np.concatenate(parts_y)

    if isinstance(X_train, pd.DataFrame):
        X_aug = pd.DataFrame(X_aug, columns=X_train.columns)
    if isinstance(y_train, pd.Series):
        y_aug = pd.Series(y_aug, name=y_train.name)

    print(
        f"Augmented training samples: {len(y_aug)} "
        f"({n_copies} noisy copies, noise_frac={noise_frac})"
    )

    return X_aug, y_aug