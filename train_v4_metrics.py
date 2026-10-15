import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================
FEATURE_CSV = "dataset_v3_with_features.csv"

TARGET_COLUMNS = [
    "Missing_Teeth",
    "Wisdom_Teeth_Status",
]

FEATURE_COLS = ["q1", "q2", "q3", "q4"]

SAVE_DIR = "saved_models"
BUNDLE_NAME = "opg_bundle_v4_metrics.pkl"


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Loading feature dataset: {FEATURE_CSV}")
    df = pd.read_csv(FEATURE_CSV)

    required_cols = FEATURE_COLS + TARGET_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    X = df[FEATURE_COLS].values

    bundle = {
        "models": {},
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLUMNS,
    }

    print("\nTraining and evaluating Random Forest models...\n")

    for target in TARGET_COLUMNS:
        print("=" * 60)
        print(f"Target: {target}")
        print("=" * 60)

        y = df[target].values

        unique_classes = np.unique(y)
        stratify_y = y if len(unique_classes) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y,
        )

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {acc:.4f} ({acc * 100:.2f}%)\n")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nFeature Importance:")
        for feature, importance in zip(FEATURE_COLS, clf.feature_importances_):
            print(f"{feature}: {importance:.4f}")

        print("\n")

        bundle["models"][target] = clf

    model_path = f"{SAVE_DIR}/{BUNDLE_NAME}"
    joblib.dump(bundle, model_path)

    print(f"Evaluation completed.")
    print(f"Metrics model bundle saved at: {model_path}")


if __name__ == "__main__":
    main()