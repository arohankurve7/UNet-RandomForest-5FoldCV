import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
FEATURE_CSV = "dataset_v3_with_features.csv"

TARGET_COLUMNS = [
    "Missing_Teeth",
    "Wisdom_Teeth_Status",
]

FEATURE_COLS = ["q1", "q2", "q3", "q4"]

N_SPLITS = 5
RANDOM_STATE = 42


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

    print("\nRunning 5-Fold Cross Validation...\n")

    for target in TARGET_COLUMNS:
        print("=" * 70)
        print(f"Target: {target}")
        print("=" * 70)

        y = df[target].values

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        scores = cross_val_score(
            clf,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        print("\nFold-wise Accuracy:")
        for i, score in enumerate(scores, start=1):
            print(f"Fold {i}: {score:.4f} ({score * 100:.2f}%)")

        print(f"\nMean Accuracy: {scores.mean():.4f} ({scores.mean() * 100:.2f}%)")
        print(f"Standard Deviation: {scores.std():.4f} ({scores.std() * 100:.2f}%)")

        y_pred = cross_val_predict(
            clf,
            X,
            y,
            cv=cv,
        )

        overall_acc = accuracy_score(y, y_pred)

        print(f"\nOverall Cross-Validated Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

        print("\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        print("\n")


if __name__ == "__main__":
    main()