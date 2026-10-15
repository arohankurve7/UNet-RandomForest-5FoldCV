import os
import re
from typing import Dict

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


DATASET_CSV = "dataset_v3_final_116.csv"
IMAGES_DIR = "images"
MODEL_PATH = "model/dental_xray_seg.h5"
SAVE_DIR = "saved_models"
BUNDLE_NAME = "opg_bundle_v6.pkl"

TARGET_COLUMNS = ["Missing_Teeth", "Wisdom_Teeth_Status"]

FEATURE_COLS = [
    "q1", "q2", "q3", "q4",
    "total_area",
    "component_count",
    "largest_component_area",
    "left_right_diff",
    "upper_lower_diff",
    "gap_count",
]

N_SPLITS = 5
RANDOM_STATE = 42


def extract_numeric_id(name: str) -> str:
    base = os.path.splitext(str(name).strip())[0]
    match = re.search(r"(\d+)$", base)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract numeric ID from: {name}")


def find_image_path(images_dir: str, image_name: str) -> str:
    target_id = extract_numeric_id(image_name)
    for fname in os.listdir(images_dir):
        try:
            if extract_numeric_id(fname) == target_id:
                return os.path.join(images_dir, fname)
        except Exception:
            continue
    raise FileNotFoundError(f"Image not found for: {image_name}")


def count_gaps(mask: np.ndarray) -> int:
    col_sum = np.sum(mask, axis=0)
    tooth_cols = col_sum > 0

    gaps = 0
    in_gap = False
    gap_len = 0

    start = int(mask.shape[1] * 0.10)
    end = int(mask.shape[1] * 0.90)

    for has_tooth in tooth_cols[start:end]:
        if not has_tooth:
            in_gap = True
            gap_len += 1
        else:
            if in_gap and gap_len > 10:
                gaps += 1
            in_gap = False
            gap_len = 0

    if in_gap and gap_len > 10:
        gaps += 1

    return gaps


def extract_features_from_image(image_path: str, seg_model) -> Dict[str, float]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape
    mid_x, mid_y = w // 2, h // 2

    img_resized = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_resized, axis=0), axis=-1)

    pred = seg_model.predict(img_tensor, verbose=0)[0, :, :, 0]
    mask = cv2.resize(
        (pred > 0.5).astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    q1_raw = float(np.sum(mask[:mid_y, :mid_x]))
    q2_raw = float(np.sum(mask[:mid_y, mid_x:]))
    q3_raw = float(np.sum(mask[mid_y:, mid_x:]))
    q4_raw = float(np.sum(mask[mid_y:, :mid_x]))

    total = q1_raw + q2_raw + q3_raw + q4_raw

    if total == 0:
        q1 = q2 = q3 = q4 = 0.0
    else:
        q1 = round((q1_raw / total) * 100.0, 4)
        q2 = round((q2_raw / total) * 100.0, 4)
        q3 = round((q3_raw / total) * 100.0, 4)
        q4 = round((q4_raw / total) * 100.0, 4)

    total_area = round((total / (h * w)) * 100.0, 4)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    component_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 20:
            component_areas.append(area)

    component_count = len(component_areas)

    largest_component_area = max(component_areas) if component_areas else 0
    largest_component_area = round((largest_component_area / (h * w)) * 100.0, 4)

    left_density = q1 + q4
    right_density = q2 + q3
    upper_density = q1 + q2
    lower_density = q3 + q4

    left_right_diff = round(abs(left_density - right_density), 4)
    upper_lower_diff = round(abs(upper_density - lower_density), 4)

    gap_count = count_gaps(mask)

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "total_area": total_area,
        "component_count": component_count,
        "largest_component_area": largest_component_area,
        "left_right_diff": left_right_diff,
        "upper_lower_diff": upper_lower_diff,
        "gap_count": gap_count,
    }


def build_feature_table(df: pd.DataFrame, seg_model) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        image_name = str(row["Image"]).strip()
        image_path = find_image_path(IMAGES_DIR, image_name)

        features = extract_features_from_image(image_path, seg_model)

        rows.append({
            "Image": image_name,
            **features,
            "Missing_Teeth": int(row["Missing_Teeth"]),
            "Wisdom_Teeth_Status": int(row["Wisdom_Teeth_Status"]),
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} images...")

    return pd.DataFrame(rows)


def evaluate_with_cross_validation(feature_df: pd.DataFrame):
    X = feature_df[FEATURE_COLS].values

    print("\nRunning 5-Fold Cross Validation...\n")

    for target in TARGET_COLUMNS:
        print("=" * 70)
        print(f"Target: {target}")
        print("=" * 70)

        y = feature_df[target].values

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

        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        print("\nFold-wise Accuracy:")
        for i, score in enumerate(scores, start=1):
            print(f"Fold {i}: {score:.4f} ({score * 100:.2f}%)")

        print(f"\nMean Accuracy: {scores.mean():.4f} ({scores.mean() * 100:.2f}%)")
        print(f"Standard Deviation: {scores.std():.4f} ({scores.std() * 100:.2f}%)")

        y_pred = cross_val_predict(clf, X, y, cv=cv)

        print(f"\nOverall Cross-Validated Accuracy: {accuracy_score(y, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        clf.fit(X, y)
        print("\nFeature Importance:")
        for feat, imp in zip(FEATURE_COLS, clf.feature_importances_):
            print(f"{feat}: {imp:.4f}")
        print("\n")


def train_final_models(feature_df: pd.DataFrame):
    X = feature_df[FEATURE_COLS].values

    bundle = {
        "models": {},
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLUMNS,
    }

    for target in TARGET_COLUMNS:
        print(f"Training final model for {target}...")
        y = feature_df[target].values

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

        clf.fit(X, y)
        bundle["models"][target] = clf

    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, BUNDLE_NAME)
    joblib.dump(bundle, model_path)

    print(f"\nSaved model bundle: {model_path}")


def main():
    print(f"Loading dataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)

    print(f"Loading U-Net model: {MODEL_PATH}")
    seg_model = tf.keras.models.load_model(MODEL_PATH)

    print("Extracting enhanced features...")
    feature_df = build_feature_table(df, seg_model)

    output_csv = "dataset_v6_with_more_features.csv"
    feature_df.to_csv(output_csv, index=False)
    print(f"\nSaved enhanced feature table: {output_csv}")

    evaluate_with_cross_validation(feature_df)
    train_final_models(feature_df)


if __name__ == "__main__":
    main()