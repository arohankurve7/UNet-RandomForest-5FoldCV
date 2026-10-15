import os
import re
from typing import List

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================
DATASET_CSV = "dataset_v3_final_116.csv"
IMAGES_DIR = "images"
MODEL_PATH = "model/dental_xray_seg.h5"
SAVE_DIR = "saved_models"
BUNDLE_NAME = "opg_bundle_v3.pkl"

TARGET_COLUMNS = [
    "Missing_Teeth",
    "Wisdom_Teeth_Status",
    "Impaction_Type",
    "Alignment_issue",
]

FEATURE_COLS = ["q1", "q2", "q3", "q4"]


# =========================================================
# HELPERS
# =========================================================
def extract_numeric_id(name: str) -> str:
    """
    Extract numeric part from names like:
    Image_1 -> 1
    image_25 -> 25
    7.png -> 7
    """
    name = str(name).strip()
    base = os.path.splitext(name)[0]
    match = re.search(r"(\d+)$", base)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract numeric ID from: {name}")


def find_image_path(images_dir: str, image_name: str) -> str:
    """
    Finds actual image file by numeric ID.
    Example:
    CSV: Image_1
    File: 1.png
    """
    target_id = extract_numeric_id(image_name)

    for fname in os.listdir(images_dir):
        file_id = None
        try:
            file_id = extract_numeric_id(fname)
        except Exception:
            continue

        if file_id == target_id:
            return os.path.join(images_dir, fname)

    raise FileNotFoundError(f"Image not found for: {image_name}")


def extract_q_features(image_path: str, seg_model) -> List[float]:
    """
    Runs U-Net segmentation on the image and computes q1-q4 density features.
    """
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

    q1 = float(np.sum(mask[:mid_y, :mid_x]))
    q2 = float(np.sum(mask[:mid_y, mid_x:]))
    q3 = float(np.sum(mask[mid_y:, mid_x:]))
    q4 = float(np.sum(mask[mid_y:, :mid_x]))

    total = q1 + q2 + q3 + q4
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        round((q1 / total) * 100.0, 4),
        round((q2 / total) * 100.0, 4),
        round((q3 / total) * 100.0, 4),
        round((q4 / total) * 100.0, 4),
    ]


def build_feature_table(df: pd.DataFrame, images_dir: str, seg_model) -> pd.DataFrame:
    """
    Creates dataframe with q1-q4 features + labels.
    """
    rows = []

    for _, row in df.iterrows():
        image_name = str(row["Image"]).strip()
        image_path = find_image_path(images_dir, image_name)
        q1, q2, q3, q4 = extract_q_features(image_path, seg_model)

        rows.append({
            "Image": image_name,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "Missing_Teeth": int(row["Missing_Teeth"]),
            "Wisdom_Teeth_Status": int(row["Wisdom_Teeth_Status"]),
            "Impaction_Type": int(row["Impaction_Type"]),
            "Alignment_issue": int(row["Alignment_issue"]),
        })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Loading dataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)

    required_cols = ["Image"] + TARGET_COLUMNS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Images folder not found: {IMAGES_DIR}\n"
            f"Put all OPG images inside '{IMAGES_DIR}' or update IMAGES_DIR."
        )

    print(f"Loading U-Net model: {MODEL_PATH}")
    seg_model = tf.keras.models.load_model(MODEL_PATH)

    print("Extracting q1-q4 features from images...")
    feature_df = build_feature_table(df, IMAGES_DIR, seg_model)

    feature_csv_path = "dataset_v3_with_features.csv"
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"Saved feature table: {feature_csv_path}")

    X = feature_df[FEATURE_COLS].values

    bundle = {
        "models": {},
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLUMNS,
    }

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("\nTraining Random Forest models...\n")
    for target in TARGET_COLUMNS:
        y = feature_df[target].values

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

        print(f"✅ Accuracy for {target}: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        bundle["models"][target] = clf

    model_path = os.path.join(SAVE_DIR, BUNDLE_NAME)
    joblib.dump(bundle, model_path)
    print(f"\n✅ Bundle saved at: {model_path}")


if __name__ == "__main__":
    main()