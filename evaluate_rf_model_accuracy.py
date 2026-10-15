import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# ===== Load bundle =====
file_path = "saved_models/opg_bundle_v3.pkl"
data = joblib.load(file_path)

models = data["models"]
feature_columns = data["feature_columns"]
target_columns = data["target_columns"]

print("✅ Bundle Loaded Successfully\n")

# ===== Load dataset =====
df = pd.read_csv("dataset_v3_with_features.csv")

# ===== Features (IMPORTANT FIX HERE) =====
X = df[feature_columns].values

print("🎯 Accuracy Results:\n")

# ===== Loop for each target =====
for target_name in target_columns:
    if target_name in df.columns:
        y_true = df[target_name].values
        model = models[target_name]

        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)

        print(f"{target_name} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\n✅ Evaluation Completed")