# inspect_rf_model.py

import joblib

# ===== Load bundle =====
file_path = "saved_models/opg_bundle_v3.pkl"
data = joblib.load(file_path)

print("✅ Bundle Loaded Successfully\n")

# ===== Bundle-level info =====
print("Type of loaded object:", type(data))
print("\n📦 Keys inside bundle:")
print(data.keys())

feature_columns = data.get("feature_columns", [])
target_columns = data.get("target_columns", [])
models = data.get("models", {})

print("\n📌 Feature Columns:")
print(feature_columns)

print("\n🎯 Target Columns:")
print(target_columns)

# ===== Inspect each Random Forest model =====
for target_name, model in models.items():
    print("\n" + "=" * 60)
    print(f"TARGET: {target_name}")
    print("=" * 60)

    # 1. Model type
    print("1. Model Type:", type(model))

    # 2. Number of trees
    if hasattr(model, "n_estimators"):
        print("2. Number of Trees:", model.n_estimators)

    # 3. Feature importance
    if hasattr(model, "feature_importances_"):
        print("3. Feature Importances:")
        for fname, importance in zip(feature_columns, model.feature_importances_):
            print(f"   {fname}: {importance:.4f}")

    # 4. Feature names
    print("4. Feature Names:", feature_columns)

    # Extra useful details
    if hasattr(model, "class_weight"):
        print("Class Weight:", model.class_weight)

    if hasattr(model, "random_state"):
        print("Random State:", model.random_state)

print("\n✅ Inspection Completed Successfully")