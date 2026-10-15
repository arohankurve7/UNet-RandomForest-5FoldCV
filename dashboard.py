import os
import cv2
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
SEG_MODEL_PATH = "model/dental_xray_seg.h5"
RF_BUNDLE_PATH = "saved_models/opg_bundle_v6.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading U-Net segmentation model...")
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH)

print("Loading Random Forest v6 bundle...")
bundle = joblib.load(RF_BUNDLE_PATH)
rf_models = bundle["models"]
FEATURE_COLS = bundle["feature_columns"]

MISSING_MAP = {0: "None", 1: "Yes"}
WISDOM_MAP = {0: "Absent", 1: "Present"}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tooth analysis using Panoramic Dental X-rays</title>
<style>
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;text-align:center;background-color:#f3f4f6;padding:20px;margin:0}
.container{max-width:1200px;margin:auto;background:white;padding:30px;border-radius:20px;box-shadow:0 10px 30px rgba(0,0,0,0.08)}
h1{color:#0b7cc1;margin-bottom:10px;font-size:32px}
.upload-section{background:#e5e7eb;padding:20px;border-radius:16px;margin:20px 0 30px 0}
input[type="file"]{font-size:16px;margin-right:20px}
button{background:#0b84c6;color:white;border:none;padding:14px 36px;border-radius:12px;cursor:pointer;font-size:16px;font-weight:600}
.analysis-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-top:20px}
.clinical-box{text-align:left;padding:18px 20px;border-radius:16px;box-shadow:0 3px 10px rgba(0,0,0,0.05);min-height:120px}
.stat-title{font-weight:800;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.stat-value{font-size:22px;font-weight:800;line-height:1.35;word-break:break-word}
.density-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-top:20px}
.density-box{background:#f8fafc;padding:16px;border-radius:14px;border:1px solid #e2e8f0;text-align:left}
.density-title{color:#0b7cc1;font-weight:700;font-size:14px}
.density-value{font-size:22px;color:#1f2937;font-weight:800;margin-top:6px}
.image-comparison{display:flex;justify-content:center;gap:20px;margin-top:35px;flex-wrap:wrap}
.image-box{flex:1;min-width:45%}
.image-box h3{color:#334155;margin-bottom:12px;font-size:20px}
img{width:100%;border-radius:14px;border:2px solid #cbd5e1;box-shadow:0 8px 18px rgba(0,0,0,0.08)}
</style>
</head>
<body>
<div class="container">
<h1>🦷 Teeth Analysis using 2D Panoramic Dental X-rays</h1>

<div class="upload-section">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<button type="submit">Analyze OPG</button>
</form>
</div>

{% if stats %}
<div class="analysis-grid">
<div class="clinical-box" style="border-left:6px solid #ef4444;background:#fef2f2;">
<div class="stat-title" style="color:#b91c1c;">Missing Teeth Detected</div>
<div class="stat-value" style="color:#991b1b;">{{ stats.missing }}</div>
</div>

<div class="clinical-box" style="border-left:6px solid #10b981;background:#ecfdf5;">
<div class="stat-title" style="color:#047857;">Wisdom Tooth</div>
<div class="stat-value" style="color:#065f46;">{{ stats.wisdom }}</div>
</div>

<div class="clinical-box" style="border-left:6px solid #0ea5a4;background:#ecfeff;">
<div class="stat-title" style="color:#0f766e;">Clinical Symmetry Insight</div>
<div class="stat-value" style="color:#134e4a;">{{ stats.symmetry }}</div>
</div>

<div class="clinical-box" style="border-left:6px solid #f59e0b;background:#fffbeb;">
<div class="stat-title" style="color:#b45309;">Anatomy Insight</div>
<div class="stat-value" style="color:#92400e;">{{ stats.anatomy }}</div>
</div>
</div>

<div class="density-grid">
<div class="density-box"><div class="density-title">Q1 Density</div><div class="density-value">{{ stats.q1 }}%</div></div>
<div class="density-box"><div class="density-title">Q2 Density</div><div class="density-value">{{ stats.q2 }}%</div></div>
<div class="density-box"><div class="density-title">Q3 Density</div><div class="density-value">{{ stats.q3 }}%</div></div>
<div class="density-box"><div class="density-title">Q4 Density</div><div class="density-value">{{ stats.q4 }}%</div></div>
</div>

<div class="image-comparison">
<div class="image-box">
<h3>Original OPG</h3>
<img src="{{ original }}" alt="Original OPG">
</div>
<div class="image-box">
<h3>Segmented OPG (with AI analysis)</h3>
<img src="{{ segmented }}" alt="Segmented OPG">
</div>
</div>
{% endif %}
</div>
</body>
</html>
"""


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


def predict_labels(features):
    X = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)

    missing_pred = int(rf_models["Missing_Teeth"].predict(X)[0])
    wisdom_pred = int(rf_models["Wisdom_Teeth_Status"].predict(X)[0])

    return {
        "missing": MISSING_MAP.get(missing_pred, "Unknown"),
        "wisdom": WISDOM_MAP.get(wisdom_pred, "Unknown"),
    }


def compute_clinical_symmetry(q1, q2, q3, q4):
    total = q1 + q2 + q3 + q4
    if total == 0:
        return "Not Applicable"

    left = q2 + q3
    right = q1 + q4
    diff = round(abs(left - right), 2)

    if diff < 10:
        return f"Balanced distribution ({diff}% variance)"
    elif diff < 25:
        return f"Mild asymmetry ({diff}% variance): Possible uneven tooth distribution."
    else:
        return f"Asymmetry detected ({diff}% variance): Possible unilateral tooth loss or localized impaction."


def detect_anatomy(mask, original_img):
    tooth_pixels = original_img[mask > 0]

    if len(tooth_pixels) == 0:
        return "Edentulous state: No natural tooth structures detected"

    tooth_region = cv2.bitwise_and(original_img, original_img, mask=mask)
    _, metal_bin = cv2.threshold(tooth_region, 240, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(metal_bin)

    findings = []
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        if width > 80:
            return "Metal Bridge (Horizontal connectivity)"

        if area > 450:
            findings.append("Metal Prosthetic/Crown")

    max_intensity = float(np.max(tooth_pixels))
    mean_intensity = float(np.mean(tooth_pixels))

    if max_intensity > 235:
        findings.append("High-density Restoration")
    elif mean_intensity > 165 and max_intensity > 210:
        findings.append("Potential Root Canal Filling")

    return " | ".join(set(findings)) if findings else "Normal anatomical distribution"


def draw_styled_text(image, text, pos, scale, thick, color=(255, 255, 0), align="left"):
    font = cv2.FONT_HERSHEY_TRIPLEX

    if align == "right":
        text_w = cv2.getTextSize(text, font, scale, thick)[0][0]
        actual_pos = (pos[0] - text_w, pos[1])
    else:
        actual_pos = pos

    cv2.putText(image, text, actual_pos, font, scale, (0, 0, 0), thick + 5, cv2.LINE_AA)
    cv2.putText(image, text, actual_pos, font, scale, (40, 40, 40), thick + 2, cv2.LINE_AA)
    cv2.putText(image, text, actual_pos, font, scale, color, thick, cv2.LINE_AA)


def perform_fusion_analysis(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read uploaded image.")

    h, w = img.shape
    mid_x, mid_y = w // 2, h // 2

    img_resized = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_resized, axis=0), axis=-1)

    pred = seg_model.predict(img_tensor, verbose=0)[0, :, :, 0]
    mask = cv2.resize((pred > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    q1_sum = float(np.sum(mask[:mid_y, :mid_x]))
    q2_sum = float(np.sum(mask[:mid_y, mid_x:]))
    q3_sum = float(np.sum(mask[mid_y:, mid_x:]))
    q4_sum = float(np.sum(mask[mid_y:, :mid_x]))

    total = q1_sum + q2_sum + q3_sum + q4_sum

    if total == 0:
        q1_p = q2_p = q3_p = q4_p = 0.0
    else:
        q1_p = round((q1_sum / total) * 100, 1)
        q2_p = round((q2_sum / total) * 100, 1)
        q3_p = round((q3_sum / total) * 100, 1)
        q4_p = round((q4_sum / total) * 100, 1)

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

    left_density = q1_p + q4_p
    right_density = q2_p + q3_p
    upper_density = q1_p + q2_p
    lower_density = q3_p + q4_p

    left_right_diff = round(abs(left_density - right_density), 4)
    upper_lower_diff = round(abs(upper_density - lower_density), 4)

    gap_count = count_gaps(mask)

    features = {
        "q1": q1_p,
        "q2": q2_p,
        "q3": q3_p,
        "q4": q4_p,
        "total_area": total_area,
        "component_count": component_count,
        "largest_component_area": largest_component_area,
        "left_right_diff": left_right_diff,
        "upper_lower_diff": upper_lower_diff,
        "gap_count": gap_count,
    }

    preds = predict_labels(features)
    symmetry_text = compute_clinical_symmetry(q1_p, q2_p, q3_p, q4_p)
    anatomy_text = detect_anatomy(mask, img)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    base_scale = h / 1000.0
    title_scale = max(0.95, base_scale * 1.45)
    dens_scale = max(0.62, base_scale * 0.98)
    line_thick = max(2, int(base_scale * 3.2))

    cv2.line(overlay, (mid_x, 0), (mid_x, h), (255, 255, 0), 2)
    cv2.line(overlay, (0, mid_y), (w, mid_y), (255, 255, 0), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), max(2, int(base_scale * 2.4)))

    margin_x = int(w * 0.02)
    margin_top = int(h * 0.06)
    margin_bottom = int(h * 0.08)
    line_gap = int(h * 0.05)

    draw_styled_text(overlay, "Upper Right (Q1)", (margin_x, margin_top), title_scale, line_thick, align="left")
    draw_styled_text(overlay, f"Density: {q1_p}%", (margin_x, margin_top + line_gap), dens_scale, max(2, line_thick - 1), color=(0, 255, 255), align="left")

    draw_styled_text(overlay, "Upper Left (Q2)", (w - margin_x, margin_top), title_scale, line_thick, align="right")
    draw_styled_text(overlay, f"Density: {q2_p}%", (w - margin_x, margin_top + line_gap), dens_scale, max(2, line_thick - 1), color=(0, 255, 255), align="right")

    draw_styled_text(overlay, "Lower Right (Q4)", (margin_x, h - margin_bottom - line_gap), title_scale, line_thick, align="left")
    draw_styled_text(overlay, f"Density: {q4_p}%", (margin_x, h - margin_bottom), dens_scale, max(2, line_thick - 1), color=(0, 255, 255), align="left")

    draw_styled_text(overlay, "Lower Left (Q3)", (w - margin_x, h - margin_bottom - line_gap), title_scale, line_thick, align="right")
    draw_styled_text(overlay, f"Density: {q3_p}%", (w - margin_x, h - margin_bottom), dens_scale, max(2, line_thick - 1), color=(0, 255, 255), align="right")

    cv2.imwrite(output_path, overlay)

    return {
        "q1": q1_p,
        "q2": q2_p,
        "q3": q3_p,
        "q4": q4_p,
        "missing": preds["missing"],
        "wisdom": preds["wisdom"],
        "symmetry": symmetry_text,
        "anatomy": anatomy_text,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    stats = None
    original_url = None
    segmented_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            rand_id = str(np.random.randint(1000, 9999))
            input_name = f"input_{rand_id}.png"
            output_name = f"output_{rand_id}.png"

            input_path = os.path.join(UPLOAD_FOLDER, input_name)
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            file.save(input_path)

            stats = perform_fusion_analysis(input_path, output_path)
            original_url = f"/uploads/{input_name}"
            segmented_url = f"/outputs/{output_name}"

    return render_template_string(
        HTML_PAGE,
        stats=stats,
        original=original_url,
        segmented=segmented_url,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5002)