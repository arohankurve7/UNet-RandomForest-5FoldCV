# Dental OPG Analysis Using U-Net and Random Forest

A research-oriented dental panoramic radiograph (OPG) analysis system combining U-Net-based tooth segmentation, handcrafted structural feature extraction, and Random Forest classification.

The project explores automated screening of missing teeth and wisdom-tooth status from panoramic dental X-ray images.

## Key Features

- Tooth-region segmentation using a pretrained U-Net model
- Structural feature extraction from segmented OPG images
- Random Forest classification
- Missing-teeth and wisdom-teeth screening
- Five-fold stratified cross-validation
- Flask-based web dashboard for visualization and predictions

## Methodology

1. Input panoramic dental radiograph (OPG).
2. Preprocess the image for model inference.
3. Segment tooth regions using U-Net.
4. Extract structural and regional features.
5. Classify dental findings using Random Forest.
6. Display predictions and visualizations through the dashboard.

## Feature Engineering

The extended V6 pipeline includes structural features such as:

- Total segmented area
- Connected-component information
- Largest component area
- Left-right structural difference
- Upper-lower structural difference
- Gap count

## Model Training and Evaluation

The extended training pipeline uses:

- Random Forest with 500 estimators
- Balanced class weights
- Five-fold StratifiedKFold cross-validation
- Random state: 42
- Accuracy and classification metrics

The dataset used during development contains 116 panoramic OPG images.

## Dashboard and Model Versions

The verified working dashboard is:

`web_dashboard.py`

It loads:

- `model/dental_xray_seg.h5` — pretrained U-Net segmentation model
- `saved_models/opg_bundle_v3.pkl` — V3 Random Forest bundle

The repository also contains later training and evaluation experiments, including five-fold cross-validation and V6 feature engineering.

**Important:** The working dashboard uses the V3 Random Forest bundle. Results from later experiments should not be interpreted as the dashboard's independently verified performance.

## Installation

Developed using Python 3.8.10 on macOS Apple Silicon.

Install dependencies:

```bash
pip install -r requirements.txt

## Running the Dashboard

Required model files:

- model/dental_xray_seg.h5
- saved_models/opg_bundle_v3.pkl

Run the Flask dashboard:

    python web_dashboard.py

Open the local URL displayed in the terminal.

## Repository Structure

- model/ — U-Net architecture and model utilities
- templates/ — Dashboard HTML template
- train_v3.py — Initial training pipeline
- train_v4_metrics.py — Metrics-oriented training
- train_v5_cross_validation.py — Cross-validation experiment
- train_v6_more_features.py — Extended feature engineering and evaluation
- web_dashboard.py — Working Flask dashboard
- dashboard.py — Additional dashboard implementation
- requirements.txt — Python dependencies

## Data and Model Availability

Medical images, datasets, trained model weights, uploads, and generated outputs are excluded from this repository.

The required model files must be supplied separately to run inference.

## Disclaimer

This project is intended for academic research and exploratory dental screening only. It is not a clinically validated diagnostic device and must not replace professional dental assessment.
