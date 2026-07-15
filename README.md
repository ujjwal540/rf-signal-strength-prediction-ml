# RF Signal Strength Prediction using Machine Learning

A production-quality, portfolio-ready Electronics & Communication Engineering project that predicts **Received Signal Strength Indicator (RSSI)** from wireless channel and system parameters using classical machine learning.

---

## 1) Introduction

Reliable RSSI estimation is essential in wireless network planning, adaptive power control, handover optimization, and indoor/outdoor coverage analysis. This project builds a modular machine learning pipeline to estimate RSSI from:

- Distance
- Frequency
- Transmit Power
- Antenna Gain
- Obstacle Loss
- Environment Type

If no real dataset is provided, the pipeline automatically generates a realistic synthetic RF dataset based on propagation principles.

---

## 2) Project Architecture

```text
rf-signal-strength-prediction-ml/
│
├── main.py
├── train.py
├── predict.py
├── dataset.py
├── preprocessing.py
├── model.py
├── evaluation.py
├── visualization.py
├── config.py
├── utils.py
│
├── data/
│   ├── rf_dataset.csv
│   └── sample_prediction.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── outputs/
│   ├── prediction.csv
│   ├── heatmap.png
│   ├── scatter.png
│   ├── feature_importance.png
│   ├── residual_plot_<model>.png
│   ├── actual_vs_predicted_<model>.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── metrics.txt
│
├── screenshots/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 3) Workflow Diagram

```text
[Load/Generate Dataset] -> [Preprocess Features] -> [Train 4 Models]
                -> [Evaluate Metrics] -> [Select Best Model]
                -> [Save .pkl Models + Reports + Figures]
                -> [Predict on Sample Input] -> [Export prediction.csv]
```

---

## 4) Installation

```bash
git clone https://github.com/ujjwal540/rf-signal-strength-prediction-ml.git
cd rf-signal-strength-prediction-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5) Usage

Run full pipeline:

```bash
python main.py
```

This command will automatically:
1. Generate/load RF dataset
2. Train all required regressors
3. Evaluate using MAE, MSE, RMSE, R²
4. Save trained models (`.pkl`)
5. Generate and save all plots
6. Export predictions CSV and metrics report

---

## 6) Mathematical Background

Synthetic RSSI generation follows a path-loss based model:

\[
\text{FSPL(dB)} = 32.44 + 20\log_{10}(d_{km}) + 20\log_{10}(f_{MHz})
\]

\[
\text{RSSI(dBm)} = P_t + G_t - \text{FSPL} - L_{obstacle} - L_{environment} + \epsilon
\]

Where:
- \(P_t\): Transmit power in dBm
- \(G_t\): Antenna gain in dBi
- \(d\): Distance
- \(f\): Carrier frequency
- \(\epsilon\): Shadowing / fading noise term

---

## 7) Machine Learning Methodology

Implemented models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Processing strategy:
- Standardize numerical features
- One-hot encode environment type
- Use sklearn `Pipeline` + `ColumnTransformer`
- Compare models with unified metrics

---

## 8) Results

Performance summary is exported to:
- `outputs/metrics.txt`

Predictions exported to:
- `outputs/prediction.csv`

Best model is selected automatically using lowest RMSE.

---

## 9) Visualizations (Publication-Quality)

Generated in `outputs/`:
- RSSI vs Distance (`scatter.png`)
- Actual vs Predicted (`actual_vs_predicted_<model>.png`)
- Residual Plot (`residual_plot_<model>.png`)
- Feature Importance (`feature_importance.png`)
- Correlation Matrix (`correlation_matrix.png`)
- Model Comparison (`model_comparison.png`)
- Signal Strength Heatmap (`heatmap.png`)

---

## 10) Screenshots

Add representative outputs in `screenshots/` and embed below for portfolio presentation.

Example markdown:

```markdown
![Model Comparison](screenshots/model_comparison.png)
![Signal Heatmap](screenshots/heatmap.png)
```

---

## 11) Future Work

- Add XGBoost/LightGBM/CatBoost benchmark models
- Incorporate terrain/building GIS data
- Add confidence intervals and uncertainty estimation
- Introduce hyperparameter tuning (Optuna/GridSearchCV)
- Deploy with FastAPI + Docker for real-time inference
- Integrate MLOps tracking (MLflow)

---

## 12) License

This project is licensed under the MIT License.
