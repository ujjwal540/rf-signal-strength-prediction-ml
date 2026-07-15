# RF Signal Strength Prediction using Machine Learning

An end-to-end Python machine-learning project for predicting received signal
strength indicator (RSSI, dBm) from wireless-channel and system parameters.
It includes synthetic RF data generation, preprocessing, model training,
evaluation, visualizations, and batch prediction export.

## Features

- Generates a reproducible 5,000-row synthetic RF dataset when one is absent.
- Predicts RSSI from distance, frequency, transmit power, antenna gain,
  obstacle loss, and environment type.
- Trains Linear Regression, Decision Tree, Random Forest, and Gradient
  Boosting regressors.
- Saves fitted pipelines, metrics, predictions, and diagnostic plots.
- Automatically selects the model with the lowest RMSE for batch inference.

## Project structure

```text
.
|-- main.py                           # End-to-end pipeline entry point
|-- rf_signal_strength_prediction.py  # Compatibility entry point
|-- config.py                         # Paths, features, and model configuration
|-- dataset.py                         # Synthetic data and sample-input creation
|-- preprocessing.py                   # Scaling and one-hot encoding
|-- model.py                           # Regressor and pipeline factories
|-- train.py                           # Training and artifact generation
|-- evaluation.py                      # Regression metrics and report writer
|-- predict.py                         # Batch inference from CSV
|-- visualization.py                   # Plot generation helpers
|-- utils.py                           # Logging, directories, and pickle helpers
|-- requirements.txt                   # Python dependencies
|-- data/                              # Generated data and sample input
|-- models/                            # Serialized sklearn pipelines
|-- outputs/                           # Metrics, predictions, and generated charts
`-- screenshots/                       # README portfolio images
```

The generated contents of `data/`, `models/`, and `outputs/` are reproducible
and ignored by Git. Representative images are committed in `screenshots/` for
the README.

## Workflow

```text
Load or generate RF data
  -> preprocess numerical and categorical features
  -> train four regressors
  -> evaluate MAE, MSE, RMSE, and R2
  -> save models and visualizations
  -> choose the lowest-RMSE model
  -> predict the sample input and export prediction.csv
```

The synthetic target uses free-space path loss (FSPL):

```text
FSPL(dB) = 32.44 + 20 log10(distance_km) + 20 log10(frequency_MHz)
RSSI(dBm) = transmit_power + antenna_gain - FSPL
            - obstacle_loss - environment_loss + fading_noise
```

## Installation and usage

```bash
git clone https://github.com/ujjwal540/rf-signal-strength-prediction-ml.git
cd rf-signal-strength-prediction-ml
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

For headless systems, use a non-interactive Matplotlib backend:

```bash
# Windows PowerShell
$env:MPLBACKEND = "Agg"; python main.py

# macOS/Linux
MPLBACKEND=Agg python main.py
```

## Verified output

The included dataset was processed successfully. Gradient Boosting was selected
as the best model by RMSE.

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 4.688803 | 6.083295 | 0.883984 |
| Decision Tree | 6.826181 | 8.714845 | 0.761900 |
| Random Forest | 5.064458 | 6.359464 | 0.873211 |
| Gradient Boosting | **4.002769** | **4.955520** | **0.923013** |

`outputs/prediction.csv` contains 25 batch predictions. Example:

```csv
distance_m,frequency_mhz,environment_type,predicted_rssi_dbm
4803.328943,700,Suburban,-100.247887
1678.452541,3500,Rural,-98.385515
3725.452999,2600,Indoor,-123.534565
```

## Screenshots

### Model comparison

![Model comparison showing RMSE for all trained models](screenshots/model_comparison.png)

### Signal-strength heatmap

![Mean RSSI by distance and frequency bins](screenshots/heatmap.png)

### Actual vs predicted RSSI

![Actual versus predicted RSSI for Gradient Boosting](screenshots/actual_vs_predicted_gradient_boosting.png)

### Residual plots

| Decision Tree | Gradient Boosting |
|---|---|
| ![Decision Tree residual plot](screenshots/residual_plot_decision_tree.png) | ![Gradient Boosting residual plot](screenshots/residual_plot_gradient_boosting.png) |

| Linear Regression | Random Forest |
|---|---|
| ![Linear Regression residual plot](screenshots/residual_plot_linear_regression.png) | ![Random Forest residual plot](screenshots/residual_plot_random_forest.png) |

## Generated artifacts

After `python main.py`, inspect:

- `models/*.pkl` - trained preprocessing-and-model pipelines.
- `outputs/metrics.txt` - MAE, MSE, RMSE, and R2 for every model.
- `outputs/prediction.csv` - sample input with `predicted_rssi_dbm` appended.
- `outputs/scatter.png` and `outputs/correlation_matrix.png` - data diagnostics.
- `outputs/heatmap.png`, `outputs/model_comparison.png`, and
  `outputs/feature_importance.png` - summary visualizations.
- `outputs/actual_vs_predicted_<model>.png` and
  `outputs/residual_plot_<model>.png` - per-model evaluation plots.

## License

This project is licensed under the [MIT License](LICENSE).
