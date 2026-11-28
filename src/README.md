# Model Training (`src/`)

`train_models.py` trains, evaluates, and compares five era-prediction classifiers on `data/train_data.csv` (120,000 tracks) and `data/test_data.csv` (30,000 tracks), incorporating domain feature engineering, ordinal era framing, and neural-network benchmarking:

- **Shared Feature Engineering (`src/modeling_utils.py`)**: `AudioFeatureEngineer` accepts the 11 raw Spotify columns and adds 6 interaction ratios (`energy_x_loudness`, `loudness_per_acousticness`, `energy_per_acousticness`, `valence_x_danceability`, `acoustic_x_instrumental`, `loudness_per_energy`).
- **Models Compared**:
  1. **Balanced Logistic Regression** (`class_weight="balanced"`)
  2. **Balanced Random Forest** (`class_weight="balanced"`, `max_depth=10`, `min_samples_leaf=15`)
  3. **Balanced Gradient Boosted Decision Trees** (`HistGradientBoostingClassifier` with `class_weight="balanced"`)
  4. **Ordinal Gradient Boosted Trees (`OrdinalGBDTClassifier`)** — Frank & Hall cumulative ordinal decomposition across `<2000` < `2000-2010` < `2011-2020` < `2021+`
  5. **Balanced MLP Neural Network (`BalancedMLPClassifier`)** — `(128, 64)` MLP benchmark confirming how tabular neural nets compare against tree ensembles

## Usage

From the repository root:

```bash
python src/train_models.py
```

## Outputs

- `models/best_era_model.joblib` — Recommended lightweight production `Pipeline` ready for app/API inference
- `models/{logistic_regression,random_forest,gradient_boosted,ordinal_gbdt,mlp}_era.joblib` — Individual fitted pipelines
- `results/metrics_summary.json` & `results/classification_reports.txt` — Classification + ordinal metrics (`within_1_era_accuracy`, `era_mae`)
- `results/confusion_matrices.png` & `results/feature_importance_comparison.png` — Visual evaluation dashboards
