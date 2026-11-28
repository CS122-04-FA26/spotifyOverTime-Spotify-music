# Source Code (`src/`)

## 1. Model Training (`src/train_models.py` & `src/modeling_utils.py`)

Trains, evaluates, and compares five era-prediction classifiers on `data/train_data.csv` (120,000 tracks) and `data/test_data.csv` (30,000 tracks):
- **Shared Feature Engineering (`AudioFeatureEngineer`)**: Computes 6 interaction features (`energy_x_loudness`, `loudness_per_acousticness`, `energy_per_acousticness`, `valence_x_danceability`, `acoustic_x_instrumental`, `loudness_per_energy`).
- **Models**: Balanced Logistic Regression, Balanced Random Forest, Balanced Gradient Boosted Decision Trees, Frank & Hall Ordinal GBDT (`OrdinalGBDTClassifier`), and Balanced MLP Neural Network (`BalancedMLPClassifier`).

```bash
python src/train_models.py
```

## 2. Flask Backend (`src/app.py`)

Serves `frontend/index.html` and connects it to `models/best_era_model.joblib`:
- `GET /` — Frontend web UI
- `GET /api/health` — Model & catalog health check
- `GET /api/search?q=<title>` — Top song matches from `data/spotify_full.csv`
- `POST /api/predict` — Predict era from `{"query": "Blinding Lights"}` or `{"features": {...}}`
- `POST /api/predict-audio` — Extract audio features via `librosa` from an uploaded `.mp3`/`.wav` file and predict era

```bash
python src/app.py
```
