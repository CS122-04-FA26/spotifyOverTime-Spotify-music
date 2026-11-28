# spotifyOverTime-Spotify-music
CS122 Group 8 Final Project — Predicting a song's musical time period (`<2000`, `2000-2010`, `2011-2020`, `2021+`) from Spotify audio features.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train models (generates `models/best_era_model.joblib` and `results/`):**
   ```bash
   python src/train_models.py
   ```
3. **Run the Flask web app:**
   ```bash
   python src/app.py
   ```
   Then open `http://127.0.0.1:5000` in your browser to search songs or upload audio files.
