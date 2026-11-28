"""
Flask backend for Spotify Over Time era prediction.

Serves `frontend/index.html` and exposes REST endpoints to:
  - Search songs in `data/spotify_full.csv` (`GET /api/search?q=...`)
  - Predict a song's era by song title or raw audio features (`POST /api/predict`)
  - Extract audio features from an uploaded audio file and predict era (`POST /api/predict-audio`)

Usage (from repository root):
    python src/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.modeling_utils import ERA_ORDER, RAW_INPUT_FEATURES

FRONTEND_DIR = REPO_ROOT / "frontend"
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

CATALOG_COLS = [
    "title",
    "title_normalized",
    "track_artist",
    "release_year",
    "era",
    "track_popularity",
    *RAW_INPUT_FEATURES,
]

ERA_DESCRIPTIONS = {
    "<2000": "pre-2000 recordings, characterized by higher average valence, strong groove/danceability, and lower digital loudness compression",
    "2000-2010": "the 2000–2010 era, marked by rising acoustic catalog diversity alongside louder mainstream mastering",
    "2011-2020": "the 2011–2020 streaming decade, defined by peak production energy/loudness and lower average valence",
    "2021+": "the 2021+ modern streaming era, featuring moodier/lower-valence songwriting, crisp dynamics, and diverse acoustic-electronic hybrids",
}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

_MODEL = None
_MODEL_NAME = None
_CATALOG_DF: pd.DataFrame | None = None
_CATALOG_LOADED = False


@app.after_request
def _add_cors_headers(response):
    """Allow requests even if index.html is opened via file:// or another local port."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def get_model():
    """Lazily load the trained era prediction pipeline."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL, _MODEL_NAME

    preferred = os.environ.get("ERA_MODEL_FILE", "best_era_model.joblib")
    candidates = [
        MODELS_DIR / preferred,
        MODELS_DIR / "best_era_model.joblib",
        MODELS_DIR / "gradient_boosted_era.joblib",
        MODELS_DIR / "random_forest_era.joblib",
        MODELS_DIR / "logistic_regression_era.joblib",
    ]
    for path in candidates:
        if path.exists():
            _MODEL = joblib.load(path)
            _MODEL_NAME = path.name
            print(f"[SpotifyOverTime] Loaded model: {_MODEL_NAME}")
            return _MODEL, _MODEL_NAME

    raise FileNotFoundError(
        "No trained model found in models/. Run `python src/train_models.py` first."
    )


def get_catalog() -> pd.DataFrame | None:
    """Lazily load and index song metadata + audio features from data/spotify_full.csv."""
    global _CATALOG_DF, _CATALOG_LOADED
    if _CATALOG_LOADED:
        return _CATALOG_DF

    _CATALOG_LOADED = True
    csv_path = DATA_DIR / "spotify_full.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 10_000:
        print("[SpotifyOverTime] data/spotify_full.csv not available (or LFS pointer); song title lookup disabled.")
        return None

    print("[SpotifyOverTime] Loading song catalog from data/spotify_full.csv...")
    dtype_map = {
        "energy": "float32",
        "valence": "float32",
        "danceability": "float32",
        "acousticness": "float32",
        "instrumentalness": "float32",
        "speechiness": "float32",
        "liveness": "float32",
        "loudness": "float32",
        "tempo": "float32",
    }
    df = pd.read_csv(csv_path, usecols=CATALOG_COLS, dtype=dtype_map, low_memory=False)
    df = df.dropna(subset=["title", *RAW_INPUT_FEATURES]).copy()
    df["key"] = df["key"].astype("int16")
    df["mode"] = df["mode"].astype("int8")
    df["track_popularity"] = pd.to_numeric(df["track_popularity"], errors="coerce").fillna(0).astype("float32")
    df["title_str"] = df["title"].astype(str)
    df["artist_str"] = df["track_artist"].fillna("Unknown Artist").astype(str)
    df["title_lower"] = df["title_normalized"].fillna(df["title_str"]).astype(str).str.strip().str.lower()
    df["artist_lower"] = df["artist_str"].str.strip().str.lower()
    df = (
        df.sort_values("track_popularity", ascending=False)
        .drop_duplicates(subset=["title_lower", "artist_lower"], keep="first")
        .reset_index(drop=True)
    )
    _CATALOG_DF = df
    print(f"[SpotifyOverTime] Indexed {len(_CATALOG_DF):,} tracks for search.")
    return _CATALOG_DF


def find_song_in_catalog(query: str) -> pd.Series | None:
    """Find the most popular matching track for `title` or `title - artist`."""
    catalog = get_catalog()
    if catalog is None or not query.strip():
        return None

    raw_q = query.strip().lower()
    # Support "Title - Artist" or "Title by Artist" queries from autocomplete
    title_q, artist_q = raw_q, None
    if " — " in raw_q:
        title_q, artist_q = [p.strip() for p in raw_q.split(" — ", 1)]
    elif " - " in raw_q:
        title_q, artist_q = [p.strip() for p in raw_q.split(" - ", 1)]

    if artist_q:
        exact_both = catalog[(catalog["title_lower"] == title_q) & (catalog["artist_lower"] == artist_q)]
        if not exact_both.empty:
            return exact_both.iloc[0]

    exact_title = catalog[catalog["title_lower"] == title_q]
    if not exact_title.empty:
        return exact_title.iloc[0]

    starts = catalog[catalog["title_lower"].str.startswith(raw_q, na=False)]
    if not starts.empty:
        return starts.iloc[0]

    contains = catalog[catalog["title_lower"].str.contains(raw_q, regex=False, na=False)]
    if not contains.empty:
        return contains.iloc[0]

    return None


def build_explanation(features: dict[str, float], predicted_era: str) -> str:
    """Generate a concise natural-language summary of the song's audio profile."""
    energy = features["energy"]
    valence = features["valence"]
    acoustic = features["acousticness"]
    dance = features["danceability"]
    loudness = features["loudness"]
    tempo = features["tempo"]

    energy_desc = "high energy" if energy >= 0.68 else ("moderate energy" if energy >= 0.40 else "gentle energy")
    valence_desc = "upbeat valence" if valence >= 0.58 else ("balanced mood" if valence >= 0.36 else "moody, low-valence tone")
    acoustic_desc = "rich acoustic texture" if acoustic >= 0.50 else ("hybrid instrumentation" if acoustic >= 0.15 else "electronic/studio-driven production")
    dance_desc = "strong danceability" if dance >= 0.62 else "steady rhythmic feel"

    return (
        f"This track's {energy_desc} ({energy:.2f}), {valence_desc} ({valence:.2f}), "
        f"{acoustic_desc} ({acoustic:.2f}), and {dance_desc} ({dance:.2f} at {tempo:.0f} BPM, "
        f"{loudness:.1f} dB) most closely align with {ERA_DESCRIPTIONS[predicted_era]}."
    )


def run_era_inference(features: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Run model inference on an 11-feature dictionary and return (predicted_era, probabilities)."""
    model, _ = get_model()
    row_df = pd.DataFrame([{col: float(features[col]) for col in RAW_INPUT_FEATURES}])
    row_df["key"] = row_df["key"].round().clip(0, 11).astype(int)
    row_df["mode"] = row_df["mode"].round().clip(0, 1).astype(int)

    proba_raw = model.predict_proba(row_df)[0]
    proba_series = pd.Series(proba_raw, index=model.classes_)[ERA_ORDER]
    proba_dict = {era: round(float(proba_series[era]), 4) for era in ERA_ORDER}

    # Select argmax over probabilities so UI top prediction always matches the tallest confidence bar
    predicted_era = max(proba_dict, key=proba_dict.get)
    return predicted_era, proba_dict


def extract_audio_features_from_file(audio_path: str) -> dict[str, float]:
    """Extract approximate Spotify-style audio features from an uploaded audio file using librosa."""
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60.0)
    if y.size == 0:
        raise ValueError("Uploaded audio file appears to be empty.")

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    loudness_db = float(np.clip(20.0 * np.log10(max(rms_mean, 1e-5)), -50.0, 0.0))
    energy = float(np.clip((loudness_db + 35.0) / 32.0, 0.02, 0.99))

    tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_est)[0])
    if tempo <= 0:
        tempo = 120.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = float(np.std(onset_env) / (np.mean(onset_env) + 1e-5))
    tempo_Factor = float(np.exp(-((tempo - 118.0) ** 2) / (2.0 * 38.0 ** 2)))
    danceability = float(np.clip(0.35 + 0.35 * tempo_Factor + 0.15 * min(pulse, 2.0) / 2.0, 0.08, 0.96))

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_ratio = float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y)) + 1e-6))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    acousticness = float(np.clip(harm_ratio * (1.0 - min(spec_cent / 4500.0, 0.95)), 0.01, 0.98))

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    speechiness = float(np.clip(zcr * 1.4, 0.025, 0.65))
    instrumentalness = float(np.clip(1.0 - min(speechiness * 5.5, 0.95) - flatness * 2.0, 0.0, 0.95))
    liveness = float(np.clip(flatness * 4.5 + 0.08, 0.05, 0.85))

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = int(np.argmax(chroma_mean))

    # Compare major vs minor triad energy around detected tonic
    major_score = chroma_mean[key] + chroma_mean[(key + 4) % 12] + chroma_mean[(key + 7) % 12]
    minor_score = chroma_mean[key] + chroma_mean[(key + 3) % 12] + chroma_mean[(key + 7) % 12]
    mode = 1 if major_score >= minor_score else 0

    valence = float(
        np.clip(0.28 + 0.25 * danceability + 0.18 * energy + (0.14 if mode == 1 else -0.05), 0.05, 0.96)
    )

    return {
        "energy": round(energy, 4),
        "valence": round(valence, 4),
        "danceability": round(danceability, 4),
        "acousticness": round(acousticness, 4),
        "instrumentalness": round(instrumentalness, 4),
        "speechiness": round(speechiness, 4),
        "liveness": round(liveness, 4),
        "loudness": round(loudness_db, 3),
        "tempo": round(tempo, 2),
        "key": key,
        "mode": mode,
    }


@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    _, model_name = get_model()
    catalog = get_catalog()
    return jsonify(
        {
            "status": "ok",
            "model": model_name,
            "catalog_songs": int(len(catalog)) if catalog is not None else 0,
            "eras": ERA_ORDER,
        }
    )


@app.route("/api/search", methods=["GET"])
def search_songs():
    q = (request.args.get("q") or "").strip().lower()
    if len(q) < 2:
        return jsonify({"results": []})

    catalog = get_catalog()
    if catalog is None:
        return jsonify({"results": []})

    matches = catalog[catalog["title_lower"].str.startswith(q, na=False)].head(8)
    if len(matches) < 8:
        extra = catalog[
            catalog["title_lower"].str.contains(q, regex=False, na=False)
            & ~catalog.index.isin(matches.index)
        ].head(8 - len(matches))
        matches = pd.concat([matches, extra], ignore_index=True)

    results = [
        {
            "title": row["title_str"],
            "artist": row["artist_str"],
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "era": row["era"],
            "display": f"{row['title_str']} — {row['artist_str']}",
        }
        for _, row in matches.iterrows()
    ]
    return jsonify({"results": results})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    _, model_name = get_model()

    # Mode 1: Direct raw audio features dictionary
    if "features" in payload and isinstance(payload["features"], dict):
        feats = payload["features"]
        missing = [c for c in RAW_INPUT_FEATURES if c not in feats]
        if missing:
            return jsonify({"error": f"Missing audio features: {missing}"}), 400

        clean_feats = {c: float(feats[c]) for c in RAW_INPUT_FEATURES}
        predicted_era, probabilities = run_era_inference(clean_feats)
        return jsonify(
            {
                "source": "features",
                "model": model_name,
                "matched_song": payload.get("title", "Custom Audio Features"),
                "predicted_era": predicted_era,
                "probabilities": probabilities,
                "features": clean_feats,
                "explanation": build_explanation(clean_feats, predicted_era),
            }
        )

    # Mode 2: Song name lookup in Spotify catalog
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Please enter a song name or upload an audio file."}), 400

    song = find_song_in_catalog(query)
    if song is None:
        return (
            jsonify(
                {
                    "error": f'Could not find "{query}" in the local Spotify catalog. Try another song title or upload an audio file.'
                }
            ),
            404,
        )

    clean_feats = {c: float(song[c]) for c in RAW_INPUT_FEATURES}
    predicted_era, probabilities = run_era_inference(clean_feats)
    year = int(song["release_year"]) if pd.notna(song["release_year"]) else None

    return jsonify(
        {
            "source": "catalog",
            "model": model_name,
            "matched_song": song["title_str"],
            "artist": song["artist_str"],
            "release_year": year,
            "actual_era": song["era"],
            "predicted_era": predicted_era,
            "probabilities": probabilities,
            "features": clean_feats,
            "explanation": build_explanation(clean_feats, predicted_era),
        }
    )


@app.route("/api/predict-audio", methods=["POST", "OPTIONS"])
def predict_audio():
    if request.method == "OPTIONS":
        return ("", 204)

    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No audio file uploaded."}), 400

    uploaded = request.files["file"]
    suffix = Path(uploaded.filename).suffix or ".mp3"
    _, model_name = get_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        uploaded.save(tmp_path)

    try:
        feats = extract_audio_features_from_file(tmp_path)
        predicted_era, probabilities = run_era_inference(feats)
        return jsonify(
            {
                "source": "audio_upload",
                "model": model_name,
                "matched_song": uploaded.filename,
                "predicted_era": predicted_era,
                "probabilities": probabilities,
                "features": feats,
                "explanation": build_explanation(feats, predicted_era),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to analyze audio file: {exc}"}), 400
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    get_model()
    get_catalog()
    print(f"\nSpotify Over Time running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
