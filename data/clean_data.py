import pandas as pd
import os

df = pd.read_csv('data/spotify_full.csv', low_memory=False)
print("Original shape:", df.shape)
print(df.isnull().sum())

df_clean = df.dropna(subset=['energy','valence','danceability','tempo','acousticness','decade'])
df_clean = df_clean.drop_duplicates()

keep_cols = ['track_id','track_popularity','energy','valence','danceability',
             'acousticness','instrumentalness','speechiness','liveness','loudness',
             'tempo','key','mode','decade','era','genre_l1','mood_cluster','emotional_arc',
             'mean_energy_by_decade','mean_valence_by_decade','mean_tempo_by_decade',
             'mean_loudness_by_decade','acousticness_decline_idx','danceability_trend']
df_clean = df_clean[keep_cols]

sample = df_clean.sample(n=150000, random_state=42)
sample.to_csv('data/cleaned_csv.csv', index=False)

print("Cleaned sample shape:", sample.shape)
print("File size (MB):", os.path.getsize('data/cleaned_csv.csv') / 1e6)