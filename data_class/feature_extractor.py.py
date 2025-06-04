import pandas as pd
import os
from data_class.Audio import Audio

root_folder = "Data/genres_original"

header_written = False
csv_path = "new_features_audio.csv"

for genre_folder in os.listdir(root_folder):
    genre_path = os.path.join(root_folder, genre_folder)
    for audio_file in os.listdir(genre_path):
        audio_path = os.path.join(genre_path, audio_file)
        try:
            audio = Audio(path=audio_path, label=genre_folder)
            feats = audio.extract_features()
            df_row = pd.DataFrame([feats])
            # Enregistrement dans le CSV
            df_row.to_csv(csv_path, mode='a', index=False, header=not header_written)
            header_written = True
            print(f"{audio_path} OK")
        except Exception as e:
            print(f"Erreur avec {audio_path}: {e}")