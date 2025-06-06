import argparse
import os
import numpy as np
import pandas as pd
from data_class import AudioDataFrame, AudioFeatureExtractor, Audio
from models.recommender import AudioSimilarityRecommender
from tools.preprocessing import Preprocessor

def main():
    parser = argparse.ArgumentParser(description='Système de recommandation audio')
    parser.add_argument('--data_path', type=str, default='Data/genres_original',
                        help='Chemin vers le dossier contenant les fichiers audio')
    parser.add_argument('--query_path', type=str, required=True,
                        help='Chemin vers le fichier audio de requête')
    parser.add_argument('--n_recommendations', type=int, default=5,
                        help='Nombre de recommandations à retourner')
    parser.add_argument('--scale', action='store_true',
                        help='Appliquer le scaling des features')
    parser.add_argument('--force_extract', action='store_true',
                        help='Forcer la réextraction des features même si le CSV existe')
    args = parser.parse_args()

    # 1. Chargement ou extraction du DataFrame complet
    csv_cache = os.path.join(args.data_path, 'audio_features.csv')
    if os.path.exists(csv_cache) and not args.force_extract:
        print(f"Chargement du cache : {csv_cache}")
        audio_df = AudioDataFrame.from_dataframe(pd.read_csv(csv_cache))
    else:
        print(f"Extraction des features depuis {args.data_path}")
        audio_df = AudioDataFrame.from_folder(args.data_path)
        audio_df.to_csv(csv_cache, index=False)
        print(f"Features extraites et sauvegardées dans {csv_cache}")

    # 2. Détection des colonnes de features numériques à utiliser
    exclude_cols = ['label', 'path']
    feature_cols = [col for col in audio_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(audio_df[col])]

    # 3. Scaling si demandé
    preproc = Preprocessor()
    if args.scale:
        print("Application du scaling aux features...")
        audio_df = preproc.scale_data(audio_df)

    # 4. Extraction et preprocessing du fichier query
    print(f"Extraction des features du fichier de requête : {args.query_path}")
    query_df = AudioDataFrame.from_path(args.query_path) # df a une ligne
    if args.scale:
        query_df = preproc.scale_data(query_df)

    # 5. Recommendation
    recommender = AudioSimilarityRecommender(audio_df, feature_cols=feature_cols, path_col="path")
    paths = recommender.recommend(query_df, k=args.n_recommendations)
    print("\nRecommandations (chemins des fichiers) :")
    for p in paths:
        print(p)

if __name__ == "__main__":
    main()
