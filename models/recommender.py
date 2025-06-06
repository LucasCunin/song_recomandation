import numpy as np
from typing import List
from data_class import AudioDataFrame

class AudioSimilarityRecommender:
    def __init__(self, audio_df, feature_cols: List[str], path_col: str = "path"):
        """
        audio_df : AudioDataframe (héritant de pd.DataFrame)
        feature_cols : liste des colonnes numériques/features à utiliser pour la similarité
        path_col : colonne contenant le chemin du fichier audio
        """
        self.audio_df = audio_df
        self.feature_cols = feature_cols
        self.path_col = path_col
        # On extrait déjà les features en numpy array pour l'efficacité
        self.X = np.array(audio_df[feature_cols])
        self.paths = audio_df[path_col].values

    def recommend(self, query_df: AudioDataFrame, k: int = 5, metric: str = "euclidean") -> List[str]:
        """
        query_vector : 1D numpy array des features du fichier à comparer (après preprocessing !)
        k : nombre de voisins à retourner
        metric : "euclidean" ou "cosine"
        Retourne : liste des chemins des k fichiers les plus similaires
        """
        # Calcul des distances
        if metric == "euclidean":
            dists = np.linalg.norm(self.X - query_df[self.feature_cols].values, axis=1)
        elif metric == "cosine":
            # 1 - cos sim pour retrouver la "distance"
            X_norm = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
            q_norm = query_df[self.feature_cols].values / np.linalg.norm(query_df[self.feature_cols].values)
            dists = 1 - np.dot(X_norm, q_norm)
        else:
            raise ValueError("metric doit être 'euclidean' ou 'cosine'")

        idx = np.argsort(dists)[:k]
        return list(self.paths[idx])  # Ou si tu veux tout (label, path, distance) -> cf. plus bas

    def recommend_full(self, query_vector: np.ndarray, k: int = 5, metric: str = "euclidean"):
        """
        Version qui retourne tout le DataFrame des k plus proches, AVEC la distance, trié.
        """
        if metric == "euclidean":
            dists = np.linalg.norm(self.X - query_vector, axis=1)
        elif metric == "cosine":
            X_norm = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
            q_norm = query_vector / np.linalg.norm(query_vector)
            dists = 1 - np.dot(X_norm, q_norm)
        else:
            raise ValueError("metric doit être 'euclidean' ou 'cosine'")

        idx = np.argsort(dists)[:k]
        df_result = self.audio_df.iloc[idx].copy()
        df_result["distance"] = dists[idx]
        return df_result.sort_values("distance")
