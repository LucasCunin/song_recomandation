from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd

class CosineSimilarityRecommender:
    def __init__(self, n_components: int = 10):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.features_df = None
        self.feature_columns = None
        
    def fit(self, features_df: pd.DataFrame, feature_columns: List[str]):
        """Entraîne le modèle de recommandation."""
        self.feature_columns = feature_columns
        self.features_df = features_df.copy()
        
        # Normalisation des features
        X = self.scaler.fit_transform(features_df[feature_columns])
        
        # Réduction de dimensionnalité
        X_pca = self.pca.fit_transform(X)
        
        # Stockage des features transformées
        self.features_df['features_pca'] = [x for x in X_pca]
        
    def recommend(self, audio_features: Dict[str, Any], n_recommendations: int = 5) -> pd.DataFrame:
        """Recommande des morceaux similaires."""
        # Préparation des features d'entrée
        input_features = np.array([audio_features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Normalisation et transformation
        input_features_scaled = self.scaler.transform(input_features)
        input_features_pca = self.pca.transform(input_features_scaled)
        
        # Calcul des similarités
        similarities = cosine_similarity(
            input_features_pca,
            np.vstack(self.features_df['features_pca'])
        )[0]
        
        # Récupération des indices des morceaux les plus similaires
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        # Création du DataFrame de résultats
        recommendations = self.features_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['label', 'similarity_score']]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features dans la PCA."""
        importance = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_columns
        )
        return importance 