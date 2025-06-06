from data_class import AudioDataFrame
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class Preprocessor:
    def __init__(self, n_components: int = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.numeric_cols = None

    def get_numeric_cols(self, audio_df: AudioDataFrame):
        num_cols = audio_df.select_dtypes(include=[np.number]).columns.tolist()
        return num_cols

    def scale_data(self, audio_df: AudioDataFrame) -> AudioDataFrame:
        """Retourne un NOUVEAU AudioDataframe avec les features numériques rescalées."""
        df = audio_df.copy()
        self.numeric_cols = self.get_numeric_cols(df)
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        return AudioDataFrame.from_dataframe(df)

    def pca_data(self, audio_df: AudioDataFrame) -> AudioDataFrame:
        """Retourne un NOUVEAU AudioDataframe avec les composantes principales et conserve les métadonnées."""
        df = audio_df.copy()
        self.numeric_cols = self.get_numeric_cols(df)
        features = df[self.numeric_cols]
        features_pca = self.pca.fit_transform(features)
        pca_cols = [f'PC{i+1}' for i in range(self.pca.n_components_)]
        # Garder les colonnes non numériques (ex: label)
        meta_cols = [col for col in df.columns if col not in self.numeric_cols]
        df_pca = pd.DataFrame(features_pca, columns=pca_cols, index=df.index)
        # Concaténer label et PCA
        df_result = pd.concat([df[meta_cols], df_pca], axis=1)
        return AudioDataFrame.from_dataframe(df_result)

    def get_pca_components(self) -> np.ndarray:
        return self.pca.components_

    def get_explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_
