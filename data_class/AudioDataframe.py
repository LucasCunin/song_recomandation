from pydantic import BaseModel
import pandas as pd
from typing import List
import os
from .Audio import AudioFeatureExtractor, AudioDataRaw, Audio


class AudioDataFrame(pd.DataFrame):
    @classmethod
    def from_folder(cls, folder_path: str):
        """Scanne un dossier et sous-dossiers : 1 ligne par fichier audio valide."""
        rows = []
        for genre_folder in os.listdir(folder_path):
            genre_path = os.path.join(folder_path, genre_folder)
            if not os.path.isdir(genre_path):
                continue
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                try:
                    audio = Audio(path=file_path, label=genre_folder)
                    extractor = AudioFeatureExtractor(audio=audio)
                    row = extractor.extract_features()
                    rows.append(row.model_dump())
                    print(f"{file_path} OK")
                except Exception as e:
                    print(f"Erreur avec {file_path}: {e}")
        return cls(rows)
    
    @classmethod
    def from_path(cls, file_path: str) -> 'AudioDataFrame':
        """Crée un AudioDataFrame à partir d'un seul fichier audio."""
        audio = Audio(path=file_path, label="query")
        extractor = AudioFeatureExtractor(audio=audio)
        row = extractor.extract_features()
        # Créer un DataFrame avec un index explicite
        df = pd.DataFrame([row.model_dump()], index=[0])
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'AudioDataFrame':
        """Crée un AudioDataFrame à partir d'un DataFrame pandas existant."""
        instance = cls(df)
        for col in AudioDataRaw.model_fields:
            if col not in instance.columns:
                instance[col] = None
        return instance

    def __init__(self, data):
        """
        Initialise un AudioDataFrame à partir de données.
        
        Args:
            data: Soit une liste d'AudioDataRaw, soit un DataFrame pandas
        """
        if isinstance(data, pd.DataFrame):
            super().__init__(data)
        else:
            super().__init__(data)
            for col in AudioDataRaw.model_fields:
                if col not in self.columns:
                    self[col] = None

    # @classmethod
    # def from_csv(cls, csv_path: str, path_col: str = "path", label_col: str = "label"):
    #     """CSV avec chemin et label (peu importe le nombre de colonnes, on utilise les deux précisés)."""
    #     df = pd.read_csv(csv_path)
    #     rows = []
    #     for _, row in df.iterrows():
    #         try:
    #             extractor = AudioFeatureExtractor(row[path_col], label=row[label_col])
    #             feat = extractor.extract_features()
    #             rows.append(feat.model_dump())
    #             print(f"{row[path_col]} OK")
    #         except Exception as e:
    #             print(f"Erreur avec {row[path_col]}: {e}")
        # return cls(rows)
    

    
    
    
    