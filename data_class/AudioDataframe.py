from pydantic import BaseModel, PrivateAttr
import pandas as pd

class AudioDataRaw(BaseModel):
    # General
    label: str
    zcr: float
    rms: float
    centroid: float
    bandwidth: float
    rolloff: float
    tempo: float
    spectral_contrast: float

    # MFCC
    mfcc_1_mean: float
    mfcc_1_var: float
    mfcc_2_mean: float
    mfcc_2_var: float
    mfcc_3_mean: float
    mfcc_3_var: float
    mfcc_4_mean: float
    mfcc_4_var: float
    mfcc_5_mean: float
    mfcc_5_var: float
    mfcc_6_mean: float
    mfcc_6_var: float
    mfcc_7_mean: float
    mfcc_7_var: float
    mfcc_8_mean: float
    mfcc_8_var: float
    mfcc_9_mean: float
    mfcc_9_var: float
    mfcc_10_mean: float
    mfcc_10_var: float
    mfcc_11_mean: float
    mfcc_11_var: float
    mfcc_12_mean: float
    mfcc_12_var: float
    mfcc_13_mean: float
    mfcc_13_var: float
    
    # F0
    f0_mean: float
    f0_median: float
    f0_std: float

class AudioDataframe(BaseModel, pd.DataFrame):
        
    feature_columns: list[AudioDataRaw]
    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, feature_columns: list[AudioDataRaw]):
        super().__init__()
        self.feature_columns = feature_columns
        self._df = pd.DataFrame(feature_columns)

    def __getitem__(self, key):
        return self._df[key]

    
    
    