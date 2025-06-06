import librosa
import numpy as np

from pydantic import BaseModel

class AudioDataRaw(BaseModel):
    # General
    path: str
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

class Audio(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    path: str
    label: str
    sr: int = None
    y: np.ndarray = None
    frame_length: int = 2048
    hop_length: int = 512

    def model_post_init(self, __context):
        """Charge le fichier audio après l'initialisation du modèle."""
        if self.y is None:
            self.y, self.sr = librosa.load(self.path)

class AudioFeatureExtractor(BaseModel):
    audio: Audio
    
    def get_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(self.audio.y, frame_length=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(zcr)

    def get_rms(self):
        rms = librosa.feature.rms(y=self.audio.y, frame_length=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(rms)

    def get_centroid(self):
        centroid = librosa.feature.spectral_centroid(y=self.audio.y, sr=self.audio.sr, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(centroid)

    def get_bandwidth(self):
        bandwidth = librosa.feature.spectral_bandwidth(y=self.audio.y, sr=self.audio.sr, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(bandwidth)

    def get_rolloff(self):
        rolloff = librosa.feature.spectral_rolloff(y=self.audio.y, sr=self.audio.sr, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(rolloff)

    def get_mfcc_mean(self, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=self.audio.y, sr=self.audio.sr, n_mfcc=n_mfcc, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(mfcc, axis=1)
    
    def get_mfcc_var(self, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=self.audio.y, sr=self.audio.sr, n_mfcc=n_mfcc, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.var(mfcc, axis=1)
    
    def get_f0_stats(self):
        f0, _, _ = librosa.pyin(self.audio.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size > 0:
            return {
                'f0_mean': float(np.mean(f0_valid)),
                'f0_median': float(np.median(f0_valid)),
                'f0_std': float(np.std(f0_valid)),
            }
        else:
            return {'f0_mean': 0.0, 'f0_median': 0.0, 'f0_std': 0.0}

    def get_tempo(self):
        onset_env = librosa.onset.onset_strength(y=self.audio.y, sr=self.audio.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.audio.sr)
        return float(tempo[0])  # tempo est un array

    def get_spectral_contrast(self):
        contrast = librosa.feature.spectral_contrast(y=self.audio.y, sr=self.audio.sr, n_fft=self.audio.frame_length, hop_length=self.audio.hop_length)
        return np.mean(contrast)

    def extract_features(self) -> AudioDataRaw:
        features = {
            'path': self.audio.path,
            'zcr': self.get_zcr(),
            'rms': self.get_rms(),
            'centroid': self.get_centroid(),
            'bandwidth': self.get_bandwidth(),
            'rolloff': self.get_rolloff(),
            'tempo': self.get_tempo(),
            'spectral_contrast': self.get_spectral_contrast(),
            'label': self.audio.label
        }
        mfcc_means = self.get_mfcc_mean()
        mfcc_vars = self.get_mfcc_var()
        for i, (mean, var) in enumerate(zip(mfcc_means, mfcc_vars)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_var'] = var

        features.update(self.get_f0_stats())

        return AudioDataRaw(**features)