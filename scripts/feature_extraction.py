import librosa
import numpy as np
from scripts.preprocess import preprocess_audio

def extract_features(file_path):
    audio, sr = preprocess_audio(file_path)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Combine all features
    feature_vector = np.hstack([
        mfcc_mean,
        chroma_mean,
        zcr_mean,
        spec_centroid_mean
    ])

    return feature_vector
