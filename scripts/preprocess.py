import librosa
import numpy as np

def preprocess_audio(file_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # Remove silence
    audio, _ = librosa.effects.trim(audio)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    return audio, sr