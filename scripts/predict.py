import sys
import numpy as np
import joblib
from feature_extraction import extract_features

# Load trained model and scaler
model = joblib.load("../models/svm_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_audio(file_path):
    # Extract features from audio
    features = extract_features(file_path)

    # Scale features
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled).max()

    if prediction == 0:
        result = "REAL VOICE"
    else:
        result = "FAKE VOICE"

    print("\nPrediction:", result)
    print("Confidence:", round(probability * 100, 2), "%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.wav>")
    else:
        predict_audio(sys.argv[1])
