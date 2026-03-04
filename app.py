from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import librosa
import soundfile as sf
from scripts.feature_extraction import extract_features

app = Flask(__name__)
CORS(app)

model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    original_filename = file.filename
    temp_input_path = "temp_input"
    file.save(temp_input_path)

    try:
        # Convert ANY audio format to WAV
        audio, sr = librosa.load(temp_input_path, sr=22050)
        temp_wav_path = "converted.wav"
        sf.write(temp_wav_path, audio, sr)

        # Extract features
        features = extract_features(temp_wav_path)
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled).max()

        result = "REAL VOICE" if prediction == 0 else "FAKE VOICE"

        os.remove(temp_input_path)
        os.remove(temp_wav_path)

        return jsonify({
            "prediction": result,
            "confidence": round(probability * 100, 2),
            "features": features.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)