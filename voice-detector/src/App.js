import React, { useState } from "react";
import axios from "axios";
import { FaMicrophone } from "react-icons/fa";
import "./App.css";
function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [audioURL, setAudioURL] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setAudioURL(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload an audio file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );
      setResult(response.data);
      setLoading(false);
    } catch (error) {
      setLoading(false);
      alert("Error predicting file");
    }
  };

  return (
    <div className="app">
      <div className="card">
        <h1 className="title">
          <FaMicrophone /> Deepfake Voice Detection
        </h1>

        <p className="subtitle">
          Upload an audio file to analyze whether the voice is genuine or AI-generated.
        </p>

        <label className="file-upload">
          <input
            type="file"
            accept=".wav,.mp3,.flac,.m4a"
            onChange={handleFileChange}
          />
          {file ? file.name : "Choose Audio File"}
        </label>

        {audioURL && (
          <audio controls className="audio-player">
            <source src={audioURL} />
          </audio>
        )}

        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Analyzing Voice..." : "Analyze Voice"}
        </button>

        {loading && <div className="loader"></div>}

        {result && (
          <div
            className={`result ${
              result.prediction === "REAL VOICE" ? "real" : "fake"
            }`}
          >
            <h2>{result.prediction}</h2>

            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>

            <p>Confidence: {result.confidence}%</p>
          </div>
        )}
        <div className="footer">
          Built using Machine Learning (SVM) + ReactJS + Flask
        </div>
      </div>
    </div>
  );
}

export default App;