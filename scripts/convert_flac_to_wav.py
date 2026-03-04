import os
import librosa
import soundfile as sf

dataset_path = r"C:\Users\dhruv\OneDrive\Desktop\FVD\dataset"

for label in ["real", "fake"]:
    folder_path = os.path.join(dataset_path, label)

    for file in os.listdir(folder_path):
        if file.endswith(".flac"):
            file_path = os.path.join(folder_path, file)

            # Load audio
            audio, sr = librosa.load(file_path, sr=22050)

            # New filename
            new_filename = file.replace(".flac", ".wav")
            new_path = os.path.join(folder_path, new_filename)

            # Save as wav
            sf.write(new_path, audio, sr)

            # Optional: remove original flac
            os.remove(file_path)

print("Conversion complete!")
