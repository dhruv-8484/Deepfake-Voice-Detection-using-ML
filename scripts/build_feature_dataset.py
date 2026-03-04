import os
import numpy as np
from feature_extraction import extract_features

dataset_path = r"C:\Users\dhruv\OneDrive\Desktop\FVD\dataset"

X = []
y = []

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature extraction complete!")
print("Total samples:", len(X))
print("Feature shape:", X.shape)

np.save("../features/X.npy", X)
np.save("../features/y.npy", y)

print("Saved features successfully!")
