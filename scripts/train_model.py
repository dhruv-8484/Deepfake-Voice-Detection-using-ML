import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load features
X = np.load("../features/X.npy")
y = np.load("../features/y.npy")

print("Dataset shape:", X.shape)
print("Label distribution:", np.bincount(y))

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (with random state + stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=1,   # You can change this to test stability
    stratify=y         # Keeps real/fake ratio balanced
)

# Create SVM model
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model and scaler
joblib.dump(model, "../models/svm_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model and scaler saved successfully!")
