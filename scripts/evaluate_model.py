import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained model
model = joblib.load("../models/svm_model.pkl")

# Dummy test data (placeholder)
X_test = np.random.rand(20, 27)
y_test = np.random.randint(0, 2, 20)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
