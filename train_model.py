import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os


# Load data

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------
# Train / validation split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------
# Pipeline: scaler + MLP
# -------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    ))
])

# -------------------------
# Train
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {acc * 100:.2f}%")

# -------------------------
# Save model
# -------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/gesture_model.joblib")

print("Model saved to model/gesture_model.joblib")
