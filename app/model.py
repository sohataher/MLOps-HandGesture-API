# app/model.py
import joblib
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .utils import process_landmarks

# Load model and metadata
try:
    model_data = joblib.load("saved_models/best_model_with_metadata.pkl")
    model = model_data["model"]  # Make sure this is a model trained on 63 features

    # Load label encoder
    with open("saved_models/labels.json", "r") as f:
        classes = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

except Exception as e:
    raise RuntimeError(f"Error loading model or labels: {e}")


def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    label = label_encoder.inverse_transform([prediction])[0]
    return label, float(probability)