# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    dummy_landmarks = [0.1] * 63  # Simulated 63 landmark values
    response = client.post("/predict", json={"landmarks": dummy_landmarks})
    assert response.status_code == 200
    data = response.json()
    assert "gesture" in data
    assert "confidence" in data