# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from starlette.responses import Response

from .schema import LandmarkInput, PredictionResponse
from .model import model, label_encoder
from .utils import process_landmarks, PredictionStabilizer
from .metrics import REQUEST_COUNT, ERROR_COUNT, PREDICTION_LATENCY, FEATURE_MEAN, FEATURE_STD

import numpy as np
import time

app = FastAPI(title="Hand Gesture Recognition API")
stabilizer = PredictionStabilizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: LandmarkInput):
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        features = process_landmarks(request.landmarks)

        # Log data metrics
        FEATURE_MEAN.set(np.mean(features))
        FEATURE_STD.set(np.std(features))

        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0].max()
        gesture = label_encoder.inverse_transform([stabilizer.stabilize(pred)])[0]

        latency = time.time() - start
        PREDICTION_LATENCY.observe(latency)

        return {"gesture": gesture, "confidence": float(proba)}
    
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))