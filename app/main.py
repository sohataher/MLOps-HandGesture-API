# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schema import LandmarkInput, PredictionResponse
from .model import model, label_encoder
from app.utils import process_landmarks, PredictionStabilizer
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Model-related: track prediction latency
prediction_latency = Summary('prediction_latency_seconds', 'Time spent processing prediction')

# Data-related: track first feature value
input_feature_0 = Gauge('input_feature_0_value', 'Value of the first input feature')

stabilizer = PredictionStabilizer()

app = FastAPI(title="Hand Gesture Recognition API")

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/metrics")
@prediction_latency.time()
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: LandmarkInput):
    try:
        features = process_landmarks(request.landmarks)  # Returns 63 features
        proba = model.predict_proba([features])[0]
        pred = model.predict([features])[0]

        stabilized_pred = stabilizer.stabilize(pred)
        gesture = label_encoder.inverse_transform([stabilized_pred])[0]
        
        return {
            "gesture": gesture,
            "confidence": float(proba.max())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))