# schema.py
from pydantic import BaseModel
from typing import List

class LandmarkInput(BaseModel):
    landmarks: List[float]  # length = 63

class PredictionResponse(BaseModel):
    gesture: str
    confidence: float
