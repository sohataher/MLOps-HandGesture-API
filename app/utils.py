# app/utils.py
import numpy as np

def process_landmarks(landmarks):
    """
    Takes 63 values (21x3), returns all 63 normalized values.
    """
    landmarks = np.array(landmarks).reshape(21, 3)

    # Normalize x,y relative to wrist
    wrist = landmarks[0].copy()
    landmarks[:, :2] -= wrist[:2]

    # Scale by middle finger length
    scale = np.linalg.norm(landmarks[9, :2])
    if scale > 0:
        landmarks[:, :2] /= scale

    # Return all 63 normalized values (flattened)
    return landmarks.flatten().tolist()

# # Prediction stabilizer class
class PredictionStabilizer:
    def __init__(self, window_size=7):
        self.window = []

    def stabilize(self, current_pred):
        self.window.append(current_pred)
        if len(self.window) > 7:
            self.window.pop(0)
        return max(set(self.window), key=self.window.count)


