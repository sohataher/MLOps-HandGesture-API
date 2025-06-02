# app/metrics.py
from prometheus_client import Counter, Histogram, Summary

# Model metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Total failed prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time for a prediction")

# Data metrics
from prometheus_client import Gauge

FEATURE_MEAN = Gauge("input_feature_0_mean", "Mean of feature 0")
FEATURE_STD = Gauge("input_feature_0_std", "Std Dev of feature 0")
