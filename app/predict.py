import time
import json
import joblib
import numpy as np
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge

# =====================================================
# Load Model
# =====================================================

MODEL_PATH = Path("model/model.pkl")
SCALER_PATH = Path("model/scaler.pkl")
METRICS_PATH = Path("artifacts/evaluation.json")

model = joblib.load(MODEL_PATH)

scaler = None
if SCALER_PATH.exists():
    scaler = joblib.load(SCALER_PATH)

# =====================================================
# Prometheus Metrics
# =====================================================

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total Fraud Predictions"
)

NON_FRAUD_PREDICTIONS = Counter(
    "nonfraud_predictions_total",
    "Total Non-Fraud Predictions"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current Model Accuracy"
)

# =====================================================
# Load Accuracy
# =====================================================

if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
        MODEL_ACCURACY.set(metrics.get("accuracy", 0.0))
else:
    MODEL_ACCURACY.set(0)

# =====================================================
# Prediction Function
# =====================================================

def predict_fraud(features):

    start = time.time()

    data = np.array(features).reshape(1, -1)

    if scaler is not None:
        data = scaler.transform(data)

    prediction = int(model.predict(data)[0])

    probability = float(model.predict_proba(data)[0][1])

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)

    if prediction == 1:
        FRAUD_PREDICTIONS.inc()
    else:
        NON_FRAUD_PREDICTIONS.inc()

    return prediction, probability