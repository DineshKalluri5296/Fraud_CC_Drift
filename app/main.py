from fastapi import FastAPI
from fastapi.responses import Response
import time

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from app.metrics import (
    REQUEST_COUNT,
    FRAUD_PREDICTIONS,
    NON_FRAUD_PREDICTIONS,
    PREDICTION_LATENCY,
    refresh_drift_metrics,
)

from app.schema import FraudRequest
from app.predict import predict_fraud
from model.detect_data_drift import detect_data_drift


app = FastAPI(
    title="Fraud Detection API",
    version="2.0",
    description="Production Fraud Detection API"
)


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running",
        "version": "2.0"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(request: FraudRequest):

    start_time = time.time()

    # Count total prediction requests
    REQUEST_COUNT.inc()

    values = [
        request.distance_from_home,
        request.distance_from_last_transaction,
        request.ratio_to_median_purchase_price,
        request.repeat_retailer,
        request.used_chip,
        request.used_pin_number,
        request.online_order,
    ]

    # Make prediction
    prediction, probability = predict_fraud(values)

    # Update prediction counters
    if prediction == 1:
        FRAUD_PREDICTIONS.inc()
    else:
        NON_FRAUD_PREDICTIONS.inc()

    # Record prediction latency
    PREDICTION_LATENCY.observe(
        time.time() - start_time
    )

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(
            float(probability),
            4
        )
    }


# ============================================================
# Data Drift Detection
# ============================================================

@app.post("/drift")
def run_drift_detection():

    # Run drift detection
    result = detect_data_drift()

    # Refresh Prometheus drift metrics
    refresh_drift_metrics()

    return {
        "message": "Data drift detection completed",
        "result": result
    }


# ============================================================
# Prometheus Metrics
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
