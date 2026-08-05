from fastapi import FastAPI
from fastapi.responses import Response

import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from app.schema import FraudRequest
from app.predict import predict_fraud


app = FastAPI(
    title="Fraud Detection API",
    version="2.0",
    description="Production Fraud Detection API"
)

############################################################
# Prometheus Metrics
############################################################

TOTAL_REQUESTS = Counter(
    "prediction_requests_total",
    "Total Prediction Requests"
)

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total Fraud Predictions"
)

NON_FRAUD_PREDICTIONS = Counter(
    "nonfraud_predictions_total",
    "Total Non Fraud Predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Latency"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current Model Accuracy"
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current Data Drift Score"
)

MODEL_DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Current Model Drift Score"
)

############################################################
# Initial Values
############################################################

MODEL_ACCURACY.set(0.999)

DATA_DRIFT_SCORE.set(0.02)

MODEL_DRIFT_SCORE.set(0.01)


############################################################
# Home
############################################################

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API Running"
    }


############################################################
# Health Check
############################################################

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


############################################################
# Prediction Endpoint
############################################################

@app.post("/predict")
def predict(request: FraudRequest):

    start = time.time()

    TOTAL_REQUESTS.inc()

    values = [

        request.distance_from_home,
        request.distance_from_last_transaction,
        request.ratio_to_median_purchase_price,
        request.repeat_retailer,
        request.used_chip,
        request.used_pin_number,
        request.online_order

    ]

    prediction, probability = predict_fraud(values)

    if prediction == 1:
        FRAUD_PREDICTIONS.inc()
    else:
        NON_FRAUD_PREDICTIONS.inc()

    PREDICTION_LATENCY.observe(
        time.time() - start
    )

    return {

        "fraud_prediction": int(prediction),

        "fraud_probability": round(
            probability,
            4
        )

    }


############################################################
# Prometheus Metrics
############################################################

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )