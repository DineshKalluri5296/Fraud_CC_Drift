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
    MODEL_ACCURACY,
    DATA_DRIFT_SCORE,
    MODEL_DRIFT_SCORE,
)
from app.schema import FraudRequest
from app.predict import predict_fraud


app = FastAPI(
    title="Fraud Detection API",
    version="2.0",
    description="Production Fraud Detection API"
)

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
