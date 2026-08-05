"""
Prometheus Metrics for Fraud Detection API
"""

import json
from pathlib import Path

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

# ==========================================================
# Request Metrics
# ==========================================================

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# ==========================================================
# Prediction Metrics
# ==========================================================

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud predictions"
)

NON_FRAUD_PREDICTIONS = Counter(
    "nonfraud_predictions_total",
    "Total non-fraud predictions"
)

# ==========================================================
# Latency
# ==========================================================

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
        5
    )
)

# ==========================================================
# Model Metrics
# ==========================================================

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Current model precision"
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Current model recall"
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "Current model F1 Score"
)

# ==========================================================
# Drift Metrics
# ==========================================================

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current data drift score"
)

MODEL_DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Current model drift score"
)

# ==========================================================
# Load Evaluation Metrics
# ==========================================================

METRICS_FILE = Path("artifacts/evaluation.json")

if METRICS_FILE.exists():

    with open(METRICS_FILE, "r") as f:

        metrics = json.load(f)

    MODEL_ACCURACY.set(metrics.get("accuracy", 0))

    MODEL_PRECISION.set(metrics.get("precision", 0))

    MODEL_RECALL.set(metrics.get("recall", 0))

    MODEL_F1_SCORE.set(metrics.get("f1_score", 0))

else:

    MODEL_ACCURACY.set(0)

    MODEL_PRECISION.set(0)

    MODEL_RECALL.set(0)

    MODEL_F1_SCORE.set(0)


# ==========================================================
# Helper Functions
# ==========================================================

def update_model_metrics(metrics: dict):
    """
    Update model evaluation metrics.
    """

    MODEL_ACCURACY.set(metrics.get("accuracy", 0))
    MODEL_PRECISION.set(metrics.get("precision", 0))
    MODEL_RECALL.set(metrics.get("recall", 0))
    MODEL_F1_SCORE.set(metrics.get("f1_score", 0))


def update_drift_metrics(data_drift: float, model_drift: float):
    """
    Update drift metrics.
    """

    DATA_DRIFT_SCORE.set(data_drift)
    MODEL_DRIFT_SCORE.set(model_drift)