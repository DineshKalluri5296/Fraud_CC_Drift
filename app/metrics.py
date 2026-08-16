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
# Artifact Files
# ==========================================================

METRICS_FILE = Path("artifacts/evaluation.json")
DATA_DRIFT_FILE = Path("artifacts/data_drift.json")
DRIFT_REPORT_FILE = Path("artifacts/drift_report.json")


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
    "Current statistical data drift score"
)

DATA_DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "Whether data drift was detected"
)

SCHEMA_DRIFT_DETECTED = Gauge(
    "schema_drift_detected",
    "Whether schema drift was detected"
)

# ==========================================================
# Helper Functions
# ==========================================================

def update_model_metrics(metrics: dict):
    """
    Update model evaluation metrics.
    """

    MODEL_ACCURACY.set(
        metrics.get("accuracy", 0)
    )

    MODEL_PRECISION.set(
        metrics.get("precision", 0)
    )

    MODEL_RECALL.set(
        metrics.get("recall", 0)
    )

    MODEL_F1_SCORE.set(
        metrics.get("f1_score", 0)
    )


def update_drift_metrics(data_drift: float):
    """
    Update statistical data drift score.
    """

    DATA_DRIFT_SCORE.set(data_drift)


def load_model_metrics():
    """
    Load model evaluation metrics from evaluation.json.
    """

    if not METRICS_FILE.exists():

        update_model_metrics({})

        return

    with open(METRICS_FILE, "r") as file:

        metrics = json.load(file)

    update_model_metrics(metrics)


def load_data_drift_status():

    if not DATA_DRIFT_FILE.exists():

        DATA_DRIFT_DETECTED.set(0)
        SCHEMA_DRIFT_DETECTED.set(0)

        return

    with open(DATA_DRIFT_FILE, "r") as file:

        drift = json.load(file)

    detected = drift.get(
        "data_drift_detected",
        False
    )

    drift_type = drift.get(
        "drift_type",
        ""
    )

    DATA_DRIFT_DETECTED.set(
        1 if detected else 0
    )

    SCHEMA_DRIFT_DETECTED.set(
        1 if drift_type == "schema_drift" else 0
    )
    

def load_drift_score():
    """
    Load statistical drift score from drift_report.json.
    """

    if not DRIFT_REPORT_FILE.exists():

        DATA_DRIFT_SCORE.set(0)

        return

    with open(DRIFT_REPORT_FILE, "r") as file:

        drift_report = json.load(file)

    DATA_DRIFT_SCORE.set(
        drift_report.get(
            "overall_drift_score",
            0
        )
    )


# ==========================================================
# Load Metrics at Application Startup
# ==========================================================

load_model_metrics()

load_data_drift_status()

load_drift_score()


def refresh_drift_metrics():

    load_data_drift_status()

    load_drift_score()


