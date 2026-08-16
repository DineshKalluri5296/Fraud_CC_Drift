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
# Base Directory
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# Artifact Files
# ==========================================================

METRICS_FILE = (
    BASE_DIR / "artifacts" / "evaluation.json"
)

DATA_DRIFT_FILE = (
    BASE_DIR / "artifacts" / "data_drift.json"
)

DRIFT_REPORT_FILE = (
    BASE_DIR / "artifacts" / "drift_report.json"
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
# Prediction Latency
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
# Data Drift Metrics
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
# Update Model Metrics
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


# ==========================================================
# Load Model Metrics
# ==========================================================

def load_model_metrics():
    """
    Load model evaluation metrics
    from artifacts/evaluation.json.
    """

    if not METRICS_FILE.exists():

        update_model_metrics({})

        return

    try:

        with open(
            METRICS_FILE,
            "r"
        ) as file:

            metrics = json.load(file)

        update_model_metrics(metrics)

    except (
        json.JSONDecodeError,
        OSError
    ):

        update_model_metrics({})


# ==========================================================
# Load Data Drift Status
# ==========================================================

def load_data_drift_status():
    """
    Load data drift and schema drift
    status from data_drift.json.
    """

    if not DATA_DRIFT_FILE.exists():

        DATA_DRIFT_DETECTED.set(0)

        SCHEMA_DRIFT_DETECTED.set(0)

        return

    try:

        with open(
            DATA_DRIFT_FILE,
            "r"
        ) as file:

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

    except (
        json.JSONDecodeError,
        OSError
    ):

        DATA_DRIFT_DETECTED.set(0)

        SCHEMA_DRIFT_DETECTED.set(0)


# ==========================================================
# Load Data Drift Score
# ==========================================================

def load_drift_score():
    """
    Load overall data drift score
    from artifacts/drift_report.json.
    """

    if not DRIFT_REPORT_FILE.exists():

        DATA_DRIFT_SCORE.set(0)

        return

    try:

        with open(
            DRIFT_REPORT_FILE,
            "r"
        ) as file:

            drift_report = json.load(file)

        drift_score = drift_report.get(
            "overall_drift_score",
            0
        )

        DATA_DRIFT_SCORE.set(
            float(drift_score)
        )

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError
    ):

        DATA_DRIFT_SCORE.set(0)


# ==========================================================
# Refresh Drift Metrics
# ==========================================================

def refresh_drift_metrics():
    """
    Refresh drift-related Prometheus metrics
    after running data drift detection.
    """

    load_data_drift_status()

    load_drift_score()


# ==========================================================
# Application Startup
# ==========================================================

load_model_metrics()

load_data_drift_status()

load_drift_score()
