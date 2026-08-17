"""
Data Drift Detection for Fraud Detection Model
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from pandas.errors import EmptyDataError


# ==========================================================
# Base Directory
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# Dataset Paths
# ==========================================================

REFERENCE_DATA = BASE_DIR / "data" / "reference.csv"
CURRENT_DATA = BASE_DIR / "data" / "current.csv"


# ==========================================================
# Output Paths
# ==========================================================

ARTIFACTS_DIR = BASE_DIR / "artifacts"

DRIFT_REPORT_FILE = ARTIFACTS_DIR / "drift_report.json"
DATA_DRIFT_FILE = ARTIFACTS_DIR / "data_drift.json"


# ==========================================================
# Configuration
# ==========================================================

P_VALUE_THRESHOLD = 0.05
KS_THRESHOLD = 0.1

TARGET_COLUMN = "fraud"

DATASET_VALIDATION = "Dataset validation"


# ==========================================================
# Dataset Loading
# ==========================================================

def load_datasets():
    """
    Load reference and current datasets.

    Returns:
        tuple: Reference dataframe, current dataframe, error type.
    """

    try:
        reference_data = pd.read_csv(REFERENCE_DATA)
        current_data = pd.read_csv(CURRENT_DATA)

        return reference_data, current_data, None

    except FileNotFoundError:
        return None, None, "missing_dataset"

    except EmptyDataError:
        return None, None, "empty_dataset"


# ==========================================================
# Dataset Validation
# ==========================================================

def build_dataset_validation_report(validation_type):
    """
    Build report for invalid datasets.
    """

    return {
        "report_date": pd.Timestamp.now("UTC").isoformat(),
        "drift_detected": True,
        "data_drift_detected": True,
        "drift_type": validation_type,
        "missing_columns": [],
        "extra_columns": [],
        "overall_drift_score": 0.0,
        "features": {},
        "recommendation": "Retraining may be required.",
    }


def validate_datasets(reference_data, current_data):
    """
    Validate reference and current datasets.

    Returns:
        dict or None: Validation report when invalid.
    """

    if reference_data is None or current_data is None:
        return None

    if reference_data.empty or current_data.empty:
        return build_dataset_validation_report("empty_dataset")

    return None


# ==========================================================
# Schema Validation
# ==========================================================

def get_schema_differences(reference_data, current_data):
    """
    Identify missing and extra columns.
    """

    reference_columns = set(reference_data.columns)
    current_columns = set(current_data.columns)

    missing_columns = sorted(reference_columns - current_columns)
    extra_columns = sorted(current_columns - reference_columns)

    return missing_columns, extra_columns


def build_schema_drift_report(missing_columns, extra_columns):
    """
    Build schema drift report.
    """

    return {
        "report_date": pd.Timestamp.now("UTC").isoformat(),
        "drift_detected": True,
        "data_drift_detected": True,
        "drift_type": "schema_drift",
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "overall_drift_score": 0.0,
        "features": {},
        "recommendation": "Retraining may be required.",
    }


# ==========================================================
# Feature Validation
# ==========================================================

def is_numeric_feature(reference_values, current_values):
    """
    Check whether both feature columns are numeric.
    """

    return (
        pd.api.types.is_numeric_dtype(reference_values)
        and pd.api.types.is_numeric_dtype(current_values)
    )


def clean_feature_values(reference_values, current_values):
    """
    Remove missing values from feature data.
    """

    reference_values = reference_values.dropna()
    current_values = current_values.dropna()

    return reference_values, current_values


# ==========================================================
# Statistical Drift Analysis
# ==========================================================

def analyze_feature(reference_values, current_values):
    """
    Perform KS test for one feature.

    Returns:
        dict or None: Feature drift result.
    """

    if not is_numeric_feature(reference_values, current_values):
        return None

    reference_values, current_values = clean_feature_values(
        reference_values,
        current_values,
    )

    if reference_values.empty or current_values.empty:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift": False,
            "status": "insufficient_data",
        }

    statistic, p_value = ks_2samp(
        reference_values,
        current_values,
    )

    # Explicitly convert NumPy/SciPy boolean to native Python bool.
    column_drift = bool(
        p_value < P_VALUE_THRESHOLD
        and statistic > KS_THRESHOLD
    )

    return {
        "ks_statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "drift": column_drift,
        "status": (
            "drift_detected"
            if column_drift
            else "no_drift"
        ),
    }


# ==========================================================
# Feature Analysis
# ==========================================================

def get_common_columns(reference_data, current_data):
    """
    Return columns available in both datasets.
    """

    return [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]


def analyze_features(reference_data, current_data):
    """
    Analyze all common numeric features.
    """

    feature_report = {}
    drift_scores = []
    drift_detected = False

    common_columns = get_common_columns(
        reference_data,
        current_data,
    )

    for column in common_columns:

        if column == TARGET_COLUMN:
            continue

        result = analyze_feature(
            reference_data[column],
            current_data[column],
        )

        if result is None:
            continue

        feature_report[column] = result

        statistic = result.get("ks_statistic")

        if statistic is not None:
            drift_scores.append(float(statistic))

        if result.get("drift", False):
            drift_detected = True

    return feature_report, drift_scores, bool(drift_detected)


# ==========================================================
# Overall Drift Score
# ==========================================================

def calculate_overall_drift_score(drift_scores):
    """
    Calculate average KS statistic.
    """

    if not drift_scores:
        return 0.0

    return float(sum(drift_scores) / len(drift_scores))


# ==========================================================
# Final Report
# ==========================================================

def build_final_report(
    drift_detected,
    overall_drift_score,
    feature_report,
):
    """
    Build final statistical drift report.
    """

    drift_detected = bool(drift_detected)

    return {
        "report_date": pd.Timestamp.now("UTC").isoformat(),

        "drift_detected": drift_detected,

        "data_drift_detected": drift_detected,

        "drift_type": (
            "statistical_drift"
            if drift_detected
            else "no_drift"
        ),

        "overall_drift_score": round(
            float(overall_drift_score),
            6,
        ),

        "features": feature_report,

        "recommendation": (
            "Retraining may be required."
            if drift_detected
            else "No retraining required."
        ),
    }


# ==========================================================
# JSON Serialization
# ==========================================================

def json_serializer(obj):
    """
    Convert Pandas/NumPy scalar values to JSON-compatible types.
    """

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if hasattr(obj, "item"):
        return obj.item()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


# ==========================================================
# Save Report
# ==========================================================

def save_drift_report(report):
    """
    Save drift report to artifacts directory.
    """

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DRIFT_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=json_serializer,
        )


def save_data_drift_status(report):
    """
    Save simplified data drift status.
    """

    status = {
        "data_drift_detected": bool(
            report.get(
                "data_drift_detected",
                False,
            )
        ),

        "drift_type": report.get(
            "drift_type",
            "no_drift",
        ),
    }

    with open(
        DATA_DRIFT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            status,
            file,
            indent=4,
            default=json_serializer,
        )


# ==========================================================
# Main Data Drift Detection
# ==========================================================

def detect_data_drift():
    """
    Detect schema and statistical data drift.

    Returns:
        dict: Data drift report.
    """

    reference_data, current_data, load_error = load_datasets()

    if load_error:
        return build_dataset_validation_report(load_error)

    validation_report = validate_datasets(
        reference_data,
        current_data,
    )

    if validation_report:
        return validation_report

    missing_columns, extra_columns = get_schema_differences(
        reference_data,
        current_data,
    )

    if missing_columns or extra_columns:
        return build_schema_drift_report(
            missing_columns,
            extra_columns,
        )

    feature_report, drift_scores, drift_detected = analyze_features(
        reference_data,
        current_data,
    )

    overall_drift_score = calculate_overall_drift_score(
        drift_scores,
    )

    return build_final_report(
        drift_detected,
        overall_drift_score,
        feature_report,
    )


# ==========================================================
# Run Detection
# ==========================================================

if __name__ == "__main__":

    report = detect_data_drift()

    save_drift_report(report)

    save_data_drift_status(report)

    print(
        json.dumps(
            report,
            indent=4,
            default=json_serializer,
        )
    )
