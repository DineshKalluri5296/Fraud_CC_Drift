"""
Data Drift Detection for Fraud Detection Model.

Uses:
    - Schema comparison
    - Kolmogorov-Smirnov (KS) test

Outputs:
    - artifacts/data_drift.json
    - artifacts/drift_report.json
"""

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from scipy.stats import ks_2samp


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_DATA = (
    BASE_DIR / "data" / "reference.csv"
)

CURRENT_DATA = (
    BASE_DIR / "data" / "current.csv"
)

ARTIFACTS_DIR = (
    BASE_DIR / "artifacts"
)

DATA_DRIFT_OUTPUT = (
    ARTIFACTS_DIR / "data_drift.json"
)

DRIFT_REPORT_OUTPUT = (
    ARTIFACTS_DIR / "drift_report.json"
)


# ==========================================================
# Drift Configuration
# ==========================================================

P_VALUE_THRESHOLD = 0.05

KS_THRESHOLD = 0.01


# Create artifacts directory
ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# JSON Helper
# ==========================================================

def save_json(path: Path, data: dict):
    """
    Save dictionary as formatted JSON.
    """

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ==========================================================
# Data Drift Detection
# ==========================================================

def detect_data_drift():
    """
    Detect schema and statistical data drift.

    Returns:
        dict: Data drift report.
    """

    # ======================================================
    # Load Datasets
    # ======================================================

    try:

        ref = pd.read_csv(
            REFERENCE_DATA
        )

        cur = pd.read_csv(
            CURRENT_DATA
        )

    except FileNotFoundError as error:

        report = {
            "data_drift_detected": True,
            "drift_type": "missing_dataset",
            "missing_columns": [],
            "extra_columns": [],
            "overall_drift_score": 1.0,
            "threshold": KS_THRESHOLD,
            "method": "Dataset validation",
            "recommendation": (
                "Reference or current dataset "
                "is missing."
            )
        }

        save_json(
            DATA_DRIFT_OUTPUT,
            report
        )

        save_json(
            DRIFT_REPORT_OUTPUT,
            report
        )

        print(
            f"Dataset error: {error}"
        )

        print(
            json.dumps(
                report,
                indent=4
            )
        )

        return report

    except EmptyDataError:

        report = {
            "data_drift_detected": True,
            "drift_type": "empty_dataset",
            "missing_columns": [],
            "extra_columns": [],
            "overall_drift_score": 1.0,
            "threshold": KS_THRESHOLD,
            "method": "Dataset validation",
            "recommendation": (
                "Reference or current dataset "
                "is empty."
            )
        }

        save_json(
            DATA_DRIFT_OUTPUT,
            report
        )

        save_json(
            DRIFT_REPORT_OUTPUT,
            report
        )

        print(
            json.dumps(
                report,
                indent=4
            )
        )

        return report

    # ======================================================
    # Empty DataFrame Check
    # ======================================================

    if ref.empty or cur.empty:

        report = {
            "data_drift_detected": True,
            "drift_type": "empty_dataset",
            "missing_columns": [],
            "extra_columns": [],
            "overall_drift_score": 1.0,
            "threshold": KS_THRESHOLD,
            "method": "Dataset validation",
            "recommendation": (
                "Reference or current dataset "
                "contains no records."
            )
        }

        save_json(
            DATA_DRIFT_OUTPUT,
            report
        )

        save_json(
            DRIFT_REPORT_OUTPUT,
            report
        )

        print(
            json.dumps(
                report,
                indent=4
            )
        )

        return report

    # ======================================================
    # Schema Drift Detection
    # ======================================================

    ref_columns = set(
        ref.columns
    )

    cur_columns = set(
        cur.columns
    )

    missing_columns = sorted(
        ref_columns - cur_columns
    )

    extra_columns = sorted(
        cur_columns - ref_columns
    )

    # ======================================================
    # Schema Drift
    # ======================================================

    if missing_columns or extra_columns:

        report = {

            "report_date":
                pd.Timestamp.utcnow().isoformat(),

            "drift_detected": True,

            "data_drift_detected": True,

            "drift_type": "schema_drift",

            "missing_columns":
                missing_columns,

            "extra_columns":
                extra_columns,

            "overall_drift_score": 1.0,

            "threshold": KS_THRESHOLD,

            "method":
                "Schema comparison",

            "recommendation": (
                "Schema drift detected. "
                "Review missing or extra columns "
                "before using the dataset for "
                "model inference."
            )
        }

        save_json(
            DATA_DRIFT_OUTPUT,
            report
        )

        save_json(
            DRIFT_REPORT_OUTPUT,
            report
        )

        print(
            json.dumps(
                report,
                indent=4
            )
        )

        return report

    # ======================================================
    # Statistical Drift Detection
    # ======================================================

    feature_report = {}

    drift_detected = False

    drift_scores = []

    # Use common columns only
    common_columns = [
        column
        for column in ref.columns
        if column in cur.columns
    ]

    for column in common_columns:

        # Skip target column
        if column == "fraud":
            continue

        # --------------------------------------------------
        # Numeric Feature Validation
        # --------------------------------------------------

        if not (
            pd.api.types.is_numeric_dtype(
                ref[column]
            )
            and
            pd.api.types.is_numeric_dtype(
                cur[column]
            )
        ):

            feature_report[column] = {
                "ks_statistic": None,
                "p_value": None,
                "drift": False,
                "status": "skipped_non_numeric"
            }

            continue

        # --------------------------------------------------
        # Remove Missing Values
        # --------------------------------------------------

        reference_values = (
            ref[column]
            .dropna()
        )

        current_values = (
            cur[column]
            .dropna()
        )

        # --------------------------------------------------
        # Empty Column Check
        # --------------------------------------------------

        if (
            reference_values.empty
            or current_values.empty
        ):

            feature_report[column] = {
                "ks_statistic": None,
                "p_value": None,
                "drift": True,
                "status": "empty_column"
            }

            drift_detected = True

            drift_scores.append(1.0)

            continue

        # --------------------------------------------------
        # KS Test
        # --------------------------------------------------

        statistic, p_value = ks_2samp(
            reference_values,
            current_values
        )

        statistic = float(
            statistic
        )

        p_value = float(
            p_value
        )

        # --------------------------------------------------
        # Drift Decision
        # --------------------------------------------------

        column_drift = (
            p_value < P_VALUE_THRESHOLD
            and statistic > KS_THRESHOLD
        )

        feature_report[column] = {

            "ks_statistic":
                round(statistic, 6),

            "p_value":
                round(p_value, 6),

            "drift":
                column_drift,

            "status":
                "drift_detected"
                if column_drift
                else "no_drift"
        }

        drift_scores.append(
            statistic
        )

        if column_drift:

            drift_detected = True

    # ======================================================
    # Overall Drift Score
    # ======================================================

    if drift_scores:

        overall_drift_score = (
            sum(drift_scores)
            / len(drift_scores)
        )

    else:

        overall_drift_score = 0.0

    # ======================================================
    # Final Report
    # ======================================================

    final_report = {

        "report_date":
            pd.Timestamp.utcnow().isoformat(),

        "drift_detected":
            drift_detected,

        "data_drift_detected":
            drift_detected,

        "drift_type": (
            "statistical_drift"
            if drift_detected
            else "no_drift"
        ),

        "overall_drift_score":
            round(
                overall_drift_score,
                4
            ),

        "p_value_threshold":
            P_VALUE_THRESHOLD,

        "ks_threshold":
            KS_THRESHOLD,

        "method":
            "Kolmogorov-Smirnov Test",

        "features":
            feature_report,

        "recommendation": (
            "Retraining may be required."
            if drift_detected
            else
            "No retraining required."
        )
    }

    # ======================================================
    # Save Reports
    # ======================================================

    save_json(
        DATA_DRIFT_OUTPUT,
        final_report
    )

    save_json(
        DRIFT_REPORT_OUTPUT,
        final_report
    )

    # ======================================================
    # Console Output
    # ======================================================

    print(
        json.dumps(
            final_report,
            indent=4
        )
    )

    return final_report


# ==========================================================
# Script Entry Point
# ==========================================================

if __name__ == "__main__":

    detect_data_drift()
