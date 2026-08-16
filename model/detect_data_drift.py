import json
import os

import pandas as pd
from scipy.stats import ks_2samp
from pandas.errors import EmptyDataError

REFERENCE_DATA = "data/reference.csv"
CURRENT_DATA = "data/current.csv"

DATA_DRIFT_OUTPUT = "artifacts/data_drift.json"
DRIFT_REPORT_OUTPUT = "artifacts/drift_report.json"

P_VALUE_THRESHOLD = 0.05
KS_THRESHOLD = 0.01

os.makedirs("artifacts", exist_ok=True)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def detect_data_drift():

    try:
        ref = pd.read_csv(REFERENCE_DATA)
        cur = pd.read_csv(CURRENT_DATA)

    except EmptyDataError:

        report = {
            "data_drift_detected": True,
            "drift_type": "empty_dataset",
            "missing_columns": [],
            "extra_columns": [],
            "overall_drift_score": 1.0,
            "threshold": 0.05
        }

        save_json(DATA_DRIFT_OUTPUT, report)
        save_json(DRIFT_REPORT_OUTPUT, report)

        print(json.dumps(report, indent=4))

        return report

    # ======================================================
    # Schema Drift Detection
    # ======================================================

    ref_columns = set(ref.columns)
    cur_columns = set(cur.columns)

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
            "data_drift_detected": True,
            "drift_type": "schema_drift",

            "missing_columns": missing_columns,
            "extra_columns": extra_columns,

            "overall_drift_score": 1.0,

            "threshold": 0.05,

            "method": "Schema comparison",

            "recommendation": (
                "Schema drift detected. "
                "Review missing or extra columns before "
                "using the dataset for model inference."
            )
        }

        save_json(DATA_DRIFT_OUTPUT, report)
        save_json(DRIFT_REPORT_OUTPUT, report)

        print(json.dumps(report, indent=4))

        return report

    # ======================================================
    # Statistical Drift Detection
    # ======================================================

    report = {}
    drift_detected = False
    drift_scores = []

    for column in ref.columns:

        if column == "fraud":
            continue

        statistic, p_value = ks_2samp(
            ref[column],
            cur[column]
        )

        statistic = float(statistic)
        p_value = float(p_value)

        column_drift = (
            p_value < P_VALUE_THRESHOLD
            and statistic > KS_THRESHOLD
        )

        report[column] = {
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift": column_drift
        }

        drift_scores.append(statistic)

        if column_drift:
            drift_detected = True

    # ======================================================
    # Overall Drift Score
    # ======================================================

    if drift_scores:
        overall_drift_score = sum(drift_scores) / len(drift_scores)
    else:
        overall_drift_score = 0.0

    final_report = {

        "report_date": pd.Timestamp.utcnow().isoformat(),

        "drift_detected": drift_detected,

        "data_drift_detected": drift_detected,

        "drift_type": (
            "statistical_drift"
            if drift_detected
            else "no_drift"
        ),

        "overall_drift_score": round(
            overall_drift_score,
            4
        ),

        "threshold": 0.05,

        "method": "Kolmogorov-Smirnov Test",

        "features": report,

        "recommendation": (
            "Retraining may be required."
            if drift_detected
            else "No retraining required."
        )
    }

    save_json(
        DATA_DRIFT_OUTPUT,
        final_report
    )

    save_json(
        DRIFT_REPORT_OUTPUT,
        final_report
    )

    print(json.dumps(final_report, indent=4))

    return final_report


if __name__ == "__main__":
    detect_data_drift()
