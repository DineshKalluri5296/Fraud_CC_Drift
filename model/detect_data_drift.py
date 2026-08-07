import json
import os

import pandas as pd
from scipy.stats import ks_2samp
from pandas.errors import EmptyDataError

REFERENCE_DATA = "data/reference.csv"
CURRENT_DATA = "data/current.csv"
OUTPUT = "artifacts/data_drift.json"

P_VALUE_THRESHOLD = 0.05
KS_THRESHOLD = 0.01

os.makedirs("artifacts", exist_ok=True)


def detect_data_drift():

    try:
        ref = pd.read_csv(REFERENCE_DATA)
        cur = pd.read_csv(CURRENT_DATA)
    except EmptyDataError:
        report = {
            "data_drift_detected": True,
            "drift_type": "empty_dataset",
            "reason": "Current or reference dataset is empty"
        }

        with open(OUTPUT, "w") as f:
            json.dump(report, f, indent=4)

        print(json.dumps(report, indent=4))
        return

    # ==============================
    # Schema Drift Detection
    # ==============================

    ref_columns = set(ref.columns)
    cur_columns = set(cur.columns)

    missing_columns = sorted(list(ref_columns - cur_columns))
    extra_columns = sorted(list(cur_columns - ref_columns))

    if missing_columns or extra_columns:

        report = {
            "data_drift_detected": True,
            "drift_type": "schema_drift",
            "missing_columns": missing_columns,
            "extra_columns": extra_columns
        }

        with open(OUTPUT, "w") as f:
            json.dump(report, f, indent=4)

        print(json.dumps(report, indent=4))
        return

    # ==============================
    # Statistical Drift Detection
    # ==============================

    report = {}
    drift_detected = False

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

        drift_detected |= column_drift

    final_report = {
        "data_drift_detected": drift_detected,
        "drift_type": "statistical_drift" if drift_detected else "no_drift",
        "columns": report
    }

    with open(OUTPUT, "w") as f:
        json.dump(final_report, f, indent=4)

    print(json.dumps(final_report, indent=4))


if __name__ == "__main__":
    detect_data_drift()
