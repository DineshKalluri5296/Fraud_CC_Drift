import json
import os

import pandas as pd
from scipy.stats import ks_2samp


REFERENCE_DATA = "data/reference.csv"
CURRENT_DATA = "data/current.csv"

OUTPUT = "artifacts/data_drift.json"

# Drift detection thresholds
P_VALUE_THRESHOLD = 0.05
KS_THRESHOLD = 0.01

os.makedirs("artifacts", exist_ok=True)


def detect_data_drift():
    ref = pd.read_csv(REFERENCE_DATA)
    cur = pd.read_csv(CURRENT_DATA)

    report = {}
    drift_detected = False

    for column in ref.columns:

        if column == "fraud":
            continue

        statistic, p_value = ks_2samp(
            ref[column],
            cur[column]
        )

        # Require both statistical significance and meaningful difference
        column_drift = (
            p_value < P_VALUE_THRESHOLD
            and statistic > KS_THRESHOLD
        )

        report[column] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift": column_drift
        }

        if column_drift:
            drift_detected = True

    final_report = {
        "data_drift_detected": drift_detected,
        "columns": report
    }

    with open(OUTPUT, "w") as f:
        json.dump(final_report, f, indent=4)

    print(json.dumps(final_report, indent=4))


if __name__ == "__main__":
    detect_data_drift()
