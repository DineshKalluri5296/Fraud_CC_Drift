import json
import pandas as pd
from scipy.stats import ks_2samp
import os


REFERENCE_DATA = "data/reference.csv"
CURRENT_DATA = "data/current.csv"

OUTPUT = "artifacts/data_drift.json"

os.makedirs("artifacts", exist_ok=True)


def detect_data_drift():

    ref = pd.read_csv(REFERENCE_DATA)

    cur = pd.read_csv(CURRENT_DATA)

    report = {}

    drift = False

    for column in ref.columns:

        if column == "fraud":
            continue

        statistic, p_value = ks_2samp(
            ref[column],
            cur[column]
        )

        report[column] = {

            "ks_statistic": round(float(statistic), 4),

            "p_value": round(float(p_value), 4),

            "drift": bool(p_value < 0.05)

        }

        if p_value < 0.05:
            drift = True

    final_report = {

        "data_drift_detected": drift,

        "columns": report

    }

    with open(OUTPUT, "w") as f:

        json.dump(final_report, f, indent=4)

    print(final_report)


if __name__ == "__main__":

    detect_data_drift()