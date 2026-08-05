import json
import os

BASELINE_FILE = "artifacts/baseline_metrics.json"

CURRENT_FILE = "artifacts/evaluation.json"

OUTPUT = "artifacts/model_drift.json"

THRESHOLD = 0.03

os.makedirs("artifacts", exist_ok=True)


def detect_model_drift():

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    with open(CURRENT_FILE) as f:
        current = json.load(f)

    baseline_acc = baseline["accuracy"]

    current_acc = current["accuracy"]

    difference = baseline_acc - current_acc

    drift = difference > THRESHOLD

    report = {

        "baseline_accuracy": baseline_acc,

        "current_accuracy": current_acc,

        "accuracy_drop": round(difference, 4),

        "model_drift_detected": drift

    }

    with open(OUTPUT, "w") as f:

        json.dump(report, f, indent=4)

    print(report)


if __name__ == "__main__":

    detect_model_drift()