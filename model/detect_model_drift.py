import json
import os

BASELINE_FILE = "artifacts/evaluation.json"
CURRENT_FILE = "artifacts/evaluation.json"

OUTPUT = "artifacts/model_drift.json"

THRESHOLD = 0.03

os.makedirs("artifacts", exist_ok=True)


def detect_model_drift():

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    with open(CURRENT_FILE) as f:
        current = json.load(f)

    baseline_acc = float(
        baseline["metrics"]["accuracy"]
    )

    current_acc = float(
        current["metrics"]["accuracy"]
    )

    difference = baseline_acc - current_acc

    drift = difference > THRESHOLD

    report = {
        "baseline_accuracy": baseline_acc,
        "current_accuracy": current_acc,
        "accuracy_drop": round(difference, 4),
        "threshold": THRESHOLD,
        "model_drift_detected": drift
    }

    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    detect_model_drift()
