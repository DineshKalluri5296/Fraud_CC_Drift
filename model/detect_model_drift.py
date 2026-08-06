import json
import os

BASELINE_FILE = "artifacts/evaluation.json"
CURRENT_FILE = "artifacts/evaluation.json"

OUTPUT = "artifacts/model_drift.json"

THRESHOLD = 0.03

os.makedirs("artifacts", exist_ok=True)


def get_accuracy(data):
    if "metrics" in data:
        return float(data["metrics"]["accuracy"])
    elif "accuracy" in data:
        return float(data["accuracy"])
    else:
        raise KeyError(
            f"Accuracy not found. Keys are: {list(data.keys())}"
        )


def detect_model_drift():

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    with open(CURRENT_FILE) as f:
        current = json.load(f)

    print("Baseline JSON:")
    print(json.dumps(baseline, indent=2))

    print("Current JSON:")
    print(json.dumps(current, indent=2))

    baseline_acc = get_accuracy(baseline)
    current_acc = get_accuracy(current)

    difference = baseline_acc - current_acc

    report = {
        "baseline_accuracy": baseline_acc,
        "current_accuracy": current_acc,
        "accuracy_drop": round(difference, 4),
        "threshold": THRESHOLD,
        "model_drift_detected": difference > THRESHOLD
    }

    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    detect_model_drift()
