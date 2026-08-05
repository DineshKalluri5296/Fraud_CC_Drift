import subprocess
import json
import os


THRESHOLD = 0.95


# ==========================================================
# Check Metrics
# ==========================================================

if not os.path.exists("artifacts/evaluation.json"):

    print("No evaluation found")

    subprocess.run(["python", "model/train.py"])

    exit()


with open("artifacts/evaluation.json") as f:

    metrics = json.load(f)


accuracy = metrics["accuracy"]

print(f"Current Accuracy : {accuracy}")


# ==========================================================
# Retrain
# ==========================================================

if accuracy < THRESHOLD:

    print("Accuracy below threshold")

    subprocess.run(["python", "model/train.py"])

else:

    print("No Retraining Required")