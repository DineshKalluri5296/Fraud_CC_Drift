import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split


# ==========================================================
# Load Data
# ==========================================================

df = pd.read_csv("data/card_transdata.csv")

X = df.drop("fraud", axis=1)

y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# Load Model
# ==========================================================

model = joblib.load("model/model.pkl")


# ==========================================================
# Prediction
# ==========================================================

pred = model.predict(X_test)


accuracy = accuracy_score(y_test, pred)

precision = precision_score(y_test, pred)

recall = recall_score(y_test, pred)

f1 = f1_score(y_test, pred)


metrics = {

    "accuracy": round(accuracy, 4),

    "precision": round(precision, 4),

    "recall": round(recall, 4),

    "f1_score": round(f1, 4)

}


print(metrics)


# ==========================================================
# Save Metrics
# ==========================================================

with open("artifacts/evaluation.json", "w") as f:

    json.dump(metrics, f, indent=4)


# ==========================================================
# Threshold Check
# ==========================================================

THRESHOLD = 0.95

if accuracy < THRESHOLD:

    raise Exception(
        f"Accuracy dropped to {accuracy:.4f}"
    )

print("Model Passed")
