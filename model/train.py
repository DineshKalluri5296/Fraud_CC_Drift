import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from lightgbm import LGBMClassifier


# ==========================================================
# Paths
# ==========================================================

DATA_PATH = "data/card_transdata.csv"

MODEL_DIR = "model"

ARTIFACT_DIR = "artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

X = df.drop("fraud", axis=1)

y = df["fraud"]


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# Model
# ==========================================================

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================================
# Evaluation
# ==========================================================

pred = model.predict(X_test)

metrics = {

    "accuracy": round(accuracy_score(y_test, pred), 4),

    "precision": round(precision_score(y_test, pred), 4),

    "recall": round(recall_score(y_test, pred), 4),

    "f1_score": round(f1_score(y_test, pred), 4)

}

print(metrics)


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(model, "model/model.pkl")


# ==========================================================
# Save Metrics
# ==========================================================

with open("artifacts/evaluation.json", "w") as f:

    json.dump(metrics, f, indent=4)


print("\nTraining Completed")