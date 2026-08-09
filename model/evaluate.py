# import json
# import os
# import joblib
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
# )
# from sklearn.model_selection import train_test_split

# # =========================================================
# # Configuration
# # ==========================================================

# DATA_PATH = "data/card_transdata.csv"
# MODEL_PATH = "model/model.pkl"
# OUTPUT_PATH = "artifacts/evaluation.json"
# ACCURACY_THRESHOLD = 0.95


# # ==========================================================
# # Load Data
# # ==========================================================

# df = pd.read_csv(DATA_PATH)

# features = df.drop("fraud", axis=1)
# target = df["fraud"]

# features_train, features_test, target_train, target_test = train_test_split(
#     features,
#     target,
#     test_size=0.2,
#     random_state=42,
#     stratify=target,
# )
# # ==========================================================
# # Load Model
# # ==========================================================
# model = joblib.load(MODEL_PATH)

# # ==========================================================
# # Prediction
# # ==========================================================

# predictions = model.predict(features_test)

# # ==========================================================
# # Calculate Metrics
# # ==========================================================

# accuracy = accuracy_score(target_test, predictions)
# precision = precision_score(target_test, predictions)
# recall = recall_score(target_test, predictions)
# f1 = f1_score(target_test, predictions)

# metrics = {
#     "accuracy": round(accuracy, 4),
#     "precision": round(precision, 4),
#     "recall": round(recall, 4),
#     "f1_score": round(f1, 4),
# }

# print(json.dumps(metrics, indent=4))

# # ==========================================================
# # Save Metrics
# # ==========================================================

# os.makedirs("artifacts", exist_ok=True)

# with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
#     json.dump(metrics, file, indent=4)


# # ==========================================================
# # Accuracy Threshold Check
# # ==========================================================

# if accuracy < ACCURACY_THRESHOLD:
#     raise ValueError(
#         f"Model accuracy {accuracy:.4f} is below "
#         f"the required threshold {ACCURACY_THRESHOLD:.2f}"
#     )

# print("Model Passed")

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/card_transdata.csv"
MODEL_PATH = "model/model.pkl"
OUTPUT_PATH = "artifacts/evaluation.json"
ACCURACY_THRESHOLD = 0.95


def evaluate_model():
    """Evaluate the fraud detection model."""

    # Load data
    dataframe = pd.read_csv(DATA_PATH)

    features = dataframe.drop("fraud", axis=1)
    target = dataframe["fraud"]

    # Create test dataset
    _, features_test, _, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    predictions = model.predict(features_test)

    # Calculate metrics
    accuracy = accuracy_score(target_test, predictions)
    precision = precision_score(target_test, predictions)
    recall = recall_score(target_test, predictions)
    f1 = f1_score(target_test, predictions)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(json.dumps(metrics, indent=4))

    # Save evaluation results
    os.makedirs("artifacts", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Model quality check
    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(
            f"Model accuracy {accuracy:.4f} is below "
            f"the required threshold {ACCURACY_THRESHOLD:.2f}"
        )

    print("Model Passed")

    return metrics


if __name__ == "__main__":
    evaluate_model()

