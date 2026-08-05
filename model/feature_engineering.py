from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


def create_features(X, y):
    """
    Train-Test Split
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "model/scaler.pkl")

    return X_train, X_test, y_train, y_test