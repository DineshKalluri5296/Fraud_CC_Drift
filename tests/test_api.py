import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sample_payload():
    return {
        "distance_from_home": 12.5,
        "distance_from_last_transaction": 3.2,
        "ratio_to_median_purchase_price": 2.5,
        "repeat_retailer": 1,
        "used_chip": 1,
        "used_pin_number": 0,
        "online_order": 1
    }


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 200

    data = response.json()

    assert "fraud_prediction" in data
    assert "fraud_probability" in data
    assert isinstance(data["fraud_prediction"], int)
    assert isinstance(data["fraud_probability"], float)


def test_invalid_request():

    payload = {}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_metrics():

    response = client.get("/metrics")

    assert response.status_code == 200
