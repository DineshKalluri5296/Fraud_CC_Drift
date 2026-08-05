import joblib
import numpy as np


MODEL_PATH = "model/model.pkl"


def test_model_load():

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    sample = np.array([[
        12.5,
        3.2,
        2.5,
        1,
        1,
        0,
        1
    ]])

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]


def test_prediction_probability():

    model = joblib.load(MODEL_PATH)

    sample = np.array([[
        12.5,
        3.2,
        2.5,
        1,
        1,
        0,
        1
    ]])

    probability = model.predict_proba(sample)

    assert probability.shape == (1, 2)


def test_prediction_type():

    model = joblib.load(MODEL_PATH)

    sample = np.array([[
        10,
        2,
        1.5,
        0,
        0,
        0,
        0
    ]])

    pred = model.predict(sample)[0]

    assert isinstance(int(pred), int)