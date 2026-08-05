import numpy as np
from scipy.stats import ks_2samp


def calculate_drift(reference, current, threshold=0.05):
    """
    Detect drift using the Kolmogorov-Smirnov test.

    Returns:
        bool: True if drift is detected, otherwise False.
    """
    _, p_value = ks_2samp(reference, current)

    # Convert NumPy bool to Python bool
    return bool(p_value < threshold)


# ==========================
# Unit Tests
# ==========================

def test_no_drift():

    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)

    assert calculate_drift(reference, current) is False


def test_drift_detected():

    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(3, 1, 1000)

    assert calculate_drift(reference, current) is True


def test_same_distribution():

    reference = np.arange(100)
    current = np.arange(100)

    assert calculate_drift(reference, current) is False


def test_different_distribution():

    reference = np.arange(100)
    current = np.arange(100) + 100

    assert calculate_drift(reference, current) is True
