from pydantic import BaseModel, Field


# ==========================================================
# Request Schema
# ==========================================================

class FraudRequest(BaseModel):
    """
    Request body for fraud prediction
    """

    distance_from_home: float = Field(
        ...,
        description="Distance from customer's home",
        examples=[15.32]
    )

    distance_from_last_transaction: float = Field(
        ...,
        description="Distance from previous transaction",
        examples=[5.36]
    )

    ratio_to_median_purchase_price: float = Field(
        ...,
        description="Ratio to median purchase price",
        examples=[3.25]
    )

    repeat_retailer: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = Repeat retailer, 0 = New retailer",
        examples=[1]
    )

    used_chip: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = Chip used",
        examples=[0]
    )

    used_pin_number: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = PIN used",
        examples=[1]
    )

    online_order: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = Online transaction",
        examples=[0]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance_from_home": 15.32,
                "distance_from_last_transaction": 5.36,
                "ratio_to_median_purchase_price": 3.25,
                "repeat_retailer": 1,
                "used_chip": 0,
                "used_pin_number": 1,
                "online_order": 0
            }
        }
    }


# ==========================================================
# Prediction Response
# ==========================================================

class PredictionResponse(BaseModel):
    """
    Prediction response
    """

    fraud_prediction: int = Field(
        description="0 = Genuine Transaction, 1 = Fraud"
    )

    fraud_probability: float = Field(
        description="Fraud probability"
    )


# ==========================================================
# Health Check Response
# ==========================================================

class HealthResponse(BaseModel):
    """
    Health endpoint response
    """

    status: str
    model_loaded: bool
    version: str


# ==========================================================
# Model Metrics Response
# ==========================================================

class ModelMetricsResponse(BaseModel):
    """
    Model evaluation metrics
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float


# ==========================================================
# Drift Metrics Response
# ==========================================================

class DriftResponse(BaseModel):
    """
    Drift detection response
    """

    data_drift_score: float
    model_drift_score: float
    drift_detected: bool


# ==========================================================
# Generic Error Response
# ==========================================================

class ErrorResponse(BaseModel):
    detail: str