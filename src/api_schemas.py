"""Pydantic schemas for the churn prediction API.

These define the contract for the /predict and /health endpoints:
- What fields are required in a request
- What types and values they may have
- What the response looks like

FastAPI uses these to auto-generate /docs and to validate incoming JSON
before your endpoint code runs. Invalid requests get a detailed 422 error
automatically — you don't write validation code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Allowed categorical values (mirrors src.schema.CATEGORICAL_FEATURES) ──
# We list them here as Literal types so FastAPI enforces them at the API
# layer. src.schema.py validates them again at the pipeline layer — both
# checks are cheap and catching errors early means better error messages.

YesNo = Literal["Yes", "No"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerRecord(BaseModel):
    """A single customer's features for churn scoring."""

    model_config = ConfigDict(
        # Reject unknown fields. Strict, but what we want for a prod API.
        extra="forbid",
        # Allow schema to ship an example for /docs.
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
                "TotalCharges": 190.0,
            }
        },
    )

    # Demographics
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="1 if senior citizen, else 0"
    )
    Partner: YesNo
    Dependents: YesNo

    # Account
    tenure: int = Field(..., ge=0, description="Months with the company")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # Services
    PhoneService: YesNo
    MultipleLines: YesNoNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet

    # Financials
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: float = Field(..., ge=0.0)


class PredictionResponse(BaseModel):
    """Model output for a single customer."""

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted probability of churn in [0, 1]"
    )
    churn_prediction: Literal[0, 1] = Field(
        ..., description="1 = predicted to churn at the tuned threshold"
    )
    threshold: float = Field(
        ..., ge=0.0, le=1.0,
        description="Decision threshold used for churn_prediction"
    )
    risk_band: Literal["low", "medium", "high"]


class BatchPredictionRequest(BaseModel):
    """Score several customers at once."""

    customers: list[CustomerRecord] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Matched output for a batch request — same order as the input."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Returned by /health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    threshold: float | None = None
    test_roc_auc: float | None = None
    version: str
