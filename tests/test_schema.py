"""Tests for src/schema.py — the input contract."""

import pandas as pd
import pytest

from src.schema import ALL_FEATURES, validate_features


def _valid_row() -> dict:
    """A minimal well-formed customer record."""
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 425.0,
    }


def test_valid_row_passes():
    df = pd.DataFrame([_valid_row()])
    result = validate_features(df)
    assert result.ok, f"Unexpected errors: {result.errors}"


def test_missing_column_is_caught():
    row = _valid_row()
    del row["Contract"]
    result = validate_features(pd.DataFrame([row]))
    assert not result.ok
    assert any("Contract" in e for e in result.errors)


def test_unknown_category_is_caught():
    """Case-sensitive: 'Fiber Optic' (capital O) must fail."""
    row = _valid_row()
    row["InternetService"] = "Fiber Optic"
    result = validate_features(pd.DataFrame([row]))
    assert not result.ok
    assert any("Fiber Optic" in e for e in result.errors)


def test_non_numeric_value_is_caught():
    row = _valid_row()
    row["tenure"] = "five"
    result = validate_features(pd.DataFrame([row]))
    assert not result.ok


def test_nan_in_numeric_is_caught():
    row = _valid_row()
    row["MonthlyCharges"] = float("nan")
    result = validate_features(pd.DataFrame([row]))
    assert not result.ok


def test_raise_if_failed_raises():
    row = _valid_row()
    del row["tenure"]
    result = validate_features(pd.DataFrame([row]))
    with pytest.raises(ValueError, match="Schema validation failed"):
        result.raise_if_failed()
