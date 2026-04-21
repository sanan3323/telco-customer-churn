"""Tests for src/features.py — the preprocessor."""

import pandas as pd

from src.features import build_preprocessor
from src.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {**{c: "Yes" for c in ["Partner","Dependents","PhoneService","PaperlessBilling"]},
         "gender": "Female", "MultipleLines": "No",
         "InternetService": "Fiber optic",
         "OnlineSecurity": "No", "OnlineBackup": "No",
         "DeviceProtection": "No", "TechSupport": "No",
         "StreamingTV": "Yes", "StreamingMovies": "Yes",
         "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
         "SeniorCitizen": 0, "tenure": 5,
         "MonthlyCharges": 70.0, "TotalCharges": 350.0},
        {**{c: "No" for c in ["Partner","Dependents","PhoneService","PaperlessBilling"]},
         "gender": "Male", "MultipleLines": "No phone service",
         "InternetService": "DSL",
         "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
         "DeviceProtection": "Yes", "TechSupport": "Yes",
         "StreamingTV": "No", "StreamingMovies": "No",
         "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)",
         "SeniorCitizen": 1, "tenure": 60,
         "MonthlyCharges": 90.0, "TotalCharges": 5400.0},
    ])


def test_preprocessor_fit_transform_shape():
    X = _sample_frame()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    # Numerics stay 1-per-column; categoricals expand via one-hot.
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)  # at least numeric + some OHE


def test_preprocessor_handles_unknown_category():
    """handle_unknown='ignore' means an unseen category maps to all-zeros,
    not a crash. Schema validation should catch this earlier in prod, but
    the encoder is the last line of defense."""
    X_train = _sample_frame()
    pre = build_preprocessor()
    pre.fit(X_train)

    X_bad = _sample_frame().iloc[[0]].copy()
    X_bad["PaymentMethod"] = "Cash"  # not in the training vocab
    Xt = pre.transform(X_bad)
    assert Xt.shape == (1, pre.transform(X_train.iloc[[0]]).shape[1])
