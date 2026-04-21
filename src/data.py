"""Data loading and cleaning for the Telco Churn project.

The raw CSV has one well-known data quality issue: TotalCharges is stored
as an object (string) dtype because roughly 11 rows contain a blank
space for customers with tenure == 0 (signed up but not yet billed).
We coerce to numeric and treat the blanks as 0.0.
"""

from pathlib import Path

import pandas as pd

from src.schema import ALL_FEATURES, TARGET, validate_features


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV without any cleaning.

    Useful for EDA and for inspecting the dataset as it arrives.
    """
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed cleaning steps.

    1. TotalCharges: coerce to numeric, blanks -> 0.0.
    2. customerID: dropped (no predictive signal, high cardinality).

    The input DataFrame is not mutated — we operate on a copy.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    return out


def load_and_clean(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """Load and clean in one call.

    If `validate=True` (default), the result is checked against the schema.
    Turn validation off only when you know the file may be partial (EDA on
    a truncated sample, for instance).
    """
    df = clean(load_raw(path))
    if validate:
        # We only validate the feature columns; the target is separate.
        feature_df = df.drop(columns=[TARGET]) if TARGET in df.columns else df
        validate_features(feature_df).raise_if_failed()
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X and y. The target is binarised to {0, 1}."""
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not in DataFrame")
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    # Guard against column drift between training runs.
    missing = set(ALL_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Feature columns missing from training data: {sorted(missing)}")
    return X[ALL_FEATURES], y  # canonical column order
