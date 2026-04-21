"""Preprocessing pipeline builder for the Telco Churn project."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step: scale numerics, one-hot encode categoricals.

    The feature lists come from schema.py — changing a feature there (adding
    a new column, changing allowed categories) automatically propagates here.

    Using OneHotEncoder(handle_unknown='ignore') as a second line of defense
    after schema validation: if somehow an unknown category reaches the
    encoder, it's zero-encoded rather than crashing the request. Schema
    validation in production should reject these before they ever get here.
    """
    categorical_cols = list(CATEGORICAL_FEATURES.keys())

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
             categorical_cols),
        ],
        remainder="drop",   # anything not in the schema is silently dropped
        verbose_feature_names_out=True,
    )
