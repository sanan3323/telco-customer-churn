"""Input schema contract for the churn model.

Both training and inference validate incoming data against this schema.
When the contract changes (new feature, renamed column), change it here
once — everything downstream picks it up.
"""

from dataclasses import dataclass, field


# Columns the model consumes. Order doesn't matter for DataFrame inputs,
# but listing them here is the canonical source of truth.
NUMERIC_FEATURES: list[str] = [
    "SeniorCitizen",   # 0/1 but stored as int in the source file
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# For categorical features, we list the values seen at training time.
# Inference-time values must be a subset, otherwise we reject the row.
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "gender":           ["Female", "Male"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "MultipleLines":    ["Yes", "No", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["Yes", "No", "No internet service"],
    "OnlineBackup":     ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport":      ["Yes", "No", "No internet service"],
    "StreamingTV":      ["Yes", "No", "No internet service"],
    "StreamingMovies":  ["Yes", "No", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# All feature names, in no particular order.
ALL_FEATURES: list[str] = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())

TARGET: str = "Churn"
TARGET_VALUES: list[str] = ["Yes", "No"]


@dataclass
class ValidationResult:
    """Outcome of a schema check. `ok == True` means safe to pass to the model."""
    ok: bool
    errors: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Schema validation failed:\n  - " + "\n  - ".join(self.errors))


def validate_features(df) -> ValidationResult:
    """Check that a DataFrame conforms to the feature schema.

    Rules:
      1. All required columns must be present.
      2. Numeric features must be numeric-coercible with no NaN.
      3. Categorical features must contain only known category values.

    Returns a ValidationResult. Call `.raise_if_failed()` to hard-fail.
    """
    import pandas as pd  # local import keeps module cheap to load

    errors: list[str] = []

    # 1. Required columns
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return ValidationResult(ok=False, errors=errors)  # no point continuing

    # 2. Numeric columns
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
        elif df[col].isna().any():
            errors.append(f"Column '{col}' contains NaN values")

    # 3. Categorical columns — values must be in the allowed set
    for col, allowed in CATEGORICAL_FEATURES.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            errors.append(
                f"Column '{col}' contains unknown categories {sorted(unknown)}; "
                f"allowed: {allowed}"
            )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
