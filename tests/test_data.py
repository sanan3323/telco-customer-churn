"""Tests for src/data.py."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import clean, load_and_clean, split_features_target
from src.schema import TARGET

FIXTURE = Path(__file__).parent / "fixtures" / "mini_churn.csv"


def test_clean_coerces_blank_total_charges_to_zero():
    raw = pd.read_csv(FIXTURE)
    cleaned = clean(raw)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].notna().all()
    zero_row = cleaned[cleaned["tenure"] == 0]
    assert zero_row["TotalCharges"].iloc[0] == 0.0


def test_clean_drops_customer_id():
    raw = pd.read_csv(FIXTURE)
    cleaned = clean(raw)
    assert "customerID" not in cleaned.columns


def test_clean_does_not_mutate_input():
    raw = pd.read_csv(FIXTURE)
    raw_copy = raw.copy()
    _ = clean(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)


def test_load_and_clean_preserves_row_count():
    cleaned = load_and_clean(FIXTURE, validate=False)
    raw = pd.read_csv(FIXTURE)
    assert len(cleaned) == len(raw)


def test_split_features_target_separates_columns():
    df = load_and_clean(FIXTURE, validate=False)
    X, y = split_features_target(df)
    assert TARGET not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)
