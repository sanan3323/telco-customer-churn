"""Tests for src/threshold.py."""

import numpy as np

from src.threshold import find_optimal_threshold


def test_threshold_prefers_recall_when_fn_is_expensive():
    """With a very high FN cost, the optimiser should pick a low threshold
    (classify more as positive, catch more true positives)."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.3, size=1000)
    # Model is calibrated-ish: prob for positives centred higher.
    y_proba = np.where(y_true == 1,
                       rng.beta(6, 4, size=1000),
                       rng.beta(4, 6, size=1000))

    low_fn   = find_optimal_threshold(y_true, y_proba, fn_cost=1.0,  fp_cost=1.0)
    high_fn  = find_optimal_threshold(y_true, y_proba, fn_cost=20.0, fp_cost=1.0)

    assert high_fn["threshold"] < low_fn["threshold"]


def test_threshold_grid_is_returned_for_plotting():
    y_true  = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    result = find_optimal_threshold(y_true, y_proba)
    assert len(result["thresholds"]) == len(result["costs"])
    assert result["cost"] == result["costs"].min()
