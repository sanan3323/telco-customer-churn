"""Business-cost-aware decision threshold tuning."""

import numpy as np
from sklearn.metrics import confusion_matrix


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    grid_size: int = 91,
) -> dict:
    """Find the probability threshold that minimises expected cost.

    Sweeps thresholds in [0.05, 0.95] and returns the one minimising
    `fn_cost * FN + fp_cost * FP` on the provided predictions.

    Parameters
    ----------
    y_true
        Ground-truth binary labels.
    y_proba
        Predicted probability of the positive class.
    fn_cost, fp_cost
        Relative cost of each error type. Default 5:1 reflects churn being
        ~5x more expensive than an unnecessary retention offer.
    grid_size
        Number of thresholds to try in the sweep.

    Returns
    -------
    dict with keys:
      threshold : float   the best threshold
      cost      : float   the minimum cost achieved
      thresholds: ndarray full grid (for plotting)
      costs     : ndarray cost at each grid point (for plotting)
    """
    thresholds = np.linspace(0.05, 0.95, grid_size)
    costs = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        costs[i] = fp_cost * fp + fn_cost * fn

    best_idx = int(np.argmin(costs))
    return {
        "threshold":  float(thresholds[best_idx]),
        "cost":       float(costs[best_idx]),
        "thresholds": thresholds,
        "costs":      costs,
    }
