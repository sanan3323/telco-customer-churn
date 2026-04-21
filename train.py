"""Train the churn model end-to-end and save the artifact.

Usage:
    python train.py
    python train.py --data-path data/raw/CustChurn.csv --output-path models/churn_pipeline.joblib

This script is deterministic: same inputs + same seed + same library versions
produce byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import (
    build_pipeline,
    evaluate,
    find_optimal_threshold,
    format_report,
    load_and_clean,
    split_features_target,
)

# ── Defaults ─────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5
SCORING_METRIC = "roc_auc"

# Grid-searched hyperparameters. Winners in the notebook were
# learning_rate=0.05, max_depth=2, n_estimators=200 — we keep the search
# range narrow around those so retrains stay fast but can still adapt if
# the data shifts.
PARAM_GRID = {
    "clf__n_estimators":  [150, 200, 300],
    "clf__max_depth":     [2, 3],
    "clf__learning_rate": [0.05, 0.1],
}

# Cost ratio for threshold tuning. Supplied by finance in production.
FN_COST = 5.0
FP_COST = 1.0

PROJECT_ROOT     = Path(__file__).resolve().parent
DEFAULT_DATA     = PROJECT_ROOT / "data" / "raw" / "CustChurn.csv"
DEFAULT_OUTPUT   = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
DEFAULT_METRICS  = PROJECT_ROOT / "models" / "training_metrics.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def main(data_path: Path, output_path: Path, metrics_path: Path) -> None:
    # ── Load ────────────────────────────────────────────────────────────
    log.info("Loading data from %s", data_path)
    df = load_and_clean(data_path)
    X, y = split_features_target(df)
    log.info("Loaded %d rows, %d features, churn rate %.2f%%",
             len(X), X.shape[1], y.mean() * 100)

    # ── Split ───────────────────────────────────────────────────────────
    # We tune the decision threshold on a validation slice of the training
    # data, NOT on the test set. This is the fix for the issue the senior
    # review flagged in the notebook.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20,  # 20% of 80% = 16% of total
        stratify=y_trainval, random_state=RANDOM_SEED,
    )
    log.info("Splits — train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # ── Fit: grid search on the training set ────────────────────────────
    pipe = build_pipeline(random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    grid = GridSearchCV(pipe, PARAM_GRID, cv=cv,
                        scoring=SCORING_METRIC, n_jobs=-1, refit=True, verbose=1)
    log.info("Fitting grid search (%d CV folds × %d combos)...",
             CV_FOLDS, len(list(_grid_iter(PARAM_GRID))))
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    log.info("Best params: %s", grid.best_params_)
    log.info("Best CV %s: %.4f", SCORING_METRIC, grid.best_score_)

    # ── Tune threshold on the validation set ────────────────────────────
    val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    thr_result = find_optimal_threshold(
        y_val.values, val_proba, fn_cost=FN_COST, fp_cost=FP_COST,
    )
    threshold = thr_result["threshold"]
    log.info("Cost-tuned threshold (on validation): %.3f (cost=%.0f)",
             threshold, thr_result["cost"])

    # ── Evaluate on the held-out test set ───────────────────────────────
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    report = evaluate(y_test.values, test_proba, threshold=threshold)
    log.info("Test metrics at cost-tuned threshold:\n%s", format_report(report))

    # ── Persist ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline":    best_pipeline,
        "threshold":   threshold,
        "metrics":     report.as_dict(),
        "best_params": grid.best_params_,
        "seed":        RANDOM_SEED,
    }, output_path)
    log.info("Saved artifact to %s", output_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "test_metrics": report.as_dict(),
            "best_params":  grid.best_params_,
            "best_cv_score": grid.best_score_,
            "fn_cost": FN_COST, "fp_cost": FP_COST,
        }, f, indent=2)
    log.info("Saved metrics JSON to %s", metrics_path)


def _grid_iter(grid: dict):
    """Count grid combinations for logging."""
    from itertools import product
    keys, vals = list(grid.keys()), list(grid.values())
    yield from (dict(zip(keys, combo)) for combo in product(*vals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path",    type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-path",  type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    main(args.data_path, args.output_path, args.metrics_path)
