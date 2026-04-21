"""Score a CSV of customers using the trained pipeline.

Usage:
    python predict.py --input customers.csv --output predictions.csv

Expected input columns: see src.schema.ALL_FEATURES.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.data import clean
from src.schema import ALL_FEATURES, validate_features

PROJECT_ROOT   = Path(__file__).resolve().parent
DEFAULT_MODEL  = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")


def predict_churn(
    df: pd.DataFrame,
    artifact_path: Path = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Score a DataFrame of customer records.

    Parameters
    ----------
    df
        DataFrame with the columns defined in src.schema.ALL_FEATURES.
        Rows with invalid data are rejected (ValueError).
    artifact_path
        Path to the joblib file produced by train.py.

    Returns
    -------
    pd.DataFrame with columns:
      - all input columns preserved
      - churn_probability : float in [0, 1]
      - churn_prediction  : int 0 or 1 at the saved threshold
      - risk_band         : str   'low' | 'medium' | 'high'
    """
    artifact = joblib.load(artifact_path)
    pipeline  = artifact["pipeline"]
    threshold = artifact["threshold"]

    # Clean first (handles TotalCharges), then validate schema.
    cleaned = clean(df)
    validate_features(cleaned[ALL_FEATURES]).raise_if_failed()

    proba = pipeline.predict_proba(cleaned[ALL_FEATURES])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"]  = (proba >= threshold).astype(int)
    out["risk_band"] = pd.cut(
        proba,
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["low", "medium", "high"],
    )
    return out


def main(input_path: Path, output_path: Path, artifact_path: Path) -> None:
    log.info("Loading input: %s", input_path)
    df = pd.read_csv(input_path)
    log.info("Scoring %d rows against %s", len(df), artifact_path)
    scored = predict_churn(df, artifact_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    log.info("Wrote predictions to %s", output_path)

    # Summary to stdout for human sanity-check.
    n_high = (scored["risk_band"] == "high").sum()
    log.info("Summary: %d high-risk / %d medium / %d low",
             n_high,
             (scored["risk_band"] == "medium").sum(),
             (scored["risk_band"] == "low").sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",    type=Path, required=True)
    parser.add_argument("--output",   type=Path, default=PROJECT_ROOT / "predictions.csv")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    main(args.input, args.output, args.artifact)
