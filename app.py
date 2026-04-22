"""FastAPI service for the churn prediction model.

Run locally:
    uvicorn app:app --reload
Then visit http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.api_schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerRecord,
    HealthResponse,
    PredictionResponse,
)
from src.data import clean
from src.schema import ALL_FEATURES, validate_features

API_VERSION = "0.3.0"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# Module-level state: populated in lifespan, read via get_model() dependency.
_MODEL: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python train.py` before starting the API."
        )
    artifact = joblib.load(MODEL_PATH)
    _MODEL["pipeline"] = artifact["pipeline"]
    _MODEL["threshold"] = artifact["threshold"]
    _MODEL["metrics"] = artifact.get("metrics", {})
    log.info("Model loaded. threshold=%.3f test_roc_auc=%.4f",
             _MODEL["threshold"],
             _MODEL["metrics"].get("roc_auc", float("nan")))
    yield
    log.info("Shutting down")
    _MODEL.clear()


def get_model() -> dict[str, Any]:
    """FastAPI dependency: returns the loaded model.

    Tests can override this via app.dependency_overrides[get_model] = ...
    to inject a stub without touching disk.
    """
    if not _MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _MODEL


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predicts customer churn for a telecom subscriber. "
        "Returns a probability in [0, 1], a binary decision at the cost-tuned "
        "threshold, and a risk band for prioritisation."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# ── Helpers ──────────────────────────────────────────────────────────────
def _score_dataframe(df: pd.DataFrame, model: dict[str, Any]) -> list[PredictionResponse]:
    """Clean → validate → predict. Shared by /predict and /predict/batch."""
    cleaned = clean(df)
    result = validate_features(cleaned[ALL_FEATURES])
    if not result.ok:
        raise HTTPException(status_code=422, detail={"errors": result.errors})

    proba = model["pipeline"].predict_proba(cleaned[ALL_FEATURES])[:, 1]
    threshold = model["threshold"]

    def band(p: float) -> str:
        return "low" if p < 0.3 else "medium" if p < 0.6 else "high"

    return [
        PredictionResponse(
            churn_probability=float(p),
            churn_prediction=int(p >= threshold),
            threshold=float(threshold),
            risk_band=band(p),
        )
        for p in proba
    ]


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health(model: dict = Depends(get_model)) -> HealthResponse:
    """Liveness and readiness check."""
    return HealthResponse(
        status="ok",
        model_loaded=True,
        threshold=model["threshold"],
        test_roc_auc=model.get("metrics", {}).get("roc_auc"),
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerRecord,
    model: dict = Depends(get_model),
) -> PredictionResponse:
    """Score a single customer."""
    try:
        df = pd.DataFrame([customer.model_dump()])
        return _score_dataframe(df, model)[0]
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    model: dict = Depends(get_model),
) -> BatchPredictionResponse:
    """Score up to 1000 customers in a single call."""
    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        predictions = _score_dataframe(df, model)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
