"""FastAPI service for the churn prediction model.

Run locally with:
    uvicorn app:app --reload

Then visit http://localhost:8000/docs for interactive API docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data import clean
from src.schema import ALL_FEATURES, validate_features

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# ── Model loading ────────────────────────────────────────────────────────
# The model is loaded ONCE at startup and kept in memory. Loading a joblib
# artifact takes ~100ms which is fine at boot but would be fatal if we did
# it on every request. `lifespan` is FastAPI's way to run setup/teardown.
MODEL: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Loading model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python train.py` before starting the API."
        )
    artifact = joblib.load(MODEL_PATH)
    MODEL["pipeline"] = artifact["pipeline"]
    MODEL["threshold"] = artifact["threshold"]
    MODEL["metrics"] = artifact.get("metrics", {})
    log.info("Model loaded. Threshold=%.3f  Test ROC-AUC=%.4f",
             MODEL["threshold"], MODEL["metrics"].get("roc_auc", float("nan")))
    yield
    # Shutdown
    log.info("Shutting down")
    MODEL.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn for a telecom subscriber.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check. Returns 200 if the service is up and the model loaded."""
    return {
        "status": "ok",
        "model_loaded": bool(MODEL),
        "threshold": MODEL.get("threshold"),
        "test_roc_auc": MODEL.get("metrics", {}).get("roc_auc"),
    }


@app.post("/predict")
def predict(customer: dict) -> dict[str, Any]:
    """Score a single customer.

    Expects a JSON body with the fields listed in src.schema.ALL_FEATURES.
    Returns churn probability and a binary decision at the tuned threshold.
    """
    try:
        df = pd.DataFrame([customer])
        cleaned = clean(df)
        result = validate_features(cleaned[ALL_FEATURES])
        if not result.ok:
            raise HTTPException(status_code=422, detail={"errors": result.errors})

        proba = float(MODEL["pipeline"].predict_proba(cleaned[ALL_FEATURES])[0, 1])
        decision = int(proba >= MODEL["threshold"])
        return {
            "churn_probability": proba,
            "churn_prediction": decision,
            "threshold": MODEL["threshold"],
            "risk_band": (
                "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
            ),
        }
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing field: {e}")
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))