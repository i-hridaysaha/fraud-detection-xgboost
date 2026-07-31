"""
api.py — FastAPI service exposing the trained XGBoost fraud model for
real-time scoring. Designed for the shape of request a payment/transaction
system would send at authorization time.

Two ways to supply the time-dependent features (velocity, recency, device):

  1. Store-backed (production path): set "use_feature_store": true and the
     service fetches those features from the online feature store
     (src/online_features.py) using the transaction's ids — computed live from
     per-customer streaming state, not passed by the caller. Only the raw
     transaction fields (V1-V28, Amount, ids, timestamp) are required.

  2. Caller-provided (legacy path, default): the caller passes all streaming
     features pre-computed in `features`. Kept for backward compatibility.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

Example request (store-backed):
    POST /score
    {
      "transaction_id": "TXN_00012345",
      "customer_id": "CUST_004821",
      "merchant_id": 118,
      "merchant_category": "electronics",
      "device_id": "DEV_002_CUST_004821",
      "amount": 482.10,
      "timestamp": "2026-07-09T14:32:00Z",
      "use_feature_store": true,
      "ingest": true,
      "features": { "V1": -1.2, "V2": 0.5, ... }   # V1-V28 only in this mode
    }
"""

import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

from src.config import PATHS, FEATURES
from src.explainability import explain_single_transaction
from src.online_features import MerchantRisk, get_feature_store

# ---- artifacts + online store, loaded once at startup (see lifespan) ----
_model = None
_feature_cols = None
_threshold_info = None
_merchant_risk_maps = None
_device_map = None
_explainer = None
_feature_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and construct the online feature store once, at
    process start. Replaces the deprecated @app.on_event("startup") hook with
    the supported lifespan handler.
    """
    global _model, _feature_cols, _threshold_info, _merchant_risk_maps
    global _device_map, _explainer, _feature_store

    _model = joblib.load(PATHS.xgb_model_path)
    _feature_cols = joblib.load(PATHS.feature_list_path)
    _threshold_info = joblib.load(PATHS.threshold_path)
    _merchant_risk_maps = joblib.load(PATHS.merchant_risk_map_path)
    _device_map = joblib.load(PATHS.device_map_path)

    # TreeExplainer is cheap relative to scoring volume; build once and cache.
    import shap
    _explainer = shap.TreeExplainer(_model)

    # Online feature store (Redis or in-memory, per FEATURE_STORE_BACKEND).
    # Reuse the already-loaded merchant-risk map rather than loading it twice.
    _feature_store = get_feature_store(merchant_risk=MerchantRisk(_merchant_risk_maps))

    print("Model, feature list, threshold, encoders, and feature store loaded.")
    yield
    # nothing to tear down explicitly; Redis connections are pooled


app = FastAPI(title="Enterprise Fraud Detection API", version="1.1.0", lifespan=lifespan)


class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: int
    merchant_category: str
    device_id: str
    amount: float
    timestamp: str
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Model input features. In store-backed mode (use_feature_store=true) "
                    "supply only V1-V28 — the velocity/recency/device features come from "
                    "the online feature store. In legacy mode supply all of them "
                    "pre-computed (V1-V28, seconds_since_last_txn, velocity_*_10m/60m/1440m, "
                    "is_new_device, distinct_device_count_so_far).",
    )
    use_feature_store: bool = Field(
        default=False,
        description="If true, fetch streaming features from the online store instead of "
                    "requiring the caller to pass them.",
    )
    ingest: bool = Field(
        default=False,
        description="Store-backed mode only: after scoring, commit this transaction into "
                    "the feature store so it counts toward future transactions' features. "
                    "Scoring always happens BEFORE the commit (causal ordering).",
    )
    explain: Optional[bool] = False


class IngestRequest(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: int
    merchant_category: str
    device_id: str
    amount: float
    timestamp: str


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    latency_ms: float
    top_factors: Optional[list] = None


def _txn_dict(req) -> dict:
    """Normalise a request into the dict shape the feature store expects."""
    return {
        "transaction_id": req.transaction_id,
        FEATURES.customer_col: req.customer_id,
        FEATURES.merchant_col: req.merchant_id,
        FEATURES.merchant_category_col: req.merchant_category,
        FEATURES.device_col: req.device_id,
        FEATURES.amount_col: req.amount,
        "timestamp": req.timestamp,
    }


def _finalize_row(row: dict) -> pd.DataFrame:
    """Validate all model columns are present and order them for the model."""
    missing = [c for c in _feature_cols if c not in row]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. "
                   f"Expected all of: {_feature_cols}",
        )
    return pd.DataFrame([{c: row[c] for c in _feature_cols}])


def _build_feature_row_legacy(req: TransactionRequest) -> pd.DataFrame:
    """Legacy path: caller passes all streaming features pre-computed."""
    row = dict(req.features)
    row["Amount"] = req.amount

    merchant = _merchant_risk_maps["merchant_risk"].get(
        req.merchant_id, _merchant_risk_maps["global_mean"]
    )
    category = _merchant_risk_maps["category_risk"].get(
        req.merchant_category, _merchant_risk_maps["global_mean"]
    )
    row.setdefault("merchant_risk_score", merchant)
    row.setdefault("merchant_category_risk_score", category)
    return _finalize_row(row)


def _build_feature_row_from_store(req: TransactionRequest) -> pd.DataFrame:
    """Store-backed path: stateful features come from the online store; the
    caller supplies only the raw transaction inputs (V1-V28) plus Amount.
    """
    row = dict(req.features)              # V1-V28
    row["Amount"] = req.amount
    row.update(_feature_store.get_features(_txn_dict(req)))  # velocity/recency/device/merchant
    return _finalize_row(row)


@app.post("/score", response_model=ScoreResponse)
def score_transaction(req: TransactionRequest):
    start = time.perf_counter()

    if req.use_feature_store:
        X = _build_feature_row_from_store(req)
    else:
        X = _build_feature_row_legacy(req)

    proba = float(_model.predict_proba(X)[:, 1][0])
    threshold = _threshold_info["best_threshold"]
    is_flagged = proba >= threshold

    top_factors = None
    if req.explain:
        top_factors = explain_single_transaction(_explainer, X)

    # Causal ordering: the transaction is scored against state that EXCLUDES it,
    # then (optionally) committed so it counts for future transactions.
    if req.use_feature_store and req.ingest:
        _feature_store.commit(_txn_dict(req))

    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(proba, 6),
        is_flagged=is_flagged,
        threshold_used=threshold,
        latency_ms=round(latency_ms, 2),
        top_factors=top_factors,
    )


@app.post("/ingest")
def ingest_transaction(req: IngestRequest):
    """Fold a transaction into the online feature store without scoring it.
    Used to warm per-customer state (e.g. load test warmup, backfill) and as a
    pure ingestion path separate from scoring.
    """
    _feature_store.commit(_txn_dict(req))
    return {"transaction_id": req.transaction_id, "ingested": True}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
