"""
api.py — FastAPI service exposing the trained XGBoost fraud model for
real-time scoring. Designed for the shape of request a payment/transaction
system would send at authorization time.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

Example request:
    POST /score
    {
      "transaction_id": "TXN_00012345",
      "customer_id": "CUST_004821",
      "merchant_id": 118,
      "merchant_category": "electronics",
      "device_id": "DEV_002_CUST_004821",
      "amount": 482.10,
      "timestamp": "2026-07-09T14:32:00Z",
      "features": { "V1": -1.2, "V2": 0.5, ... }
    }
"""

import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

from src.config import PATHS, FEATURES
from src.explainability import compute_shap_values, explain_single_transaction

app = FastAPI(title="Enterprise Fraud Detection API", version="1.0.0")

# ---- Load artifacts once at startup ----
_model = None
_feature_cols = None
_threshold_info = None
_merchant_risk_maps = None
_device_map = None
_explainer = None


@app.on_event("startup")
def load_artifacts():
    global _model, _feature_cols, _threshold_info, _merchant_risk_maps, _device_map, _explainer
    _model = joblib.load(PATHS.xgb_model_path)
    _feature_cols = joblib.load(PATHS.feature_list_path)
    _threshold_info = joblib.load(PATHS.threshold_path)
    _merchant_risk_maps = joblib.load(PATHS.merchant_risk_map_path)
    _device_map = joblib.load(PATHS.device_map_path)
    # NOTE: TreeExplainer construction is cheap relative to scoring volume;
    # for very high QPS, precompute once and cache (done here via startup hook).
    import shap
    _explainer = shap.TreeExplainer(_model)
    print("Model, feature list, threshold, and encoders loaded.")


class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: int
    merchant_category: str
    device_id: str
    amount: float
    timestamp: str
    features: Dict[str, float] = Field(
        ..., description="Precomputed streaming features: V1-V28, seconds_since_last_txn, "
                          "velocity_count_10m, velocity_amount_10m, velocity_count_60m, "
                          "velocity_amount_60m, velocity_count_1440m, velocity_amount_1440m, "
                          "is_new_device, distinct_device_count_so_far. In production these "
                          "come from a streaming feature store (e.g. Feast/Flink), not computed "
                          "per-request from a cold start."
    )
    explain: Optional[bool] = False


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    latency_ms: float
    top_factors: Optional[list] = None


def _build_feature_row(req: TransactionRequest) -> pd.DataFrame:
    row = dict(req.features)
    row["Amount"] = req.amount

    merchant_risk = _merchant_risk_maps["merchant_risk"].get(
        req.merchant_id, _merchant_risk_maps["global_mean"]
    )
    merchant_category_risk = _merchant_risk_maps["category_risk"].get(
        req.merchant_category, _merchant_risk_maps["global_mean"]
    )
    row.setdefault("merchant_risk_score", merchant_risk)
    row.setdefault("merchant_category_risk_score", merchant_category_risk)

    missing = [c for c in _feature_cols if c not in row]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. "
                   f"Expected all of: {_feature_cols}",
        )

    return pd.DataFrame([{c: row[c] for c in _feature_cols}])


@app.post("/score", response_model=ScoreResponse)
def score_transaction(req: TransactionRequest):
    start = time.perf_counter()

    X = _build_feature_row(req)
    proba = float(_model.predict_proba(X)[:, 1][0])
    threshold = _threshold_info["best_threshold"]
    is_flagged = proba >= threshold

    top_factors = None
    if req.explain:
        top_factors = explain_single_transaction(_explainer, X)

    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(proba, 6),
        is_flagged=is_flagged,
        threshold_used=threshold,
        latency_ms=round(latency_ms, 2),
        top_factors=top_factors,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
