"""
test_streaming_parity.py — the proof that the online (streaming) feature store
produces the SAME features as the offline batch pipeline for the same ordered
stream of transactions.

This is the single most important correctness guarantee for the serving path:
if the online features drift from what the model was trained on, the model is
being fed a distribution it never saw, and offline metrics become meaningless.
So we run one fixed toy transaction set through both paths and assert the
resulting feature rows are equal (tiny tolerance on summed float amounts).

Both backends are checked; the in-memory backend always runs, and the Redis
backend runs too when a server is reachable (skipped otherwise so the suite
stays green with zero infra).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(".")
from src.config import FEATURES
from src.feature_engineering import (
    add_time_since_last_transaction,
    add_transaction_velocity,
    add_device_consistency,
    apply_merchant_risk_map,
)
from src.online_features import (
    InMemoryFeatureStore,
    RedisFeatureStore,
    MerchantRisk,
)


# ---- fixed toy stream -------------------------------------------------------

def make_toy_stream() -> pd.DataFrame:
    """A small hand-built stream that deliberately exercises every stateful
    feature: bursts inside the 10m window, gaps across the 60m/1440m windows,
    repeat vs. new devices, first-ever transactions, and multiple customers
    interleaved in time. Timestamps are distinct per customer to keep window
    boundaries unambiguous.
    """
    base = pd.Timestamp("2026-01-01 00:00:00")
    rows = [
        # customer A: a tight burst then a big gap
        ("A", 0,     100.0, "devA1", 10, "electronics"),
        ("A", 120,   50.0,  "devA1", 10, "electronics"),   # +2m, same device
        ("A", 300,   25.0,  "devA2", 11, "grocery"),       # +5m, NEW device
        ("A", 3600,  200.0, "devA1", 10, "electronics"),   # +60m, old device back
        ("A", 90000, 75.0,  "devA3", 12, "crypto"),        # +25h, NEW device, windows empty
        # customer B: interleaved, steady cadence
        ("B", 60,    500.0, "devB1", 20, "travel"),
        ("B", 600,   500.0, "devB1", 20, "travel"),        # +10m exactly (boundary)
        ("B", 1200,  10.0,  "devB2", 21, "grocery"),       # NEW device
        ("B", 2000,  10.0,  "devB1", 20, "travel"),
        # customer C: single transaction ever (first-txn sentinel path)
        ("C", 500,   42.0,  "devC1", 30, "grocery"),
    ]
    df = pd.DataFrame(
        [
            {
                "transaction_id": f"TXN_{i:04d}",
                "customer_id": c,
                "timestamp": base + pd.Timedelta(seconds=s),
                "Amount": amt,
                "device_id": dev,
                "merchant_id": mid,
                "merchant_category": cat,
                "Class": 0,
            }
            for i, (c, s, amt, dev, mid, cat) in enumerate(rows)
        ]
    )
    return df


def make_toy_risk_maps(df: pd.DataFrame) -> dict:
    """A fixed merchant-risk map. Merchant risk is a static offline lookup, not
    streaming state, so both paths just need the SAME map — the exact values
    don't matter for parity, only that batch and streaming agree.
    """
    merchants = sorted(df[FEATURES.merchant_col].unique())
    categories = sorted(df[FEATURES.merchant_category_col].unique())
    global_mean = 0.05
    return {
        "merchant_risk": {m: 0.01 * (i + 1) for i, m in enumerate(merchants)},
        "category_risk": {c: 0.02 * (i + 1) for i, c in enumerate(categories)},
        "global_mean": global_mean,
    }


# ---- feature columns compared ----------------------------------------------

def stateful_feature_cols():
    cols = ["seconds_since_last_txn", "is_new_device", "distinct_device_count_so_far",
            "merchant_risk_score", "merchant_category_risk_score"]
    for w in FEATURES.velocity_windows_minutes:
        cols += [f"velocity_count_{w}m", f"velocity_amount_{w}m"]
    return cols


def batch_features(df: pd.DataFrame, risk_maps: dict) -> pd.DataFrame:
    """Run the OFFLINE pipeline pieces exactly as build_feature_pipeline does,
    with an empty known-device map so device history comes purely from the
    stream (the streaming store also starts from empty state).
    """
    d = df.copy()
    d = add_time_since_last_transaction(d)
    d = add_transaction_velocity(d)
    d = apply_merchant_risk_map(d, risk_maps)
    d = add_device_consistency(d, known_devices={})
    return d.set_index("transaction_id")


def streaming_features(df: pd.DataFrame, store) -> pd.DataFrame:
    """Run the ONLINE path: process transactions in timestamp order, calling
    get_features (state as of now, excluding current txn) then commit.
    """
    out = {}
    for row in df.sort_values("timestamp").itertuples():
        txn = {
            "transaction_id": row.transaction_id,
            "customer_id": row.customer_id,
            "timestamp": row.timestamp,
            "Amount": row.Amount,
            "device_id": row.device_id,
            "merchant_id": row.merchant_id,
            "merchant_category": row.merchant_category,
        }
        feats = store.get_features(txn)
        store.commit(txn)
        out[row.transaction_id] = feats
    return pd.DataFrame.from_dict(out, orient="index")


def assert_parity(batch: pd.DataFrame, stream: pd.DataFrame):
    for col in stateful_feature_cols():
        b = batch[col].astype(float)
        s = stream[col].astype(float).reindex(b.index)
        # tiny tolerance on summed float amounts; counts/flags are exact
        np.testing.assert_allclose(
            s.values, b.values, rtol=1e-9, atol=1e-6,
            err_msg=f"online vs batch mismatch in column {col!r}",
        )


# ---- tests ------------------------------------------------------------------

def test_in_memory_matches_batch():
    df = make_toy_stream()
    risk_maps = make_toy_risk_maps(df)
    batch = batch_features(df, risk_maps)
    store = InMemoryFeatureStore(MerchantRisk(risk_maps))
    stream = streaming_features(df, store)
    assert_parity(batch, stream)


def test_first_txn_uses_sentinel_not_zero():
    df = make_toy_stream()
    risk_maps = make_toy_risk_maps(df)
    store = InMemoryFeatureStore(MerchantRisk(risk_maps))
    stream = streaming_features(df, store)
    # customer C has exactly one txn; its first (only) txn is first-ever
    first_txns = (
        df.sort_values("timestamp").groupby("customer_id").head(1)["transaction_id"]
    )
    assert (stream.loc[first_txns, "seconds_since_last_txn"] == 1e6).all()


def test_redis_backend_matches_batch_fakeredis():
    """Exercises the actual RedisFeatureStore code (sorted sets, pipelines,
    MULTI commits) against an in-process fake Redis, so the Redis path is
    covered even when no real server is available."""
    fakeredis = pytest.importorskip("fakeredis")
    df = make_toy_stream()
    risk_maps = make_toy_risk_maps(df)
    batch = batch_features(df, risk_maps)
    store = RedisFeatureStore(MerchantRisk(risk_maps), fakeredis.FakeStrictRedis())
    stream = streaming_features(df, store)
    assert_parity(batch, stream)


def _redis_client_or_skip():
    redis = pytest.importorskip("redis")
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.Redis.from_url(url)
        client.ping()
    except Exception:
        pytest.skip("no reachable Redis server for parity test")
    return client


def test_redis_matches_batch():
    client = _redis_client_or_skip()
    df = make_toy_stream()
    risk_maps = make_toy_risk_maps(df)
    batch = batch_features(df, risk_maps)
    # isolate this test's keys
    for k in client.scan_iter("fs:*"):
        client.delete(k)
    store = RedisFeatureStore(MerchantRisk(risk_maps), client)
    stream = streaming_features(df, store)
    assert_parity(batch, stream)
    for k in client.scan_iter("fs:*"):
        client.delete(k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
