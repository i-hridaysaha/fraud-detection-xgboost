"""
feature_engineering.py

All features here are computed CAUSALLY: for any given transaction, a
feature only uses information available strictly BEFORE that transaction's
timestamp. This is essential for fraud detection — at inference time you
will never have access to "future" transactions from the same customer, so
training on features that peek ahead would produce a model that looks great
offline and fails in production.

Feature groups implemented:
  1. Transaction velocity  (rolling count/sum of a customer's txns in windows)
  2. Time since last transaction (per customer)
  3. Merchant risk score (historical fraud rate per merchant, causally computed)
  4. Device consistency (is this the customer's usual device? how many
     distinct devices has this customer used so far?)
"""

import numpy as np
import pandas as pd
import joblib
import sys

sys.path.append(".")
from src.config import FEATURES, PATHS

# Single source of truth for the two definitions the online (streaming) feature
# store must reproduce exactly. Both the batch pipeline below and
# src/online_features.py import these so the two paths can never silently drift.
#
# FIRST_TXN_SENTINEL: a customer's first-ever transaction has no prior history,
# so seconds_since_last_txn gets a large value (not 0 — 0 would falsely read as
# "just transacted a moment ago").
FIRST_TXN_SENTINEL = 1e6


def velocity_window_seconds() -> dict:
    """Maps each configured velocity window (in minutes) to its width in
    seconds. Kept here, next to the batch rolling logic, so the streaming
    aggregator windows against identical boundaries.
    """
    return {w: w * 60 for w in FEATURES.velocity_windows_minutes}


def add_time_since_last_transaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    df["prev_txn_time"] = df.groupby(FEATURES.customer_col)["timestamp"].shift(1)
    df["seconds_since_last_txn"] = (
        (df["timestamp"] - df["prev_txn_time"]).dt.total_seconds()
    )
    # First-ever transaction for a customer: no prior history. Use a large
    # sentinel value rather than 0 (0 would falsely look "just transacted").
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(FIRST_TXN_SENTINEL)
    df = df.drop(columns=["prev_txn_time"])
    return df


def add_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """For each window (e.g. 10m/1h/24h), count how many transactions and
    total $ amount this customer made in the preceding window, EXCLUDING
    the current transaction itself.
    """
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    df = df.set_index("timestamp")

    velocity_frames = []
    for w in FEATURES.velocity_windows_minutes:
        window_str = f"{w}min"
        grouped = df.groupby(FEATURES.customer_col)
        # rolling count/sum over trailing window, then shift by 1 txn to
        # exclude the current row from its own window
        count_col = f"velocity_count_{w}m"
        sum_col = f"velocity_amount_{w}m"

        roll_count = grouped[FEATURES.amount_col].rolling(window_str).count()
        roll_sum = grouped[FEATURES.amount_col].rolling(window_str).sum()

        roll_count.index = roll_count.index.droplevel(0)
        roll_sum.index = roll_sum.index.droplevel(0)

        # subtract self (rolling window is inclusive of current row)
        velocity_frames.append((count_col, roll_count - 1))
        velocity_frames.append((sum_col, roll_sum - df[FEATURES.amount_col]))

    df = df.reset_index()
    for col_name, series in velocity_frames:
        df[col_name] = series.reset_index(drop=True).clip(lower=0)

    return df


def fit_merchant_risk_map(train_df: pd.DataFrame, target_col: str = "Class",
                           smoothing: float = 20.0) -> dict:
    """Fits a smoothed target-encoding of historical fraud rate per merchant,
    using ONLY the training split (never val/test) to avoid leakage.
    Smoothing pulls sparse merchants' estimates toward the global mean.
    """
    global_rate = train_df[target_col].mean()
    stats = train_df.groupby(FEATURES.merchant_col)[target_col].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_rate * smoothing) / (stats["count"] + smoothing)
    risk_map = smoothed.to_dict()
    risk_map["__global_mean__"] = global_rate

    category_stats = train_df.groupby(FEATURES.merchant_category_col)[target_col].mean().to_dict()

    return {"merchant_risk": risk_map, "category_risk": category_stats, "global_mean": global_rate}


def apply_merchant_risk_map(df: pd.DataFrame, risk_maps: dict) -> pd.DataFrame:
    global_mean = risk_maps["global_mean"]
    df["merchant_risk_score"] = df[FEATURES.merchant_col].map(risk_maps["merchant_risk"]).fillna(global_mean)
    df["merchant_category_risk_score"] = (
        df[FEATURES.merchant_category_col].map(risk_maps["category_risk"]).fillna(global_mean)
    )
    return df


def fit_device_map(train_df: pd.DataFrame) -> dict:
    """Records, per customer, the set of devices seen in TRAINING data —
    used to derive whether a future device is 'new' relative to history
    known at model-fit time. At inference time this is combined with a
    causal running count within the val/test period itself (see
    add_device_consistency), so a device that's new in training but becomes
    familiar over time in later data is still tracked correctly.
    """
    device_sets = train_df.groupby(FEATURES.customer_col)[FEATURES.device_col].apply(set).to_dict()
    return device_sets


def add_device_consistency(df: pd.DataFrame, known_devices: dict) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    seen_devices_running = {}
    is_new_device = np.zeros(len(df), dtype=int)
    distinct_device_count = np.zeros(len(df), dtype=int)

    for i, row in enumerate(df.itertuples()):
        cust = getattr(row, FEATURES.customer_col)
        dev = getattr(row, FEATURES.device_col)

        prior_known = known_devices.get(cust, set()) | seen_devices_running.get(cust, set())
        is_new_device[i] = int(dev not in prior_known)
        seen_devices_running.setdefault(cust, set()).add(dev)
        distinct_device_count[i] = len(prior_known | {dev})

    df["is_new_device"] = is_new_device
    df["distinct_device_count_so_far"] = distinct_device_count
    return df


def build_feature_pipeline(train_df, val_df, test_df):
    """Fits leakage-safe encoders on train only, then applies feature
    engineering consistently across train/val/test.
    """
    risk_maps = fit_merchant_risk_map(train_df)
    device_map = fit_device_map(train_df)

    processed = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d = part.copy()
        d = add_time_since_last_transaction(d)
        d = add_transaction_velocity(d)
        d = apply_merchant_risk_map(d, risk_maps)
        d = add_device_consistency(d, device_map)
        processed[name] = d

    joblib.dump(risk_maps, PATHS.merchant_risk_map_path)
    joblib.dump(device_map, PATHS.device_map_path)

    return processed["train"], processed["val"], processed["test"], risk_maps, device_map


def get_feature_columns() -> list:
    velocity_cols = []
    for w in FEATURES.velocity_windows_minutes:
        velocity_cols += [f"velocity_count_{w}m", f"velocity_amount_{w}m"]

    cols = (
        FEATURES.pca_cols
        + [FEATURES.amount_col, "seconds_since_last_txn"]
        + velocity_cols
        + ["merchant_risk_score", "merchant_category_risk_score",
           "is_new_device", "distinct_device_count_so_far"]
    )
    return cols
