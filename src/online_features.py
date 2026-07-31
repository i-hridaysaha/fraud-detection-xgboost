"""
online_features.py — real-time, stateful feature store for serving.

The batch pipeline in feature_engineering.py computes velocity / recency /
device features by scanning the WHOLE dataset. That's correct for training but
cannot run live at 1M+ txns/day: at authorization time you have one transaction
and milliseconds to score it, not the customer's full history in a DataFrame.

This module maintains that history as incrementally-updated per-customer state
and serves the same features the batch pipeline produces — with strict
train/serve parity (proven by tests/test_streaming_parity.py).

Two-phase, causal update contract
---------------------------------
    feats = store.get_features(txn)   # state AS OF NOW, excluding this txn
    ...score...
    store.commit(txn)                 # fold this txn in, for FUTURE txns

get_features never sees the current transaction in its own state, exactly
mirroring the batch semantics (velocity excludes the current row; is_new_device
is evaluated against prior devices only). commit is what makes a transaction
count for the *next* one. Doing it in this order is what preserves causality —
the same guarantee the offline pipeline gets from sorting + shifting.

Backends
--------
- RedisFeatureStore: production path. Sliding-window velocity via sorted sets
  (ZADD/ZRANGEBYSCORE/ZREMRANGEBYSCORE), last-txn string, device set. Commits
  are atomic per customer via a MULTI pipeline. TTLs keep memory bounded.
- InMemoryFeatureStore: pure-Python fallback with identical behaviour and a
  per-customer lock, so the repo runs and tests pass with no Redis server.

Select the backend with FEATURE_STORE_BACKEND=redis|memory (see get_feature_store).
"""

import os
import sys
import threading
from collections import defaultdict
from typing import Dict, Optional

import joblib
import pandas as pd

sys.path.append(".")
from src.config import FEATURES, PATHS
from src.feature_engineering import FIRST_TXN_SENTINEL, velocity_window_seconds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_epoch(ts) -> float:
    """Normalise any accepted timestamp form (epoch float, ISO string,
    pd.Timestamp, datetime) to epoch seconds — the score space used for the
    velocity sorted sets. Using one converter everywhere keeps batch and
    streaming on the same clock.
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    return pd.Timestamp(ts).timestamp()


class MerchantRisk:
    """Static merchant-risk lookup, identical to apply_merchant_risk_map /
    the API's _build_feature_row: smoothed target-encoded fraud rate per
    merchant (and per category) fit offline on the training split, with a
    global-mean fallback for unseen entities. This is NOT streaming state —
    it's a fixed map refreshed on retrain — so it's just a fast dict lookup.
    """

    def __init__(self, risk_maps: dict):
        self._merchant = risk_maps["merchant_risk"]
        self._category = risk_maps["category_risk"]
        self._global_mean = risk_maps["global_mean"]

    @classmethod
    def load(cls, path: str = None) -> "MerchantRisk":
        return cls(joblib.load(path or PATHS.merchant_risk_map_path))

    def scores(self, merchant_id, merchant_category) -> Dict[str, float]:
        return {
            "merchant_risk_score": self._merchant.get(merchant_id, self._global_mean),
            "merchant_category_risk_score": self._category.get(
                merchant_category, self._global_mean
            ),
        }


class BaseFeatureStore:
    """Common feature assembly. Backends implement only the raw state
    primitives (_velocity_state, _last_txn_epoch, _device_state, and their
    commit counterparts); this base turns them into the model's feature row so
    the two backends can never disagree on feature *definitions*.
    """

    def __init__(self, merchant_risk: MerchantRisk):
        self.merchant_risk = merchant_risk
        self.window_seconds = velocity_window_seconds()

    # ---- feature assembly (shared) ----

    def get_features(self, txn: dict) -> Dict[str, float]:
        """Full stateful feature vector for `txn`, computed from state as of now
        and EXCLUDING txn itself. PCA (V1-V28) and Amount are supplied by the
        caller/request — they aren't stateful — so they're not returned here.
        """
        cust = txn[FEATURES.customer_col]
        now = _to_epoch(txn["timestamp"])
        device = txn[FEATURES.device_col]

        feats: Dict[str, float] = {}

        # velocity: count + summed amount over each trailing window, priors only
        vel = self._velocity_state(cust, now)
        for w in FEATURES.velocity_windows_minutes:
            count, amount_sum = vel[w]
            feats[f"velocity_count_{w}m"] = float(count)
            feats[f"velocity_amount_{w}m"] = float(max(amount_sum, 0.0))

        # recency: sentinel on first-ever txn, never 0
        last = self._last_txn_epoch(cust)
        feats["seconds_since_last_txn"] = (
            FIRST_TXN_SENTINEL if last is None else float(now - last)
        )

        # device consistency: evaluate is_new BEFORE this device is committed
        seen, distinct_prior = self._device_state(cust, device)
        feats["is_new_device"] = int(not seen)
        feats["distinct_device_count_so_far"] = int(
            distinct_prior + (0 if seen else 1)
        )

        feats.update(
            self.merchant_risk.scores(
                txn[FEATURES.merchant_col], txn[FEATURES.merchant_category_col]
            )
        )
        return feats

    def commit(self, txn: dict) -> None:
        """Fold `txn` into per-customer state so it counts for future txns."""
        raise NotImplementedError

    # ---- state primitives (backend-specific) ----

    def _velocity_state(self, cust, now) -> Dict[int, tuple]:
        raise NotImplementedError

    def _last_txn_epoch(self, cust) -> Optional[float]:
        raise NotImplementedError

    def _device_state(self, cust, device) -> tuple:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend (no Redis required)
# ---------------------------------------------------------------------------

class InMemoryFeatureStore(BaseFeatureStore):
    """Process-local state. Behaviourally identical to the Redis backend; used
    for tests and for running the repo without a Redis server. A per-customer
    lock makes commits atomic under concurrent workers (mirrors Redis MULTI).
    """

    def __init__(self, merchant_risk: MerchantRisk):
        super().__init__(merchant_risk)
        self._events = defaultdict(list)      # cust -> list[(epoch, amount)]
        self._last = {}                        # cust -> epoch
        self._devices = defaultdict(set)       # cust -> {device_id}
        self._locks = defaultdict(threading.Lock)
        self._max_window = max(self.window_seconds.values())

    def _velocity_state(self, cust, now):
        events = self._events.get(cust, [])
        out = {}
        for w in FEATURES.velocity_windows_minutes:
            lower = now - self.window_seconds[w]
            # left-exclusive, right-inclusive to match pandas rolling(closed="right")
            in_window = [a for (ts, a) in events if lower < ts <= now]
            out[w] = (len(in_window), sum(in_window))
        return out

    def _last_txn_epoch(self, cust):
        return self._last.get(cust)

    def _device_state(self, cust, device):
        devs = self._devices.get(cust, set())
        return (device in devs, len(devs))

    def commit(self, txn: dict) -> None:
        cust = txn[FEATURES.customer_col]
        now = _to_epoch(txn["timestamp"])
        amount = float(txn[FEATURES.amount_col])
        with self._locks[cust]:
            events = self._events[cust]
            events.append((now, amount))
            # trim anything older than the largest window keeps memory bounded
            cutoff = now - self._max_window
            if events and events[0][0] <= cutoff:
                self._events[cust] = [(ts, a) for (ts, a) in events if ts > cutoff]
            self._last[cust] = now
            self._devices[cust].add(txn[FEATURES.device_col])


# ---------------------------------------------------------------------------
# Redis backend (production path)
# ---------------------------------------------------------------------------

class RedisFeatureStore(BaseFeatureStore):
    """Redis-backed sliding-window aggregator.

    Key schema (per customer_id):
      fs:vel:{cid}  sorted set  score = txn epoch seconds
                                member = "{epoch}|{amount}|{txn_id}"
                    -> ZRANGEBYSCORE (now-w, now] gives the window's members;
                       count = len, amount = sum of the parsed amount field.
                       The txn_id keeps members unique when a customer has two
                       txns at the same epoch+amount.
      fs:last:{cid} string      last txn epoch (seconds_since_last_txn source)
      fs:dev:{cid}  set         distinct device_ids seen so far

    Why sorted sets: a time-scored ZSET is the natural sliding window — old
    events age out with one ZREMRANGEBYSCORE and any window is a single
    range query, all O(log N + K). TTLs bound memory for dormant customers.
    """

    # velocity events only need to live as long as the largest window; last-txn
    # and device history are longer-lived signals, kept for a rolling horizon.
    DEVICE_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days

    def __init__(self, merchant_risk: MerchantRisk, client):
        super().__init__(merchant_risk)
        self.r = client
        self._max_window = max(self.window_seconds.values())

    @staticmethod
    def _k_vel(cust): return f"fs:vel:{cust}"

    @staticmethod
    def _k_last(cust): return f"fs:last:{cust}"

    @staticmethod
    def _k_dev(cust): return f"fs:dev:{cust}"

    def _velocity_state(self, cust, now):
        key = self._k_vel(cust)
        out = {}
        # one range read per window; each is left-exclusive, right-inclusive
        pipe = self.r.pipeline(transaction=False)
        windows = FEATURES.velocity_windows_minutes
        for w in windows:
            lower = now - self.window_seconds[w]
            pipe.zrangebyscore(key, f"({lower}", now)
        results = pipe.execute()
        for w, members in zip(windows, results):
            total = 0.0
            for m in members:
                m = m.decode() if isinstance(m, bytes) else m
                # member = "epoch|amount|txn_id"
                total += float(m.split("|", 2)[1])
            out[w] = (len(members), total)
        return out

    def _last_txn_epoch(self, cust):
        v = self.r.get(self._k_last(cust))
        if v is None:
            return None
        return float(v.decode() if isinstance(v, bytes) else v)

    def _device_state(self, cust, device):
        key = self._k_dev(cust)
        pipe = self.r.pipeline(transaction=False)
        pipe.sismember(key, device)
        pipe.scard(key)
        seen, count = pipe.execute()
        return (bool(seen), int(count))

    def commit(self, txn: dict) -> None:
        cust = txn[FEATURES.customer_col]
        now = _to_epoch(txn["timestamp"])
        amount = float(txn[FEATURES.amount_col])
        txn_id = txn.get("transaction_id", "")
        vel_key = self._k_vel(cust)
        member = f"{now}|{amount}|{txn_id}"

        # MULTI makes the whole commit atomic for this customer, so concurrent
        # workers can't interleave a partial update.
        pipe = self.r.pipeline(transaction=True)
        pipe.zadd(vel_key, {member: now})
        pipe.zremrangebyscore(vel_key, "-inf", now - self._max_window)
        pipe.expire(vel_key, int(self._max_window * 2))
        pipe.set(self._k_last(cust), now, ex=self.DEVICE_TTL_SECONDS)
        pipe.sadd(self._k_dev(cust), txn[FEATURES.device_col])
        pipe.expire(self._k_dev(cust), self.DEVICE_TTL_SECONDS)
        pipe.execute()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_feature_store(
    backend: str = None,
    merchant_risk: MerchantRisk = None,
    redis_url: str = None,
) -> BaseFeatureStore:
    """Build the feature store selected by config/env.

    FEATURE_STORE_BACKEND = "redis" | "memory" (default "memory" so the repo
    runs with zero infra). REDIS_URL configures the Redis connection.
    """
    backend = (backend or os.getenv("FEATURE_STORE_BACKEND", "memory")).lower()
    merchant_risk = merchant_risk or MerchantRisk.load()

    if backend == "redis":
        import redis
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url)
        client.ping()  # fail fast if Redis is unreachable
        return RedisFeatureStore(merchant_risk, client)

    if backend == "memory":
        return InMemoryFeatureStore(merchant_risk)

    raise ValueError(f"Unknown FEATURE_STORE_BACKEND: {backend!r} (use 'redis' or 'memory')")
