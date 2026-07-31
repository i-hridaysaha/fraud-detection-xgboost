"""
locustfile.py — interactive alternative to loadtest.py, for exploring the
latency/throughput curve with Locust's live web UI.

The scripted loadtest.py is what produces loadtest/results.md (deterministic,
reports p50/p95/p99 in one shot). Use this when you want to ramp users
interactively and watch the distribution move.

Run:
    locust -f loadtest/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089

Each simulated user warms one customer's history on start, then repeatedly
scores store-backed transactions for that customer (score-then-ingest).
"""

import random
import time
from datetime import datetime, timezone

from locust import HttpUser, task, between

CATEGORIES = ["grocery", "electronics", "travel", "crypto", "gambling", "restaurant"]


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _features():
    return {f"V{i}": random.gauss(0, 1) for i in range(1, 29)}


class FraudScoringUser(HttpUser):
    wait_time = between(0.0, 0.05)

    def on_start(self):
        # give this virtual user its own customer with some recent history so
        # velocity lookups are non-empty from the first scored request
        self.customer_id = f"CUST_{random.randint(0, 100000):06d}"
        now = time.time()
        for _ in range(5):
            self.client.post("/ingest", json={
                "transaction_id": f"WARM_{self.customer_id}_{random.random()}",
                "customer_id": self.customer_id,
                "merchant_id": random.randint(1, 500),
                "merchant_category": random.choice(CATEGORIES),
                "device_id": f"{self.customer_id}_dev{random.randint(0,2)}",
                "amount": round(random.uniform(1, 800), 2),
                "timestamp": _iso(now - random.uniform(0, 24 * 3600)),
            })

    @task
    def score(self):
        self.client.post("/score", json={
            "transaction_id": f"LOAD_{random.random()}",
            "customer_id": self.customer_id,
            "merchant_id": random.randint(1, 500),
            "merchant_category": random.choice(CATEGORIES),
            "device_id": f"{self.customer_id}_dev{random.randint(0,3)}",
            "amount": round(random.uniform(1, 800), 2),
            "timestamp": _iso(time.time()),
            "use_feature_store": True,
            "ingest": True,
            "features": _features(),
        })
