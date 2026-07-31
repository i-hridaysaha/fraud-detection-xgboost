# Load test

Proves the store-backed `/score` latency claim under concurrency. Two runners
hit the same endpoint:

- **`loadtest.py`** (asyncio + httpx) — scripted, deterministic; warms the
  store, drives a fixed number of requests, and writes `results.md` with
  throughput and p50/p95/p99. This is what produced [results.md](results.md).
- **`locustfile.py`** — interactive Locust UI for ramping users by hand.

## Run it

Start the API (Redis-backed via docker-compose, or in-memory locally):

```bash
# in-memory, single worker (matches results.md)
FEATURE_STORE_BACKEND=memory uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Then, in another shell:

```bash
python loadtest/loadtest.py \
    --host http://127.0.0.1:8000 \
    --customers 2000 --warmup-per-customer 5 \
    --requests 20000 --concurrency 8 \
    --hardware "your machine" --backend "in-memory" --server "uvicorn 1 worker"
```

Or interactively:

```bash
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
# open http://localhost:8089
```

## Notes

- The load test **warms per-customer history first** so velocity windows are
  populated — otherwise the benchmark would measure an unrealistically empty
  store.
- Each `/score` request runs score-then-commit (`ingest: true`), so state grows
  during the run exactly as it would in production.
- Throughput is bounded by a single CPU-bound Python worker; see the scaling
  section in [results.md](results.md) for the multi-worker + Redis path.
