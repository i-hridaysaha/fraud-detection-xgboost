# Enterprise Fraud Detection Platform

A production-style credit card fraud detection pipeline built around XGBoost,
designed for the scale and constraints of a real payments environment
(1M+ transactions/day, ~0.1-0.3% fraud rate, sub-100ms scoring latency).

This repo demonstrates the full lifecycle: data → leakage-safe feature
engineering → imbalance handling → model comparison → threshold tuning →
SHAP explainability → API serving → drift monitoring.

## Results

Results below are on synthetic data (see [Data](#about-the-data) section) — the
PCA-style features are randomly generated with only a modest fraud/legit
signal, which caps absolute precision. The pipeline's design (leakage-safe
splits, causal features, threshold tuning, SHAP) is what should be evaluated
here; running the same code on the real Kaggle dataset (see below) would be
expected to yield meaningfully stronger separation, since real fraud leaves a
much stronger signal in the underlying PCA components than this synthetic
approximation does.

![SHAP Summary](reports/shap_summary.png)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.542 | 0.101 | 0.026 | 0.465 | 0.050 |
| Random Forest | 0.756 | 0.127 | 0.050 | 0.697 | 0.093 |
| XGBoost (tuned threshold, F1-optimal) | 0.723 | 0.110 | 0.067 | 0.242 | 0.105 |
| XGBoost (default 0.5 threshold) | 0.723 | 0.110 | 0.052 | 0.560 | 0.095 |

**Reading these results:** XGBoost gives the best PR-AUC/F1 tradeoff overall.
Random Forest achieves the highest recall (catches 70% of fraud) at the cost
of a very high false-positive rate — illustrating the precision/recall
tradeoff a real fraud team would tune based on analyst review capacity, which
is exactly what the threshold-tuning logic in `src/imbalance.py` is for.

## Why this isn't "just call `.fit()`"

Fraud detection has a few properties that make naive ML approaches actively
dangerous in production, and this repo is structured specifically to avoid
them:

1. **Extreme class imbalance** (~0.1-0.3% positive rate) — accuracy is a
   useless metric here (99.8% accuracy = predict "not fraud" always).
   We report Precision/Recall/F1/PR-AUC, not accuracy.
2. **Temporal leakage** — a random train/test split leaks future
   information (e.g. rolling features computed with future transactions).
   Every split and every feature in this pipeline is **time-based and
   causal** — see `src/data_loader.py::time_based_split` and the docstring
   in `src/feature_engineering.py`.
3. **Delayed ground truth** — confirmed fraud (chargebacks) often arrives
   days or weeks after a transaction. `src/monitoring.py` treats data drift
   (available immediately) and concept drift / performance decay (only
   available once delayed labels arrive) as two separate monitoring
   problems, because they are.
4. **Explainability is a compliance requirement**, not a nice-to-have, in
   financial services. SHAP is used both for global model validation and
   for per-transaction analyst-facing explanations.

## Project structure

```
fraud_detection_platform/
├── data/
│   └── generate_synthetic_data.py   # synthetic Kaggle-schema-compatible data generator
├── src/
│   ├── config.py               # all tunable parameters in one place
│   ├── data_loader.py           # loading + time-based split
│   ├── feature_engineering.py   # velocity, recency, merchant risk, device consistency
│   ├── imbalance.py             # class weighting, SMOTE, threshold tuning
│   ├── models.py                # LR / RF / XGBoost factories
│   ├── evaluate.py              # precision/recall/F1/AUC reporting
│   ├── explainability.py        # SHAP (global + per-transaction)
│   ├── online_features.py       # real-time streaming feature store (Redis + in-memory)
│   ├── api.py                   # FastAPI real-time scoring service
│   └── monitoring.py            # PSI-based data drift + delayed performance tracking
├── tests/
│   ├── test_pipeline.py         # leakage/sanity tests
│   └── test_streaming_parity.py # proves online features == batch features
├── loadtest/                    # concurrency load test + real latency results
│   ├── loadtest.py              # asyncio+httpx scripted runner
│   ├── locustfile.py            # interactive Locust runner
│   └── results.md               # actual RPS + p50/p95/p99 from a real run
├── train.py                     # end-to-end training entry point
├── Dockerfile
├── docker-compose.yml           # Redis + API together
├── requirements.txt
└── reports/                     # generated metrics + SHAP plots land here
```

## About the data

The canonical public dataset for this problem is the Kaggle
["Credit Card Fraud Detection"](https://www.kaggle.com/mlg-ulb/creditcardfraud)
dataset: ~285K European cardholder transactions from September 2013, with
`Time`, 28 PCA-anonymized features (`V1`-`V28`), `Amount`, and `Class`
(1 = fraud). It's the standard benchmark for this problem, but it is
**fully anonymized** — no customer, merchant, or device identifiers — so it
cannot, by itself, demonstrate velocity, merchant-risk, or
device-consistency features, which need entity identifiers to compute.

This repo ships `data/generate_synthetic_data.py`, which:
- generates transactions with the **same schema** as the Kaggle set
  (`Time`, `V1`-`V28`, `Amount`, `Class`), with fraud/legit distributions
  modeled after the real dataset's known structure (fraud separates from
  legitimate transactions on a handful of the PCA components), **and**
- augments every row with `customer_id`, `merchant_id`,
  `merchant_category`, `device_id`, and a real `timestamp`, so the full
  feature set can be built and demonstrated end-to-end.

**To use the real Kaggle data instead:** download `creditcard.csv` from the
link above, place it at `data/transactions.csv`, and run `train.py` as
normal — `src/data_loader.py` detects that the file lacks identity columns
and synthesizes them on top of the *real* `Amount`/`V1-V28`/`Class` values,
using the exact same logic as the synthetic generator. This keeps the fraud
signal genuine while still enabling the full feature set. This is a
reasonable stand-in for a real internal data warehouse join (in production,
`customer_id`/`merchant_id`/`device_id` would come from your transaction
system, not be synthesized).

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (or supply your own transactions.csv — see above)
python data/generate_synthetic_data.py --n_rows 500000 --fraud_rate 0.0017

# 2. Train all three models, evaluate, and generate SHAP report
python train.py --imbalance_strategy class_weight
#    (or --imbalance_strategy smote / none)

# 3. Serve the trained XGBoost model
#    (in-memory feature store, no infra needed)
FEATURE_STORE_BACKEND=memory uvicorn src.api:app --host 0.0.0.0 --port 8000
#    ...or Redis-backed API + Redis together:
docker compose up --build

# 4. Run tests
pytest tests/ -v

# 5. Load test the store-backed /score path (writes loadtest/results.md)
python loadtest/loadtest.py --host http://localhost:8000
```

## Feature engineering

All features are computed **causally** (using only information available
strictly before each transaction's timestamp):

| Feature | Description |
|---|---|
| `seconds_since_last_txn` | Time since this customer's previous transaction. First-ever transaction gets a large sentinel value, not 0. |
| `velocity_count_{10,60,1440}m` | Count of this customer's transactions in the trailing 10min/1hr/24hr window, excluding the current transaction. |
| `velocity_amount_{10,60,1440}m` | Sum of `$` amount in the same trailing windows. |
| `merchant_risk_score` | Smoothed target-encoded historical fraud rate per merchant, fit on the training split only. |
| `merchant_category_risk_score` | Same, aggregated at the merchant-category level (groceries vs. crypto exchange vs. gambling, etc.). |
| `is_new_device` | Whether this is a device not previously seen for this customer (as of this point in time). |
| `distinct_device_count_so_far` | Running count of distinct devices used by this customer. |

## Imbalance handling

Configurable via `--imbalance_strategy`:
- `class_weight` (default): `scale_pos_weight` for XGBoost / `class_weight="balanced"` for sklearn models — reweights the loss rather than touching the data.
- `smote`: oversamples the minority class via SMOTE **after** the time-based split (never before — fitting SMOTE before splitting would leak synthetic points derived from val/test fraud examples into training).
- `none`: no correction, useful as a baseline to see how much the above actually help.

Decision threshold is tuned on the **validation set only** (`src/imbalance.py::tune_threshold_for_f1`), reporting both the F1-optimal threshold and a precision-constrained "high recall" alternative (useful when the business wants to cap analyst false-positive review load at a minimum precision).

## Models compared

1. **Logistic Regression** — fast, interpretable baseline / sanity check that the features carry signal at all.
2. **Random Forest** — stronger nonlinear baseline, useful cross-check for feature importance.
3. **XGBoost** — final production model. Chosen for its accuracy/latency tradeoff at high transaction volume, native `scale_pos_weight` support, and first-class SHAP `TreeExplainer` support (exact, fast SHAP values rather than approximations).

`train.py` trains and evaluates all three on the same held-out, chronologically-last test split and writes a comparison table to `reports/model_comparison.json`.

## Explainability

`src/explainability.py` uses SHAP's `TreeExplainer` for two purposes:
- **Global**: `reports/shap_summary.png` and `reports/shap_top_features.csv` — which features drive the model overall (for model governance / validation review).
- **Local**: `explain_single_transaction()` — surfaces the top contributing factors for one transaction, returned directly in the `/score` API response when `explain: true` is passed. This is what a fraud analyst's alert-triage UI would show.

## Real-time feature store

The velocity/recency/device features are cheap to compute in batch over a whole
DataFrame but impossible to recompute per-request at 1M+ txns/day. `src/online_features.py`
is the online counterpart: a **stateful streaming aggregator** that maintains
per-customer state incrementally and serves the exact same features the batch
pipeline produces.

**Two-phase, causal contract.** Every transaction is handled in two steps:

```python
feats = store.get_features(txn)   # state AS OF NOW, excluding this txn
# ...score the model...
store.commit(txn)                 # fold this txn in, for FUTURE txns
```

`get_features` never sees the current transaction in its own state — velocity
excludes the current row, `is_new_device` is evaluated against prior devices
only — exactly mirroring the batch semantics (which get the same guarantee from
sort + shift). `commit` is what makes a transaction count for the *next* one.
This ordering is the causality guarantee; scoring always happens before the
commit.

**Redis data structures** (per `customer_id`):

| Key | Type | Purpose |
|---|---|---|
| `fs:vel:{cid}` | **sorted set**, score = txn epoch seconds | Sliding-window velocity. `ZRANGEBYSCORE (now-w, now]` gives a window's events in `O(log N + K)`; `ZREMRANGEBYSCORE` ages out anything past the largest window. Member is `"{epoch}|{amount}|{txn_id}"` so amounts are recoverable for windowed sums and members stay unique. |
| `fs:last:{cid}` | string | Last txn epoch → `seconds_since_last_txn` (large sentinel, never 0, on first-ever txn). |
| `fs:dev:{cid}` | set | Distinct device ids → `is_new_device` + `distinct_device_count_so_far`. |

A time-scored sorted set *is* a sliding window: old events age out with one
range delete and any window (10m/60m/1440m) is a single range query. TTLs keep
memory bounded for dormant customers. Commits run inside a `MULTI` pipeline so
concurrent workers can't interleave a partial update. Merchant risk is a static
offline lookup (from `merchant_risk_map.joblib`), not streaming state, so it
stays a plain dict lookup.

**In-memory fallback.** `InMemoryFeatureStore` implements the same interface
(with a per-customer lock) so the repo runs and all tests pass with **no Redis
server**. Select the backend with `FEATURE_STORE_BACKEND=redis|memory` and
`REDIS_URL`.

**Train/serve parity — the guardrail.** `tests/test_streaming_parity.py` runs a
fixed transaction stream through *both* the batch pipeline and the streaming
store and asserts the feature rows are identical (both backends). This is the
proof that the serving path didn't silently drift from what the model was
trained on. Batch and streaming reference a single definition of the window
widths and the first-txn sentinel (`src/feature_engineering.py`) rather than two
copies.

## Deployment

`src/api.py` is a FastAPI service exposing `POST /score`, with two ways to
supply the time-dependent features:

- **Store-backed** (`"use_feature_store": true`): the service fetches
  velocity/recency/device features from the online store above using the
  transaction's ids; the caller passes only the raw txn fields (V1-V28, Amount,
  ids, timestamp). Set `"ingest": true` to commit the txn after scoring.
- **Caller-provided** (default): the caller passes all features pre-computed —
  the original contract, kept for backward compatibility.

`POST /ingest` folds a transaction into the store without scoring (warmup /
backfill). `GET /health` is for liveness/readiness probes.

Bring up Redis + the API together with `docker compose up --build`.

### Load test / latency

`loadtest/` drives concurrent store-backed `/score` requests after warming the
store with realistic per-customer history. On an **Apple M4 laptop, single
uvicorn worker, in-memory backend**, the measured result over 20,000 requests
(warm store, concurrency 8) was:

| Throughput | p50 | p95 | p99 | errors |
|---|---|---|---|---|
| **435 RPS** | 18.1 ms | 24.2 ms | **27.8 ms** | 0 |

p99 well under the 100ms target. Throughput is bounded by a single CPU-bound
Python worker; the production config (N workers + Redis for shared state, via
`docker-compose.yml`) scales RPS roughly linearly across cores. Full numbers,
the concurrency sweep, and the scaling analysis are in
[loadtest/results.md](loadtest/results.md) — every number there is from a real
run, not hand-written.

## Monitoring

`src/monitoring.py` separates two distinct failure modes, since they need
different responses:

- **Data drift** (`compute_feature_drift_report`, `compute_prediction_drift`): Population Stability Index per feature and on the model's output score distribution. Available immediately, no ground truth needed. PSI < 0.1 stable, 0.1-0.25 investigate, > 0.25 significant shift.
- **Concept drift / performance decay** (`compute_delayed_performance_report`): recomputes precision/recall/F1 once delayed ground-truth labels (chargebacks, confirmed fraud reports) arrive for a past batch, run as a scheduled job. This is the actual retraining trigger — data drift alone doesn't necessarily mean the model got worse.

## Known limitations / next steps for a real deployment

- The synthetic data generator's fraud/legit separation is simplified; a real deployment would validate against confirmed fraud investigation outcomes, not just PR-AUC on a held-out set.
- The batch pipeline (`src/feature_engineering.py`) remains the source of truth for feature definitions during training. The live serving path uses the streaming aggregator in `src/online_features.py` (see [Real-time feature store](#real-time-feature-store)), kept in lockstep with batch by `tests/test_streaming_parity.py`. A full production deployment would back it with a managed Redis (or Flink/Kafka Streams for very high fan-in) rather than a single node.
- No hyperparameter search is included (fixed, reasonable XGBoost defaults are used) — wire in Optuna/Ray Tune before treating this as final.
- No formal model card / bias audit across customer segments is included — worth adding before production use in a regulated environment.
