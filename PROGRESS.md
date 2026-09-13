# PROGRESS.md — ProximaScale

---

## Person B — LSTM Prediction Engine

### Week 10–11 (Apr 28 – May 2, 2026)
- Fixed critical scaler persistence bug: scaler now saved/loaded via joblib
- Fixed bare imports: all model files work from project root
- Populated anomaly.py as standalone module
- Updated evaluate.py to accept optional real CSV via --data flag
- Fixed misindented print in train.py
- Switched keras imports to tf_keras for TF 2.16.1 compatibility
- Pinned all library versions in requirements.txt
- Verified: predict import OK, RMSE=5.48, MAE=4.39, chart saved

### Status
Semester 1 complete. Model loads, predicts, evaluates correctly.
Ready for Person C and Person D integration.
scaler.pkl committed alongside model weights.

---

## Person A — App + Monitoring

### Week 1–2 (Environment + Schema)
- Set up virtual environment, installed Flask, psutil, Locust
- Agreed data schema with team: `{timestamp, cpu_percent, memory_percent, request_rate}`
- Created initial `app/app.py` with CPU-load routes and request counter

### Week 3–4 (Flask App + Collector skeleton)
- Flask routes `/`, `/heavy` generating measurable CPU load via math loop
- `before_request` hook incrementing `_request_count`
- `monitoring/collector.py` skeleton — polls psutil every 5s, writes to CSV
- **Issue found:** `request_rate` was hardcoded to `0` — `/metrics` endpoint missing

### Week 5 (Fixes + Full module completion)
- Added `/metrics` JSON endpoint to `app.py` exposing live `request_rate`
- Fixed `app.run(host='0.0.0.0')` so Flask is reachable inside Docker
- Refactored `collector.py` — separated `collect_metrics(window)` from `run_collector()`
- Added `monitoring/schema.py` — MetricRecord dataclass
- Added `monitoring/storage.py` — append_row and read_last_n
- Added `app/Dockerfile`
- Added Locust scenarios: spike, gradual ramp, normal load

### TODO
- [ ] Regenerate metrics.csv — run all 3 Locust scenarios
- [ ] Target: ≥1,000 rows with real variance
- [ ] Hand off metrics.csv to Person B for LSTM training

# ProximaScale — Progress Report

## Overview
ProximaScale is a proactive autoscaling load predictor combining a multivariate
LSTM and Prophet in a hybrid ensemble, forecasting CPU load ~90 seconds ahead
of real spikes. Two novelty contributions:
- **MC Dropout uncertainty estimation** (primary novelty) — quantifies forecast
  confidence via stochastic Dropout-active forward passes (mean, std, upper bound).
- **Counterfactual correction** (original novelty) — corrects the self-masking
  bias that occurs when the system's own proactive scaling suppresses the load
  signal it's trying to learn from.

## Completed
- **Data pipeline**: chronological train/test split, MinMax scaling (fit on
  train only), sliding windows of shape (10, 3) → (3,)
- **Multivariate LSTM**: LSTM(64)→Dropout(0.2)→LSTM(32)→Dropout(0.2)→Dense(3),
  trained on cpu/memory/request-rate
- **Prophet model**: univariate CPU forecaster, 3-step horizon
- **Hybrid ensemble**: weighted blend of LSTM + Prophet forecasts
- **MC Dropout uncertainty**: 30 stochastic forward passes → mean, std, upper
  bound (mean + 2·std)
- **Anomaly detection**: rolling Z-score (threshold 2.5) on live CPU readings
- **Counterfactual correction**: blends post-scaling readings toward a
  pre-scaling baseline (0.6 actual / 0.4 baseline)
- **Training pipeline**: end-to-end script, saves scaler/model/Prophet artifacts
- **Shared interface**: `predict_load(window) -> (predicted_load, upper_bound,
  is_anomaly)` — the integration point for Person D's main loop; lazy-loads
  all artifacts, never crashes (falls back to a safe heuristic on failure)
- **Evaluation harness**: 4-way comparison (Reactive / Univariate LSTM /
  Multivariate LSTM / Hybrid) on RMSE, MAE, scaling lead time, oscillation count
- **requirements.txt + commit checklist** for `feature/lstm-model`

## Current status
First real-data evaluation run (82 training rows, 5 test windows):

| Method | RMSE | MAE |
|---|---|---|
| Reactive (baseline) | 38.95 | 32.36 |
| Univariate LSTM | 44.48 | 39.92 |
| Multivariate LSTM | 16.97 | 14.63 |
| Hybrid (LSTM+Prophet) | 22.06 | 21.90 |

Multivariate LSTM clearly beats the naive baseline. Hybrid currently
underperforms the raw LSTM — traced to Prophet's forecast being static
(trained once, doesn't adapt per-window), diluting an otherwise strong live
signal at the default 0.7/0.3 blend.

## Known limitations
- Real dataset is currently very small (82 training rows / 5 test windows) —
  RMSE numbers are preliminary, not statistically conclusive.
- Scaling lead time could not be measured on this dataset (test split
  contained no threshold-crossing spike to measure advance warning against).
- Prophet's static forecast is a known architectural limitation of the
  current ensemble design.

## Next steps
- [ ] Get a larger real dataset (in progress, with a collaborator)
- [ ] Retrain Prophet + LSTM on the larger dataset
- [ ] Re-run evaluation; re-check whether Hybrid-underperforms-LSTM still holds
- [ ] Sweep LSTM_WEIGHT/PROPHET_WEIGHT (candidates: 0.9/0.1, 0.85/0.15, 0.95/0.05)
- [ ] Finalize and push to `feature/lstm-model`