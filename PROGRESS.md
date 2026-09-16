# ProximaScale — Progress Report

## Overview
ProximaScale is a proactive autoscaling load predictor combining a multivariate LSTM and Prophet in a hybrid ensemble, forecasting CPU load ~90 seconds ahead of real spikes.

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

---

## Person B — LSTM Prediction Engine

### Completed
- Data pipeline: chronological train/test split, MinMax scaling, sliding windows.
- Multivariate LSTM trained on CPU, memory and request rate.
- Prophet model and hybrid ensemble.
- MC Dropout uncertainty estimation.
- Anomaly detection.
- Counterfactual correction.
- Training and evaluation pipelines.
- Shared predict_load(window) interface.
- Evaluation harness comparing Reactive, Univariate LSTM, Multivariate LSTM and Hybrid.

### Current status
First real-data evaluation showed Multivariate LSTM outperforming the naive baseline.

| Method | RMSE | MAE |
|---|---:|---:|
| Reactive (baseline) | 38.95 | 32.36 |
| Univariate LSTM | 44.48 | 39.92 |
| Multivariate LSTM | 16.97 | 14.63 |
| Hybrid (LSTM+Prophet) | 22.06 | 21.90 |

### Known limitations
- Real dataset is currently small, so results are preliminary.
- Scaling lead time could not yet be measured.
- Prophet's static forecast is a current architectural limitation.

### Next steps
- [ ] Get a larger real dataset
- [ ] Retrain Prophet + LSTM
- [ ] Re-run evaluation
- [ ] Sweep LSTM_WEIGHT/PROPHET_WEIGHT
- [ ] Finalize and push to feature/lstm-model

---

## Person C — Decision Engine + Actuator

### Week 9, Sem 1 (Complete)
- decision/engine.py — evaluate() with static threshold, anomaly bypass, cooldown
- decision/hysteresis.py — extracted Hysteresis class, 3 min cooldown
- actuator/docker_scaler.py — Docker SDK, min/max replica guards, pull guard, scale_down selects the newest worker container (configurable via scale_down_strategy); selection is deterministic via Created timestamp sort.
- actuator/scaler_interface.py — abstract base class
- main.py — orchestration loop, signal normaliser, dummy predictions
- tests/test_decision.py — 5 pytest cases, Docker mocked
- tests/test_integration.py — 9 smoke tests

### Status
Semester 1 complete. Pending: swap dummy predictions for Person B's predict(records) call in main.py.

### Semester 2 — Person C (decision & actuator)

- `decision/engine.py` — `evaluate(predicted_cpu, upper_bound=None, anomaly_flag=False)`.
  Locked 3-arg interface. Risk-aware scale-up branch (`upper_bound > upper_thresh`)
  inserted between anomaly and cooldown checks. Returns
  `scale_up` / `scale_down` / `hold` / `hold_cooldown`.
- `decision/adaptive_threshold.py` — rolling-window mean ± k·σ with static fallback
  under `min_samples`, floor/ceil clamps, and `min_band` guard against zero-variance
  collapse. Replaces static config thresholds inside the engine.
- `decision/scaling_log.py` — atomic append to `data/scaling_events.json`
  (temp-file + `.replace()`, thread lock, required-key validation). One entry per
  `scale_up` / `scale_down`; consumed by Person B's counterfactual correction.
- `actuator/docker_scaler.py` — `cpu_quota=50000`, `cpu_period=100000`,
  `mem_limit="256m"` passed into `containers.run()`. Deterministic `scale_down`
  selection via sort by `attrs["Created"]`; strategy configurable via
  `scale_down_strategy` in `config.yaml` (or `SCALE_DOWN_STRATEGY` env var).
- `config.yaml` — `max_containers` corrected 5 → 10 to match the locked spec.
- Tests: `tests/test_adaptive_threshold.py` (3), `tests/test_scaling_log.py` (3),
  3 new cases in `tests/test_decision.py` covering `upper_bound` branch and
  event-log writing. **26/26 pass.**

#### Scale-down target — spec conflict resolution

`c.md` specifies "remove the most recently added container" (`containers[-1]`).
`PROXIMASCALE_MASTER_PLAN_1.md` says "stops oldest extra container."
These conflict.

**Resolution (adopted):** default to **newest** (follows `c.md`, the stricter doc),
configurable via `scale_down_strategy: newest|oldest` in `config.yaml`, or
`SCALE_DOWN_STRATEGY` env var. Deterministic selection via sort on
`container.attrs["Created"]`; `_pick_down_target()` respects the configured value.
Team to formally reconcile the two documents before Week 5 handoff.

#### Scope note — main.py and tests/test_integration.py

Per the master plan, `main.py` and `tests/test_integration.py` are **Person D's
files** ("DO NOT TOUCH" for Person C). I wrote them as temporary integration
scaffolds so we could demonstrate the full pipeline end-to-end before
Person D's actuator dispatch landed. Ownership remains Person D's; request
sign-off / handoff before Week 5.

> **Person D note (integration merge):** main.py has since been reconciled —
> it now runs Person B's Semester-2 predict_load() feeding Person C's 3-arg
> evaluate(predicted_cpu, upper_bound, anomaly_flag) interface directly.
> See the Person D section below.

---

## Person D — Integration (Semester 2, interface merge)

### Status
- Merged Person B's Semester-2 model/predict_load() (LSTM + MC Dropout + Prophet ensemble, counterfactual correction, anomaly check) into main.py.
- Merged Person C's Semester-2 decision/actuator work (adaptive threshold, scaling event log, upper_bound-aware evaluate()) into main.py alongside it.
- The real loop now calls predict_load() and unpacks (predicted_cpu, upper_bound, anomaly_flag), feeding all three into DecisionEngine.evaluate() so upper_bound is now consumed, not just logged.
- Renamed model/saved/evaluation_chart.png to evaluation_chart_DUMMY_DATA_semester1.png and added model/saved/README.md.