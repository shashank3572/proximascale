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
---

## Person C � Decision Engine + Actuator

### Week 9 � Sem 1 (Complete)
- decision/engine.py � evaluate() with static threshold, anomaly bypass, cooldown
- decision/hysteresis.py � extracted Hysteresis class, 3 min cooldown
- actuator/docker_scaler.py � Docker SDK, min/max replica guards, pull guard, scale_down selects the newest worker container (configurable via scale_down_strategy); selection is deterministic via Created timestamp sort.
- actuator/scaler_interface.py � abstract base class
- main.py � orchestration loop, signal normaliser, dummy predictions
- tests/test_decision.py � 5 pytest cases, Docker mocked
- tests/test_integration.py � 9 smoke tests

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