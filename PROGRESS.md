# PROGRESS.md â€” ProximaScale

---

## Person B â€” LSTM Prediction Engine

### Week 10â€“11 (Apr 28 â€“ May 2, 2026)
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

## Person A â€” App + Monitoring

### Week 1â€“2 (Environment + Schema)
- Set up virtual environment, installed Flask, psutil, Locust
- Agreed data schema with team: `{timestamp, cpu_percent, memory_percent, request_rate}`
- Created initial `app/app.py` with CPU-load routes and request counter

### Week 3â€“4 (Flask App + Collector skeleton)
- Flask routes `/`, `/heavy` generating measurable CPU load via math loop
- `before_request` hook incrementing `_request_count`
- `monitoring/collector.py` skeleton â€” polls psutil every 5s, writes to CSV
- **Issue found:** `request_rate` was hardcoded to `0` â€” `/metrics` endpoint missing

### Week 5 (Fixes + Full module completion)
- Added `/metrics` JSON endpoint to `app.py` exposing live `request_rate`
- Fixed `app.run(host='0.0.0.0')` so Flask is reachable inside Docker
- Refactored `collector.py` â€” separated `collect_metrics(window)` from `run_collector()`
- Added `monitoring/schema.py` â€” MetricRecord dataclass
- Added `monitoring/storage.py` â€” append_row and read_last_n
- Added `app/Dockerfile`
- Added Locust scenarios: spike, gradual ramp, normal load

### TODO
- [ ] Regenerate metrics.csv â€” run all 3 Locust scenarios
- [ ] Target: â‰¥1,000 rows with real variance
- [ ] Hand off metrics.csv to Person B for LSTM training
---

## Person C — Decision Engine + Actuator

### Week 9 — Sem 1 (Complete)
- decision/engine.py — evaluate() with static threshold, anomaly bypass, cooldown
- decision/hysteresis.py — extracted Hysteresis class, 3 min cooldown
- actuator/docker_scaler.py — Docker SDK, min/max replica guards, pull guard, sorted scale_down
- actuator/scaler_interface.py — abstract base class
- main.py — orchestration loop, signal normaliser, dummy predictions
- tests/test_decision.py — 5 pytest cases, Docker mocked
- tests/test_integration.py — 9 smoke tests

### Status
Semester 1 complete. Pending: swap dummy predictions for Person B's predict(records) call in main.py.
