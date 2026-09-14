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
- Resolved Keras 2/3 framework mismatch — model retrained using tf_keras throughout
- proximascale_lstm.h5 and scaler.pkl regenerated and committed (saved in .h5 format for tf_keras compatibility)
- predict() loads and runs correctly — verified end-to-end
- Note: RMSE=5.48, MAE=4.39 metrics were from synthetic prototype run; model has since been retrained with corrected data ranges

### Status
Semester 1 complete. Model loads, predicts, and evaluates correctly.
scaler.pkl committed alongside model weights. Ready for integration.

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
- metrics.csv regenerated — 2,339 rows, full load range captured

## Person C Decision Engine + Actuator

### Week 9  Sem 1 (Complete)
- decision/engine.py  evaluate() with static threshold, anomaly bypass, cooldown
- decision/hysteresis.py  extracted Hysteresis class, 3 min cooldown
- actuator/docker_scaler.py  Docker SDK, min/max replica guards, pull guard, sorted scale_down
- actuator/scaler_interface.py  abstract base class
- main.py  orchestration loop, signal normaliser, dummy predictions
- tests/test_decision.py  5 pytest cases, Docker mocked
- tests/test_integration.py  9 smoke tests

### Status
Semester 1 complete. Pending: swap dummy predictions for Person B's predict(records) call in main.py.

## Person D — Integration (Semester 2, interface merge)

### Status (this branch: integration-fix, off dev)
- Merged Person B's Semester-2 `model/predict_load()` (LSTM+MC Dropout+Prophet
  ensemble, counterfactual correction, anomaly check) into `main.py`. The real
  loop now calls `predict_load()` and unpacks `(predicted_cpu, upper_bound,
  anomaly_flag)` instead of the old Semester-1 `predict()` dict.
- `upper_bound` is logged but not yet consumed by `DecisionEngine.evaluate()`
  — evaluate() still takes only `(predicted_cpu, anomaly_flag)`. Wiring the
  uncertainty bound into the scaling decision is open, not done.
- Renamed `model/saved/evaluation_chart.png` -> `evaluation_chart_DUMMY_DATA_semester1.png`
  and added `model/saved/README.md` — that chart was generated from synthetic
  data, not the real Semester-2 dataset, and was at risk of being cited in the
  report as a real result.

### Still open (blocked or not this integration's scope)
- No real-data evaluation run/committed (`data/collected/metrics.csv` is 81
  rows — blocked on Person A delivering >=1,000 rows per their own TODO below)
- Experiment 4 (with/without counterfactual ablation) not implemented —
  evaluate.py currently applies correction uniformly to all 4 methods
  (Person B)
- No pytest coverage for model/*.py — only `__main__` self-tests exist
  (Person B)
- Provenance of committed `model/saved/*.h5`/`.pkl` (real 81-row data vs.
  `metrics_dummy_backup.csv`) is unrecorded (Person B)
