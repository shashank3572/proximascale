# ProximaScale — Progress Report

## Overview
ProximaScale is a proactive autoscaling load predictor combining a multivariate LSTM and Prophet in a hybrid ensemble, forecasting CPU load ~90 seconds ahead of real spikes.

---

## Person A — App + Monitoring

### Week 1–2 (Environment + Schema)
- Set up virtual environment, installed Flask, Locust, Docker SDK
- Agreed data schema with team: `{timestamp, cpu_percent, memory_percent, request_rate, post_scaling}`
- Created `app/app.py` with CPU-load routes and a request counter

### Week 3–4 (Flask app + monitoring)
- Flask routes `/work/light`, `/work/heavy`, `/health` — measurable CPU load via math loop
- `before_request` hook increments a SQLite-backed request counter (`monitoring/metrics.py`), not a hardcoded value
- `monitoring/collector.py` polls **container** CPU/memory via the Docker SDK (cgroup deltas) — not host `psutil`
- `monitoring/schema.py` — `MetricRecord` dataclass (`to_dict`/`from_csv_row`/`from_dict`)
- `monitoring/storage.py` — `append_row()` / `read_last_n()`, thread-safe CSV persistence

### Week 5 (Sem 2: post_scaling + integration audit fixes)
- [x] `post_scaling` field added to `MetricRecord` and wired through the collector
- [x] `is_post_scaling()` reads `data/scaling_events.json` (guarded against malformed/missing JSON)
- [fix] `collector.py` now calls `storage.append_row()` instead of duplicating CSV-writing logic — `storage.py` is no longer dead code
- [fix] `collector.py` now calls `reset_request_count()` once per cycle (atomic read-and-reset) instead of a separate `get_request_count()` + `reset_request_count()`, which could drop a request that arrived in between
- [fix] `requirements.txt` had an unresolved git merge conflict (unrelated `ml/` deps pasted in) — resolved to just this module's real dependencies (Flask, docker, locust, requests); unused `psutil` pin removed
- [fix] `app/Dockerfile` referenced files (`requirements.txt`, `monitoring/`) outside its own build context — Dockerfile now documents the correct build command: `docker build -f app/Dockerfile -t proximascale-app .` run from the repo root
- [removed] An unauthorized `ml/` shortcut pipeline (RandomForest predictor + a 14MB committed `.pkl`) and edits to `main.py` / `actuator/docker_scaler.py` were introduced by mistake in commit `06806fb`. These were outside this module's scope — real LSTM+Prophet work belongs to Person B, and `main.py`/the actuator belong to Person D/Person C. Removed; `main.py` and `actuator/docker_scaler.py` reverted to empty placeholders for their owners to fill in.

### TODO (Week 5–6) — still outstanding, needs a real run
- [ ] Run the full collection cycle for real: `docker build`, `docker run --cpus=0.5 --memory=256m`, all three Locust scenarios, collector active — this needs an actual machine with Docker, which is why it isn't done here
- [ ] Target: **≥1,500 rows**, spanning **≥3 days**, covering normal/spike/ramp/quiet periods, with genuine container CPU excursions into the 70–90%+ range and `post_scaling` populated
- [ ] `data/collected/metrics.csv` currently has 121 real rows (correct 5-column schema, generated 2026-09-12) — a valid start, but well short of the 1,500-row / 3-day target. Do not treat this file as the final handoff dataset.
- [ ] Multi-laptop Locust master/worker test — not yet evidenced anywhere in this repo
- [ ] Hand off the final `metrics.csv` to Person B and get explicit confirmation it's usable for LSTM training

### Commit log (this week)
```
[monitoring] add post-scaling field to metric schema
[monitoring] update storage for post-scaling field
[monitoring] align locust scenarios
[monitoring] add collected metrics dataset (partial)
[fix] resolve requirements.txt merge conflict
[fix] correct Dockerfile build-context documentation
[fix] wire collector.py to storage.append_row(), atomic request-count reset
[fix] remove unauthorized ml/ pipeline; revert main.py and actuator/docker_scaler.py to placeholders
[fix] update PROGRESS.md to match actual code
```

> **Person D note (integration merge):** `app.py`'s routes (`/work/light`, `/work/heavy`)
> and SQLite-backed counter are the ones actually wired to `monitoring/collector.py` and
> the Locust scenarios — an older `/`, `/heavy` + in-memory-counter + `/metrics`-endpoint
> version that had drifted out of sync with the rest of the pipeline was removed during
> the integration merge (it also had a latent duplicate `/health` route definition that
> would have crashed Flask on startup). Also fixed: `collector.py`'s poll loop was
> hardcoded to `time.sleep(60)` instead of using its own `POLL_INTERVAL = 30` constant —
> now consistent with `main.py`'s assumed 30s cadence.

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
- Merged Person A's Semester-2 app-monitoring work: adopted their `/work/light`/`/work/heavy` routes and SQLite-backed request counter (the ones actually wired to collector.py and the Locust scenarios) over an older, disconnected in-memory-counter version; fixed a latent duplicate `/health` route definition; fixed `collector.py`'s hardcoded `time.sleep(60)` to use its own `POLL_INTERVAL` constant; fixed `app/Dockerfile`'s stale base image and an internal inconsistency between two different app.py copy strategies.