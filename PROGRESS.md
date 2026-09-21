# ProximaScale — Progress Report

## Overview
ProximaScale is a proactive autoscaling load predictor combining Prophet (base forecast) and a multivariate LSTM that predicts Prophet's residual (residual stacking), forecasting CPU load ~90 seconds ahead of real spikes.

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
- [ ] Real dataset still short of target. The 121 real rows (5-column schema, collected 2026-09-12 to 09-14) now live in `data/collected/metrics_real_dev.csv`. `data/collected/metrics.csv` was replaced in commit 2da9d70e by a 90,000-row **synthetic** dataset from `model/generate_dataset.py` — see `data/collected/README.md`. Do not present it as measured data.
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
- Prophet base model and residual stacking (LSTM predicts Prophet's residual; final = Prophet + LSTM residual). The earlier fixed-weight LSTM/Prophet ensemble is kept in `model/ensemble.py` as a deprecated record of the approach that was tried first.
- MC Dropout uncertainty estimation.
- Anomaly detection.
- Counterfactual correction.
- Training and evaluation pipelines.
- Shared predict_load(window) interface.
- Evaluation harness comparing Reactive, Univariate LSTM, Multivariate LSTM and Hybrid.

### Status at first real-data evaluation (HISTORICAL)
> Measured on the 121 real rows with the **old fixed-weight ensemble**, before the switch to residual stacking. These numbers are superseded and must not be cited as current results; re-run `python model/evaluate.py` for the current comparison.

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
- Merged Person B's Semester-2 model/predict_load() (Prophet + LSTM residual stacking, MC Dropout, counterfactual correction, anomaly check) into main.py.
- Merged Person C's Semester-2 decision/actuator work (adaptive threshold, scaling event log, upper_bound-aware evaluate()) into main.py alongside it.
- The real loop now calls predict_load() and unpacks (predicted_cpu, upper_bound, anomaly_flag), feeding all three into DecisionEngine.evaluate() so upper_bound is now consumed, not just logged.
- Renamed model/saved/evaluation_chart.png to evaluation_chart_DUMMY_DATA_semester1.png and added model/saved/README.md.
- Merged Person A's Semester-2 app-monitoring work: adopted their `/work/light`/`/work/heavy` routes and SQLite-backed request counter (the ones actually wired to collector.py and the Locust scenarios) over an older, disconnected in-memory-counter version; fixed a latent duplicate `/health` route definition; fixed `collector.py`'s hardcoded `time.sleep(60)` to use its own `POLL_INTERVAL` constant; fixed `app/Dockerfile`'s stale base image and an internal inconsistency between two different app.py copy strategies.

---

## Repair pass — audit fixes (2026-09-20)

Each item was reproduced against the `dev` branch before fixing; every fix has tests (full suite: 95 passed in a fresh venv built from `requirements.txt`).

| # | Problem (verified) | Fix |
|---|---|---|
| 1 | `is_post_scaling()` read a single JSON object but the log is an array (error swallowed → always `False`); window also hard-coded ×60 (would be 1 h) | reads the array via `scaling_log`, window = `expires_after_steps × tick_seconds` (180 s) |
| 2 | `main.py` real loop called `collect_metrics(window=10)` (takes no args; TypeError) and used an out-of-scope `engine` global | reads `storage.read_last_n(10)`; collector runs as its own process; engine passed in; skips stale/short data |
| 3 | `docker.from_env()` at import time in the collector | lazy client |
| 4/5 | `MODEL_PATH` pointed at the stale pre-residual `.keras` model (unloadable, legacy Keras 2 config) so every prediction fell back; the residual-stacking model is the `.h5`; `predict.py` docstring contradicted the code | `MODEL_PATH` → `proximascale_lstm.h5`; fallback now logged loudly; docstring fixed; `pyarrow` added (needed to unpickle the Prophet artifact) |
| 6 | dashboard read `logs/events.csv` but nothing wrote it | `main.py` `log_event()` writes it each decision |
| 7 | `/predict` route imported a nonexistent `predict` | uses `predict_load` (works from a full checkout; the Docker image excludes `model/`) |
| 8 | tests could not tell a broken model from a working one (fallback also returns floats), wrong skip guard, a tautological test, no test of residual stacking or counterfactual correction | tests strengthened/added, incl. held-out "stacked beats Prophet-only" |
| 9 | setup guide required Python 3.10 (pinned numpy/pandas/sklearn need ≥3.11); unused `tf-keras`; unpinned pytest | requirements/docs corrected; verified in a fresh venv |
| 10 | `model/ensemble.py` docstring/self-test disagreed with its weights | kept as a deprecated historical record; unused by the pipeline (test enforces) |
| 12 | `shap.DeepExplainer` fails on Keras 3 models | `KernelExplainer` |
| 11 | `cleanup.py` filtered `proximascale_worker_` but containers are named `proximascale-worker-…` (removed nothing); README had pasted-in text; docs described the old architecture, `--data` and chart options that do not exist, and no worker-image tag | fixed (the worker-image tag is untested without a Docker daemon) |

Still open: `evaluate.py` has not been re-run since the architecture switch (the historical table above is stale); real container data is limited to 121 rows; CI does not run the TensorFlow/Prophet tests; no end-to-end run against a live Docker daemon has been done in this pass.

## Live-demo fixes (2026-09-21)

Found during a real `python main.py --config config_demo.yaml` demo run (spike via `/work/heavy`). Every item reproduced from live logs before fixing; full suite now 113 passed.

| # | Symptom (observed live) | Root cause | Fix |
|---|---|---|---|
| 1 | After a spike, `scale_up` fired on 6 consecutive polls even while predicted CPU was *falling*, then `scale_down`/`cooldown` alternated | `engine.evaluate()` checked the `upper_bound_risk` branch BEFORE the cooldown gate; the MC-Dropout bound (mean + 2σ) stays above threshold for many polls after the mean has turned, so every poll re-scaled | cooldown now gates every branch except `anomaly` (one scale-up per cooldown window per spike); regression-tested in `tests/test_live_demo_fixes.py` |
| 2 | Predicted CPU logged as 124.8 / upper bound 156.7 | Prophet + LSTM residual + 2σ is unbounded; container CPU is physically ≤100% | `predict_load()` clamps mean and upper bound to 0–100 (fallback path too); ordering preserved so a pegged bound still reads as high risk |
| 3 | Fleet stalled at 1 extra container through multiple `scale_down`s | Not a bug: `min_containers: 1` (correct per spec). The demo just couldn't show full collapse | `config_demo.yaml` sets `min_containers: 0` (demo-only; real `config.yaml` unchanged) |
| 4 | Collector `Requests: 0` forever, even under load | Collector read the SQLite file directly; the Flask app runs in a container, so the host and container see *different* `metrics.db` files — the host copy is never written | app gains `POST /request-rate/reset` (read-and-reset, atomic); collector calls it over HTTP first, SQLite stays as the fallback for a host-run app; returns −1 (not a fake 0) when neither channel works |
| 5 | Collector CPU flapped 0.01% ↔ 100.0% | A single `stats(stream=False)` diffs against `precpu_stats` — Docker's cached snapshot of the previous call, which is stale on the first call and after idle periods | two fresh snapshots 2 s apart are diffed (`_cpu_percent_from_stats`); still clamped to 100 |
| 6 | Flask page showed a day-long cumulative count ("1271 requests since last monitoring reset") | It rendered the raw counter, which the collector only resets every 30 s | label now says "requests in the current monitoring window"; the page polls every 3 s so it tracks the live window; `/request-rate/reset` added to the endpoint list |
| 7 | Dashboard SHAP panel permanently showed the "isn't wired into the live control loop" notice | `main.py` never called `dashboard/shap_explain.py` | `run_real_loop` computes attribution on every scale-up via `_compute_shap()`: reuses `prepare_scaled_window()` + the cached model (identical input to prediction), KernelExplainer nsamples=256, window + neutral baseline as background; failures log a warning and record zeros — never kill the loop |
| 8 | Dashboard "Recent events" gave no reason for each action | events.csv carried no reason | events rows gain a `reason` column (engine's `last_reason`); dashboard table shows it; `generate_demo()` supplies it too |

Note: `logs/events.csv` from runs before this pass lack `reason` and the dashboard tolerates both formats. `config_demo.yaml` is still demo-only — run evaluation numbers against the real `config.yaml`.

## Current 4-Way Evaluation (Residual Stacking) — Verified <21/09/2026>

Run on clean 90k synthetic CSV. Command: `python model/evaluate.py`
Windows evaluated: 17990

| Method | RMSE | MAE | Avg Lead Time (s) | Oscillations |
|---|---|---|---|---|
| Reactive | 6.64 | 3.92 | 0.0 | 72 |
| Univariate LSTM | 5.13 | 3.03 | 38.3 | 102 |
| Prophet Alone | 10.15 | 4.67 | 0.0 | 0 |
| **Hybrid (Full)** | **5.09** | **3.06** | **57.5** | **96** |

Target Scaling Lead Time: >= 60s
Hybrid RMSE improves on Prophet Alone (10.15) — LSTM's residual correction is adding value.

Limitation (Live-mode predictions): The Prophet base model is anchored to the synthetic training timeline. During live-container demos with real workloads, predicted CPU may temporarily diverge from actual CPU, particularly during severe spikes. The LSTM residual model partially compensates for this, but the system is best demonstrated via --simulate mode or with pre-recorded synthetic data for the dashboard.

