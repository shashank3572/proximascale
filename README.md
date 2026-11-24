# ProximaScale

**Proactive Docker auto-scaling with Prophet + LSTM residual stacking**

Final year project — AMC Engineering College, Bengaluru (CSE-AIML, VTU)
Guide: Prof. Vinaya S Kavalgi

ProximaScale forecasts container CPU load about 90 seconds ahead and scales
Docker containers *before* demand spikes arrive, unlike reactive scalers that
only act after a threshold is breached.

## Team

| Person | Name | Roll No | Module |
|--------|------|---------|--------|
| A | Shravani V | 1AM23CI140 | App + Monitoring |
| B | Shreyas R | 1AM23CI141 | LSTM + ML |
| C | Soubhagya Ranjan Behera | 1AM23CI148 | Decision + Actuator |
| D | Shashank B | 1AM23CI138 | Integration + Dashboard + Repo Owner |

## Architecture

```text
Monitoring (monitoring/collector.py, separate process)
    -> Data collection / storage (data/collected/metrics.csv)
    -> Preprocessing (30-step window = 15 min, scaling, counterfactual correction)
    -> Base forecast: Prophet (also given to the LSTM as an input channel)
    -> Residual = actual - Prophet fitted
    -> Residual model: 3 LSTMs (seed ensemble), predict the next 3 residuals
    -> Residual stacking: final = Prophet forecast + LSTM residual forecast
    -> Anomaly detection + MC-Dropout uncertainty (calibrated 95% upper bound)
    -> Decision engine (decision/engine.py)
    -> Docker actuator (actuator/docker_scaler.py)
    -> Application scaling
    -> Dashboard / observability (logs/events.csv -> dashboard/live_plot.py)
```

`main.py` orchestrates the loop: read the last `HISTORY_SIZE` (30) readings,
`predict_load()`, `engine.evaluate()`, execute the signal, log the decision for
the dashboard.

Interfaces:

- `predict_load(window) -> (predicted_cpu, upper_bound, anomaly_flag)` — `model/predict.py`
- `engine.evaluate(predicted_cpu, upper_bound=None, anomaly_flag=False, current_cpu=None) -> signal`
  (`scale_up` / `scale_down` / `hold` / `hold_cooldown` / ...; see `docs/data_schema.md`)

### The served model

`predict_load()` serves the stacked bundle in `model/saved/proximascale_v4/`
(`model/forecaster.py`) whenever it is given at least 30 readings, and the
original 10-reading model (`proximascale_lstm.h5`) for 10-29 readings (the
first 15 minutes after a cold start). v4 is trained on the original data plus
data with varied spike shapes. `PROXIMASCALE_MODEL=v3` / `=v2` serve earlier
bundles and `PROXIMASCALE_MODEL=legacy` the original model only.

On the test split: RMSE 4.35 (original model 5.09), lead time 58 s, no
forecast false alarms; with the tuned engine 36/36 spikes get a scale-up
30-90 s ahead, with no false scale-ups and 2.7 replicas on average (original
live config: 20/36, 44 false scale-ups a day, 5.9 replicas). All synthetic data.
How the bundle was chosen, and every result behind it, is in
`docs/model_evaluation.md`.

The decision engine has three switches (`config.yaml`), each chosen on the
validation period by `model/tune_decisions.py`: `use_anomaly_bypass: false`,
`use_upper_bound_risk: false`, and `scale_down_max_current_cpu: 75` (no
scale-down while CPU is still above 75%). `scale_up_on_current_cpu` (a
reactive floor) exists but is off: it made every evaluated period worse.

### Why residual stacking

An earlier version blended LSTM and Prophet forecasts with a fixed weighted
average. That blend could never beat the LSTM alone, so we replaced it: Prophet
models trend/seasonality, and the LSTM only learns what Prophet gets wrong.
The old blend is kept as `model/ensemble.py` (deprecated, unused by the
pipeline) as a record that the ensemble approach was tried first.

### Counterfactual correction

Rows collected right after a scaling event (`post_scaling = True`) are
blended with the last pre-scaling value so the model does not learn the
scaler's own effect as if it were natural load.

## Quick start

Python 3.11 – 3.13 (tested on 3.12). Docker Desktop for real mode.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# TensorFlow must be imported before scikit-learn on Windows, or the process
# dies with an access violation inside `import tensorflow`. tools/sitecustomize.py
# makes that ordering automatic for every process using this interpreter.
# Copy it into the venv (redo this whenever .venv is recreated):
cp tools/sitecustomize.py .venv/Lib/site-packages/sitecustomize.py

pytest tests -q                     # full suite
python main.py --simulate           # demo loop, no Docker, no TensorFlow
streamlit run dashboard/live_plot.py -- --demo
```

Real mode needs four things running (see `docs/setup_guide.md`):

```powershell
# 1. the app image, tagged for both the monitored app and the scaled workers
docker build -f app/Dockerfile -t proximascale-app -t proximascale-worker:latest .
docker run -d --name proximascale-app -p 5000:5000 proximascale-app

# 2. the collector (own terminal) -- writes data/collected/metrics_live.csv
#    (metrics.csv stays the frozen 90k synthetic training dataset)
python -m monitoring.collector

# 3. the control loop (own terminal) -- also writes logs/events.csv
python main.py

# 4. generate real load, or the loop just holds -- e.g. the Flask page's
#    /work/heavy button or the Locust scenarios in step 6 of the setup guide

# optional: live dashboard reads logs/events.csv
streamlit run dashboard/live_plot.py
```

> The `proximascale-worker-<n>-<ts>` containers are extra replicas the actuator
> creates and removes; `min_containers: 1` (config.yaml) keeps one alive at idle
> by design. `config_demo.yaml` sets `min_containers: 0` so a demo can scale all
> the way down — it is for demos only, never for report numbers.
>
> Request counting works across the container boundary via `POST
> /request-rate/reset`: the collector reads-and-resets the app's counter once per
> 30 s poll, so `request_rate` in the CSV is the number of workload requests that
> arrived during that specific window (not a cumulative daily count).

`main.py` skips a cycle (and warns) if the newest collected row is older than
three poll intervals, so it never scales on stale data. Decision logs
(`logs/events.csv`) include the engine's `reason` and SHAP per-feature
attribution on every scale-up.

Other commands: `python -m baseline.reactive_scaler` (reactive comparison
group), `python cleanup.py` (remove scaled worker containers).

## Data and model artifacts

- `data/collected/README.md` — which CSV is which. **`metrics.csv` is a
  synthetic dataset** produced by `model/generate_realistic_data.py`, not container
  measurements. The small real collection is `metrics_real_dev.csv`.
- `model/saved/README.md` — which artifacts are current. The pickles and the
  LSTM were produced with the pinned versions in `requirements.txt`; changing
  those pins can stop them loading.

## Repository rules

- `main` is protected. All work goes through PRs into `dev`.
- Every PR runs two GitHub Actions jobs: a **unit** job (compile, lint, config
  validation, import checks, pytest without the ML stack) and a **model** job
  that installs the pinned `requirements.txt`, runs `python model/predict.py`
  and fails if any test was *skipped* — a skip there would mean model behaviour
  is untested, so it is not treated as green.
- A red CI check blocks merge.
- Only the repo owner (Person D) merges into `dev` and promotes `dev` -> `main`.

## Documentation

- `docs/setup_guide.md` — environment, Docker setup, demo steps
- `docs/METRICS.md` — every metric's unit and definition (CPU, memory,
  requests per window vs per second, prediction, effective thresholds)
- `docs/DEMO_TROUBLESHOOTING.md` — symptom-first fixes for Docker, stale data,
  model fallbacks, request-rate failures, workers and the dashboard
- `DEMO_GUIDE.md` — the panel run, start to cleanup, with honest limitations
- `docs/data_schema.md` — metric record schema and module interfaces
- `docs/scaling_events.md` — scaling event log format
- `docs/model_evaluation.md` — model search, calibration, engine tuning,
  unseen-data tests and how to reproduce every number
- `PROGRESS.md` — milestone tracking and the audit repair log

## Behaviour contracts (post-audit)

These are the invariants the tests pin down; changing them is a design
decision, not a refactor.

| Contract | Where | What it guarantees |
|---|---|---|
| Forecast fallback | `model/predict.py` | On any internal failure, `predict_load()` returns `(cpu*1.15, cpu*1.3, False)` — clamped to 0-100, and **`is_anomaly=False`**: a broken model never masquerades as a detected spike. |
| Live calibration | `model/predict.py` | `predicted = current_cpu + forecast_delta`. The bundled Prophet was fit on synthetic data (~52 % baseline), so its absolute level is re-anchored on the measured CPU. `PROXIMASCALE_LIVE_CALIBRATION=off` restores the raw output. |
| Missing measurements | `monitoring/collector.py` | Zero CPU, a missing CPU field, a stopped container and a Docker failure are four different states. Only real measurements are written to `metrics_live.csv`. |
| Request window | `monitoring/metrics.py`, `POST /request-rate/reset` | Counting and resetting happen in one transaction, and the app reports how many seconds the count covers. |
| Adaptive band | `decision/adaptive_threshold.py` | `lower < upper`, both inside 0-100, separated by `adaptive_min_band`, limits configurable and validated at startup. |
| Decision log | `main.py` → `logs/events.csv` | Every row carries the **effective** thresholds and the forecast mode, so the dashboard cannot disagree with the engine. |
| Actuator honesty | `actuator/docker_scaler.py` | Replica count comes from Docker. `traffic_serving` is reported separately and is `False` in pool mode — no load balancer is claimed. |
