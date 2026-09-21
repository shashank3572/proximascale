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
    -> Preprocessing (10-step window, scaling, counterfactual correction)
    -> Base forecast: Prophet
    -> Residual = actual - Prophet fitted
    -> Residual model: LSTM (predicts the next 3 residuals)
    -> Residual stacking: final = Prophet forecast + LSTM residual forecast
    -> Anomaly detection + MC-Dropout uncertainty (upper bound)
    -> Decision engine (decision/engine.py)
    -> Docker actuator (actuator/docker_scaler.py)
    -> Application scaling
    -> Dashboard / observability (logs/events.csv -> dashboard/live_plot.py)
```

`main.py` orchestrates the loop: read the last 10 readings, `predict_load()`,
`engine.evaluate()`, execute the signal, log the decision for the dashboard.

Interfaces:

- `predict_load(window) -> (predicted_cpu, upper_bound, anomaly_flag)` — `model/predict.py`
- `engine.evaluate(predicted_cpu, upper_bound=None, anomaly_flag=False) -> signal`
  (`scale_up` / `scale_down` / `hold` / `hold_cooldown` / ...; see `docs/data_schema.md`)

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

pytest tests -q                     # full suite
python main.py --simulate           # demo loop, no Docker, no TensorFlow
streamlit run dashboard/live_plot.py -- --demo
```

Real mode needs four things running (see `docs/setup_guide.md`):

```powershell
# 1. the app image, tagged for both the monitored app and the scaled workers
docker build -f app/Dockerfile -t proximascale-app -t proximascale-worker:latest .
docker run -d --name proximascale-app -p 5000:5000 proximascale-app

# 2. the collector (own terminal) -- writes data/collected/metrics.csv
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
- Every PR runs pytest + flake8 via GitHub Actions (lightweight job; the
  TensorFlow/Prophet model tests skip there, so run `pytest tests -q` locally
  in the full environment before merging model changes).
- A red CI check blocks merge.
- Only the repo owner (Person D) merges into `dev` and promotes `dev` -> `main`.

## Documentation

- `docs/setup_guide.md` — environment, Docker setup, demo steps
- `docs/data_schema.md` — metric record schema and module interfaces
- `docs/scaling_events.md` — scaling event log format
- `PROGRESS.md` — milestone tracking and the audit repair log
