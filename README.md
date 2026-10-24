# ProximaScale 

**Proactive Docker Auto-Scaling with Prophet + LSTM Residual Stacking**

Final-year project — CSE-AIML, AMC Engineering College, Bengaluru (VTU)  
**Project Guide:** Prof. Vinaya S Kavalgi

ProximaScale is a proactive container auto-scaling framework that forecasts CPU load approximately **90 seconds ahead** and uses the predictions to make scaling decisions before potential demand spikes arrive. It combines time-series forecasting, residual learning, uncertainty estimation, anomaly detection, and feedback-driven decision-making with Docker-based container actuation.

Unlike purely reactive approaches that wait for resource utilization to cross a threshold, ProximaScale aims to anticipate upcoming workload changes and respond earlier.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Team](#team)
- [System Architecture](#system-architecture)
- [Why Residual Stacking?](#why-residual-stacking)
- [Counterfactual Correction](#counterfactual-correction)
- [Quick Start](#quick-start)
- [Running the System](#running-the-system)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Testing and Repository Workflow](#testing-and-repository-workflow)
- [Known Limitations](#known-limitations)
- [Documentation](#documentation)

## Project Overview

ProximaScale integrates workload monitoring, forecasting, decision-making, and container management into a single control loop.

The system collects resource and workload metrics, processes recent observations, predicts future CPU utilization, and evaluates whether scaling action is appropriate. The resulting decision is passed to a Docker actuator, while logs and dashboard visualizations provide visibility into the system's behaviour.

The primary forecasting horizon is approximately **90 seconds**, based on three future observations at a 30-second sampling interval.

## Key Features

- **Proactive workload forecasting:** Predicts CPU utilization ahead of time to support earlier scaling decisions.
- **Hybrid forecasting pipeline:** Combines Prophet's time-series forecast with an LSTM model that predicts residual errors.
- **Uncertainty estimation:** Uses Monte Carlo Dropout to estimate forecast variability and derive an uncertainty-aware upper bound.
- **Anomaly detection:** Incorporates anomaly signals into the scaling decision process.
- **Feedback-driven decisions:** Evaluates forecast outputs and configured thresholds to select scaling actions.
- **Counterfactual correction:** Adjusts selected post-scaling observations using pre-scaling information.
- **Docker-based actuation:** Creates or removes worker containers according to the scaling signal and configured limits.
- **Monitoring and observability:** Records metrics and scaling events for inspection through logs and the Streamlit dashboard.
- **Simulation support:** Allows the control loop to be demonstrated without requiring live Docker actuation.

## Team

| Person | Name | Roll No. | Module |
|---|---|---|---|
| A | Shravani V | 1AM23CI140 | Application and Monitoring |
| B | Shreyas R | 1AM23CI141 | LSTM and Machine Learning |
| C | Soubhagya Ranjan Behera | 1AM23CI148 | Decision Engine and Actuator |
| D | Shashank B | 1AM23CI138 | Integration, Dashboard and Repository Management |

## System Architecture

```text
Monitoring
(monitoring/collector.py)
        |
        v
Metric Collection and Storage
(data/collected/)
        |
        v
Preprocessing
(10-observation window, scaling,
counterfactual correction)
        |
        v
Base Forecast: Prophet
        |
        v
Residual Prediction: LSTM
        |
        v
Residual Stacking
Final Forecast = Prophet Forecast
                  + LSTM Residual Forecast
        |
        v
Anomaly Detection and
Uncertainty Estimation
        |
        v
Decision Engine
(decision/engine.py)
        |
        v
Docker Actuator
(actuator/docker_scaler.py)
        |
        v
Worker Container Scaling
        |
        v
Event Logging and Dashboard
(logs/events.csv)
```

### Main Components

| Component | Responsibility |
|---|---|
| `monitoring/collector.py` | Collects CPU, memory and request-rate observations. |
| `monitoring/storage.py` | Supports metric storage and retrieval. |
| `model/prophet_model.py` | Provides the base time-series forecast. |
| `model/lstm_model.py` | Models residual patterns not captured by Prophet. |
| `model/predict.py` | Integrates preprocessing and forecasting outputs. |
| `model/uncertainty.py` | Supports uncertainty estimation. |
| `model/anomaly.py` | Detects anomalous observations or workload behaviour. |
| `model/counterfactual.py` | Applies post-scaling correction to selected observations. |
| `decision/engine.py` | Evaluates forecast information and determines a scaling signal. |
| `actuator/docker_scaler.py` | Executes supported Docker scaling actions. |
| `dashboard/live_plot.py` | Displays monitoring information and scaling-related visualizations. |
| `main.py` | Orchestrates the monitoring-to-actuation control loop. |

### Main Interfaces

```python
predict_load(window) -> (
    predicted_cpu,
    upper_bound,
    anomaly_flag
)
```

Defined in `model/predict.py`.

```python
engine.evaluate(
    predicted_cpu,
    upper_bound=None,
    anomaly_flag=False
) -> signal
```

Defined in `decision/engine.py`.

The decision engine can produce signals such as `scale_up`, `scale_down`, `hold`, and `hold_cooldown`, depending on its implementation and configuration. Refer to `docs/data_schema.md` for the documented interfaces and event schema.

## Why Residual Stacking?

An earlier implementation combined Prophet and LSTM forecasts using a fixed weighted average. That approach did not provide an improvement over the LSTM-only model in the team's evaluation.

The current approach uses **residual stacking**:

1. Prophet produces a base forecast representing the time-series trend and seasonality captured by the model.
2. Residuals are calculated from the difference between actual observations and Prophet's fitted values.
3. The LSTM learns residual patterns and predicts future residuals.
4. The residual prediction is added to the Prophet forecast to obtain the final prediction.

Conceptually:

\[
\hat{y}_{\text{final}} =
\hat{y}_{\text{Prophet}} +
\hat{r}_{\text{LSTM}}
\]

Here, \(\hat{y}_{\text{Prophet}}\) is the base forecast and \(\hat{r}_{\text{LSTM}}\) is the predicted residual correction.

The earlier weighted-average implementation is retained in `model/ensemble.py` as a deprecated component and is not part of the current forecasting pipeline.

## Counterfactual Correction

Scaling actions can change observed CPU utilization. Consequently, observations collected after a scaling event may reflect the effect of the controller rather than workload behaviour alone.

ProximaScale includes a counterfactual correction step that blends selected post-scaling observations (`post_scaling = True`) with the last pre-scaling value. This is intended to reduce the influence of the controller's own actions on subsequent forecasting inputs.

The correction depends on the availability of suitable pre-scaling observations and should be evaluated as part of the complete pipeline.

## Quick Start

### Prerequisites

- Python 3.11–3.13; Python 3.12 is the tested version documented for this project.
- Docker Desktop for real container-scaling mode.
- Git for repository management.
- The dependencies specified in `requirements.txt`.

### Environment Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

Run the test suite:

```bash
pytest tests -q
```

### Simulation Mode

Run the control loop without live Docker actuation:

```bash
python main.py --simulate
```

Launch the dashboard in demo mode:

```bash
streamlit run dashboard/live_plot.py -- --demo
```

Check `docs/setup_guide.md` for the expected setup and behaviour of each mode.

## Running the System

Real mode requires the application, monitoring collector, control loop and workload generation to be configured correctly.

### 1. Build the Docker Images

```bash
docker build -f app/Dockerfile \
  -t proximascale-app \
  -t proximascale-worker:latest .
```

### 2. Start the Application Container

```bash
docker run -d \
  --name proximascale-app \
  -p 5000:5000 \
  proximascale-app
```

### 3. Start the Monitoring Collector

In a separate terminal:

```bash
python -m monitoring.collector
```

The collector writes live observations to `data/collected/metrics_live.csv`. The frozen `metrics.csv` dataset is retained separately for model training and evaluation.

### 4. Start the Control Loop

In another terminal:

```bash
python main.py
```

The control loop reads recent observations, invokes `predict_load()`, evaluates the forecast through the decision engine, executes the resulting signal and records scaling events in `logs/events.csv`.

The control loop skips a cycle and logs a warning if the newest metric observation is older than three polling intervals.

### 5. Generate Workload

The system needs incoming workload to demonstrate changes in resource utilization. Use the application's supported workload endpoints or the Locust scenarios documented in `docs/setup_guide.md`.

### 6. Launch the Dashboard

Optionally, run:

```bash
streamlit run dashboard/live_plot.py
```

The dashboard reads scaling-event logs and provides visibility into the recorded system behaviour.

### Container Behaviour

- The actuator creates and removes additional worker containers using names such as `proximascale-worker-<n>-<ts>`.
- The standard `config.yaml` sets `min_containers: 1`, keeping at least one container alive at idle.
- `config_demo.yaml` sets `min_containers: 0` to permit scaling all the way down in the demo configuration. It is intended for demonstrations, not for producing research benchmark results.
- The collector reads and resets the application's request counter every 30 seconds. The recorded `request_rate` represents requests counted during that collection window, not a cumulative daily total.

**Important:** Creating additional worker containers does not, by itself, distribute incoming traffic across them. Traffic routing and load distribution must be configured separately if the demonstration is intended to show that replicas share workload.

## Data and Model Artifacts

The project uses separate datasets for synthetic training/evaluation and real development observations.

| Path | Purpose |
|---|---|
| `data/collected/metrics.csv` | Synthetic dataset generated by `model/generate_realistic_data.py`; used for the documented training/evaluation workflow. |
| `data/collected/metrics_live.csv` | Live observations collected during real-mode operation. |
| `data/collected/metrics_real_dev.csv` | Small real collection retained for development and inspection. |
| `model/saved/` | Saved model artefacts and associated documentation. |
| `logs/events.csv` | Recorded decisions and scaling events used by the dashboard. |

See `data/collected/README.md` and `model/saved/README.md` for additional details.

**Reproducibility note:** Reported evaluation results should identify the dataset, train/test split, baseline, metrics and evaluation procedure used. Results from synthetic data should not be described as measurements from a live Docker deployment.

Model artefacts were generated with the dependency versions pinned in `requirements.txt`. Changing these versions may affect compatibility or loading behaviour.

## Testing and Repository Workflow

The repository uses automated checks to help maintain code quality and consistency.

- Run `pytest tests -q` locally before merging changes.
- GitHub Actions runs the configured test and lint checks, including `flake8`.
- Some model tests may be skipped in the lightweight CI environment; run the complete local suite in the full environment when changing forecasting or model code.
- Protected changes to `main` should go through the repository's documented pull-request workflow.

### Useful Commands

```bash
# Run tests
pytest tests -q

# Start simulation
python main.py --simulate

# Start the dashboard
streamlit run dashboard/live_plot.py

# Run the reactive comparison component
python -m baseline.reactive_scaler
```

The repository also contains `cleanup.py` for removing scaled worker containers. Review its behaviour and verify the intended Docker resources before running cleanup commands.

## Known Limitations

- The synthetic dataset does not represent every characteristic of a real workload or container environment.
- Live measurements and synthetic benchmark results must be evaluated and reported separately.
- Forecast accuracy alone does not establish that the scaling controller improves resource utilization, response time or cost.
- The effectiveness of residual stacking depends on the quality of the base forecast, residual data and evaluation methodology.
- Additional worker containers require suitable traffic routing if they are expected to share incoming requests.
- Scaling behaviour depends on configuration, workload generation, cooldowns, minimum and maximum replica limits, and the availability of recent observations.
- Performance claims should be supported by reproducible experiments and clearly documented baselines.

These limitations are important considerations for further testing and evaluation; they do not, by themselves, establish the performance of a particular deployment.

## Documentation

- `docs/setup_guide.md` — Environment setup, Docker configuration and demo instructions.
- `docs/data_schema.md` — Metric schemas and module interfaces.
- `docs/scaling_events.md` — Scaling-event log format.
- `PROGRESS.md` — Project milestones and repair log.
- `data/collected/README.md` — Dataset descriptions.
- `model/saved/README.md` — Model artefacts and dependency notes.

---

**ProximaScale** is a final-year engineering project exploring the integration of proactive forecasting and Docker-based scaling. Its effectiveness should be assessed through transparent, reproducible experiments and clearly documented limitations.
