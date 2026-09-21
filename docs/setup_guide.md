# ProximaScale — Setup Guide

> How to run the entire project on a fresh machine.
> All teammates have RTX gaming laptops. All steps tested on Windows + WSL2 / Ubuntu 22.04.
> **Windows Users:** If running model scripts via pipes or redirects, set `$env:PYTHONUTF8=1` in PowerShell to avoid emoji-encoding crashes. Interactive terminals are unaffected.
---

## Prerequisites

| Tool           | Version  | Install                                 |
|----------------|----------|-----------------------------------------|
| Python         | 3.11-3.13 (tested: 3.12) | python.org |
| Docker Desktop | latest   | docker.com/products/docker-desktop      |
| Git            | any      | git-scm.com                             |
| CUDA (GPU)     | 12.x     | For training only — inference works on CPU |

---

## 1. Clone the Repo

```bash
git clone https://github.com/<your-org>/proximascale.git
cd proximascale
git checkout dev
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / WSL2
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** TensorFlow picks up the RTX GPU automatically if CUDA 12 + cuDNN are installed.
> Verify with: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## 4. Run the Flask App (Person A)

```bash
# Option A: directly
python app/app.py

# Option B: inside Docker (from project root)
docker build -f app/Dockerfile -t proximascale-app -t proximascale-worker:latest .
docker run -d --name proximascale-app -p 5000:5000 proximascale-app
```

> The second tag matters: `config.yaml` tells the actuator to launch scaled workers from
> `proximascale-worker:latest`, and the collector monitors the container named `proximascale-app`.
> If that image does not exist locally, `scale_up` cannot start a worker.

Test it:
```bash
curl http://localhost:5000/health
curl http://localhost:5000/metrics
```

---

## 5. Start the Monitoring Collector (Person A)

In a **separate terminal** (with venv active):
```bash
python -m monitoring.collector
```

This polls CPU/memory/request_rate every 30 seconds and appends to `data/collected/metrics.csv`.

**Real mode (`python main.py`) reads this file**, so the collector must be running. `main.py` skips
prediction and warns if the newest row is more than 3 poll intervals old.

> The committed `metrics.csv` is a 90,000-row *synthetic* dataset (see `data/collected/README.md`).
> Running the collector appends real rows to it; to collect a clean real dataset, point the
> collector at a new file or back up/move the committed one first.

---

## 6. Generate Load with Locust (Person A)

In a **third terminal**:
```bash
# Normal steady load (30 min)
locust -f data/locust_scenarios/normal_load.py --host=http://localhost:5000 \
       --headless -u 20 -r 2 --run-time 30m

# Spike scenario (8 min)
locust -f data/locust_scenarios/spike_load.py --host=http://localhost:5000 \
       --headless --run-time 8m

# Gradual ramp (12 min)
locust -f data/locust_scenarios/gradual_ramp.py --host=http://localhost:5000 \
       --headless --run-time 12m
```

Run all three back-to-back to get >1,000 rows of varied data in the CSV.

---

## 7. Train the Models (Person B)

Training is a two-step residual-stacking pipeline and **overwrites the committed artifacts** in
`model/saved/`. You do not need to retrain to run inference.

```bash
python model/prophet_model.py   # 1. fit Prophet, saves proximascale_prophet.pkl (must come first)
python model/train.py           # 2. train the LSTM on Prophet's residuals
```

Both read `data/collected/metrics.csv` (there is no `--data` option). Outputs in `model/saved/`:
- `proximascale_prophet.pkl`
- `proximascale_lstm.h5`
- `scaler.pkl`, `scaler_residual.pkl`

> **The trained models are already committed** — see `model/saved/README.md` for provenance.
> Back up `model/saved/` before retraining.

---

## 8. Evaluate the Model (Person B)

```bash
python model/evaluate.py
```

Prints a 4-way comparison table (Reactive / Univariate LSTM / Multivariate LSTM / Hybrid
residual-stacking) on the chronological test split. It does not write a chart, and it can take
several minutes. If the cached univariate baseline is missing it trains and saves one.

---

## 9. Run the Full System (Person D)

```bash
# Simulation mode (no TensorFlow required — good for quick demo)
python main.py --simulate

# Real mode (requires Docker Desktop running, the collector from step 5, and the worker image from step 4)
python main.py

# Real mode with custom poll interval (keep 30: the model is trained at a 30s sampling rate)
python main.py --interval 30
```

Each decision is appended to `logs/events.csv` (git-ignored). View it live:

```bash
streamlit run dashboard/live_plot.py            # live, reads logs/events.csv
streamlit run dashboard/live_plot.py -- --demo  # synthetic demo data
```

The SHAP panel shows "No SHAP values recorded" in live mode: `ShapExplainer` is fixed and tested
but not yet called from `main.py`.

---

## 10. Run Tests

```bash
# Decision engine tests (Docker mocked — no Docker Desktop needed)
pytest tests/test_decision.py -v

# Integration smoke tests
pytest tests/test_integration.py -v

# All tests
pytest tests/ -v
```

---

## 11. Clean Up Docker Containers After Testing

```bash
python cleanup.py
```

---

## Folder Structure

```
proximascale/
├── app/                  # Flask app + Dockerfile (Person A)
├── monitoring/           # Collector, storage, schema (Person A)
├── model/                # LSTM, training, predict, evaluate (Person B)
│   └── saved/            # Committed Prophet + LSTM artifacts + scalers (see its README)
├── decision/             # Decision engine + hysteresis (Person C)
├── actuator/             # Docker scaler (Person C)
├── data/
│   ├── collected/        # metrics.csv (synthetic, 90k rows) + small real samples; see its README
│   └── locust_scenarios/ # Load test scripts (Person A)
├── dashboard/            # Live plot — Semester 2 (Person D)
├── tests/                # Pytest suites (Person D)
├── docs/                 # This file, data_schema.md, scaling_events.md
├── logs/                 # events.csv written by main.py (git-ignored)
├── main.py               # Orchestration loop (Person D)
├── cleanup.py            # Remove Docker containers
├── config.yaml           # Scaling thresholds
└── requirements.txt      # Pinned dependencies
```
