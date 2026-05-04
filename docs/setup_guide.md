# ProximaScale — Setup Guide

> How to run the entire project on a fresh machine.
> All teammates have RTX gaming laptops. All steps tested on Windows + WSL2 / Ubuntu 22.04.

---

## Prerequisites

| Tool           | Version  | Install                                 |
|----------------|----------|-----------------------------------------|
| Python         | 3.10.x   | python.org or `apt install python3.10`  |
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
docker build -f app/Dockerfile -t proximascale-app .
docker run -p 5000:5000 proximascale-app
```

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

This polls CPU/memory/request_rate every 10 seconds and appends to `data/collected/metrics.csv`.

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

## 7. Train the LSTM Model (Person B)

```bash
# Train on synthetic data (no real CSV needed)
python model/train.py

# Train on real data from Person A
python model/train.py --data data/collected/metrics.csv   # (add --data arg if needed)
```

Outputs:
- `model/saved/proximascale_lstm.keras`
- `model/saved/scaler.pkl`

> **The trained model is already committed** — you don't need to retrain to run inference.

---

## 8. Evaluate the Model (Person B)

```bash
python model/evaluate.py
# Chart saved to model/saved/evaluation_chart.png
```

---

## 9. Run the Full System (Person D)

```bash
# Simulation mode (no TensorFlow required — good for quick demo)
python main.py --simulate

# Real mode (requires Docker Desktop running + trained model)
python main.py

# Real mode with custom poll interval
python main.py --interval 30
```

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
│   └── saved/            # Committed model weights + scaler
├── decision/             # Decision engine + hysteresis (Person C)
├── actuator/             # Docker scaler (Person C)
├── data/
│   ├── collected/        # metrics.csv (auto-generated)
│   └── locust_scenarios/ # Load test scripts (Person A)
├── dashboard/            # Live plot — Semester 2 (Person D)
├── tests/                # Pytest suites (Person D)
├── docs/                 # This file, data_schema.md
├── main.py               # Orchestration loop (Person D)
├── cleanup.py            # Remove Docker containers
├── config.yaml           # Scaling thresholds
└── requirements.txt      # Pinned dependencies
```
