# ProximaScale

**AI-Powered Hybrid LSTM Framework for Proactive Auto-Scaling**

Final year project — AMC Engineering College, Bengaluru (CSE-AIML, VTU)
Guide: Prof. Vinaya S Kavalgi

ProximaScale forecasts container workload with a hybrid LSTM–Prophet
model and scales Docker containers *before* demand spikes arrive —
unlike reactive scalers that only act after a threshold is breached.

## Team

| Person | Name | Roll No | Module |
|--------|------|---------|--------|
| A | Shravani V | 1AM23CI140 | App + Monitoring |
| B | Shreyas R | 1AM23CI141 | LSTM + ML |
| C | Soubhagya Ranjan Behera | 1AM23CI148 | Decision + Actuator |
| D | Shashank B | 1AM23CI138 | Integration + Dashboard + Repo Owner |

## Architecture
monitoring/collector → model/predict → decision/engine → actuator/docker_scaler
(Person A) (Person B) (Person C) (Person C)
↓
main.py orchestration ← Person D
↓
dashboard/live_plot.py ← Person D

text

Prediction interface: `predict(window) → {"predicted_cpu": [...], "anomaly": bool}`  
Decision interface: `evaluate(predicted_cpu, anomaly_flag=...) → "scale_up" | "scale_down" | "hold"`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Or the demo loop (no Docker, no TensorFlow)
python main.py --simulate

# Dashboard (demo mode)
streamlit run dashboard/live_plot.py -- --demo

# Reactive baseline (comparison group)
python -m baseline.reactive_scaler

# Tests
pytest tests/ -v
Repository rules
main is protected. All work goes through PRs into dev.

Every PR runs pytest + flake8 via GitHub Actions.

A red CI check blocks merge.

Only the repo owner (Person D) merges into dev and promotes dev → main.

Documentation
docs/setup_guide.md — environment, Docker setup, demo steps

docs/data_schema.md — metric record schema and module interfaces

PROGRESS.md — milestone tracking

text

### 8.3 `PROGRESS.md`

```markdown
# ProximaScale — Progress

## Status legend
✅ done · 🟡 in progress · ⬜ not started

## Milestones

| # | Item | Owner | Status |
|---|------|-------|--------|
| 1 | Monitoring collector + CSV storage | A | ✅ |
| 2 | LSTM + Prophet hybrid model | B | 🟡 |
| 3 | Decision engine + hysteresis + actuator | C | ✅ |
| 4 | Orchestration loop (`main.py`) | D | ✅ |
| 5 | Reactive baseline scaler | D | ✅ |
| 6 | SHAP explainability module | D | ✅ |
| 7 | Streamlit live dashboard | D | ✅ |
| 8 | Results charts + sample CSVs | D | ✅ |
| 9 | Integration tests | D | ✅ |
| 10 | CI pipeline (pytest + flake8) | D | ✅ |
| 11 | Experiment 1 — accuracy | B + D | ⬜ |
| 12 | Experiment 2 — lead time | D | ⬜ |
| 13 | Experiment 3 — latency | D | ⬜ |
| 14 | Experiment 4 — oscillation | D + C | ⬜ |
| 15 | Final report assembly | D | ⬜ |