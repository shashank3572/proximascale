# ProximaScale

> AI-driven proactive auto-scaling system that uses a multivariate hybrid LSTM-Prophet model
> with multi-step prediction, adaptive thresholding, and anomaly detection to forecast workloads
> and scale Docker containers **before** demand spikes occur.

**AMC Engineering College, Bengaluru | CSE-AIML | VTU | 2025-26**

---

## What It Does

Traditional auto-scalers react *after* CPU spikes — containers spin up too late, SLOs are missed.
ProximaScale predicts workload 3 minutes ahead using an LSTM trained on CPU%, memory%, and request rate,
then scales Docker containers **proactively** before the spike hits.

```
Flask App  →  Monitoring Collector  →  LSTM Prediction  →  Decision Engine  →  Docker Actuator
(load gen)     (CPU/MEM/req_rate)      (3-step forecast)   (threshold logic)   (scale_up/down)
```

---

## Key Features

| Feature | Description | Novelty |
|---|---|---|
| Multivariate LSTM | CPU + Memory + Request Rate as inputs | Paper [1] |
| Multi-step prediction | Forecasts next 3 minutes (not just 1 step) | Paper [2] |
| Anomaly detection | Z-score spike flag triggers instant scale-up | IEEE HPCC 2025 |
| Adaptive threshold | Dynamic threshold based on rolling traffic window | Sem 2 |
| Hybrid LSTM+Prophet | Ensemble for better long-term trend capture | Sem 2 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run in simulation mode (no Docker or GPU needed)
python main.py --simulate

# 3. Run tests
pytest tests/ -v
```

See [docs/setup_guide.md](docs/setup_guide.md) for full setup instructions.

---

## Team

| Person | Role | Branch |
|---|---|---|
| Person A | Flask App + Monitoring | `feature/app-monitoring` |
| Person B | LSTM Model (core ML) | `feature/lstm-model` |
| Person C | Decision Engine + Actuator | `feature/decision-actuator` |
| Person D | Integration + Dashboard + Repo | `feature/integration-dashboard` |

---

## References

1. Dang-Quang & Yoo, "Multivariate Bi-LSTM Autoscaling", *Applied Sciences* 2022. DOI: 10.3390/app12073523
2. Guruge & Priyadarshana, "Prophet + LSTM Kubernetes Autoscaling", *Frontiers in CS* 2025. DOI: 10.3389/fcomp.2025.1509165
3. Lanciano et al., "Predictive Autoscaling with OpenStack Monasca", IEEE/ACM UCC 2021.
4. Golshani et al., "TCN-based Proactive Autoscaling", JPDC 2021. DOI: 10.1016/j.jpdc.2021.04.006
