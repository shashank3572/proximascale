# ProximaScale — Progress Report

## Overview
ProximaScale is a proactive autoscaling load predictor combining a multivariate LSTM and Prophet in a hybrid ensemble, forecasting CPU load ~90 seconds ahead of real spikes.

## Person C Decision Engine + Actuator

### Week 9 Sem 1 (Complete)
- decision/engine.py evaluate() with static threshold, anomaly bypass, cooldown
- decision/hysteresis.py extracted Hysteresis class, 3 min cooldown
- actuator/docker_scaler.py Docker SDK, min/max replica guards, pull guard, sorted scale_down
- actuator/scaler_interface.py abstract base class
- main.py orchestration loop, signal normaliser, dummy predictions
- tests/test_decision.py 5 pytest cases, Docker mocked
- tests/test_integration.py 9 smoke tests

### Status
Semester 1 complete. Pending: swap dummy predictions for Person B's predict(records) call in main.py.

## Person D — Integration (Semester 2, interface merge)

### Status
- Merged Person B's Semester-2 model/predict_load() (LSTM + MC Dropout + Prophet ensemble, counterfactual correction, anomaly check) into main.py.
- The real loop now calls predict_load() and unpacks (predicted_cpu, upper_bound, anomaly_flag).
- upper_bound is logged but not yet consumed by DecisionEngine.evaluate().
- Renamed model/saved/evaluation_chart.png to evaluation_chart_DUMMY_DATA_semester1.png and added model/saved/README.md.

## Person B — LSTM Model

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
