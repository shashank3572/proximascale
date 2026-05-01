# PROGRESS.md — feature/app-monitoring (Person A)

## Week 1–2 (Environment + Schema)
- Set up virtual environment, installed Flask, psutil, Locust
- Agreed data schema with team: `{timestamp, cpu_percent, memory_percent, request_rate}`
- Created initial `app/app.py` with CPU-load routes and request counter

## Week 3–4 (Flask App + Collector skeleton)
- Flask routes `/`, `/heavy` generating measurable CPU load via math loop
- `before_request` hook incrementing `_request_count`
- `monitoring/collector.py` skeleton — polls psutil every 5s, writes to CSV
- **Issue found:** `request_rate` was hardcoded to `0` — `/metrics` endpoint missing

## Week 5 (Fixes + Full module completion) ← current
- [fix] Added `/metrics` JSON endpoint to `app.py` exposing live `request_rate`
- [fix] Fixed `app.run(host='0.0.0.0')` so Flask is reachable inside Docker
- [fix] `collector.py` — separated `collect_metrics(window)` (returns list of dicts)
  from `run_collector()` (infinite polling loop). Calls `/metrics` instead of hardcoding 0.
- [new] `monitoring/schema.py` — `MetricRecord` dataclass with `to_dict()` and `from_csv_row()`
- [new] `monitoring/storage.py` — `append_row()` and `read_last_n(n)` with CSV header management
- [new] `app/Dockerfile` — `python:3.10-slim`, copies app.py, installs flask+psutil, EXPOSE 5000
- [new] `data/locust_scenarios/spike_load.py` — `SpikeShape` (0→100 users instantly, 2 spikes)
- [new] `data/locust_scenarios/gradual_ramp.py` — `GradualRampShape` (+10 users/30s to 100)
- [fix] `data/locust_scenarios/normal_load.py` — added `/heavy` task with weight 1:3 ratio
- [new] `requirements.txt` — pinned flask, psutil, locust, requests

## TODO (Week 5–6)
- [ ] Regenerate `metrics.csv` — run all 3 Locust scenarios with collector active
- [ ] Target: ≥1,000 rows with real variance in all 3 columns
- [ ] Hand off `metrics.csv` to Person B for LSTM training

## Commit log (this week)
```
[monitoring] fix request_rate: add /metrics endpoint to app.py
[monitoring] fix app.run host to 0.0.0.0 for Docker compatibility
[monitoring] refactor collector: separate collect_metrics(window) from run_collector()
[monitoring] add schema.py: MetricRecord dataclass with to_dict/from_csv_row
[monitoring] add storage.py: append_row and read_last_n with CSV header
[monitoring] add Dockerfile for app container
[monitoring] add spike_load.py: SpikeShape LoadTestShape
[monitoring] add gradual_ramp.py: GradualRampShape LoadTestShape
[monitoring] fix normal_load.py: add /heavy task
[monitoring] add requirements.txt with pinned versions
[monitoring] update PROGRESS.md
```
