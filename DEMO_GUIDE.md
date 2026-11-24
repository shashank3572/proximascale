# ProximaScale — demo and review guide

Windows + PowerShell + Docker Desktop, separate terminal windows. Commands are
copy-paste ready. A quoted number is either **measured** (dated) or a **shape**
to expect (`~`); nothing is invented and no run has been back-filled.

---

## 1. What this system actually does

```
app container (proximascale-app, :5000)
        │  Docker stats every 30 s
        ▼
monitoring/collector.py ──► data/collected/metrics_live.csv
        ▼
main.py ── predict_load(window) ──► (predicted_cpu, upper_bound, is_anomaly)
        │        Prophet + residual LSTM, re-anchored on the measured CPU
        ▼
decision/engine.py ── adaptive band ──► scale_up / scale_down / hold
        ▼
actuator/docker_scaler.py ── creates/removes proximascale-worker-* containers
        ▼
logs/events.csv ──► dashboard/live_plot.py (Streamlit)
```

| Container | Role |
|---|---|
| `proximascale-app` | The workload. This is the container the collector measures. |
| `proximascale-worker-*` | Capacity the scaler adds. Pool mode: extra capacity, **not** a traffic-sharing fleet (§10). |

---

## 2. Prerequisites

* Windows 10/11, Docker Desktop running (`docker version` prints a Server version).
* Python 3.12 with the project venv: `.venv\Scripts\python.exe`.
* The three images present:

```powershell
docker images --format "{{.Repository}}:{{.Tag}}" | Select-String proximascale
# proximascale-app:latest
# proximascale-worker:latest
# proximascale-nginx:latest
```

If `proximascale-app` is missing, build it from the repository root:

```powershell
docker build -f app/Dockerfile -t proximascale-app .
```

---

## 3. Start the workload container

```powershell
docker start proximascale-app
Start-Sleep -Seconds 3
docker ps --filter "name=proximascale-app" --format "{{.Names}}  {{.Status}}"
# proximascale-app  Up 5 seconds
curl.exe -s http://localhost:5000/health
# {"status":"ok"}
```

If it does not exist yet:

```powershell
docker run -d --name proximascale-app --restart unless-stopped `
  -p 5000:5000 proximascale-app:latest
```

`--restart unless-stopped` matters: without it Docker Desktop shutting down
leaves the workload `Exited (137)` and the collector has nothing to measure.

---

## 4. Terminal 1 — collector

```powershell
cd D:\proximascale
.\.venv\Scripts\python.exe -m monitoring.collector
```

One line every ~30 s (shape, not a recorded run):

```
2026-11-24T14:02:11.418 | Container CPU: 0.31% | Container Memory: 0.72% | Requests: 0 (0.00/s) | Post-scaling: False
```

* `Requests` is **requests per poll window**; the parenthesised value is the
  derived rate.
* `Requests: -1` means the app did not answer *and* the local counter failed.
* Container stopped → `collection skipped: container 'proximascale-app' is not
  running (state=exited, exit=137)` and **no CSV row**; the collector survives.

---

## 5. Terminal 2 — control loop

```powershell
cd D:\proximascale
.\.venv\Scripts\python.exe main.py
```

First ~5 minutes (10 readings needed; 30 for the v4 bundle):

```
WARNING Only 3 records in CSV — need 10. ~7 more samples (~210s) from collector...
```

Then one decision line per poll:

```
2026-11-24 14:07:22 [INFO] pred 0.9 | ub 2.6 | actual 0.3 | band [15.0, 60.0] | mode=delta_calibrated | anomaly=False → hold (in_band)
```

| Field | Meaning |
|---|---|
| `pred` | forecast CPU ~90 s ahead, **already re-anchored on `actual`** |
| `ub` | the forecast's uncertainty bound, shifted by the same offset |
| `band [lower, upper]` | the **effective** adaptive thresholds used for this decision |
| `mode` | `delta_calibrated`, `absolute` (`PROXIMASCALE_LIVE_CALIBRATION=off`) or `fallback` |
| `→ hold (reason)` | the decision and why |

Reason codes: `in_band`, `cooldown`, `cpu_low`, `cpu_high`, `upper_bound_risk`,
`current_cpu_high`, `anomaly`, `anomaly_interval`, `scale_down_guard`,
`min_reached`, `max_reached`, `no_actuator`, `error`.

---

## 6. Terminal 3 — dashboard

```powershell
cd D:\proximascale
.\.venv\Scripts\streamlit.exe run dashboard/live_plot.py
```

http://localhost:8501. Reads `logs/events.csv` and plots actual/predicted CPU
with the confidence band, **the effective upper/lower thresholds the engine
logged** (not a config value), scale_up/scale_down/anomaly markers, requests per
poll window, and a freshness caption or stale warning. UI preview without a live
run: `-- --demo`.

---

## 7. Generating load

```powershell
# one heavy request (saturates the container for several seconds)
Measure-Command { curl.exe -s http://localhost:5000/work/heavy | Out-Null }

# sustained spike, the run measured below (Locust is installed)
cd D:\proximascale
.\.venv\Scripts\locust.exe -f data/locust_scenarios/spike_load.py `
  --headless -u 20 -r 5 -t 120s --host http://localhost:5000
```

Measured 2026-11-24 (20 users, 120 s, `spike_load.py` → `/work/heavy`):
**194 requests, 0 failures**, mean 11.45 s, median 12 s, max 15.02 s, 1.63 req/s.
Container CPU went 0.02 % → 100 % → 0.02 %; the collector's request counter read
5, then 62 requests in one poll window while the spike was live, and 0 again one
poll after the last request.

An earlier run on 2026-09-22 (25 users, 60 s) produced 80 requests to the same
endpoint at 15.4 s average. Nothing else from that run is reused here.

---

## 8. What the demo shows, stage by stage

Poll interval is 30 s, so a panel run wants **8-10 minutes** of live time.

| Stage | Wall clock | What happens | What to point at |
|---|---|---|---|
| A. Cold start | 0-5 min | collector fills the window; the loop prints `Only N records in CSV - need 10` | no decisions on partial data |
| B. Idle | 5-8 min | `pred ≈ actual ≈ 0.0 %`, `hold (min_reached)` | band `[30, 75]` while static, `[10, 60]` once `adaptive_min_samples: 5` is met; the forecast tracks the real CPU |
| C. Spike | start Locust | `pred` and `ub` jump to 100, `scale_up (upper_bound_risk)`, a worker appears | `docker ps` shows `proximascale-worker-1-...` |
| D. Sustained | during spike | `hold (cooldown)`, then at most one further `scale_up` per `cooldown_seconds` | the cooldown is what spaces scale-ups here; `anomaly_min_interval_seconds` gates the anomaly branch only |
| E. Decay | after Locust stops | `pred` back to 0, below the lower bound → `scale_down (cpu_low)` | workers leave `docker ps`; `replicas` in `events.csv` goes 2 → 1 → 0 |

### Measured 2026-11-24, end to end

`main.py --config config_demo.yaml`, collector at 30 s, Locust `spike_load.py`
20 users for 120 s. Rows below are the real `logs/events.csv` decisions (band is
the engine's *effective* `[lower, upper]`):

```text
14:10:22  actual=0.02   pred=0.0   ub=0.83   band [30.0, 75.0]  anomaly=False → hold (min_reached)          replicas 0
14:12:38  actual=0.02   pred=0.0   ub=1.17   band [10.0, 60.0]  anomaly=False → hold (min_reached)          replicas 0   requests 5
14:13:27  actual=100.0  pred=100.0 ub=100.0  band [10.0, 72.6]  anomaly=False → scale_up (upper_bound_risk) replicas 0 → 1  requests 62
14:14:01  actual=100.0  pred=100.0 ub=100.0  band [10.0, 90.0]  anomaly=False → hold (cooldown)             replicas 1   requests 48
14:14:42  actual=100.0  pred=100.0 ub=100.0  band [10.0, 90.0]  anomaly=False → scale_up (upper_bound_risk) replicas 1 → 2  requests 46
14:15:16  actual=0.02   pred=0.0   ub=0.0    band [10.0, 90.0]  anomaly=False → hold (cooldown)             replicas 2
14:15:55  actual=0.02   pred=0.0   ub=1.32   band [10.0, 90.0]  anomaly=False → scale_down (cpu_low)        replicas 2 → 1
14:16:29  actual=0.02   pred=0.0   ub=0.86   band [10.0, 90.0]  anomaly=False → hold (cooldown)             replicas 1
14:17:08  actual=0.02   pred=0.0   ub=0.62   band [10.0, 90.0]  anomaly=False → scale_down (cpu_low)        replicas 1 → 0
14:18:16  actual=0.02   pred=0.0   ub=0.0    band [10.0, 90.0]  anomaly=False → hold (min_reached)          replicas 0
```

What that run actually shows — say exactly this much, no more:

1. **The forecast tracks the measurement in both directions.** `pred` was 0.0
   while the container was idle and 100.0 while it was pegged, with `actual`
   identical each time; the old mis-scaled ~50 % idle forecast is gone.
2. **Both scale-ups came from `upper_bound_risk`, not from an anomaly.** With
   live-scale CPU, Prophet's residual z-score stayed under
   `anomaly_z_threshold: 2.5` for the whole spike, so `anomaly=False` on every
   row. Do not claim the anomaly branch fired in this run — it did not.
3. **Two workers for one spike is the configured behaviour, not a double-fire
   bug.** `cooldown_seconds: 60` allows one scale-up per minute, and the
   container was still at 100 % at 14:14:42.
   `anomaly_min_interval_seconds: 300` only spaces the anomaly branch, which
   never fired here. (On 2026-09-22, before live calibration and the spacing
   gate, the anomaly branch did fire and drove the same 0 → 1 scale-up.)
4. **The band adapts and stays inside its configured limits:** static
   `[30, 75]` → `[10, 60]` once five samples existed → upper 72.6 from the
   spike window → capped at `adaptive_ceil_upper: 90`.
5. **Scale-down is cooldown-paced too, ~73 s apart, 2 → 1 → 0.**
   `min_containers: 0` is why the idle stage reads `hold (min_reached)` instead
   of `hold (in_band)`: at zero workers there is nothing left to remove, so the
   engine reports the limit rather than the band.
6. **SHAP ran on both scale-ups:** 65.5 % CPU / 11.9 % memory and 76.1 % CPU /
   5.8 % memory. Attribution exists only on scale-up rows; holds carry zeros.

These numbers come from `config_demo.yaml` (60 s cooldown, 8-sample band,
`min_containers: 0`). They demonstrate the control loop; they are not
production figures, and the run measured nothing about lead time or forecast
error — `lead time (s)` and MAE come from `logs/events.csv` and
`model/evaluate.py`, never from this table.

Before the panel, rerun it and write down your own numbers. Do not read this
table out as if it were live.

---

## 9. Verified commands (and what they print)

```powershell
# 1. compile + lint
.\.venv\Scripts\python.exe -m compileall -q app monitoring model decision actuator dashboard data tools main.py cleanup.py
.\.venv\Scripts\python.exe -m flake8 app monitoring model decision actuator dashboard main.py cleanup.py --select=E9,F63,F7,F82
# (no output = clean)

# 2. dependencies
.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.

# 3. model self-test (loads the real artifacts)
.\.venv\Scripts\python.exe model\predict.py
# [ProximaScale] predict_load() -> predicted_load=..., upper_bound=..., is_anomaly=False
# [ProximaScale] Fallback test (short window): (..., ..., False)
# [ProximaScale] predict_load() self-test PASSED.

# 4. full test suite (~2-3 min)
.\.venv\Scripts\python.exe -m pytest tests -q
```

The fallback assertion is `is_anomaly=False`: a degraded model must not claim to
have detected a spike.

---

## 10. Honest limitations (say these out loud)

1. **Workers do not receive traffic.** Pool mode has no load balancer;
   `worker_status()` reports `traffic_serving: False` and `describe_mode()`
   reports `has_load_balancer: False`. Claiming load-balanced scaling would be
   false. `worker_host_port_base` publishes one port per worker if you add a
   proxy (see `docs/DEMO_TROUBLESHOOTING.md`).
2. **One workload container plus workers.** The measured container is
   `proximascale-app`; adding workers does not by itself change its CPU.
3. **The bundled scaler was fit on synthetic data.** Live memory (~0.7 %) and
   request rate (0-15) sit outside its training range; the model logs that once
   per process. Retraining on live-scale data is future work.
4. **Lead time is small.** The horizon is 90 s (3 x 30 s steps). Do not claim
   more lead time than the logs show; `lead time (s)` is computed from
   `logs/events.csv`, never asserted.
5. **`config_demo.yaml` is demo-only** (min 0 workers, 60 s cooldown, 8-sample
   window, proactive switches enabled). Never quote numbers produced with it.

---

## 11. Simulation mode (no Docker, no ML)

```powershell
.\.venv\Scripts\python.exe main.py --simulate
```

Seven hardcoded scenarios print their signal with a **0 s cooldown** so every
scenario is visible. The output therefore does not contain `hold_cooldown`; that
step is documented as skipped in simulation, not as passing.

---

## 12. Cleanup

```powershell
# stop the control loop and collector with Ctrl+C in their terminals
docker ps -a --filter "label=proximascale.managed=true" --format "{{.Names}}"
docker rm -f (docker ps -aq --filter "label=proximascale.managed=true")
docker stop proximascale-app

# optional: metrics_live.csv is a ring buffer, safe to reset between runs
Remove-Item data\collected\metrics_live.csv -ErrorAction SilentlyContinue
Remove-Item logs\events.csv -ErrorAction SilentlyContinue
```

`cleanup.py` does the container side automatically.

---

## 13. If something goes wrong

See **`docs/DEMO_TROUBLESHOOTING.md`** — symptom → check → fix, covering Docker
unavailable, stopped container, stale metrics, insufficient observations,
missing model artifacts, request-rate failures, worker startup and dashboard
failures. Metric units and definitions are in **`docs/METRICS.md`**.
