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

# sustained spike: 25 users for 60 s (Locust is installed)
cd D:\proximascale
.\.venv\Scripts\locust.exe -f "<your locustfile>" --headless -u 25 -r 5 -t 60s --host http://localhost:5000
```

Measured 2026-09-22 (25 users, 60 s): 80 requests to `/work/heavy`, average
15.4 s per request; container CPU 0.02 % → 100 % and back.

---

## 8. What the demo shows, stage by stage

Poll interval is 30 s, so a panel run wants **8-10 minutes** of live time.

| Stage | Wall clock | What happens | What to point at |
|---|---|---|---|
| A. Cold start | 0-5 min | collector fills the window; the loop prints "need N records" | no decisions on partial data |
| B. Idle | 5-8 min | `pred ≈ actual ≈ 0-2 %`, `hold (in_band)` | the band `[15, 60]`; the forecast tracks the real CPU |
| C. Spike | start Locust | `pred` rises, `scale_up` fires, a worker appears | `docker ps` shows `proximascale-worker-1-...` |
| D. Sustained | during spike | `hold` — `cooldown` or `anomaly_interval` | one spike adds at most one worker |
| E. Decay | after Locust stops | `pred` falls below the lower bound → `scale_down (cpu_low)` | the worker disappears from `docker ps` |

Measured on 2026-09-22, **before** live calibration existed (the forecast level
is therefore the old absolute one; kept only as evidence that both directions
worked end to end):

```
11:43:00  actual=0.18   pred=50.22                 → hold
11:43:21  actual=100.0  pred=100.0  anomaly=True   → scale_up (anomaly)      replicas 0 → 1
11:44:22  actual=0.02   pred=51.42  ub=56.80       → scale_down (cpu_low)    replicas 1 → 0
```

The current build should show `pred ≈ actual` in stage B instead of ~50. That is
the fix for synthetic-baseline anchoring, and it is the single most important
thing to demonstrate. Re-measure and write down your own numbers before the
panel; do not reuse the 2026-09-22 table as if it were today's run.

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
