# Troubleshooting

Symptom-first. Every entry names the check that distinguishes the causes, so a
failed run is diagnosable instead of guesswork.

## Docker

**`docker ps` fails / "error during connect"**
Docker Desktop is not running (or WSL2 has not finished starting). Start Docker
Desktop, wait for "Engine running", then re-run. The collector and the engine
both degrade instead of crashing, so the *absence* of errors is not proof:
`python main.py` prints `<reason>_no_actuator` on every decision when the
actuator is unavailable.

**Workload container stopped**
`docker ps -a` shows `proximascale-app` as `Exited`. The collector logs
`collection skipped: container 'proximascale-app' is not running (state=exited,
exit=137)` and writes **no** row. Start it:

```powershell
docker start proximascale-app
curl.exe http://localhost:5000/health      # expect {"status":"ok"}
```

Exit code 137 is SIGKILL: usually Docker Desktop shutting down, or a host
OOM kill, not an application bug.

**Port 5000 already in use**
`netstat -ano | findstr :5000`. Stop whatever owns it, or run the app on
another port and point the collector at it:
`$env:PROXIMASCALE_APP_PORT="5001"`.

## Stale metrics

**`Newest record (...) is stale`**
`python main.py` refuses to scale on stale data — by design. The collector is
not running, or is failing. Check its terminal for `collection skipped`.
Confirm the file is advancing:

```powershell
Get-Content data\collected\metrics_live.csv -Tail 3
```

**Dashboard says "Data is stale: newest decision was Ns ago"**
Same cause: no decision rows. `logs/events.csv` is written by `main.py`, not by
the collector.

## Prediction / model

**`predict_load FALLBACK (...)` in the log**
Every prediction is failing and the loop is running on the degraded path
(`current_cpu * 1.15`, `is_anomaly=False`). The exception type is in the
message. Common causes: `model/saved/` artifacts missing, a TensorFlow import
conflict (make sure `tools/sitecustomize.py` is installed into the venv's
site-packages), or a window with fewer than 10 readings.

**`feature 'memory_percent' live range [...] is outside the training range`**
Expected on this build: the bundled scaler was fit on synthetic data with
~11-63 % memory, the real container uses <1 %. The forecast is still produced;
the model sees an extrapolated channel. Retraining on live-scale data is the
real fix, and this warning exists so nobody claims otherwise.

**Predictions look "too low" after an idle period**
That is the live calibration working: `predicted_cpu` is anchored to the
*measured* CPU, so an idle container forecasts ~0-2 %, not the synthetic
Prophet baseline. Set `PROXIMASCALE_LIVE_CALIBRATION=off` to compare against
the raw model output.

## Requests always read 0

1. Is the app threaded? `python app.py` must run with `threaded=True`
   (it does; single-threaded werkzeug queued the reset behind `/work/heavy`).
2. Is the counter reachable?
   `curl.exe -X POST http://localhost:5000/request-rate/reset` →
   `{"request_count": N, "window_seconds": ..., "requests_per_second": ...}`.
3. Raise the timeout if the host is slow:
   `$env:PROXIMASCALE_REQUEST_RESET_TIMEOUT="15"`.
4. A `-1` in the CSV means **both** channels failed; a `0` means the app
   answered and genuinely saw no workload requests.

## Scaling does not happen

* **scale_up never fires**: check the effective band in the decision line
  (`band [lower, upper]`). With `adaptive_floor_upper: 60` the trigger cannot
  fall below 60 %. If `actual` is above the upper bound and nothing happens,
  look at the reason — `max_reached` means `max_containers` is already running.
* **scale_down never fires**: `min_containers` may already be met
  (`min_reached`), the cooldown may still be active (`cooldown`), or the guard
  is holding capacity (`scale_down_guard`, i.e. current CPU is above
  `scale_down_max_current_cpu`). With `config.yaml` the floor is
  `min_containers: 1`, so the fleet will not reach zero — use
  `config_demo.yaml` for that.
* **`*_no_actuator`**: Docker SDK unavailable. See the Docker section.

## Workers

**Worker created but never healthy**
`docker inspect --format "{{json .State.Health}}" proximascale-worker-1-...`.
The probe is `GET /health` on port 5000 inside the container. If the image
does not run the Flask app, the probe fails forever; set
`worker_health_check: false` to skip it (the actuator then treats "running" as
ready).

**Do workers receive traffic?**
In the default **pool mode, no.** Workers are extra capacity, not a load
balanced fleet: `worker_status()` reports `traffic_serving: False` and
`describe_mode()` reports `has_load_balancer: False`. To make them addressable,
set `worker_host_port_base: 5001` in the config — worker *n* then publishes
`5001+n-1 -> 5000`. You still need a proxy in front of them; a `proximascale-
nginx:latest` image exists for that, but no nginx configuration ships with this
repository, so proxying is **not** verified here. Do not claim traffic
distribution in a report without adding and testing that proxy.

**Leftover workers after a crash**
`docker ps -a --filter "label=proximascale.managed=true"`, then
`docker rm -f <name>`. `python cleanup.py` also handles this.

## Dashboard

**`streamlit: command not found`**
Use `.venv\Scripts\streamlit.exe run dashboard/live_plot.py`.

**Dashboard is empty but events exist**
`logs/events.csv` must have a parseable `timestamp` column. A half-written
final line is dropped rather than blanking the page; check the file's tail.

**"No SHAP attribution recorded"**
SHAP is computed only on `scale_up` events and costs ~256 model evaluations.
If the last scale-up predates the explainability wiring, the row carries zeros.
Trigger one more scale-up.
