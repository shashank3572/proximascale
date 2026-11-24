# Metric definitions

Mixing these up caused audit finding C4: the collector stored a per-window
count, the dashboard drew it as if it were a rate, and a 3 s reset timeout
turned every measurement into 0 exactly when the system was saturated. Every
metric below has exactly one definition, and the place it is produced says so.

## CPU (`cpu_percent`)

* Source: Docker `stats` on the workload container (`proximascale-app`).
* Measured as the CPU delta between **two consecutive snapshots** `CPU_SAMPLE_GAP`
  seconds apart (default 2 s), scaled by `online_cpus`, capped at 100.
* Unit: percent of one CPU-equivalent second, 0-100.
* If the container is **not running**, or Docker does not return the CPU
  fields, no row is written at all. A missing measurement is never recorded as
  `0.0` — "the container used no CPU" and "we could not measure" are different
  facts, and the second one must not reach the model as a value.

## Memory (`memory_percent`)

* Source: Docker `memory_stats.usage / memory_stats.limit`.
* Unit: percent of the container's memory limit (not of host RAM).
* Note: with `mem_limit: 256m` this stays around 0.6-1.0 %, far below the
  synthetic training range of ~10.8-62.9 % used by the bundled scaler. That is
  a known distribution shift, not a bug; `model/predict.py::check_feature_ranges`
  logs it once per process and the honest fix is retraining on live-scale data.

## Requests (`request_rate`)

| Term | Definition | Where |
|------|------------|-------|
| `request_counter` | Cumulative workload requests since the last reset (SQLite, survives app restarts) | `monitoring/metrics.py` |
| `window_seconds` | Wall-clock seconds since the previous reset, measured **inside the app** | `POST /request-rate/reset` |
| `requests_per_second` | `request_count / window_seconds` | `POST /request-rate/reset` |
| `request_rate` (CSV) | **Requests per collector poll window** — a count, not a rate | `metrics_live.csv` |
| `-1` | The app and the local SQLite fallback were both unreachable | `metrics_live.csv` |

The collector prints the derived rate next to the count
(`Requests: 41 (1.37/s)`), so both readings are visible in the log.

Only workload endpoints count. `/`, `/health`, `/metrics`, `/request-rate`,
`/request-rate/reset` and `/predict` are excluded in `app/app.py`
(`_EXCLUDED_FROM_COUNT`) — otherwise every collector poll would add 1 to its
own measurement.

## Prediction (`predicted_cpu`, `upper_bound`)

* `predicted_cpu`: forecast CPU ~90 s ahead, **re-anchored on the measured CPU**
  (`current_cpu + forecast_delta`). See `model/predict.py`.
* `upper_bound`: the forecast's one-sided uncertainty bound, shifted by the same
  offset so the band keeps its width.
* Both clamped to 0-100. `forecast_mode` in `events.csv` records which path ran:
  `delta_calibrated`, `absolute` (`PROXIMASCALE_LIVE_CALIBRATION=off`) or
  `fallback`.

## Thresholds (`upper_threshold`, `lower_threshold`)

* The **effective** values the engine compared against for that decision
  (adaptive band after clamping), not the static config values.
* `mode` is `static` until `adaptive_min_samples` predictions exist, then
  `adaptive`.
* The dashboard plots these columns. The config value is only a fallback for
  logs written before this column existed.
