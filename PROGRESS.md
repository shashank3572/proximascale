# PROGRESS.md — feature/app-monitoring (Person A)

## Week 1–2 (Environment + Schema)
- Set up virtual environment, installed Flask, Locust, Docker SDK
- Agreed data schema with team: `{timestamp, cpu_percent, memory_percent, request_rate, post_scaling}`
- Created `app/app.py` with CPU-load routes and a request counter

## Week 3–4 (Flask app + monitoring)
- Flask routes `/work/light`, `/work/heavy`, `/health` — measurable CPU load via math loop
- `before_request` hook increments a SQLite-backed request counter (`monitoring/metrics.py`), not a hardcoded value
- `monitoring/collector.py` polls **container** CPU/memory via the Docker SDK (cgroup deltas) — not host `psutil`
- `monitoring/schema.py` — `MetricRecord` dataclass (`to_dict`/`from_csv_row`/`from_dict`)
- `monitoring/storage.py` — `append_row()` / `read_last_n()`, thread-safe CSV persistence

## Week 5 (Sem 2: post_scaling + integration audit fixes) ← current
- [x] `post_scaling` field added to `MetricRecord` and wired through the collector
- [x] `is_post_scaling()` reads `data/scaling_events.json` (guarded against malformed/missing JSON)
- [fix] `collector.py` now calls `storage.append_row()` instead of duplicating CSV-writing logic — `storage.py` is no longer dead code
- [fix] `collector.py` now calls `reset_request_count()` once per cycle (atomic read-and-reset) instead of a separate `get_request_count()` + `reset_request_count()`, which could drop a request that arrived in between
- [fix] `requirements.txt` had an unresolved git merge conflict (unrelated `ml/` deps pasted in) — resolved to just this module's real dependencies (Flask, docker, locust, requests); unused `psutil` pin removed
- [fix] `app/Dockerfile` referenced files (`requirements.txt`, `monitoring/`) outside its own build context — Dockerfile now documents the correct build command: `docker build -f app/Dockerfile -t proximascale-app .` run from the repo root
- [removed] An unauthorized `ml/` shortcut pipeline (RandomForest predictor + a 14MB committed `.pkl`) and edits to `main.py` / `actuator/docker_scaler.py` were introduced by mistake in commit `06806fb`. These were outside this module's scope — real LSTM+Prophet work belongs to Person B, and `main.py`/the actuator belong to Person D/Person C. Removed; `main.py` and `actuator/docker_scaler.py` reverted to empty placeholders for their owners to fill in.

## TODO (Week 5–6) — still outstanding, needs a real run
- [ ] Run the full collection cycle for real: `docker build`, `docker run --cpus=0.5 --memory=256m`, all three Locust scenarios, collector active — this needs an actual machine with Docker, which is why it isn't done here
- [ ] Target: **≥1,500 rows**, spanning **≥3 days**, covering normal/spike/ramp/quiet periods, with genuine container CPU excursions into the 70–90%+ range and `post_scaling` populated
- [ ] `data/collected/metrics.csv` currently has 81 real rows (correct 5-column schema, generated 2026-09-12) — a valid start, but well short of the 1,500-row / 3-day target. Do not treat this file as the final handoff dataset.
- [ ] Multi-laptop Locust master/worker test — not yet evidenced anywhere in this repo
- [ ] Hand off the final `metrics.csv` to Person B and get explicit confirmation it's usable for LSTM training

## Commit log (this week)
```
[monitoring] add post-scaling field to metric schema
[monitoring] update storage for post-scaling field
[monitoring] align locust scenarios
[monitoring] add collected metrics dataset (partial — 81 rows)
[fix] resolve requirements.txt merge conflict
[fix] correct Dockerfile build-context documentation
[fix] wire collector.py to storage.append_row(), atomic request-count reset
[fix] remove unauthorized ml/ pipeline; revert main.py and actuator/docker_scaler.py to placeholders
[fix] update PROGRESS.md to match actual code
```
