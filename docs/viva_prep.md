# ProximaScale — Viva Preparation Guide

Everything below is verified against the actual code on `dev` (not the older
master-plan notes, which contain a few stale numbers). If a fact isn't here,
check the file listed before saying it in the viva.

---

## 1. The 30-second pitch (everyone memorises)

> "ProximaScale forecasts container CPU load about 90 seconds ahead and scales
> Docker containers *before* demand spikes arrive. It uses residual stacking:
> Prophet models the trend and seasonality, and an LSTM learns only what
> Prophet gets wrong. On top of that we add MC-Dropout uncertainty estimation,
> anomaly detection, counterfactual correction, adaptive thresholds, and
> SHAP-based explainability of every scaling decision."

One-line version (if asked "so what IS your project?"):

> "An AI-driven proactive auto-scaling system — it predicts load before it
> happens and scales containers in advance, instead of reacting after a
> threshold is already breached."

**The one word to keep repeating: PROACTIVE.** Reactive scalers act *after*
CPU crosses 75%. We act *before*, on a forecast. Everything else in the
project exists to make that forecast trustworthy (uncertainty, anomaly
fallback) and safe (cooldown, counterfactual, guards).

---

## 2. The pipeline in plain words (the flow you should be able to draw)

```
Flask app in Docker (--cpus=0.5, --memory=256m cgroup limits)
        |  Locust generates load: normal / ramp / spike patterns
        v
monitoring/collector.py  (separate process, polls every 30 s)
   - reads CONTAINER cpu/memory via Docker SDK (cgroup deltas),
     NOT host psutil
   - request count via POST /request-rate/reset (read-and-reset over HTTP,
     works across the container boundary)
   - tags post_scaling=True for 180 s after any scaling event
        v
data/collected/metrics_live.csv  {timestamp, cpu_percent, memory_percent,
                                  request_rate, post_scaling}
   (metrics.csv stays the frozen 90k synthetic TRAINING dataset -- the
    collector never writes to it, or every model window would cross a
    synthetic→live discontinuity)
        v
main.py control loop (every 30 s):
   1. storage.read_last_n(10)          -- last 10 readings = 5 min of history
      (skips the cycle if data is stale > 3 polls, or fewer than 10 rows)
   2. predict_load(window)  [model/predict.py]:
        a. counterfactual correction on cpu_percent
        b. MinMax scaling (scaler fit on TRAIN data only)
        c. Prophet forecasts the NEXT 3 steps (30 s apart → ~90 s ahead),
           anchored to the window's own last timestamp
        d. LSTM predicts the RESIDUAL (what Prophet gets wrong)
        e. MC Dropout: 30 stochastic passes, mean + 2σ = upper_bound
        f. final = Prophet forecast + LSTM residual, clamped to 0–100
        g. anomaly check: rolling Z-score > 2.5 AND latest reading ≥ 5%
           (upward spikes only; the absolute floor keeps idle jitter —
           0.35% vs a 0.00–0.05% baseline — from firing fake scale-ups)
        → returns (predicted_cpu, upper_bound, anomaly_flag)
   3. engine.evaluate(predicted_cpu, upper_bound, anomaly_flag)  [decision/engine.py]
        1. anomaly_flag        → scale_up  (bypasses everything)
        2. cooldown active     → hold      (180 s hysteresis; gates all
                                            branches except anomaly)
        3. upper_bound > thresh→ scale_up  (risk-aware: mean safe, bound risky)
        4. predicted > thresh  → scale_up  (cpu_high)
        5. predicted < thresh  → scale_down(cpu_low)
        6. otherwise           → hold      (in_band)
      thresholds are ADAPTIVE: rolling mean ± 1.5σ of the last 30
      predictions, clamped, with a static 75/30 fallback until 10 samples
      and a minimum 10-point band so steady load can't collapse the band.
   4. actuator executes: docker SDK starts/stops worker containers
      (min 1, max 10; new ones constrained to 0.5 vCPU / 256 MB;
      scale-down removes the NEWEST worker by Created timestamp)
   5. on every scale action: data/scaling_events.json (atomic append)
      → this is what feeds post_scaling → counterfactual correction (loop!)
   6. on every decision: logs/events.csv (predicted, actual, bound, signal,
      replica count, reason, SHAP attributions) → Streamlit dashboard
   7. on scale-up only: SHAP KernelExplainer attributes the decision to
      CPU / memory / request rate
```

Know at least: **why the scaling_events.json → post_scaling → counterfactual
loop is circular.** The scaler writes an event; the collector reads it and
tags the next 180 s of rows; the model blends those rows toward pre-scaling
values so it doesn't learn its own actions as if they were natural load
changes. That's the counterfactual correction — and it's one of the
original novelties.

---

## 3. Verified results (the ONLY numbers you should quote)

From `model/evaluate.py` on the 90k-row synthetic dataset (17,990 evaluated
windows, 30 s sampling) — current residual-stacking architecture, verified
<21/09/2026:

| Method            | RMSE | MAE  | Avg lead time (s) | Oscillations |
|-------------------|-----:|-----:|------------------:|-------------:|
| Reactive baseline | 6.64 | 3.92 | 0.0               | 72           |
| Univariate LSTM   | 5.13 | 3.03 | 38.3              | 102          |
| Prophet alone     | 10.15| 4.67 | 0.0               | 0            |
| **Hybrid (ours)** | **5.09** | **3.06** | **57.5** | 96 |

How to present these honestly and confidently:

- **Hybrid beats Prophet alone by ~2× RMSE** (5.09 vs 10.15) — proof the LSTM
  residual correction adds real value.
- **Lead time 57.5 s vs 38.3 s for univariate LSTM** (+50%). Target was ≥60 s;
  we're just under it — say "approaching the 60-second target, and 50% better
  than the univariate baseline." Never claim 60 s+ as measured.
- **Reactive lead time is 0** — by definition it can only act after breach.
- **Oscillations (96 vs reactive's 72):** if asked — proactive systems take
  more scaling actions *by design*; the pathological flip-flopping (scale-up
  storms followed by scale-down) was a real bug we found in live demos and
  fixed by putting the cooldown gate before the risk branch (regression-tested
  in `tests/test_live_demo_fixes.py`).

**Do NOT quote the old table** (Reactive 38.95 / Multivariate 16.97 etc.) —
that was the 121-real-row dataset with the abandoned fixed-weight ensemble.
It's marked HISTORICAL in PROGRESS.md and superseded.

**Dataset honesty (memorise this framing, don't apologise):**

> "The 90,000-row training set is synthetic, generated with parameterised
> traffic simulation covering steady load, gradual ramps, and sudden spikes.
> Synthetic workload generation is standard in autoscaling research — the
> Google cluster traces and Alibaba traces used in the literature are
> themselves replay datasets. We also collected 121 rows of real container
> data (`metrics_real_dev.csv`) to validate the monitoring pipeline end to
> end. Never present `metrics.csv` as measured data — it's synthetic."

---

## 4. Who explains what (module ownership + scripts)

### Person A — Shravani V (1AM23CI140) — App + Monitoring + Data

**Files:** `app/app.py`, `app/Dockerfile`, `monitoring/collector.py`,
`monitoring/schema.py`, `monitoring/storage.py`, `data/locust_scenarios/*`

**Your 60–90 second script:**

> "I built the application layer and the data collection pipeline. The Flask
> app exposes `/work/light` and `/work/heavy` — real CPU-bound math loops, so
> load is measurable, not simulated — plus `/health`. A SQLite-backed request
> counter increments in a `before_request` hook. The app runs in Docker with
> `--cpus=0.5 --memory=256m`, which throttles it like a small cloud instance,
> so CPU genuinely saturates under load. The collector polls the Docker SDK
> every 30 seconds and reads *container-level* CPU and memory from cgroup
> deltas — not host metrics, which is an important distinction. One subtle
> bug we fixed live: Docker's `stats(stream=False)` diffs against a *cached*
> previous snapshot, which is stale after idle periods — CPU would flap
> between 0.01% and 100%. We now take two fresh snapshots 2 seconds apart and
> diff those. Request counts come from an HTTP read-and-reset endpoint,
> because the counter lives inside the container — the host's SQLite file is
> never written, so reading it locally showed zeros forever. Every row also
> carries a `post_scaling` tag for 180 seconds after any scaling event,
> stored via thread-safe CSV append, which feeds the counterfactual
> correction downstream."

**Likely questions:**

1. *What metrics do you collect and how?* — Container CPU% / memory% via
   Docker SDK cgroup deltas (two snapshots 2 s apart), request rate via an
   atomic HTTP read-and-reset counter, plus a post_scaling tag.
2. *Why Docker cgroup limits?* — Simulates a constrained cloud VM (like a
   t2.micro); gives real throttling so spikes into the 70–90%+ range are
   genuine.
3. *Why not host psutil?* — We scale containers; host CPU averages across
   everything and would hide the per-container saturation we need to learn
   from.
4. *What is post_scaling and why does it exist?* — A boolean stamped for
   180 s after each scaling event (read from `data/scaling_events.json`).
   It marks rows where CPU dropped because *we* scaled, not because demand
   fell — feeding counterfactual correction.
5. *How did the request counter break across the container boundary?* — The
   Flask app runs inside a container, so its SQLite `metrics.db` is inside
   the container filesystem; the host copy is never written. Fix:
   `POST /request-rate/reset` — the collector asks the app to atomically
   read-and-reset its own counter each poll. Returns −1 (not a fake 0) if
   neither channel works.
6. *What is the data schema?* — `{timestamp, cpu_percent, memory_percent,
   request_rate, post_scaling}`; 30 s poll interval, matching the model's
   trained sampling rate.
7. *How is storage made safe?* — `monitoring/storage.py` with thread-safe
   `append_row()` / `read_last_n()`; the collector calls `storage.append_row()`
   rather than writing CSV itself (was dead code before that fix).
8. *What generates load?* — Locust scenarios: `normal_load.py`,
   `spike_load.py`, `gradual_ramp.py` — the spike scenario drives CPU into
   the 70–90%+ range.
9. *What happens if the collector dies?* — main.py detects stale data (newest
   row older than 3 poll intervals) and skips the cycle with a warning — it
   never scales on stale data.

---

### Person B — Shreyas R (1AM23CI141) — LSTM + Residual Stacking (novelty core)

**Files:** `model/preprocessing.py`, `model/prophet_model.py`,
`model/lstm_model.py`, `model/train.py`, `model/uncertainty.py`,
`model/anomaly.py`, `model/counterfactual.py`, `model/predict.py`,
`model/evaluate.py`, `model/saved/`

**Your 60–90 second script:**

> "I built the prediction engine. The architecture is residual stacking, not
> a blend: Prophet models trend and seasonality and gives the base forecast;
> the residual — actual minus Prophet, computed on training data — is what
> the LSTM learns. Final prediction = Prophet forecast + LSTM residual. We
> first tried a fixed-weight ensemble and it performed *worse* than
> standalone LSTM, which is exactly why residual stacking is sounder: the
> LSTM targets Prophet's failure modes instead of averaging two independent
> opinions. The LSTM is multivariate — 10 timesteps of CPU, memory, and
> request rate — with a 3-step output, so we see ~90 seconds ahead.
> Architecture: LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) →
> Dropout(0.2) → Dense(3), trained with Adam and MSE. Preprocessing fits the
> MinMax scaler on training data only — no leakage — with a chronological
> split. Two things make the forecast trustworthy: MC-Dropout uncertainty —
> we keep dropout active at inference and run 30 stochastic passes; the
> spread gives σ, and we act on mean + 2σ — and a rolling Z-score anomaly
> check (z > 2.5, upward spikes only) that bypasses the forecast entirely
> when load jumps unexpectedly. The evaluation shows hybrid RMSE 5.09 vs
> Prophet alone at 10.15, with 57.5 s average lead time."

**Likely questions:**

1. *What is "hybrid" about it?* — Residual stacking (say the script's
   version). Crucially: final = Prophet + LSTM residual, an ADDITION, not a
   weighted average. The old blend lives in `model/ensemble.py` as a
   deprecated record — a test enforces it stays unused.
2. *Why keep Prophet at all if LSTM alone scores similar RMSE?* — Two
   reasons: RMSE isn't the only metric — the hybrid's lead time is 57.5 s vs
   38.3 s univariate, and lead time is what proactive scaling is about. And
   Prophet anchors the long-run trend/seasonality so the LSTM only has to
   learn short-term corrections, which generalises better than asking one
   model to learn everything.
3. *Why multivariate input?* — Request rate and memory lead CPU. The
   literature reports large RMSE improvements (~1.84× in the cited paper)
   for multivariate over univariate workload prediction.
4. *What does the model output?* — 3 values: predicted CPU at t+30 s, t+60 s,
   t+90 s. We act on the farthest step.
5. *Explain MC Dropout.* — Dropout is normally disabled at inference. We call
   the model with `training=True` 30 times; each pass drops different
   neurons, giving a distribution over predictions. Mean = the forecast; std
   = model uncertainty; upper_bound = mean + 2σ. The decision engine scales
   up when the *bound* is risky even if the mean is safe. No separate
   probabilistic model or ensemble needed.
6. *What is counterfactual correction?* — When we scale up, CPU drops
   artificially. Without correction the model learns "after scale-up, load
   collapses" and predicts low, causing premature scale-down. Rows tagged
   `post_scaling=True` are blended toward the last pre-scaling value before
   inference, matching how the model was trained. Not in any cited paper —
   an original contribution.
7. *What if the model crashes at runtime?* — `predict_load()` never raises:
   it falls back to (current_cpu × 1.15, current_cpu × 1.3, anomaly=True),
   logged loudly, so the control loop keeps running and errs toward
   caution. Also: a stale pre-residual `.keras` model once sat at MODEL_PATH
   making every prediction fall back silently — we now point at the residual
   `.h5` and log loudly on fallback.
8. *How do you prevent data leakage?* — MinMaxScaler fit on the training
   split only, chronological (not random) train/test split, scaler reused
   at inference — never re-fit.
9. *Why is the Prophet forecast anchored to the window's timestamp?* —
   An old bug: `predict_prophet(periods=3)` forecast from wherever Prophet's
   *training data* ended, not the live window. We now pass explicit future
   timestamps computed from the window's own last reading, so the forecast
   is correct regardless of how long ago training happened.
10. *Why clamp predictions to 0–100?* — Prophet + residual + 2σ is
    unbounded and live runs showed upper bounds of 156 while the container
    was pegged at 100. Clamping keeps decision thresholds meaningful, and
    preserves ordering so a pegged bound still reads as high risk.

---

### Person C — Soubhagya Ranjan Behera (1AM23CI148) — Decision Engine + Actuator

**Files:** `decision/engine.py`, `decision/adaptive_threshold.py`,
`decision/hysteresis.py`, `decision/scaling_log.py`,
`actuator/docker_scaler.py`, `actuator/scaler_interface.py`

**Your 60–90 second script:**

> "I turn predictions into actions. The decision engine's `evaluate()`
> takes the predicted CPU, the MC-Dropout upper bound, and the anomaly flag,
> and checks branches in a deliberate order: anomaly first — a genuine
> emergency bypasses everything; then the cooldown gate — 180 seconds of
> hysteresis that suppresses every remaining branch; then the risk-aware
> branch — if the upper bound crosses the threshold even while the mean is
> safe, we scale up preemptively; then plain cpu_high / cpu_low; otherwise
> hold. The branch order matters and we have a regression test for it: in
   live demos we saw scale-ups fire six polls in a row because the
> upper-bound branch sat *before* the cooldown — the bound decays slower
> than the mean after a spike, so every poll re-triggered. Moving the gate
> fixed the storm: one spike → one scale-up. Thresholds are adaptive: a
> rolling 30-sample window, mean ± 1.5σ, clamped to sane floors/ceilings,
> with a static 75/30 fallback until enough samples and a minimum 10-point
> band so steady load can't collapse the band to a point. The actuator uses
> the Docker SDK for real container start/stop, enforcing min 1 / max 10
> replicas, each new container constrained to 0.5 vCPU and 256 MB.
> Scale-down removes the newest worker by Created timestamp — deterministic,
> and configurable via `scale_down_strategy`. Every real action is appended
> atomically to `data/scaling_events.json`, which closes the loop with the
> counterfactual pipeline."

**Likely questions:**

1. *How does a prediction become actual scaling?* — evaluate() returns a
   signal; on scale_up/scale_down the engine calls the actuator inside
   `_do()`; only a *successful* SDK call arms the cooldown and writes the
   event log. Failed calls (max replicas reached, Docker down) return hold
   with a `max_reached` / `min_reached` / `no_actuator` reason instead.
2. *What prevents oscillation?* — Three layers: the 180 s cooldown gating
   every non-anomaly branch; counterfactual correction stopping the model
   from predicting a fake post-scale collapse; and the adaptive band's
   minimum width.
3. *Why adaptive thresholds instead of a static 75%?* — A static threshold
   is wrong in both directions across varying baselines. Rolling mean ± kσ
   adapts to the current regime — this is a documented research gap in
   reactive scalers. It degrades gracefully: static 75/30 until 10 samples,
   clamped between floors/ceilings, min band 10.
4. *Where is the "proactive" part in your code?* — Branch 3: the engine
   consumes `upper_bound` (from MC Dropout), not just the mean prediction.
   Scaling on the *bound* is preemption under uncertainty — plus the
   prediction itself is ~90 s ahead of reality.
5. *Why does cooldown come before the risk branch?* — Live-observed bug: the
   upper bound stays above threshold for many polls after the mean turns
   down, so the risk branch re-fired scale_up every poll (6 consecutive
   scale-ups on one spike). The bound is a *statistical* risk signal, not a
   measured emergency — one spike deserves one scale-up per cooldown window.
   Anomaly still bypasses, because that's a measured emergency.
6. *Why is scaling_events.json important?* — Two consumers: the collector's
   `post_scaling` tag (→ counterfactual correction) and auditability.
   Written atomically — temp file + `.replace()`, thread lock, required-key
   validation.
7. *Why remove the newest container on scale-down?* — Two spec docs
   conflicted (newest vs oldest); we adopted newest as the stricter default
   and made it configurable (`scale_down_strategy: newest|oldest`),
   selecting deterministically by sorting `container.attrs["Created"]`.
8. *Why 0.5 vCPU / 256 MB per container?* — Matches the constrained-app
   setup so worker replicas are interchangeable with the monitored app, and
   forces genuine saturation so the model has real spikes to learn from.
9. *What if Docker is down?* — The engine degrades to a no-actuator hold
   (reason `*_no_actuator` or `*_error`) — decisions still log, nothing
   crashes.
10. *What do the tests cover?* — `tests/test_decision.py` (including the
    upper_bound branch and event-log writing), `tests/test_adaptive_threshold.py`
    (fallback, clamps, min-band), `tests/test_scaling_log.py`, plus
    `tests/test_live_demo_fixes.py` for the cooldown-gate regression.

---

### Person D — Shashank B (1AM23CI138) — Integration + Dashboard + SHAP + Repo Owner

**Files:** `main.py`, `dashboard/live_plot.py`, `dashboard/shap_explain.py`,
`dashboard/results_charts.py`, `baseline/reactive_scaler.py`,
`config.yaml`/`config_demo.yaml`, `cleanup.py`, CI workflow, docs, README

**Your 60–90 second script:**

> "I own integration — making four people's modules actually run as one
> system — plus the dashboard and explainability. The control loop in
> main.py reads the last 10 readings, calls predict_load(), feeds all three
> outputs into evaluate(), executes the signal, and logs every decision to
> logs/events.csv with the predicted value, upper bound, replica count, the
> engine's *reason string*, and SHAP attributions. Stale-data guard: if the
> newest row is older than three poll intervals, the cycle is skipped — we
> never scale on old data. For explainability, every scale-up runs SHAP: we
> use KernelExplainer because DeepExplainer fails on Keras 3 models. The
> explained input is built by the exact same `prepare_scaled_window()` the
   prediction used — one source of truth, no scaling drift — against a
> background of the current window plus a neutral 50%-load reference so
> attributions don't degenerate to zero. It's capped at 256 explainer
> samples and only on scale-ups, and any failure just logs zeros — SHAP can
> never kill the control loop. The Streamlit dashboard shows actual vs
> predicted with the confidence band, scale event markers, anomaly markers,
> replica count, and a SHAP bar chart. I also built the reactive baseline
> scaler for comparison, own the Git workflow — protected main, PRs into
> dev, CI running pytest + flake8 on every PR — and ran the audit passes:
   11 verified fixes before the live demo, then 8 more found and fixed
> during a real demo run, taking the suite to 113 passing tests."

**Likely questions:**

1. *What integration problems did you actually fix?* — Real ones: real loop
   called `collect_metrics(window=10)` which takes no args (TypeError);
   the model path pointed at an unloadable legacy model so every prediction
   silently fell back; the dashboard read `logs/events.csv` that nothing
   wrote; SHAP wasn't wired into the loop at all; events lacked a reason
   column. Each was reproduced before fixing.
2. *Why KernelExplainer and not DeepExplainer/TreeExplainer?* — DeepExplainer
   doesn't support Keras 3 models. KernelExplainer is model-agnostic at a
   runtime cost, which we manage: 256 samples, only on scale-up, cached
   model, tiny background set.
3. *What's in the SHAP background set and why does it matter?* — Rows sampled
   from the frozen training distribution (Prophet's full diurnal cycle) through
   the same scaler, plus the current window as anchor. KernelExplainer's
   "importance" is distance from background members it averages over: with the
   explained window itself in the set (an earlier design), the explainer could
   trivially copy the input and attribution collapsed to the same ~77/23/0
   split on every decision. With only one background point equal to the input,
   every attribution would instead degenerate to ~0.
4. *What's on the dashboard?* — Actual vs predicted CPU, confidence band
   (mean to upper bound), scale-up ▲ / scale-down ▼ markers, anomaly dots,
   replica-count card, per-event reason, SHAP attribution bars. It tolerates
   old event files without the reason column (migrates them).
5. *How do you compare against reactive scaling?* — `baseline/reactive_scaler.py`:
   static threshold, no ML, plus the 4-way table from `model/evaluate.py`.
6. *Why does config_demo.yaml exist?* — Demo-only settings
   (`min_containers: 0` so the demo can show full scale-to-zero; relaxed
   anomaly threshold). Real `config.yaml` keeps min 1 and the production
   anomaly threshold — never generate report numbers from the demo config.
7. *How is the repo run?* — Protected `main`; all work via PRs into `dev`;
   GitHub Actions runs pytest + flake8 on every PR (TF/Prophet tests skip on
   CI, so the full suite runs locally before model merges); only the repo
   owner merges.
8. *What happens when the loop starts?* — Scaling events are cleared, the
   engine is built from config, and the loop starts; Ctrl+C stops cleanly
   with events preserved in logs/events.csv.
9. *Why is there a replica-count column in events?* — To correlate decisions
   with fleet state over time on the dashboard — you can see scale-ups move
   the count.
10. *What does cleanup.py do?* — Removes the `proximascale-worker-*`
    containers the actuator created. (An earlier version filtered the wrong
    name pattern and removed nothing — fixed in the audit pass.)

---

## 5. Cross-cutting questions (ANYONE can be asked these)

**"What is your project in one line?"** → Section 1 one-liner.

**"Walk me through what happens when a spike arrives."** — Locust drives
`/work/heavy` → container CPU climbs → collector records it (30 s) →
predict_load forecasts ~90 s ahead; MC Dropout widens the upper bound; if
bound or mean crosses the adaptive threshold (and cooldown allows), the
actuator starts a worker → scaling event logged → post_scaling tags the
next 180 s → counterfactual correction blends those rows → dashboard shows
the ▲ marker and SHAP explains which feature drove it.

**"What is your novelty?"** — Four things: (1) residual-stacking hybrid
(paper-backed), (2) multivariate 3-step-ahead input (paper-backed),
(3) MC-Dropout uncertainty for autoscaling decisions — not in any cited
paper, (4) counterfactual-corrected inference — not in any cited paper.
Plus supporting: adaptive thresholds, SHAP on live scaling decisions.

**"Why LSTM and not ARIMA/Prophet alone?"** — Workload is non-linear and
history-dependent; ARIMA/Prophet capture trend+seasonality but not sharp
spike dynamics. Our own table: Prophet alone RMSE 10.15 and lead time 0 vs
hybrid 5.09 / 57.5 s.

**"Why not just use Kubernetes HPA?"** — HPA is reactive: it scales on
*current* measured metrics after breach. Our contribution is exactly the
part HPA lacks — forecasting ahead, uncertainty-aware preemption, and
explainability. (Also: K8s was explicitly scoped out as demo risk.)

**"Why not reinforcement learning for decisions?"** — Out of scope: needs
massive interaction data and is hard to validate safely. Rule-based +
adaptive thresholds is auditable, testable, and sufficient for the forecast
quality we have. (Removed from scope deliberately.)

**"What if your prediction is wrong?"** — Three safety nets: MC-Dropout
upper bound scales early under uncertainty; Z-score anomaly detection
bypasses the model entirely on sudden spikes; the fallback path (on any
model error) reports anomaly=True, so the system errs toward scaling up,
never silently does nothing.

**"What if the model is wrong in the other direction (over-scaling)?"** —
Cooldown (180 s, one action per window), max_containers=10, min_containers=1,
and scale-down requires the *mean* prediction below the lower adaptive band.

**"Why 90 seconds / why 3 steps?"** — 30 s sampling × 3 steps. It's the
horizon from the reference paper; long enough to start a container before
breach, short enough that LSTM residual corrections stay accurate. Lead
time experiment: 57.5 s average.

**"Why 10 timesteps of history?"** — The window spans 5 minutes of recent
behaviour at 30 s sampling — enough for the LSTM to see slope and onset
shape without diluting recent dynamics.

**"Why synthetic data?"** → Section 3 framing. Add: "Our architecture is
data-agnostic — retraining on real traces is a config change, not a redesign."

**"What are the limitations?"** — Say them BEFORE being asked, it earns trust:
synthetic training data (121 real rows only); Prophet anchored to training
timeline, so live-mode predictions can diverge during severe spikes (the
LSTM residual partially compensates; demos use --simulate or recorded data
for the dashboard); local Docker, not a real cluster; CI skips TF tests.

**"Cost of MC-Dropout at inference?"** — 30 forward passes instead of 1 on a
tiny model (2 LSTM layers) — milliseconds, every 30 s. Trivial vs the
polling interval. SHAP is the expensive part, hence only on scale-up.

**"Is SHAP faithful for an LSTM?"** — KernelExplainer with a proper
background set is model-agnostic Shapley attribution — it's an approximation
(sampled coalitions, 256), not exact, and we present it as decision
*attribution*, not causation.

**"What did you try and abandon?"** — Fixed-weight LSTM/Prophet ensemble:
performed worse than standalone LSTM, replaced by residual stacking (kept
in `model/ensemble.py` as a historical record, a test enforces it's unused).
Also dropped: stress.py (replaced by Docker cgroup limits), Kubernetes, RL
decision engine, multi-service scaling — all scoped out deliberately.

**"Which result are you proudest of?"** — The lead-time gap: 57.5 s vs 0 for
reactive, with comparable RMSE — that's the *proactive* claim measured, not
asserted. Second: the cooldown-gate bug we caught live and regression-tested.

**"What's the biggest engineering lesson?"** — Integration bugs beat model
bugs: a stale model path silently degraded every prediction; a cached Docker
stats snapshot made CPU flap 0.01↔100%; a branch-ordering mistake caused
scale-up storms. All were found by running the real loop, not unit tests
alone — hence the audit passes and 113 tests.

---

## 6. Demo assignments (who says what)

| Time | Action | Spoken by |
|------|--------|-----------|
| 0:00 | Open dashboard — actual vs predicted, confidence band | D: "live monitoring — blue actual, orange predicted, shaded uncertainty band" |
| 0:30 | Trigger spike (Locust /work/heavy) | A: "30 concurrent users on the heavy endpoint; the app is throttled to 0.5 vCPU so saturation is real" |
| 1:30 | CPU climbing | A: "container CPU via Docker SDK — two fresh snapshots 2 s apart, not host metrics" |
| 2:30 | scale_up ▲ fires | C: "CPU hasn't hit 75% — that's the forecast + upper bound acting early. This is proactive" |
| 3:00 | SHAP bars update | D: "SHAP explains it — e.g. request rate drove most of this decision. Auditable AI" |
| 3:30 | Replica 1 → 2 | C: "a new constrained worker via the Docker SDK; load distributes; adaptive band recentres" |
| 4:00 | Lead time | B: "we scaled ~57 seconds before breach on average — reactive is 0 by definition" |
| 4:30 | Results table | B: "hybrid RMSE 5.09 vs Prophet alone 10.15 — the residual correction is earning its place" |
| 5:00 | Q&A | Everyone owns their module; cross-cutting per Section 5 |

**Failure fallbacks (decide before the demo):**
- Docker daemon down → run `python main.py --simulate` (mocked actuator,
  same engine logic) and narrate the decision flow from logs.
- Model load fails → the fallback path still produces decisions; narrate the
  degraded-mode design ("never raises, errs toward caution") — that's itself
  a talking point.
- WiFi dies → single-laptop setup; pre-recorded backup video is mandatory.

---

## 7. Rapid-fire numbers card (last thing you read before entering)

- Poll 30 s · window 10 steps (5 min) · horizon 3 steps (~90 s)
- LSTM: 64→Dropout(0.2)→32→Dropout(0.2)→Dense(3); Adam lr=0.001; input (1,10,3)
- MC Dropout: 30 passes, training=True, upper = mean + 2σ
- Anomaly: rolling Z > 2.5, upward only
- Thresholds: adaptive mean ± 1.5σ (30-sample window), clamp, min band 10,
  static fallback 75/30
- Cooldown 180 s · containers min 1 / max 10 · 0.5 vCPU + 256 MB each
- Data: 90k rows synthetic (17,990 eval windows) + 121 real rows
- Results: hybrid RMSE 5.09 / MAE 3.06 / lead 57.5 s vs reactive RMSE 6.64 /
  lead 0 s; Prophet alone RMSE 10.15
- Tests: 113 passing · CI: pytest + flake8 on every PR
- SHAP: KernelExplainer, 256 samples, on scale-up only
