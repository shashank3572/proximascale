# ProximaScale — Repository Validation Report

**Date of audit:** 25 November 2026
**Repository state:** `D:\proximascale`, branch head `139e42b8` (`Merge pull request #8 from shashank3572/fix/audit-h1-h3`)
**Auditor role:** repository audit, execution, measurement and manuscript reconciliation
**Artifacts produced by this audit:** `results/validation_2026-11-25/`, `paper/`

This report records what was inspected, what was executed, what passed, what
failed, what was fixed, and what could not be verified. It is written to be
read alongside `paper/CHANGELOG_PAPER_VS_IMPLEMENTATION.md` and the raw outputs
in `results/validation_2026-11-25/`.

---

## 1. Scope and method

The audit proceeded in the order requested:

1. Read the existing manuscript (`ProximaScale_IEEE_Research_Draft.docx`).
   Its extracted text is 1,000,000 characters, of which roughly 27,000 are
   prose and the rest are embedded base64 images; the extraction is truncated
   at Section IV-E, so Sections V-IX and the reference list of the original
   draft were **not available** to this audit. Those sections were rebuilt from
   the repository's measured results and verified citations.
2. Inventoried the repository (excluding `.venv`): 15,561 lines of Python in 60
   modules, 28 test modules, 2 synthetic datasets of 90,000 rows each, 121 rows
   of real container data, 8 trained model artifacts, 4 evaluation harnesses,
   Dockerfiles, Locust scenarios, CI workflows and 8 documentation files.
3. Ran the test suite, then each evaluation harness that produces a reported
   number, capturing exit status and logs.
4. Ran two additional controlled experiments not present in the repository:
   a fixed-configuration decision replay with the reactive floor disabled, and
   a counterfactual-weight sweep reproduced through the v4 bundle.
5. Regenerated every figure and table from the resulting CSVs and rebuilt the
   manuscript.

No file in `results/` was overwritten: all new outputs were written to new
directories (`results/validation_2026-11-25/`, `results/eval_dump/v4_verify/`).
No commit was created and nothing was pushed.

---

## 2. What the system actually does

### 2.1 Data flow as implemented

```
Locust / Flask /work/*  ->  proximascale-app container (:5000)
        |
        |  Docker stats, two snapshots 2 s apart, one row per 30 s poll
        v
monitoring/collector.py  ->  data/collected/metrics_live.csv
        |                     (+ POST /request-rate/reset for the request count)
        v
main.py  ->  model/predict.py::predict_load(30 rows)
        |        counterfactual correction -> feature scaling
        |        -> Prophet fitted/forecast for the window's own timestamps
        |        -> 3 seeded LSTMs, 10 MC passes each, residual prediction
        |        -> Prophet + residual -> live delta calibration -> 0-100 clamp
        |        -> calibrated 95% upper bound -> anomaly flag
        v
decision/engine.py  ->  adaptive band (mu +/- 1.5 sigma, 30 predictions)
        |               -> cooldown 180 s, scale-down guard 75%, optional floor
        v
actuator/docker_scaler.py  ->  creates/removes proximascale-worker-* containers
        |
        v
logs/events.csv  ->  dashboard/live_plot.py (Streamlit)
```

### 2.2 Functionality status

| Capability | Status | Evidence / location |
|---|---|---|
| Container CPU and memory monitoring (Docker SDK, 2-snapshot delta) | Implemented | `monitoring/collector.py`, `monitoring/metrics.py`; tests `test_monitoring.py` |
| Request counting across the container boundary (HTTP read-and-reset) | Implemented | `app/app.py` `POST /request-rate/reset`; `test_monitoring.py` |
| Missing measurement is not recorded as 0 | Implemented | collector writes no row; documented in `docs/METRICS.md` |
| Prophet baseline forecasting | Implemented | `model/prophet_model.py`, `model/saved/proximascale_prophet.pkl` |
| LSTM residual stacking (Prophet value + Prophet forecast as input channels) | Implemented | `model/forecaster.py`, bundle `model/saved/proximascale_v4/` |
| 3-seed Monte Carlo dropout ensemble (30 passes per decision) | Implemented | `forecaster.py::forecast`, `config.json` `default_passes: 30` |
| Calibrated one-sided upper bound (split conformal, sigma-binned) | Implemented | `config.json` `upper_bound`, `forecaster.py::upper` |
| Live delta calibration on measured CPU | Implemented | `model/predict.py::_apply_live_calibration` |
| Rolling z-score anomaly detection with absolute floor | Implemented | `model/anomaly.py` |
| Adaptive threshold band with clamps and inversion guard | Implemented | `decision/adaptive_threshold.py` |
| Cooldown / hysteresis, replica limits, scale-down guard | Implemented | `decision/engine.py`, `config.yaml` |
| Docker actuation with health check and readiness wait | Implemented (pool mode) | `actuator/docker_scaler.py` |
| Decision logging with effective thresholds and reasons | Implemented | `main.py::log_event`, `logs/events.csv` |
| SHAP attribution on scale-up | Implemented | `dashboard/shap_explain.py`, `main.py::_compute_shap` |
| Dashboard | Implemented | `dashboard/live_plot.py` |
| Offline evaluation harnesses | Implemented | `eval_dump.py`, `forecast_metrics.py`, `decision_replay.py`, `tune_decisions.py`, `unseen_eval.py`, `cf_ablation.py` |
| Uncertainty as an *instance-specific* confidence | **Partial** | computed and used, but raw sigma is uncalibrated (Section 5, F7) |
| Counterfactual correction | **Partial** | implemented with fixed 0.6 weight; 0.2 is better on held-out post-scaling rows (F8) |
| Feedback / closed loop | **Partial** | online feedback is the monitoring loop plus delta calibration; no online learning |
| Actuation that improves end-to-end latency | **Not implemented** | pool mode, no load balancer; `traffic_serving` reports false by design |
| Real-traffic forecasting accuracy | **Not established** | only 121 idle rows; every model worse than persistence on them |
| Online retraining / drift adaptation | **Not implemented** | model weights are static; only the level offset adapts |
| Kubernetes / multi-host scaling | **Not implemented** | single-host Docker only |
| Weighted-average LSTM+Prophet ensemble | **Removed / deprecated** | `model/ensemble.py`, excluded from the pipeline by test |
| 10-reading model | **Superseded** | still used for 10-29 readings (cold-start fallback) |
| `/metrics` endpoint, in-memory request counter, `ml/` RandomForest pipeline | **Removed** | recorded in `PROGRESS.md` |

---

## 3. Commands executed

All commands were run from `D:\proximascale` with the project virtualenv
(`.venv\Scripts\python.exe`), Python 3.12.10, on Windows 11 x64.

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest tests -q` | **210 passed**, 19 warnings, 61.05 s |
| 2 | `python model/evaluate.py` | **Failed**: `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f9e0'` (see B1) |
| 3 | `PYTHONIOENCODING=utf-8 python model/evaluate.py` | Ran, but the per-window Python loop had processed 2 of 17,988 windows after ~250 s; stopped. Used the vectorised harness instead (B1) |
| 4 | `python model/eval_dump.py --bundle model/saved/proximascale_v4 --tag v4_verify` | **Exit 0**, 277 s, 5 seeds x 30 passes, 17,988 windows; Table II check reproduced |
| 5 | `python model/forecast_metrics.py --dump results/eval_dump/v4_verify --out-dir results/validation_2026-11-25/forecast_metrics_v4` | **Exit 0**, 56 s; 13 CSVs incl. Table II, significance, regimes, uncertainty, sweeps |
| 6 | `python model/decision_replay.py --dump results/eval_dump/v4_verify --out-dir results/validation_2026-11-25/replay_v4` | **Exit 0**, 42 s, 10 policies x 5 seeds |
| 7 | `python model/decision_replay.py ... --config results/validation_2026-11-25/config_no_floor.yaml` | **Exit 0**, 42 s; controlled variant: same config with `scale_up_on_current_cpu: false` |
| 8 | `python paper/figures/make_figures.py` | 7 figures + dataset statistics |
| 9 | `python paper/build_paper.py` | DOCX (615 kB, 7 images, 7 tables), PDF (14 pages), Markdown (~8.8k words) |
| 10 | `python -m pip check` | **No broken requirements found** after restoring `pillow<11` and `reportlab==4.2.5` |

The evaluation harnesses write provenance into `meta.json` (dataset SHA-256,
artifact SHA-256, git commit, settings). `forecast_metrics.py` refuses to run
if `metrics.csv` changed since the dump, and re-derives Table II from the dump
rather than trusting it.

### 3.1 Environment side effects

To produce DOCX and PDF deliverables, `python-docx`, `reportlab` and `pypdf`
(plus `pypdfium2` for page rendering checks) were installed into the project
venv. The first `reportlab` resolution upgraded `pillow` to 13.0.0, which
conflicts with the pinned Streamlit 1.35 (`pillow<11`); this was corrected by
pinning `reportlab==4.2.5` and reinstalling `pillow<11` (10.4.0). `pip check`
is clean and the project's `requirements.txt` is unchanged. Paper-tooling
dependencies are listed separately in `paper/requirements-paper.txt` so they
never enter the runtime environment unnoticed.

---

## 4. Reproduced results

All values below are from `results/validation_2026-11-25/` (this audit) or
from committed CSVs whose provenance is pinned by checksums. Test split:
17,988 windows, 6.25 days, 36 spike onsets at the 75% threshold.

### 4.1 Forecast accuracy and event metrics (v4 dump, `forecast_metrics_v4/v4_verify/table_ii.csv`)

| Method | RMSE | MAE | Lead (s) | F1 | False alarms/day | Crossings |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 6.644 | 3.915 | 0.0 | 0.802 | 0.00 | 72 |
| Linear extrapolation | 10.247 | 7.710 | 45.8 | 0.814 | 9.29 | 196 |
| Prophet alone | 10.154 | 4.675 | 0.0 | n/a | 0.00 | 0 |
| Univariate LSTM | 5.131 | 3.032 | 38.3 | 0.815 | 1.12 | 102 |
| **ProximaScale v4** | **4.347 +/- 0.001** | **2.679 +/- 0.001** | **58.17 +/- 0.37** | **0.941** | **0.00** | **81.6** |
| Feasible ceiling | — | — | 58.33 | — | — | 34/2/0/0 at 90/60/30/missed |

Significance (Diebold-Mariano, HAC lags 2 and 60; 95% moving-block bootstrap):
dRMSE vs persistence -2.297 [-2.598, -1.996], p = 8.3e-32;
vs linear -5.899 [-6.066, -5.723], p < 1e-300;
vs Prophet -5.807 [-6.900, -4.679], p = 5.7e-38;
vs univariate -0.784 [-0.942, -0.631], p = 1.7e-14.
dLead vs univariate +19.8 s [13.3, 26.7].

### 4.2 Decision replay (`replay_v4/v4_verify/summary.csv`)

| Policy | Scale-ups | Reversals <=5 min | Spikes anticipated /36 | False ups/day | Mean replicas | % at max |
|---|---:|---:|---:|---:|---:|---:|
| P0 Reactive | 99 | 7 | 26 | 4.32 | 2.23 | 7.3 |
| P1 Univariate LSTM | 60 | 0 | 30 | 0.16 | 5.45 | 19.3 |
| P2 ProximaScale mean | 55 | 4.2 | 29 | 0.00 | 5.73 | 21.6 |
| P4 Original engine (all branches) | 149 | 99 | 7 | 18.67 | 8.71 | 80.0 |
| P5 P4 + static thresholds | 365 | 294 | 20 | 43.90 | 5.88 | 47.4 |
| P6 P4 without cooldown | 187 | 104 | 7.8 | 20.08 | 8.50 | 78.4 |
| P7 Upper bound + static thresholds | 102 | 19 | 34 | 0.00 | 3.42 | 11.1 |
| **P9 Shipped config.yaml** | **57** | **1** | **24** | **0.80** | **6.16** | **34.5** |
| P9 with reactive floor disabled | 55 | 0 | 29 | 0.00 | 5.73 | 21.6 |

### 4.3 Out-of-distribution (`results/unseen/forecast.csv`, served model = "v3 trained on both")

| Dataset | ProximaScale RMSE | Missed / onsets | Persistence RMSE | Univariate RMSE |
|---|---:|---:|---:|---:|
| fresh (seed 7) | 4.25 | 0 / 36 | 6.46 | 4.98 |
| shifted (3-min ramps, 2x) | 4.11 | 0 / 60 | 6.75 | 5.54 |
| shifted2 (2.5-min ramps, 1.5x) | 4.12 | 0 / 60 | 6.53 | 4.93 |
| real_dev (121 idle rows) | 35.96 | 1 / 1 | 22.67 | 27.40 |

### 4.4 Uncertainty

Raw MC-dropout sigma (`forecast_metrics/baseline/reliability.csv`,
`.../v4_verify/uncertainty.csv`): mean + 2 sigma covers 55.1% (10-step) and
59.5% (v4) of targets at a nominal 97.7%; Spearman(sigma, |error|) = 0.076
(10-step) and 0.033 (v4), and -0.53 / -0.20 inside spikes. The served
calibrated bound covers 94.85% overall (94.9% in the committed run) and about
87% above 75% CPU, with mean width ≈ 5 CPU points.

### 4.5 Counterfactual weight (`alpha_sweep.csv`, 360 post-scaling targets, scored against generator truth)

| Weight on observed | Configured | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| 0.0 | | 3.890 | 3.122 | +0.603 |
| 0.2 | | **3.274** | **2.642** | -0.820 |
| 0.4 | | 3.495 | 2.803 | -2.243 |
| **0.6** | **yes** | 4.430 | 3.763 | -3.665 |
| 0.8 | | 5.739 | 5.107 | -5.088 |
| 1.0 | | 7.223 | 6.511 | -6.511 |

Over all windows, correction on vs off changes RMSE by 0.0009
(`results/forecast_metrics/cf_ablation.csv`), because post-scaling rows are 2%
of the test set.

### 4.6 Anomaly detector (`anomaly_sweep.csv`, decision-time detector)

| z | Flags (% decisions) | Precision | Recall | Median delay (s) | False flags/day |
|---|---:|---:|---:|---:|---:|
| 2.0 | 6.49 | 0.130 | 1.0 | 0 | 162.5 |
| 2.5 (configured) | 3.54 | 0.175 | 1.0 | 0 | 84.1 |
| 3.0 | 2.17 | 0.269 | 1.0 | 0 | 45.6 |
| 3.5 | 1.41 | 0.402 | 1.0 | 0 | 24.3 |

### 4.7 Inference cost (`results/timing/baseline/latency.csv`, 300 calls)

predict_load: 68.4 ms (1 pass), 223.2 ms (5), 452.9 ms (10), 1360.2 ms (30);
p95 at 30 passes 1672.5 ms (4.5% of the 30 s interval). Univariate LSTM 47.1 ms;
SHAP attribution 522.4 ms per scale-up.

---

## 5. Defects and discrepancies found

Severity: **H** blocks a reported claim or a reproducible run, **M** misleads a
reader, **L** housekeeping.

| # | Sev | Finding | Evidence | Action taken |
|---|---|---|---|---|
| B1 | H | `model/evaluate.py` crashes on the Windows default console encoding: it prints an emoji (`\U0001f9e0`) and cp1252 cannot encode it, so the historical Table II cannot be regenerated by the documented command | Command 2 above | Documented; reported in the paper's implementation-validity section. Not patched, because the vectorised harness is the reproducible path and editing reporting code was outside the audit's mandate |
| B2 | H | `model/evaluate.py` iterates 17,988 windows in Python; at the observed rate (~2 windows in 250 s) a full run would take hours, contradicting `PROGRESS.md`'s claim that it is the standard evaluation command | Command 3 | Documented; used `eval_dump.py` + `forecast_metrics.py` (vectorised, 277 s + 56 s) for all reported numbers |
| B3 | H | `config.yaml` sets `scale_up_on_current_cpu: true`, while `README.md` and `docs/model_evaluation.md` state the reactive floor is **off** and was rejected as worse on every period | `config.yaml` line 56; README "exists but is off" | Not changed (user configuration is out of scope for an audit). The difference was measured with a controlled variant: shipped 24/36 anticipated, 0.8 false/day, 6.16 replicas; floor off 29/36, 0, 5.73. The paper reports both and flags the contradiction |
| B4 | H | Two decision harnesses disagree on the same dump and the same policy: `tune_decisions.py` records 36/36 anticipated, 0 false/day, 2.67 mean replicas ("chosen (validation)", `results/tuning/v4/chosen_test.csv`), while `decision_replay.py` produces 29/36, 0, 5.73 for the corresponding forecast-mean policy. The onset set and replica accounting differ between the harnesses | `chosen_test.csv` vs `replay_v4/.../summary.csv` | Reported as unresolved. The paper cites only replay-harness numbers (reproducible from the dump) and states that the tuning-harness figures should not be cited |
| B5 | H | `results/exp1_accuracy.csv` … `exp4_oscillation.csv` have no generating script anywhere in the repository (only `dashboard/results_charts.py`, which plots them). Exp 3's latency figures (1240/780/310 ms) cannot arise from the implemented pool-mode actuator, which does not route traffic to workers; Exp 2's 68-81 s lead times exceed the 90 s horizon/ramp geometry; Exp 4's 7-vs-2 oscillation counts are not produced by any harness | `git log -- results/exp*.csv` shows one commit adding charts and "sample experiment CSVs"; grep finds no writer | All four experiment CSVs are **withdrawn** from the manuscript and listed as unreproducible. No file deleted |
| B6 | M | README's headline "36/36 spikes … 2.7 replicas" is not reproducible through the replay harness (24/36 with the shipped config, 29/36 with the floor off) | Section 4.2 | Reported; README left unchanged (not the deliverable) |
| B7 | H | Raw MC-dropout variance is miscalibrated: nominal 97.7% coverage delivered as 55-59%, correlation with absolute error ≈ 0 | Section 4.4 | Reported as a negative result; the served bundle's conformal bound is described as marginal, not instance-specific |
| B8 | M | The configured counterfactual weight (0.6) is not the best on held-out post-scaling rows (0.2 gives RMSE 3.27 vs 4.43); its effect on aggregate RMSE is ~0 | Section 4.5 | Reported; weight not changed (model/training artifact, would require retraining to be meaningful) |
| B9 | M | Model scalers were pickled with scikit-learn 1.8.0 but `requirements.txt` pins 1.4.2; loading emits `InconsistentVersionWarning` | `pip`/run logs | Reported as an implementation-validity risk; artifacts load and reproduce the published numbers on this machine |
| B10 | L | Untracked `build_err.txt`, `build_out.txt`, `build_stderr.txt`, `build_stdout.txt` in the repository root; `.github/workflows/__init__.py` and `.github/__init__.py` are stray Python files in a workflows directory | `git status` | Reported only |
| B11 | M | `docs/METRICS.md` states the synthetic training memory range is ~10.8-62.9% and infers a distribution shift from it. The 3-feature scaler was actually fitted on 23.1-81.5% memory; 10.8-62.9 are the ranges of the two Prophet channels in the v4 scaler. The shift conclusion is still correct, but the stated range is wrong | `pickle` of `model/saved/scaler.pkl` (`data_min_[1]=23.1`, `data_max_[1]=81.54`) vs `docs/METRICS.md` | Reported; the paper quotes the measured 23-82% range |
| B12 | L | The historical manuscript's oscillation count (98) is not reproduced: the committed 10-reading dump gives 96.4 +/- 2.6, the served model 81.6 | `results/forecast_metrics/baseline/table_ii.csv`, `.../v4_verify/table_ii.csv` | Historical value removed from the paper; both current values reported |

---

## 6. Historical claims versus current evidence

| Claim in the earlier draft | Status | Current value / note |
|---|---|---|
| RMSE 5.08 | **Reproduced approximately** | 5.087 in the committed 10-reading dump (rounds to 5.09); the served 30-reading model gives 4.347 |
| MAE 3.06 | **Reproduced** | 3.062 (10-reading); 2.679 (served) |
| Mean lead time 57.5 s | **Not exactly reproduced** | 56.83 +/- 0.37 (10-reading); 58.17 +/- 0.37 (served); ceiling 58.33 |
| Scaling oscillations 98 | **Not reproduced** | 96.4 +/- 2.6 (10-reading); 81.6 +/- 1.7 (served) |
| Prophet alone RMSE 10.15 / MAE 4.68 | **Reproduced exactly** | 10.154 / 4.675 |
| "Input window: 10 observations" | **Changed in implementation** | 30 readings (15 min) in the served bundle; 10 still used below 30 readings |
| "Forecast horizon: 3 observations (90 s)" | **Confirmed** | horizon 3, 90 s |
| "Input features: CPU, memory, request rate" | **Changed** | 5 channels: those three plus Prophet yhat(t) and yhat(t+90 s) |
| "MC Dropout: 30 stochastic passes" | **Confirmed with change** | 30 passes, now pooled over 3 seeded models (10 each) |
| "y_corrected = 0.6 y_obs + 0.4 y_pred" | **Confirmed as code, corrected as description** | The second term is the last pre-scaling *observed* value, not a prediction: 0.6 observed + 0.4 pre-scaling baseline |
| "Uncertainty-aware decision-making" | **Weakened** | Bound is calibrated marginally; the upper-bound branch is disabled in `config.yaml`; raw sigma is uninformative |
| "Adaptive thresholds reduce oscillations" | **Supported directionally** | Replay: original engine 99 reversals vs 0-1 with cooldown + guard |
| "Feedback-driven loop" | **Narrowed** | Monitoring loop + live delta calibration; no online learning |
| "Experimental evaluation demonstrates feasibility" | **Confirmed for pipeline; not for SLOs** | No latency benefit measurable in pool mode |

---

## 7. Reproducibility instructions

```bash
# environment (Python 3.12)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp tools/sitecustomize.py .venv/Lib/site-packages/sitecustomize.py   # TF-before-sklearn on Windows

# 1. test suite
.venv/Scripts/python.exe -m pytest tests -q                      # 210 passed

# 2. full model metrics (vectorised; ~5.5 min total)
.venv/Scripts/python.exe model/eval_dump.py \
    --bundle model/saved/proximascale_v4 --tag v4_verify
.venv/Scripts/python.exe model/forecast_metrics.py \
    --dump results/eval_dump/v4_verify \
    --out-dir results/validation_2026-11-25/forecast_metrics_v4

# 3. decision replay with the shipped configuration
.venv/Scripts/python.exe model/decision_replay.py \
    --dump results/eval_dump/v4_verify \
    --out-dir results/validation_2026-11-25/replay_v4

# 4. controlled variant (reactive floor disabled)
#    results/validation_2026-11-25/config_no_floor.yaml is config.yaml with
#    scale_up_on_current_cpu: false
.venv/Scripts/python.exe model/decision_replay.py \
    --dump results/eval_dump/v4_verify \
    --out-dir results/validation_2026-11-25/replay_v4_nofloor \
    --config results/validation_2026-11-25/config_no_floor.yaml

# 5. out-of-distribution tests (regenerates results/unseen/)
.venv/Scripts/python.exe model/unseen_eval.py

# 6. figures and manuscript
.venv/Scripts/python.exe paper/figures/make_figures.py
.venv/Scripts/python.exe paper/build_paper.py
```

Notes:

* The dump uses 5 seeds x 30 passes and took 277 s on the audit machine.
  `eval_dump.py --limit N` produces a shorter dump for smoke tests.
* `model/evaluate.py` must be run with `PYTHONIOENCODING=utf-8` on Windows
  (B1), and is slow (B2); it is not the recommended path.
* The real-container path (`monitoring.collector.py`, `main.py`, Docker) needs
  Docker Desktop and the `proximascale-app` / `proximascale-worker` images. It
  was **not** executed in this audit: no Docker daemon was assumed, and doing
  so would have written live rows and created containers. The only live
  artifact examined is the existing `logs/events.csv` (158 decisions from a
  prior run), which is consistent with the documented loop.

---

## 8. What could not be verified

1. **End-to-end latency or SLO benefit.** The actuator adds pool workers that
   receive no traffic; there is no load balancer. The earlier latency table is
   withdrawn. No load test was run.
2. **Real-traffic accuracy.** Only 121 idle rows exist; every model is worse
   than persistence on them (RMSE 36.0 vs 22.7). No realistic real dataset is
   in the repository.
3. **A live Docker run in this audit.** Not executed, for the reasons above.
4. **Decision-level outcome at the policy level.** Two harnesses disagree
   (B4); only the replay harness is reproducible from the committed dump.
5. **The original manuscript's Sections V-IX and reference list.** The
   extracted text is truncated at Section IV-E because of embedded images, so
   the audit could not compare those sections line by line; they were rebuilt
   from repository evidence. The six citations visible in Sections I-II were
   verified bibliographically and retained.
6. **Training reproducibility.** The served artifacts are committed; retraining
   three seeds takes about 17 minutes per member on the audit machine
   (`train_seconds` 1027 s in `seed0/info.json`) and was not re-run, so the
   training pipeline's numerical reproducibility is asserted by the bundle's
   provenance metadata, not re-measured here.

---

## 9. Bottom line

The forecasting pipeline is real, tested and reproducible: 210 tests pass, the
served model scores RMSE 4.35 / MAE 2.68 at a 90 s horizon with 58.2 s mean
lead time against a 58.3 s ceiling, and the gains over four baselines are
statistically significant. The decision layer is real and its ablation is
informative, but its shipped configuration contradicts its own tuning notes,
and two internal harnesses disagree on the decision-level outcome. The
uncertainty mechanism exists and is calibrated only marginally. The actuation
path exists but cannot, in its current pool-mode form, improve application
latency. The earlier latency and oscillation experiments cannot be reproduced
from the repository and are withdrawn from the paper.
