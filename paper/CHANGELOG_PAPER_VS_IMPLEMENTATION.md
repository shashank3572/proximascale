# ProximaScale — Research Paper Change Log

**Revision date:** 25 November 2026
**Baseline document:** `ProximaScale_IEEE_Research_Draft.docx` (extracted text
truncated at Section IV-E; Sections V-IX and references were not present in the
extraction)
**Revised document:** `paper/ProximaScale_IEEE_Revised_2026-11-25.{docx,pdf,md}`
**Evidence base:** `results/validation_2026-11-25/` and committed result CSVs
(see `paper/REPOSITORY_VALIDATION_REPORT.md`)

Legend for *claim status*: **Retained** (verified, unchanged in substance),
**Modified** (verified but changed in value, scope or wording), **Removed**
(not supported or not reproducible), **Added** (new in this revision).

---

## 1. Front matter

| Item | Change | Why | Evidence | Status |
|---|---|---|---|---|
| Title | Unchanged | Research identity preserved; the framework still implements an uncertainty path, now described precisely | — | Retained |
| Authors | Unchanged (LaTeX-escaped emails in the original removed; roles normalised to department) | Formatting only | — | Retained |
| Abstract | Rewritten (~250 words): names the actual served architecture (30-reading window, 5 channels, 3-seed MC dropout, split-conformal bound), gives reproduced numbers (RMSE 4.35, MAE 2.68, lead 58.2 s, 33.8/36 spikes), and states three negative findings | Original abstract made feasibility claims unsupported by the code and contained the typo "on on" | `forecast_metrics_v4/v4_verify/table_ii.csv`, `replay_v4/.../summary.csv`, `uncertainty.csv` | Modified |
| Keywords | "uncertainty-aware scaling" → "uncertainty calibration"; "feedback-driven control" → "feedback control"; added Prophet | Terms must match what was measured | `model/predict.py`, `forecaster.py` | Modified |
| Fig. 1 | Redrawn as the deployed control loop (collector, CSV, preprocessor, forecaster, engine, actuator, pool, dashboard) instead of a conceptual five-stage diagram | Every element now corresponds to a module in the repository | `main.py`, `monitoring/collector.py`, `actuator/docker_scaler.py` | Modified |

## 2. Section I — Introduction

| Change | Why | Evidence | Status |
|---|---|---|---|
| Removed the claim that the paper "presents" a framework and re-framed the paper as implementation-led | The repository is the source of truth; several conceptual components were changed or are partial | `README.md`, `PROGRESS.md` | Modified |
| Fixed the stray backtick and sentence break ("does not provide a \`complete scaling strategy") | Editing | — | Modified |
| Contributions rewritten as five items: residual stacking with Prophet channels; two-part uncertainty with separate reporting; ablated decision layer; evaluation with feasibility ceiling and significance tests; explicit negative findings | The original contribution list mixed implemented, partial and aspirational components | `model/forecaster.py`, `decision/engine.py`, `forecast_metrics.py` | Modified |
| Added a paragraph stating that figures not reproducible from the current code are replaced by reproduced figures | Sets reader expectations and documents research integrity | `REPOSITORY_VALIDATION_REPORT.md` §6 | Added |

## 3. Section II — Related Work

| Change | Why | Evidence | Status |
|---|---|---|---|
| References [1]-[6] retained and bibliographically corrected (venues, volumes, pages, DOIs) | Original extraction lost the reference list; the six works named in the text were verified | Publisher pages / DOI records | Retained (metadata corrected) |
| Added [7] Gal & Ghahramani (MC dropout), [8] Angelopoulos & Bates (conformal), [9] Diebold & Mariano, [10] Taylor & Letham, [11] Verma et al. (Borg), [12] Kubernetes HPA documentation | New methods and baselines are used in the revision and must be cited | `model/uncertainty.py`, `model/forecaster.py::upper`, `forecast_metrics.py` | Added |
| New subsection E (Uncertainty Quantification and Calibration) | The framework's uncertainty claim requires positioning against calibration literature | `uncertainty.csv`, `reliability.csv` | Added |
| Research gap narrowed to three testable gaps: calibration of uncertainty, correction of post-scaling observations, feasibility ceiling for lead time | Original gap statement ("reliability of predictions") was not operationalised in the evaluation | `counterfactual.py`, `forecast_metrics.py` definitions | Modified |
| Removed the claim that ProximaScale's role "differs from a conventional hybrid predictor" as a novelty claim; replaced with a factual description of the residual-stacking arrangement and a note that the earlier weighted blend was abandoned | `model/ensemble.py` is deprecated and could never beat the LSTM alone | `README.md` "Why residual stacking", `tests/test_ensemble_deprecated.py` | Modified |

## 4. Section III — Proposed System

| Change | Why | Evidence | Status |
|---|---|---|---|
| Overview rewritten around the implemented data flow and Fig. 1 | Original described five conceptual stages with no measurement details | `main.py` orchestration | Rewritten |
| Monitoring: added 2-snapshot CPU delta, request read-and-reset across the container boundary, and the four-way distinction between zero CPU, missing measurement, stopped container and Docker failure | These are implemented contracts and materially change what a "0% CPU" row means | `monitoring/metrics.py`, `docs/METRICS.md` | Added |
| Forecasting: described Prophet-as-input-channel and residual target | The implementation differs from a post-hoc hybrid | `forecaster.py`, `config.json` | Modified |
| Uncertainty: separated the MC-dropout ensemble from the split-conformal bound and stated that the raw 2σ rule remains only in the legacy path | Original treated "uncertainty" as one mechanism | `uncertainty.py`, `forecaster.py::upper` | Modified |
| Decision engine: replaced the empty conceptual equation ("where … represents …") with three explicit branch conditions, adaptive band definition, clamps and cooldown semantics | The original equation was placeholders with missing symbols | `decision/engine.py`, `adaptive_threshold.py`, `config.yaml` | Rewritten |
| Actuation: added the pool-mode disclosure (no load balancer, `traffic_serving` false) | Without it, readers infer an SLO benefit that cannot exist | `docker_scaler.py` header, `README.md` behaviour contracts | Added |
| Feedback: split online feedback (monitoring + delta calibration) from offline feedback (counterfactual correction) and stated that there is no online learning | The original implied continuous learning from observed behaviour | `model/predict.py` calibration block | Modified |
| Guardrails subsection added (clamps, fallback contract, staleness skip, SHAP failure isolation) | These are tested invariants that define system behaviour | `tests/test_bugfix_pass.py`, `tests/test_remediation.py` | Added |
| Fig. 2 (forecasting stack) and Fig. 3 (window/horizon) added and numbered in order of appearance | The original had one conceptual figure and an image-only second figure | `forecaster.py` | Added |

## 5. Section IV — Residual-Stacking Prediction Model

| Change | Why | Evidence | Status |
|---|---|---|---|
| Input window: **10 → 30 readings**, 5 minutes → 15 minutes | Served bundle is 30 readings; the 10-reading model is a cold-start fallback only | `config.json`, `README.md` "The served model" | Modified |
| Input features: **3 → 5 channels** (added Prophet yhat(t), yhat(t+90 s)) | Model search selected this variant | `model/saved/proximascale_v4/seed0/info.json` | Modified |
| Horizon: 3 steps / 90 s | Verified | `config.json` | Retained |
| LSTM: 64/32 units and 29,923 parameters → **30,435 parameters**, dropout **0.2 → 0.1**, three seeded members | Served variant differs from the original description | `seed0/info.json`, `forecaster.py` | Modified |
| Training protocol made explicit: MSE, Adam 1e-3, batch 32, ≤50 epochs, patience 5, 80/10/10 chronological split, trained on original + varied-spike data | Required for reproducibility and to explain generalization results | `info.json`, `docs/model_evaluation.md` | Added |
| Equations (1)-(9) added with defined symbols: window, residual, final sum, live calibration, conformal bound, counterfactual blend | The original equations were present as images/blank symbols in the extraction; symbols were undefined | code paths cited per equation | Added/rewritten |
| Counterfactual equation corrected: the 0.4 term is the **last pre-scaling observed value**, not "y_predicted" | The original formulation mislabels the baseline term | `counterfactual.py` | Modified |
| New subsection F (live calibration) with the offset equation and deadband | Central to why the system works live and absent from the original | `model/predict.py` | Added |
| New subsection G (calibrated upper bound) | Served bundle uses conformal binning, not mean + 2σ | `config.json` `upper_bound` | Added |
| Added Table I (implementation parameters with source column) | Makes implementation-paper consistency checkable | config + artifact metadata | Added |

## 6. Section V — Experimental Setup (rebuilt; original section unavailable)

| Change | Why | Evidence | Status |
|---|---|---|---|
| Dataset description: 90,000 rows, 31 days, 180 spikes, 1,800 post-scaling rows, ranges, generator parameters | The original described "a synthetically generated workload dataset" without specification | `generate_realistic_data.py`, `dataset_stats.json` | Added |
| Added the varied-spike dataset and the three unseen one-week datasets | Needed for the generalization claims | `metrics_varied.csv`, `results/unseen/` | Added |
| Added the 121-row real dataset with its idle characteristics and the conclusion that it cannot support accuracy claims | Integrity: the only real data are idle | `metrics_real_dev.csv`, `forecast.csv` | Added |
| Splits and protocol: 80/10/10 chronological, validation used for all choices, test scored once | The original did not state that the published 10-reading model early-stopped on test; this is now disclosed | `docs/model_evaluation.md` | Added |
| Baselines: persistence, linear extrapolation, Prophet alone, univariate LSTM, 10-reading hybrid, served model | The original compared only Prophet and hybrid | `forecast_metrics.py` | Added |
| Metric definitions: lead time, feasible ceiling, crossings split into spurious/missed, false-alarm episodes/day, coverage, DM test, block bootstrap | The original reported RMSE/MAE/lead/oscillations with no definitions; "oscillations" was undefined | `forecast_metrics.py` DEFINITIONS | Added |
| Environment: Python 3.12.10, TF 2.16.1, Prophet 1.3.0, sklearn 1.4.2, 210 tests | Reproducibility | `pytest` run | Added |

## 7. Section VI — Results and Analysis (rebuilt)

| Claim in original | Revision | Why | Evidence |
|---|---|---|---|
| RMSE 5.08 | Supplied as 5.087 for the 10-reading model and 4.347 for the served model, with standard deviations over 5 seeds | Reproduced from dumps; two model families now exist | `baseline/table_ii.csv`, `v4_verify/table_ii.csv` |
| MAE 3.06 | 3.062 (10-reading), 2.679 (served) | Same | same |
| Lead time 57.5 s | 56.8 ± 0.4 s (10-reading), 58.2 ± 0.4 s (served); compared against a computed 58.3 s ceiling | 57.5 could not be reproduced exactly; lead time is meaningless without the ceiling | `table_ii.csv` |
| Oscillations 98 | **Removed**; replaced by alarm-state crossings 96.4 ± 2.6 (10-reading) and 81.6 ± 1.7 (served), plus spurious/missed decomposition | 98 is not reproducible and "oscillations" was undefined | same |
| Prophet alone 10.15 / 4.68 | 10.154 / 4.675 | Reproduced exactly | `table_ii.csv` |
| (no baseline table) | Added univariate LSTM 5.131/3.032 and persistence 6.644/3.915, with F1/precision/false alarms | Complete comparison requested and supported by the harness | same |
| (no significance testing) | Added DM tests and bootstrap intervals; report that all differences vs. baselines are significant | A single test split is not evidence without a test | `significance.csv` |
| (no fair-protocol caveat) | Added: under identical early stopping, the 10-reading hybrid (5.13) is **not** better than the univariate LSTM (5.06) | Critical for honest attribution of the gain | `fair/table_ii.csv` |
| (no lead-time ceiling) | Added: 33.8/36 spikes at 90 s, 2.2 at 30 s, mean 58.2 s vs ceiling 58.3 s | Prevents over-reading the 60 s target | `table_ii.csv`, `spikes.csv` |
| (no decision evaluation) | Added Table III and Fig. 6: 10-policy replay, including the shipped-config result (24/36, 0.8 false/day) and the floor-off variant (29/36, 0) | The paper's decision claims were previously unsupported | `replay_v4/.../summary.csv` |
| (no generalization) | Added Table IV: fresh/shifted/shifted2 (RMSE 4.25/4.11/4.12, 0 missed) and real_dev (36.0, worse than persistence 22.7) | Supports the varied-spike training claim and bounds it | `results/unseen/forecast.csv` |
| (counterfactual claimed beneficial) | Added Table V: configured α=0.6 gives RMSE 4.43 on post-scaling rows, α=0.2 gives 3.27; aggregate effect 0.0009 | The original implied a validated benefit; the sweep shows the weight is not optimal and the aggregate effect is negligible | `alpha_sweep.csv`, `cf_ablation.csv` |
| (anomaly detection claimed as a component) | Added Table VI: precision 0.175, recall 1.0, 84 false flags/day at z=2.5 | Explains why the bypass is disabled and contradicts an implied benefit | `anomaly_sweep.csv` |
| (no cost analysis) | Added Table VII: 1.36 s per 30-pass decision (4.5% of interval), 68 ms at 1 pass | Operational feasibility claim needs a number | `results/timing/baseline/latency.csv` |
| (uncertainty claimed as novelty) | Added Fig. 7 and calibration analysis: 2σ covers 55-59% at nominal 98%; calibrated bound 94.9% marginal, ~87% above 75% CPU | The strongest negative result; the original claim was not tested | `reliability.csv`, `uncertainty.csv`, `coverage.csv` |

## 8. Sections VII-IX (Discussion, Limitations, Conclusion) — rebuilt

| Change | Why | Evidence | Status |
|---|---|---|---|
| New Discussion separating three supported findings from three unsupported claims | The original discussion asserted benefits without qualification | Section VI | Added |
| Limitations expanded from a single sentence to four categories (internal, external, construct, implementation), including synthetic-only data, idle real data, pool-mode actuation, 6.25-day test period, harness disagreement, config/doc contradiction, sklearn pickle mismatch | Required for academic defensibility and explicitly requested | `REPOSITORY_VALIDATION_REPORT.md` §5, §8 | Added |
| Conclusion rewritten to state the measured result and the boundary of the claim; future work derived from the limitations (load balancer, real dataset, estimated counterfactual weight, conditional conformal, unified harness) | Original conclusion implied demonstrated feasibility for proactive resource management generally | Section VI | Modified |
| Removed "demonstrates the feasibility … for responsive workload adaptation" from the abstract/conclusion | Responsiveness (latency) was never measured | `docker_scaler.py` pool-mode contract | Removed |
| Added code/data availability statement | Reproducibility | `paper/`, `results/validation_2026-11-25/` | Added |

## 9. Figures and tables

| # | Original | Revised | Source |
|---|---|---|---|
| Fig. 1 | Conceptual architecture (image) | Deployed architecture with module-level boxes | `make_figures.py::fig_architecture` |
| Fig. 2 | Residual-stacking figure (image only, unreferenced in extracted text) | Served forecasting stack | `fig_forecasting_stack` |
| Fig. 3 | — | Input window / horizon timeline | `fig_timeline` |
| Fig. 4 | — | Forecast error by method | `baseline/table_ii.csv` |
| Fig. 5 | — | Lead-time distribution per spike | `baseline/spikes.csv` |
| Fig. 6 | — | Decision replay across policies | `replay_v4/.../summary.csv` |
| Fig. 7 | — | MC-dropout reliability curve | `baseline/reliability.csv` |
| Tables I-VII | Original had no data tables in the available text | Implementation parameters, accuracy/event metrics, decision replay, OOD, counterfactual sweep, anomaly sweep, latency | CSV-backed at build time |

All seven figures are referenced in the text (Fig. 1 §III-A, Fig. 2 §III-C,
Fig. 3 §IV-A, Fig. 4 §VI-A, Fig. 5 §VI-B, Fig. 6 §VI-D, Fig. 7 §VI-J) and all
seven tables are referenced (Tables I-VII in Sections IV, VI).

## 10. Reference list

| Status | Count | Detail |
|---|---:|---|
| Retained (verified) | 6 | Dang-Quang & Yoo 2022; Suleiman et al. 2024; Golshani & Ashtiani 2021; Guruge & Priyadarshana 2025; Arbat et al. 2022; Agarwal et al. 2024 |
| Added (new methods used) | 6 | Gal & Ghahramani 2016; Angelopoulos & Bates 2021; Diebold & Mariano 1995; Taylor & Letham 2018; Verma et al. 2015; Kubernetes HPA documentation |
| Removed | 0 | No reference was removed |
| Invented | 0 | Every DOI/venue was checked against a publisher or DOI record during this audit |

## 11. Claims removed or downgraded (summary)

1. **"Latency improved from 1240 ms to 310 ms" (exp3)** — removed; no generating
   script, and the pool-mode actuator cannot route traffic to new workers.
2. **"Lead time 68-81 s over five runs" (exp2)** — removed; exceeds the 90 s
   horizon geometry and has no generating script.
3. **"Oscillations reduced from 7 to 2 by counterfactual correction" (exp4)** —
   removed; not reproducible; the measured aggregate effect of the correction
   is 0.0009 RMSE.
4. **"98 scaling oscillations"** — removed; replaced by measured crossings.
5. **"Uncertainty-aware decision-making reduces risk"** — downgraded to a
   marginal calibrated bound with the branch disabled, plus a negative
   calibration result.
6. **"36/36 spikes anticipated, 2.7 replicas"** — downgraded to 24/36 (shipped)
   and 29/36 (floor off) from the reproducible harness, with the harness
   disagreement disclosed.
7. **"Feedback-driven adaptation"** — narrowed to monitoring + delta
   calibration; no online learning.

## 12. Files added by this revision

```
paper/
  ProximaScale_IEEE_Revised_2026-11-25.docx   editable manuscript
  ProximaScale_IEEE_Revised_2026-11-25.pdf    14-page PDF
  ProximaScale_IEEE_Revised_2026-11-25.md     diffable Markdown source
  REPOSITORY_VALIDATION_REPORT.md
  CHANGELOG_PAPER_VS_IMPLEMENTATION.md
  requirements-paper.txt
  build_paper.py                              DOCX/PDF/MD builder
  content_part1.py, content_part2.py          manuscript text
  figures/make_figures.py, fig1..fig7 .png
results/validation_2026-11-25/
  evaluate.log                                failed/slow evaluate.py run
  eval_dump.log                               277 s v4 dump log
  forecast_metrics_v4/v4_verify/*.csv         13 metric tables
  replay_v4/v4_verify/*.csv, replay_v4_nofloor/v4_verify/*.csv
  config_no_floor.yaml                        controlled variant
  dataset_stats.json, pdf_page1/5/8.png       dataset stats, render checks
results/eval_dump/v4_verify/                  per-window dumps + meta.json
```

No existing file was modified or deleted. No commit was made; no remote was
touched.
