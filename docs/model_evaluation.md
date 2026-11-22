# Model evaluation and model search (2026-11)

How the served model (`model/saved/proximascale_v4/`) and the engine settings
in `config.yaml` were chosen, what they score, and how to reproduce it. All
data is synthetic (`data/collected/README.md`); nothing here is a measurement
of real traffic.

## Protocol

- `metrics.csv` (90,000 rows of 30 s, seed 42): training period = first 80%,
  of which rows 0-64,799 fit the model and rows 64,800-71,999 are the
  **validation** period (early stopping, every choice below); **test** = rows
  72,000 on, scored once per finished choice.
- Every LSTM is trained with 3 seeds; served models pool the 3 seeds' MC
  Dropout passes (30 passes in total per decision).
- The original model's published numbers (RMSE 5.09, lead 57.5 s) used the
  test split for early stopping. Retrained under this protocol the same design
  scores RMSE 5.13 (the "fair" model).

## What changed, step by step

| Step | Change | Chosen by | Script |
|---|---|---|---|
| Model search | 30-reading window (15 min); Prophet's value now and 90 s ahead as LSTM inputs | validation RMSE over 20+ variants | `train_variants.py`, `variant_eval.py` |
| Round 3 | dropout 0.2 -> 0.1 (Huber loss, LR decay, Conv1D, extra residual input: no gain) | validation RMSE | same |
| Calibration | 95% upper bound = mean + a margin per sigma quintile (split conformal) instead of mean + 2 sigma | validation coverage | `tune_decisions.py` |
| Engine | upper-bound branch off, anomaly bypass off, no scale-down while CPU > 75% | validation replay grid (180 settings) | `tune_decisions.py` |
| Reactive floor | scale up on current CPU: **rejected**, worse on every period | validation | same |
| Varied spikes | train on `metrics.csv` + `metrics_varied.csv` (random spike shapes) | rule fixed before training: original validation RMSE within 2% of v3 and better varied validation RMSE | `generate_varied_data.py`, `train_variants.py --data`, `unseen_eval.py` |

## Results on the test split (Table II windows, 17,988 decisions = 6.25 days)

| | Original (published) | Original, fair | **v4 (served)** |
|---|---|---|---|
| RMSE / MAE | 5.09 / 3.06 | 5.13 / 3.09 | **4.35 / 2.68** |
| Lead time at 75% | 57.5 s | 53.3 s | **58.2 s** |
| Forecast false alarms / day | 0.48 | 0.96 | **0** |
| 95% bound coverage | 55% (raw 2 sigma) | | **94.9%** (calibrated) |

Decisions (`decision_replay.py`, P4 = original live config, P9 = tuned `config.yaml`):

| | P4, original engine | **P9, tuned engine, v4** |
|---|---|---|
| Spikes with a scale-up 30-90 s ahead | 20 / 36 | **36 / 36** |
| False scale-ups / day | 44 | **0** |
| Reversals within 5 min | 294 | **0** |
| Mean replicas | 5.9 | **2.67** |

## Data no model was trained on (`results/unseen/`)

One-week sets dated after `metrics.csv` ends (Prophet extrapolates). RMSE,
and spikes the forecast missed at 75%:

| Data | Original | v3 | **v4** |
|---|---|---|---|
| fresh (same generator, seed 7) | 5.00 | 4.21 | **4.25**, 0 missed |
| shifted (3-min ramps, 2x spikes) | 4.69, 49 missed | 5.32, 60 missed | **4.11, 0 missed** |
| shifted2 (2.5-min ramps, 1.5x; defined before any result on it) | 4.43, 24 missed | 4.57, 29 missed | **4.12, 0 missed** |
| real_dev (121 idle real readings) | 36.1 | 39.6 | 36.0 (persistence 22.7) |

v3 was fitted to the generator's single 90 s spike shape; v4 keeps v3's
accuracy on that shape (+0.9% RMSE) and forecasts the others. Note for the
decision tables there: "anticipated" counts scale-ups 30-90 s before onset;
on slow ramps the engine scales earlier, so read `capacity_up_before_onset`.

## Limitations

- Synthetic data only. On the real idle container every model is far off:
  the models need retraining on collected traffic before real use.
- The anomaly bypass is off because the synthetic spikes are forecastable; on
  real traffic with unforecastable bursts, re-enable `use_anomaly_bypass`.
- The calibrated bound covers 95% overall but ~80% of readings above 75%.

## Reproduce

```powershell
python model/train_variants.py --variants hx3_drop10 --data data/collected/metrics.csv data/collected/metrics_varied.csv --out-dir model/saved/variants_both
python model/eval_dump.py --bundle model/saved/proximascale_v4 --tag v4
python model/tune_decisions.py --bundle model/saved/proximascale_v4 --test-dump results/eval_dump/v4 --out results/tuning/v4
python model/eval_dump.py --bundle model/saved/proximascale_v4 --tag v4_tuned
python model/forecast_metrics.py --dump results/eval_dump/v4_tuned --out-dir results/forecast_metrics
python model/decision_replay.py --dump results/eval_dump/v4_tuned --out-dir results/d1_replay
python model/unseen_eval.py
```
