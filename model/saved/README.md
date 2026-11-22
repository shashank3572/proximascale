# model/saved/ — artifact provenance

Current residual-stacking artifacts (committed in `2da9d70e`, 2026-09-19,
"switch to residual-stacking hybrid, retrain on 31-day dataset"; trained on
`data/collected/metrics.csv`, the synthetic 90,000-row dataset):

| File | Role |
|---|---|
| `proximascale_prophet.pkl` | Prophet base model (needs `pyarrow` to unpickle) |
| `proximascale_lstm.h5` | LSTM that predicts the Prophet residual (`MODEL_PATH` in `model/lstm_model.py`) |
| `scaler.pkl` | feature scaler (cpu, memory, request rate) |
| `scaler_residual.pkl` | residual scaler |

They were produced with the versions pinned in `requirements.txt`
(scikit-learn 1.8.0, pandas 3, Keras 3). Other versions may fail to load them.
Regenerate with `python model/prophet_model.py` then `python model/train.py`
(this overwrites the files above).

Other files:

- `proximascale_lstm_univariate.h5`, `scaler_univariate.pkl` — cached baseline
  used by `model/evaluate.py` for the "Univariate LSTM" comparison row.
- `proximascale_lstm.keras` — **stale**. Pre-residual-stacking model from
  2026-09-14, saved with an older Keras (fails to load: `batch_input_shape` /
  `time_major`). Nothing references it. Safe to delete; kept until the team
  decides.
- `evaluation_chart_DUMMY_DATA_semester1.png` — Semester-1 chart from
  `data/generate_dummy_data.py` synthetic data. **Not a result**; do not cite.
  `model/evaluate.py` prints a table and does not generate a chart.

## Stacked bundles (evaluation model search, 2026-11-21)

`model/forecaster.py` loads a bundle: `config.json` plus one folder per
member (`lstm.h5`, `feature_scaler.pkl`, `target_scaler.pkl`, `info.json`),
copied unchanged from `model/train_variants.py`'s output. All use the
committed `proximascale_prophet.pkl`.

| Folder | What it is |
|---|---|
| `proximascale_v4/` | **Served.** v3's model trained on `metrics.csv` + `metrics_varied.csv` together (copied from `variants_both/`); its own calibrated bound in `config.json`. `PROXIMASCALE_MODEL=v3` serves v3. |
| `proximascale_v3/` | Variant `hx3_drop10`: 30-reading window, inputs CPU, memory, requests + Prophet's value now and 90 s ahead, residual target, dropout 0.1, 3 seeds pooled. `config.json` holds the calibrated 95% upper bound (fit on the validation period by `model/tune_decisions.py`). |
| `proximascale_v2/` | Previous bundle (`hx_w30_prophet`, dropout 0.2). `PROXIMASCALE_MODEL=v2` serves it. |
| `variants/` | Every model-search run (`train_variants.py`), 3 seeds each, trained on `metrics.csv`. |
| `variants_varied/` | `hx3_drop10` trained on `data/collected/metrics_varied.csv` (varied spike shapes). |
| `variants_both/` | `hx3_drop10` trained on `metrics.csv` and `metrics_varied.csv` together. |

The original artifacts above are still used for windows of 10-29 readings
and by `PROXIMASCALE_MODEL=legacy`.
