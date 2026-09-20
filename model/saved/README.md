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
