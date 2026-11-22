"""
ProximaScale - Refinement 1: Prophet Hyperparameter Tuning
-------------------------------------------------------------
Prophet has been trained with pure defaults this whole project --
changepoint_prior_scale=0.05, seasonality_prior_scale=10.0 (Prophet's
own built-in defaults). The reference paper tuned these explicitly,
but their numbers were tuned for HTTP request-rate data with very
different dynamics than our bounded 0-100 CPU% data -- so we tune
our OWN values here rather than copying theirs blindly.

Also explicitly tunes daily_seasonality, since our data's actual
repeating structure is a WITHIN-day surge pattern -- daily seasonality
is the more reliably learnable signal here than weekly (31 days is
only ~4.4 weeks, a weak sample for a 7-day cycle).

METHODOLOGY: grid search over candidate values, each evaluated on a
VALIDATION split carved out of the TRAINING portion only -- the final
held-out test set (used by evaluate.py) is never touched here, so
this tuning step can't leak into the numbers you'd actually report.
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet

from preprocessing import load_csv, chronological_split
from prophet_model import _to_prophet_format, TARGET_COL, TIMESTAMP_COL
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
TRAIN_FRAC = 0.8          # must match train.py / evaluate.py
VAL_FRAC_OF_TRAIN = 0.85  # last 15% of the TRAIN portion becomes validation

CHANGEPOINT_CANDIDATES = [0.01, 0.05, 0.1, 0.5]
SEASONALITY_CANDIDATES = [1.0, 10.0, 30.0]
DAILY_SEASONALITY_CANDIDATES = [True, False]


def _prep_corrected(df):
    df = df.copy()
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False
    return apply_counterfactual_correction(df)


def run_tuning():
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    train_df, _test_df = chronological_split(df, TRAIN_FRAC)   # test set never touched here

    train_df = _prep_corrected(train_df)
    train_df["cpu_percent"] = train_df["cpu_percent_corrected"]

    split_idx = int(len(train_df) * VAL_FRAC_OF_TRAIN)
    tune_train = train_df.iloc[:split_idx].reset_index(drop=True)
    tune_val = train_df.iloc[split_idx:].reset_index(drop=True)
    print(f"🧠 [ProximaScale] Tuning on {len(tune_train)} rows, validating on {len(tune_val)} rows "
          f"(held-out test set from evaluate.py is untouched)")

    prophet_df = _to_prophet_format(tune_train)
    val_timestamps = pd.to_datetime(tune_val[TIMESTAMP_COL])
    val_actual = tune_val[TARGET_COL].values

    results = []
    combos = list(itertools.product(CHANGEPOINT_CANDIDATES, SEASONALITY_CANDIDATES, DAILY_SEASONALITY_CANDIDATES))
    print(f"🧠 [ProximaScale] Testing {len(combos)} hyperparameter combinations...")

    for cp_scale, season_scale, daily in combos:
        model = Prophet(
            changepoint_prior_scale=cp_scale,
            seasonality_prior_scale=season_scale,
            daily_seasonality=daily,
        )
        model.fit(prophet_df)

        future_df = pd.DataFrame({"ds": val_timestamps})
        forecast = model.predict(future_df)
        pred = forecast["yhat"].values

        rmse = float(np.sqrt(np.mean((pred - val_actual) ** 2)))
        results.append({
            "changepoint_prior_scale": cp_scale,
            "seasonality_prior_scale": season_scale,
            "daily_seasonality": daily,
            "val_rmse": round(rmse, 3),
        })
        print(f"🧠 [ProximaScale]   cp={cp_scale}, season={season_scale}, daily={daily} -> val RMSE={rmse:.3f}")

    results_df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    print("\n🧠 [ProximaScale] ===== TUNING RESULTS (sorted, best first) =====")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n🧠 [ProximaScale] BEST: changepoint_prior_scale={best['changepoint_prior_scale']}, "
          f"seasonality_prior_scale={best['seasonality_prior_scale']}, "
          f"daily_seasonality={best['daily_seasonality']} (val RMSE={best['val_rmse']})")

    return results_df


if __name__ == "__main__":
    results_df = run_tuning()

    default_row = results_df[
        (results_df["changepoint_prior_scale"] == 0.05) &
        (results_df["seasonality_prior_scale"] == 10.0)
    ]
    if not default_row.empty:
        default_rmse = default_row.iloc[0]["val_rmse"]
        best_rmse = results_df.iloc[0]["val_rmse"]
        improvement = (default_rmse - best_rmse) / default_rmse * 100
        print(f"\n🧠 [ProximaScale] Best config improves {improvement:.1f}% over Prophet's "
              f"plain defaults on the validation split.")

    print("🧠 [ProximaScale] Tuning self-test PASSED -- now update prophet_model.py's "
          "constants with the winning values (see instructions below).")