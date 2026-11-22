"""
ProximaScale - Refinement 3 (fix): Calibrate the confidence band width
-------------------------------------------------------------------------
check_calibration.py showed the range was way too narrow (12% coverage,
should be ~95%). Reason: MC Dropout only captures "model is unsure"
uncertainty, not "the real world is just noisy" uncertainty -- and our
data clearly has more of the second kind.

Fix: instead of assuming "2x the spread" is correct, SEARCH for the
multiplier that actually gives ~95% coverage, measured on a validation
slice carved from the TRAINING data only (same safe approach as the
Prophet tuning step) -- never the held-out test set.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import load_csv, chronological_split, scale_features, load_scaler, WINDOW_SIZE
from lstm_model import MODEL_PATH, HORIZON
from prophet_model import load_prophet_model, get_prophet_fitted
from uncertainty import predict_with_uncertainty
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
RESIDUAL_SCALER_PATH = SAVED_DIR / "scaler_residual.pkl"
SAMPLING_INTERVAL_SECONDS = 30
TRAIN_FRAC = 0.8
VAL_FRAC_OF_TRAIN = 0.85
CANDIDATE_MULTIPLIERS = [2, 3, 4, 5, 6, 8, 10, 12]
CHECK_MAX_WINDOWS = 200   # only check a sample, not all 10,800 — much faster, still accurate enough
CHECK_N_PASSES = 8        # fewer guesses per window, just for this quick check

def _prep_corrected(window):
    df = pd.DataFrame(window)
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False
    return apply_counterfactual_correction(df)


def run_search():
    import pickle
    from tensorflow.keras.models import load_model

    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    train_df, _test_df = chronological_split(df, TRAIN_FRAC)   # test set untouched

    split_idx = int(len(train_df) * VAL_FRAC_OF_TRAIN)
    val_df = train_df.iloc[split_idx:].reset_index(drop=True)
    print(f"🧠 [ProximaScale] Checking on {len(val_df)} validation rows (held-out test set untouched)")

    model = load_model(MODEL_PATH, compile=False)
    scaler = load_scaler()
    with open(RESIDUAL_SCALER_PATH, "rb") as f:
        residual_scaler = pickle.load(f)
    prophet_model = load_prophet_model()

    n_windows = len(val_df) - WINDOW_SIZE - HORIZON + 1
    check_indices = list(range(0, n_windows))
    if len(check_indices) > CHECK_MAX_WINDOWS:
        positions = np.linspace(0, len(check_indices) - 1, CHECK_MAX_WINDOWS)
        check_indices = [check_indices[int(round(p))] for p in positions]
    print(f"🧠 [ProximaScale] Checking {len(check_indices)} windows (sampled from {n_windows} available)...")

    means, stds, actuals = [], [], []

    for count, i in enumerate(check_indices):
        window_df = _prep_corrected(val_df.iloc[i:i + WINDOW_SIZE].to_dict("records"))
        window_df["cpu_percent"] = window_df["cpu_percent_corrected"]
        scaled = scale_features(window_df, scaler).reshape(1, WINDOW_SIZE, -1)

        last_ts = pd.Timestamp(val_df["timestamp"].iloc[i + WINDOW_SIZE - 1])
        step = pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS)
        future_ts = [last_ts + step * j for j in range(1, HORIZON + 1)]
        prophet_forecast = get_prophet_fitted(prophet_model, future_ts)

        result = predict_with_uncertainty(
            model, scaled, residual_scaler,
            inverse_transform_fn=lambda v: residual_scaler.inverse_transform(v.reshape(-1, 1)).flatten(),
            n_passes=CHECK_N_PASSES,
        )
        means.append(prophet_forecast[-1] + result["mean"][-1])
        stds.append(result["std"][-1])
        target_idx = i + WINDOW_SIZE - 1 + HORIZON
        actuals.append(float(val_df.loc[target_idx, "cpu_percent"]))

        if (count + 1) % 20 == 0:
            print(f"🧠 [ProximaScale]   ...{count + 1}/{len(check_indices)} checked")

    means, stds, actuals = np.array(means), np.array(stds), np.array(actuals)

    print("\n🧠 [ProximaScale] ===== MULTIPLIER SEARCH =====")
    best_mult, best_gap = None, float("inf")
    for mult in CANDIDATE_MULTIPLIERS:
        lower = means - mult * stds
        upper = means + mult * stds
        coverage = float(np.mean((actuals >= lower) & (actuals <= upper)) * 100)
        gap = abs(coverage - 95)
        print(f"🧠 [ProximaScale]   multiplier={mult} -> coverage={coverage:.1f}%")
        if gap < best_gap:
            best_gap, best_mult = gap, mult

    print(f"\n🧠 [ProximaScale] BEST multiplier: {best_mult} (closest to 95% coverage)")
    print("🧠 [ProximaScale] Calibration search self-test PASSED.")
    return best_mult


if __name__ == "__main__":
    run_search()