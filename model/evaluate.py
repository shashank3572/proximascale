
"""
ProximaScale - Phase 10/12: Evaluation (Residual-Stacking)
------------------------------------------------------------------
REWORKED for Phase 12. The old 4-way comparison had a "Multivariate LSTM
(raw)" row that took the LSTM's output alone, with no Prophet, and compared
it directly to actual CPU%. That worked when the LSTM predicted raw CPU%
directly. It no longer works: post-Phase 12, the LSTM predicts a RESIDUAL
relative to Prophet's fit, so its raw output is a small correction number,
not a CPU% forecast -- comparing it to actual CPU% would be meaningless,
not just less accurate.

NEW 4-way comparison:
    1. Reactive       - unchanged. No model. "Predicted" = current raw reading.
    2. Univariate LSTM - unchanged. Own small LSTM trained on cpu_percent
                        alone (this project's own baseline, untouched by
                        the residual-stacking switch).
    3. Prophet Alone  - NEW, replaces "Multivariate LSTM (raw)". Prophet's
                        own g(t)+s(t)+h(t) forecast, zero LSTM correction.
                        Answers: "how good is Prophet by itself?"
    4. Hybrid (Full)  - predict_load() as-is: Prophet forecast + LSTM
                        residual correction, ADDED together. Answers:
                        "does the LSTM's residual correction actually help
                        over Prophet alone?" -- the exact question the
                        reference paper's architecture is built around.

Also fixes the two-sources-of-truth split issue: this now calls
chronological_split() directly (the same function train.py uses) instead
of an independent manual 80/20 slice that merely happened to match.

------------------------------------------------------------------
PERFORMANCE NOTES (added):
- Reactive / Univariate LSTM / Prophet Alone are now computed in a single
  BATCHED call each instead of one call per window. Single-sample
  model.predict() / Prophet.predict() calls carry heavy fixed per-call
  overhead in TF and Prophet, so looping them one at a time is much
  slower than doing the same work in one batched call -- this was
  likely a large chunk of your runtime.
- Hybrid still calls predict_load() window-by-window, since that
  function's internals (and whether it can accept a batch) live in
  predict.py, which this file doesn't control. It is very likely your
  remaining bottleneck -- see the note printed just before it runs.
- tf.keras.utils.disable_interactive_logging() is set globally so Keras
  stops printing a progress bar on every single predict() call, no
  matter which file triggers it (including inside predict_load).
- EVAL_MAX_WINDOWS below caps how many windows get evaluated at all
  (evenly sampled across the test period), independent of EVAL_STRIDE.
  This is the main lever for a fast run: set it to None for the full,
  citable run.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF's native startup logging

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import load_csv, chronological_split, WINDOW_SIZE
from lstm_model import HORIZON
from predict import predict_load, SAMPLING_INTERVAL_SECONDS
from prophet_model import load_prophet_model, get_prophet_fitted
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
UNIVARIATE_MODEL_PATH = SAVED_DIR / "proximascale_lstm_univariate.h5"
UNIVARIATE_SCALER_PATH = SAVED_DIR / "scaler_univariate.pkl"

SCALE_THRESHOLD = 75.0   # CPU% that triggers a "scale up" decision -- tune to match Person D
TRAIN_FRAC = 0.8         # must match train.py's TRAIN_FRAC / prophet_model.py's split
EVAL_N_PASSES = None     # None = full 30 MC Dropout passes, the real setting for citable numbers
EVAL_STRIDE = 1          # evaluate every window, not every 4th
EVAL_MAX_WINDOWS = None   # still capped so runtime stays predictable on 31 days of data --
                          # see below if you want the fully exhaustive version instead


# ============================================================================
# Data loading / windowing
# ============================================================================

def _load_test_split():
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    train_df, test_df = chronological_split(df, TRAIN_FRAC)
    print(f"🧠 [ProximaScale] Loaded {len(df)} rows -> {len(train_df)} train / {len(test_df)} test")
    return train_df, test_df


def _row_to_window_dicts(df, start, size):
    cols = ["timestamp", "cpu_percent", "memory_percent", "request_rate"]
    if "post_scaling" in df.columns:
        cols.append("post_scaling")
    return df.iloc[start:start + size][cols].to_dict("records")


def _prep_corrected(window):
    """Same pattern as predict.py/train.py: default post_scaling, then correct."""
    df = pd.DataFrame(window)
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False
    return apply_counterfactual_correction(df)


def _select_eval_indices(n_windows):
    """Strided candidate list, then evenly subsampled down to EVAL_MAX_WINDOWS
    if it's still bigger than that cap. Even subsampling (rather than just
    taking the first EVAL_MAX_WINDOWS) keeps coverage across the whole test
    period instead of only the earliest part of it."""
    candidates = list(range(0, n_windows, EVAL_STRIDE))
    if EVAL_MAX_WINDOWS is None or len(candidates) <= EVAL_MAX_WINDOWS:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, EVAL_MAX_WINDOWS)
    positions = sorted(set(int(round(p)) for p in positions))
    return [candidates[p] for p in positions]


# ============================================================================
# Method 1: Reactive baseline
# ============================================================================

def _reactive_predict(window):
    return float(window[-1]["cpu_percent"])


# ============================================================================
# Method 2: Univariate LSTM (trained here, cached to disk after first run)
# ============================================================================

def _get_univariate_artifacts(train_df):
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf

    # Stops Keras printing a progress bar on every predict() call anywhere in
    # the process (including inside predict_load, which this file doesn't
    # control) -- this is almost certainly the "wall of text" you were seeing.
    tf.keras.utils.disable_interactive_logging()

    if UNIVARIATE_MODEL_PATH.exists() and UNIVARIATE_SCALER_PATH.exists():
        model = load_model(UNIVARIATE_MODEL_PATH, compile=False)
        with open(UNIVARIATE_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("🧠 [ProximaScale] Univariate LSTM baseline loaded from cache")
        return model, scaler

    print("🧠 [ProximaScale] No cached univariate baseline -- training one now...")

    corrected = _prep_corrected(train_df.to_dict("records"))
    series = corrected["cpu_percent_corrected"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(len(scaled) - WINDOW_SIZE - HORIZON + 1):
        X.append(scaled[i:i + WINDOW_SIZE])
        y.append(scaled[i + WINDOW_SIZE: i + WINDOW_SIZE + HORIZON, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(HORIZON),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model.save(UNIVARIATE_MODEL_PATH)
    with open(UNIVARIATE_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"🧠 [ProximaScale] Univariate LSTM baseline trained + cached -> {UNIVARIATE_MODEL_PATH.name}")
    return model, scaler


def _univariate_predict_batch(model, scaler, windows):
    """Batched version: builds every window's scaled array up front and calls
    model.predict() ONCE for all of them, instead of once per window. Same
    math, same result, dramatically fewer TF call-overhead hits."""
    batch = []
    for w in windows:
        corrected = _prep_corrected(w)
        series = corrected["cpu_percent_corrected"].values.reshape(-1, 1)
        batch.append(scaler.transform(series))
    batch = np.array(batch)  # (n_eval, WINDOW_SIZE, 1)

    pred_scaled = model.predict(batch, verbose=0)  # (n_eval, HORIZON)
    last_col_scaled = pred_scaled[:, -1].reshape(-1, 1)
    pred_cpu = scaler.inverse_transform(last_col_scaled).flatten()
    return pred_cpu


# ============================================================================
# Method 3: Prophet Alone (NEW -- replaces the old "Multivariate LSTM raw")
# ============================================================================

def _prophet_alone_predict_batch(prophet_model, windows):
    """Batched version: one Prophet call for every window's target timestamp
    instead of one call per window. We also only request the single target
    timestamp per window (previously the whole 1..HORIZON trajectory was
    requested just to keep forecast[-1] -- the intermediate steps were never
    used)."""
    step = pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS)
    target_timestamps = [
        pd.Timestamp(w[-1]["timestamp"]) + step * HORIZON for w in windows
    ]
    forecast = get_prophet_fitted(prophet_model, target_timestamps)
    return np.asarray(forecast, dtype=float)


# ============================================================================
# Method 4: Hybrid -- predict_load() as-is, no modification
# ============================================================================
# (called directly in the loop below; no wrapper needed. This one is NOT
# batched -- predict_load() isn't ours to change here, so if this is still
# the slow part after everything else, the fix has to go into predict.py.
# A common cause of exactly this symptom -- a per-call cost of several
# seconds for a small model -- is TensorFlow retracing its graph on every
# call because the input's shape changes slightly call to call. Worth
# checking if you want that squeezed further.)


# ============================================================================
# Metrics helpers
# ============================================================================

def _detect_spike_onsets(actual_cpu, threshold):
    above = actual_cpu > threshold
    return [i for i in range(1, len(above)) if above[i] and not above[i - 1]]


def _oscillation_count(preds, threshold):
    decisions = (preds > threshold).astype(int)
    return int(np.sum(decisions[1:] != decisions[:-1]))


def _lead_time_for_method(preds_by_t, target_idx_by_t, spike_onset_idx, threshold):
    candidates = [
        t for t, pred in preds_by_t.items()
        if t < spike_onset_idx and pred > threshold and target_idx_by_t[t] >= spike_onset_idx
    ]
    if not candidates:
        return 0.0
    earliest_t = min(candidates)
    return float((spike_onset_idx - earliest_t) * SAMPLING_INTERVAL_SECONDS)


def _lead_time_indices_for_spikes(spike_onsets, n_windows):
    """
    Lead time can only ever be non-zero for t in [onset-HORIZON, onset-1] --
    _lead_time_for_method requires target_idx (t+HORIZON) >= onset AND
    t < onset simultaneously, which only HORIZON consecutive t values per
    spike can ever satisfy. An EXACT lead-time check needs exactly HORIZON
    window-starts per spike, not a wide lookback -- random subsampling
    (EVAL_MAX_WINDOWS/EVAL_STRIDE) is close to certain to miss this narrow
    window across many spikes, which is why lead time reads near-zero even
    when RMSE/MAE look good.
    """
    indices = set()
    for onset in spike_onsets:
        for t in range(max(0, onset - HORIZON), onset):
            start = t - WINDOW_SIZE + 1
            if 0 <= start < n_windows:
                indices.add(start)
    return indices


# ============================================================================
# Main evaluation
# ============================================================================

def run_evaluation():
    train_df, test_df = _load_test_split()

    prophet_model = load_prophet_model()
    print("🧠 [ProximaScale] Prophet model loaded")
    uni_model, uni_scaler = _get_univariate_artifacts(train_df)

    n_windows = len(test_df) - WINDOW_SIZE - HORIZON + 1
    if n_windows < 1:
        raise ValueError("Test split too small for WINDOW_SIZE + HORIZON -- lower TRAIN_FRAC or add more data.")

    spike_onsets = _detect_spike_onsets(test_df["cpu_percent"].values, SCALE_THRESHOLD)
    print(f"🧠 [ProximaScale] Detected {len(spike_onsets)} spike onset(s) above {SCALE_THRESHOLD}% in test split")

    eval_indices = set(_select_eval_indices(n_windows))
    lead_time_indices = _lead_time_indices_for_spikes(spike_onsets, n_windows)
    extra_for_lead_time = lead_time_indices - eval_indices
    all_indices = sorted(eval_indices | lead_time_indices)

    print(f"🧠 [ProximaScale] Evaluating {len(eval_indices)} windows for RMSE/MAE "
          f"({n_windows} available, stride={EVAL_STRIDE}, cap={EVAL_MAX_WINDOWS}), "
          f"plus {len(extra_for_lead_time)} extra windows added for exact lead-time coverage "
          f"around the {len(spike_onsets)} detected spike(s)...")

    windows = []
    t_values = []
    targets_idx = []
    actuals = []
    in_eval_subset = []
    for i in all_indices:
        w = _row_to_window_dicts(test_df, i, WINDOW_SIZE)
        t = i + WINDOW_SIZE - 1
        target_idx = t + HORIZON
        windows.append(w)
        t_values.append(t)
        targets_idx.append(target_idx)
        actuals.append(float(test_df.loc[target_idx, "cpu_percent"]))
        in_eval_subset.append(i in eval_indices)
    actuals = np.array(actuals)
    in_eval_subset = np.array(in_eval_subset)

    preds = {
        "reactive": np.array([_reactive_predict(w) for w in windows]),
        "univariate": _univariate_predict_batch(uni_model, uni_scaler, windows),
        "prophet_alone": _prophet_alone_predict_batch(prophet_model, windows),
    }

    n_eval = len(windows)
    print(f"🧠 [ProximaScale] Running Hybrid (predict_load, n_passes={EVAL_N_PASSES}) "
          f"on {n_eval} windows -- this is the part most likely to be slow...")
    hybrid_preds = []
    for count, w in enumerate(windows):
        hybrid_preds.append(predict_load(w, n_passes=EVAL_N_PASSES)[0])
        print(f"\r🧠 [ProximaScale]   Hybrid: {count + 1}/{n_eval}", end="", flush=True)
    print()
    preds["hybrid"] = np.array(hybrid_preds)

    labels = {"reactive": "Reactive", "univariate": "Univariate LSTM",
              "prophet_alone": "Prophet Alone", "hybrid": "Hybrid (Full)"}
    method_names = ["reactive", "univariate", "prophet_alone", "hybrid"]

    rows = []
    for name in method_names:
        p = np.array(preds[name])

        # RMSE/MAE/oscillations use ONLY the original eval subset, so the
        # extra spike-adjacent windows added for lead time (a deliberately
        # non-random, clustered sample) don't skew them.
        p_eval = p[in_eval_subset]
        actuals_eval = actuals[in_eval_subset]
        rmse = float(np.sqrt(np.mean((p_eval - actuals_eval) ** 2)))
        mae = float(np.mean(np.abs(p_eval - actuals_eval)))
        oscillations = _oscillation_count(p_eval, SCALE_THRESHOLD)

        # Lead time uses the FULL set, since only the extra dense windows
        # can mathematically register a nonzero lead time at all.
        preds_by_t = dict(zip(t_values, p))
        target_idx_by_t = dict(zip(t_values, targets_idx))
        lead_times = [_lead_time_for_method(preds_by_t, target_idx_by_t, onset, SCALE_THRESHOLD)
                      for onset in spike_onsets]
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

        rows.append({
            "Method": labels[name],
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "Avg Lead Time (s)": round(avg_lead_time, 1),
            "Oscillations": oscillations,
        })

    table = pd.DataFrame(rows)
    print("\n🧠 [ProximaScale] ===== 4-WAY COMPARISON (Residual-Stacking) =====")
    print(table.to_string(index=False))
    print("🧠 [ProximaScale] Target Scaling Lead Time: >= 60s")
    print("🧠 [ProximaScale] ======================================\n")
    return table


if __name__ == "__main__":
    result_table = run_evaluation()

    assert len(result_table) == 4, "Expected exactly 4 methods in the comparison"

    hybrid_row = result_table[result_table["Method"] == "Hybrid (Full)"].iloc[0]
    prophet_row = result_table[result_table["Method"] == "Prophet Alone"].iloc[0]
    if hybrid_row["RMSE"] > prophet_row["RMSE"]:
        print("🧠 [ProximaScale] ⚠️  WARNING: Hybrid RMSE is worse than Prophet Alone -- "
              "the LSTM's residual correction isn't helping. Check training/data before writing this up.")
    else:
        print(f"🧠 [ProximaScale] Hybrid RMSE ({hybrid_row['RMSE']}) improves on Prophet Alone "
              f"({prophet_row['RMSE']}) -- the LSTM's residual correction is adding value.")

    print("🧠 [ProximaScale] evaluate.py (residual-stacking) self-test PASSED.")

