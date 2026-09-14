"""
ProximaScale - Phase 10: Evaluation
------------------------------------------------
Runs a 4-way ablation comparison on a held-out test split of metrics.csv:

    1. Reactive          - no model at all. "Predicted" future load = the
                           current raw reading (what a dumb threshold-based
                           autoscaler effectively does today).
    2. Univariate LSTM   - same LSTM shape as Phase 2, trained here on
                           cpu_percent ALONE. Shows the value of adding
                           memory_percent + request_rate as features.
    3. Multivariate LSTM - the actual Phase 8 model, raw forecast only
                           (counterfactual correction applied since the
                           model requires it, but NO Prophet ensemble).
                           Isolates what the multivariate LSTM alone adds
                           on top of the univariate baseline.
    4. Hybrid (Full)     - predict_load() from Phase 9 as-is: correction +
                           multivariate LSTM + Prophet ensemble. This is
                           what ships to Person D.

Metrics per method:
    - RMSE / MAE        : forecast accuracy at the t+3 (~90s ahead) horizon.
    - Avg Lead Time (s) : for each detected spike in the test split, how many
                          seconds BEFORE the actual breach the method's
                          forecast already crossed SCALE_THRESHOLD. Target
                          >= 60s. 0 means the method never gave advance warning.
    - Oscillations       : how many times the binary "scale up?" decision
                          flips across the test set (thrashing/flapping).
                          Lower is better.

DESIGN NOTES (read before you write these up):
  - Counterfactual correction is applied to BOTH the univariate and
    multivariate LSTM inputs, since it's a data-cleaning step the models
    need, not something being ablated row-by-row. Only feature count
    (uni vs multi) and ensemble blending (raw LSTM vs +Prophet) differ
    between rows 2-4.
  - If your Phase 0 dummy data never actually sets post_scaling=True,
    correction is a no-op on this synthetic set, and rows 3 vs 4 will
    look closer than they would on real data with real scaling events.
    Worth a sentence in your report.
  - TEST_SPLIT below independently takes the LAST 20% of metrics.csv as
    held-out. If Phase 1/8 already trained on a different split ratio,
    change TEST_SPLIT to match -- otherwise you risk evaluating on rows
    the model has already seen, which would invalidate these numbers.
  - SCALE_THRESHOLD is a placeholder (75.0%). If Person D's real
    autoscaler uses a different scale-up trigger, change it here to match.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import load_csv, scale_features, load_scaler, WINDOW_SIZE
from lstm_model import MODEL_PATH, HORIZON
from predict import predict_load
from uncertainty import predict_with_uncertainty
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
UNIVARIATE_MODEL_PATH = SAVED_DIR / "proximascale_lstm_univariate.h5"
UNIVARIATE_SCALER_PATH = SAVED_DIR / "scaler_univariate.pkl"

SAMPLING_INTERVAL_SEC = 30      # must match Phase 0's dummy data generation rate
SCALE_THRESHOLD = 75.0          # CPU% that triggers a "scale up" decision -- tune to match Person D
TEST_SPLIT = 0.2                # last 20% of metrics.csv, held out chronologically


# ============================================================================
# Data loading / windowing
# ============================================================================

def _load_test_split():
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"🧠 [ProximaScale] Loaded {len(df)} rows -> {len(train_df)} train / {len(test_df)} test")
    return train_df, test_df


def _row_to_window_dicts(df, start, size):
    cols = ["cpu_percent", "memory_percent", "request_rate"]
    if "post_scaling" in df.columns:
        cols.append("post_scaling")
    return df.iloc[start:start + size][cols].to_dict("records")


def _prep_corrected(window):
    """Same pattern as predict.py: default post_scaling, then correct."""
    df = pd.DataFrame(window)
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False
    return apply_counterfactual_correction(df)


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


def _univariate_predict(model, scaler, window):
    corrected = _prep_corrected(window)
    series = corrected["cpu_percent_corrected"].values.reshape(-1, 1)
    scaled = scaler.transform(series).reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(scaled, verbose=0)[0]
    pred_cpu = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return float(pred_cpu[-1])


# ============================================================================
# Method 3: Multivariate LSTM, raw (correction applied, Prophet ablated)
# ============================================================================

def _get_multivariate_artifacts():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    scaler = load_scaler()
    print("🧠 [ProximaScale] Multivariate LSTM (main model) loaded")
    return model, scaler


def _multivariate_predict(model, scaler, window):
    df = _prep_corrected(window)
    df["cpu_percent"] = df["cpu_percent_corrected"]
    scaled = scale_features(df, scaler)
    window_scaled = scaled.reshape(1, WINDOW_SIZE, -1)
    result = predict_with_uncertainty(model, window_scaled, scaler)
    return float(result["mean"][-1])


# ============================================================================
# Method 4: Hybrid -- predict_load() as-is, no modification
# ============================================================================
# (called directly in the loop below; no wrapper needed)


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
    return float((spike_onset_idx - earliest_t) * SAMPLING_INTERVAL_SEC)


# ============================================================================
# Main evaluation
# ============================================================================

def run_evaluation():
    train_df, test_df = _load_test_split()

    mv_model, mv_scaler = _get_multivariate_artifacts()
    uni_model, uni_scaler = _get_univariate_artifacts(train_df)

    n_windows = len(test_df) - WINDOW_SIZE - HORIZON + 1
    if n_windows < 1:
        raise ValueError("Test split too small for WINDOW_SIZE + HORIZON -- lower TEST_SPLIT or add more data.")

    method_names = ["reactive", "univariate", "multivariate", "hybrid"]
    preds = {name: [] for name in method_names}
    targets_idx = []
    actuals = []

    print(f"🧠 [ProximaScale] Running {n_windows} sliding-window comparisons on the test split...")

    for i in range(n_windows):
        window = _row_to_window_dicts(test_df, i, WINDOW_SIZE)
        t = i + WINDOW_SIZE - 1
        target_idx = t + HORIZON
        actuals.append(float(test_df.loc[target_idx, "cpu_percent"]))
        targets_idx.append(target_idx)

        preds["reactive"].append(_reactive_predict(window))
        preds["univariate"].append(_univariate_predict(uni_model, uni_scaler, window))
        preds["multivariate"].append(_multivariate_predict(mv_model, mv_scaler, window))
        preds["hybrid"].append(predict_load(window)[0])

        if (i + 1) % 100 == 0:
            print(f"🧠 [ProximaScale]   ...{i + 1}/{n_windows} windows done")

    actuals = np.array(actuals)
    spike_onsets = _detect_spike_onsets(test_df["cpu_percent"].values, SCALE_THRESHOLD)
    print(f"🧠 [ProximaScale] Detected {len(spike_onsets)} spike onset(s) above {SCALE_THRESHOLD}% in test split")

    labels = {"reactive": "Reactive", "univariate": "Univariate LSTM",
              "multivariate": "Multivariate LSTM", "hybrid": "Hybrid (Full)"}

    rows = []
    for name in method_names:
        p = np.array(preds[name])
        rmse = float(np.sqrt(np.mean((p - actuals) ** 2)))
        mae = float(np.mean(np.abs(p - actuals)))
        oscillations = _oscillation_count(p, SCALE_THRESHOLD)

        preds_by_t = {i + WINDOW_SIZE - 1: val for i, val in enumerate(p)}
        target_idx_by_t = {i + WINDOW_SIZE - 1: targets_idx[i] for i in range(len(p))}
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
    print("\n🧠 [ProximaScale] ===== 4-WAY COMPARISON =====")
    print(table.to_string(index=False))
    print("🧠 [ProximaScale] Target Scaling Lead Time: >= 60s")
    print("🧠 [ProximaScale] ================================\n")
    return table


if __name__ == "__main__":
    result_table = run_evaluation()

    assert len(result_table) == 4, "Expected exactly 4 methods in the comparison"

    hybrid_row = result_table[result_table["Method"] == "Hybrid (Full)"].iloc[0]
    reactive_row = result_table[result_table["Method"] == "Reactive"].iloc[0]
    if hybrid_row["RMSE"] > reactive_row["RMSE"]:
        print("🧠 [ProximaScale] ⚠️  WARNING: Hybrid RMSE is worse than the naive "
              "Reactive baseline. Check your trained model/data before writing this up.")

    print("🧠 [ProximaScale] evaluate.py self-test PASSED (table generated successfully).")