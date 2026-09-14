"""
ProximaScale - Phase 9: Shared Interface
----------------------------------------------
THE HANDOFF TO PERSON D. predict_load(window) is the one function the rest
of the system calls -- everything from Phases 1-7 is glued together here.

Pipeline order for each call:
    Counterfactual correction (Phase 7) -> Preprocess/scale (Phase 1)
    -> LSTM + MC Dropout uncertainty (Phase 2, 5) + Prophet (Phase 3)
    -> Ensemble blend (Phase 4) -> Anomaly check (Phase 6) -> Return

Counterfactual correction runs FIRST and only on cpu_percent, because the
LSTM was TRAINED on counterfactual-corrected data (Phase 8) -- feeding it
raw, uncorrected live readings would create a train/inference mismatch.

The LSTM (via saved .h5) and scaler are loaded ONCE and cached at module
level -- never reloaded per call. Prophet is also loaded once; note that its
forecast is static relative to whenever Phase 3 last trained it, not the
live window -- that's a known limitation of a batch-trained Prophet model,
worth knowing for your project writeup. The LSTM component (70% of the
ensemble weight) IS live and responds to the current window every call.

Of the 3-step (t+1, t+2, t+3) horizon, predict_load() reports the FARTHEST
step (t+3, ~90s ahead at the 30s sampling rate) as the headline number --
that gives more lead time against the "Scaling Lead Time >= 60s" target
than reporting t+1 would.

NOTE (integration-fix, Person D): load_model() uses tf_keras rather than
tensorflow.keras.models -- this matches the Keras-compat fix already
resolved on dev (see PROGRESS.md). Using tensorflow.keras.models here
reintroduces that bug against this env's TF version.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys

import pandas as pd

# Bare imports below match the rest of model/*.py -- resolve them relative
# to this file regardless of what cwd the caller is running from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import scale_features, load_scaler, WINDOW_SIZE
from lstm_model import MODEL_PATH, HORIZON
from prophet_model import load_prophet_model, predict_prophet
from uncertainty import predict_with_uncertainty
from ensemble import ensemble_predict
from anomaly import is_anomaly
from counterfactual import apply_counterfactual_correction

# Lazy-loaded, module-level cache -- populated on first call, reused after.
_model = None
_scaler = None
_prophet_model = None


def _load_artifacts():
    """Loads the trained LSTM, scaler, and Prophet model ONCE. The .h5 file
    is tens of MB and must never be reloaded from disk on every call."""
    global _model, _scaler, _prophet_model
    if _model is None:
        from tf_keras.models import load_model
        _model = load_model(MODEL_PATH)
        print(f"[ProximaScale] LSTM model loaded from {MODEL_PATH}")
    if _scaler is None:
        _scaler = load_scaler()
        print("[ProximaScale] Scaler loaded")
    if _prophet_model is None:
        _prophet_model = load_prophet_model()
        print("[ProximaScale] Prophet model loaded")


def predict_load(window):
    """
    window: list of dicts, oldest -> newest, length == WINDOW_SIZE (10).
    Each dict needs 'cpu_percent', 'memory_percent', 'request_rate'; an
    optional 'post_scaling' bool defaults to False if missing.

    Returns (predicted_load, upper_bound, is_anomaly_flag):
        predicted_load : float, forecast CPU % ~90s ahead
        upper_bound    : float, MC-Dropout mean + 2*std for that same step
        is_anomaly_flag: bool, whether the latest (corrected) reading is a spike

    Never raises -- on any internal failure, falls back to
    (current_cpu * 1.15, current_cpu * 1.3, True) so Person D's main loop
    never dies because of us.
    """
    current_cpu = float(window[-1].get("cpu_percent", 0.0)) if window else 0.0

    try:
        _load_artifacts()

        if len(window) != WINDOW_SIZE:
            raise ValueError(f"window must have exactly {WINDOW_SIZE} readings, got {len(window)}")

        df = pd.DataFrame(window)
        if "post_scaling" not in df.columns:
            df["post_scaling"] = False

        # Counterfactual correction FIRST (cpu_percent only), matching how
        # the model was trained.
        df = apply_counterfactual_correction(df)
        df["cpu_percent"] = df["cpu_percent_corrected"]

        # Preprocess with the SAME scaler fit during training.
        scaled = scale_features(df, _scaler)
        window_scaled = scaled.reshape(1, WINDOW_SIZE, -1)

        # LSTM forecast + MC Dropout uncertainty (Phase 5).
        uncertainty = predict_with_uncertainty(_model, window_scaled, _scaler)

        # Prophet forecast (Phase 3, static since its last training run).
        prophet_forecast = predict_prophet(_prophet_model, periods=HORIZON)

        # Ensemble blend across all 3 horizon steps (Phase 4).
        blended_mean = ensemble_predict(uncertainty["mean"], prophet_forecast)

        predicted_load = blended_mean[-1]             # farthest step, ~90s ahead
        upper_bound = uncertainty["upper_bound"][-1]   # LSTM-only uncertainty bound

        anomaly_flag = is_anomaly(df["cpu_percent"].values)

        return float(predicted_load), float(upper_bound), bool(anomaly_flag)

    except Exception as e:
        print(f"[ProximaScale] ERROR in predict_load, using fallback: {e}")
        return float(current_cpu * 1.15), float(current_cpu * 1.3), True


if __name__ == "__main__":
    from pathlib import Path
    from preprocessing import load_csv

    THIS_DIR = Path(__file__).parent
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)

    last_10 = df.tail(WINDOW_SIZE)[["cpu_percent", "memory_percent", "request_rate"]]
    window = last_10.to_dict("records")

    predicted_load, upper_bound, anomaly_flag = predict_load(window)
    print(f"[ProximaScale] predict_load() -> predicted_load={predicted_load:.2f}, "
          f"upper_bound={upper_bound:.2f}, is_anomaly={anomaly_flag}")

    assert isinstance(predicted_load, float), "predicted_load must be a native float"
    assert isinstance(upper_bound, float), "upper_bound must be a native float"
    assert isinstance(anomaly_flag, bool), "is_anomaly must be a native bool"

    print("[ProximaScale] Type checks passed.")

    fallback_result = predict_load(window[:5])
    print(f"[ProximaScale] Fallback test (short window): {fallback_result}")
    assert fallback_result[2] is True, "Fallback should always report is_anomaly=True"

    print("[ProximaScale] predict_load() self-test PASSED.")
