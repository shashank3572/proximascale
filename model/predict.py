"""
ProximaScale - Phase 9/12: Shared Interface (Residual-Stacking)
----------------------------------------------------------------
THE HANDOFF TO PERSON D. predict_load(window) is the one function the rest
of the system calls.

REWORKED for Phase 12's residual-stacking architecture:
    Counterfactual correction (Phase 7) -> Preprocess/scale (Phase 1)
    -> Prophet forecast for the window's OWN future timestamps (Phase 3/12)
    -> LSTM predicts the RESIDUAL + MC Dropout uncertainty (Phase 2, 5, 12)
    -> final = Prophet forecast + LSTM residual (ADDITION, not a weighted
       average -- ensemble.py's weighted blend is no longer used here)
    -> Anomaly check (Phase 6) -> Return

BREAKING CONTRACT CHANGE: every window dict now REQUIRES a 'timestamp' key
(datetime or ISO string), in addition to cpu_percent/memory_percent/
request_rate. Prophet needs real calendar timestamps to forecast against --
this is a genuine interface change Person D's live monitoring loop must
supply. Missing timestamps will trip the safe fallback path below, not crash.

Also fixes a real bug from the old version: predict_prophet(periods=3) used
to forecast forward from wherever Prophet's OWN training data ended, not
from the live window's actual current time. get_prophet_fitted() with
explicit future timestamps (computed from the window's own last reading)
anchors the forecast correctly regardless of how long ago Prophet was
trained relative to now.
"""

import pickle
from pathlib import Path

import pandas as pd

from preprocessing import scale_features, load_scaler, WINDOW_SIZE
from lstm_model import MODEL_PATH, HORIZON
from prophet_model import load_prophet_model, get_prophet_fitted
from uncertainty import predict_with_uncertainty
from anomaly import is_anomaly
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
RESIDUAL_SCALER_PATH = SAVED_DIR / "scaler_residual.pkl"   # must match train.py exactly

SAMPLING_INTERVAL_SECONDS = 30   # must match prophet_model.py's DEFAULT_FREQ ("30s")

# Lazy-loaded, module-level cache -- populated on first call, reused after.
_model = None
_scaler = None
_residual_scaler = None
_prophet_model = None


def _load_artifacts():
    """Loads the trained LSTM, feature scaler, residual scaler, and Prophet
    model ONCE. The .h5 file is tens of MB and must never be reloaded from
    disk on every call."""
    global _model, _scaler, _residual_scaler, _prophet_model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH, compile=False)
        print(f"🧠 [ProximaScale] LSTM model loaded from {MODEL_PATH}")
    if _scaler is None:
        _scaler = load_scaler()
        print("🧠 [ProximaScale] Feature scaler loaded")
    if _residual_scaler is None:
        with open(RESIDUAL_SCALER_PATH, "rb") as f:
            _residual_scaler = pickle.load(f)
        print("🧠 [ProximaScale] Residual scaler loaded")
    if _prophet_model is None:
        _prophet_model = load_prophet_model()
        print("🧠 [ProximaScale] Prophet model loaded")


def predict_load(window,n_passes=None):
    """
    window: list of dicts, oldest -> newest, length == WINDOW_SIZE (10).
    Each dict needs 'timestamp', 'cpu_percent', 'memory_percent',
    'request_rate'; an optional 'post_scaling' bool defaults to False.

    Returns (predicted_load, upper_bound, is_anomaly_flag):
        predicted_load : float, forecast CPU % ~90s ahead
                         (= Prophet's forecast + LSTM's residual forecast)
        upper_bound    : float, (Prophet forecast + LSTM residual mean)
                         + 2*std of the LSTM's residual uncertainty, for
                         that same step
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
        if "timestamp" not in df.columns:
            raise ValueError("window dicts must include 'timestamp' for Phase 12's Prophet lookup")
        if "post_scaling" not in df.columns:
            df["post_scaling"] = False

        # Counterfactual correction FIRST (cpu_percent only), matching how
        # the model was trained.
        df = apply_counterfactual_correction(df)
        df["cpu_percent"] = df["cpu_percent_corrected"]

        # Prophet forecast, anchored to THIS window's own last timestamp --
        # not to wherever Prophet's training data happened to end.
        last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        step = pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS)
        future_timestamps = [last_ts + step * i for i in range(1, HORIZON + 1)]
        prophet_forecast = get_prophet_fitted(_prophet_model, future_timestamps)

        # Preprocess with the SAME feature scaler fit during training.
        scaled = scale_features(df, _scaler)
        window_scaled = scaled.reshape(1, WINDOW_SIZE, -1)

        # LSTM forecast of the RESIDUAL + MC Dropout uncertainty (Phase 5),
        # inverse-transformed with the residual scaler (1 column, no padding
        # trick needed -- unlike the old 3-feature cpu_percent scaler).
        uncertainty_kwargs = {"n_passes": n_passes} if n_passes is not None else {}
        uncertainty = predict_with_uncertainty(
            _model, window_scaled, _residual_scaler,
            inverse_transform_fn=lambda values: _residual_scaler.inverse_transform(
                values.reshape(-1, 1)
            ).flatten(),
            **uncertainty_kwargs,
        )

        # Final = Prophet's forecast + LSTM's residual forecast (ADDITION,
        # per the reference paper's architecture -- not a weighted average).
        print(f"🧠 [ProximaScale] DEBUG window's own cpu_percent (corrected): "f"{[round(v, 2) for v in df['cpu_percent'].values]}")
        print(f"🧠 [ProximaScale] DEBUG window's post_scaling flags: {list(df['post_scaling'].values)}")
        print(f"🧠 [ProximaScale] DEBUG window's own cpu_percent (corrected): "f"{[round(v, 2) for v in df['cpu_percent'].values]}")
        print(f"🧠 [ProximaScale] DEBUG window's post_scaling flags: {list(df['post_scaling'].values)}")
        print(f"🧠 [ProximaScale] DEBUG Prophet forecast: {[round(v, 2) for v in prophet_forecast]}")
        print(f"🧠 [ProximaScale] DEBUG LSTM residual mean: {[round(v, 2) for v in uncertainty['mean']]}")
        final_mean = [p + r for p, r in zip(prophet_forecast, uncertainty["mean"])]
        final_upper_bound = [p + r for p, r in zip(prophet_forecast, uncertainty["upper_bound"])]

        predicted_load = final_mean[-1]            # farthest step, ~90s ahead
        upper_bound = final_upper_bound[-1]

        anomaly_flag = is_anomaly(df["cpu_percent"].values)

        return float(predicted_load), float(upper_bound), bool(anomaly_flag)

    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in predict_load, using fallback: {e}")
        return float(current_cpu * 1.15), float(current_cpu * 1.3), True


if __name__ == "__main__":
    from preprocessing import load_csv

    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)

    last_10 = df.iloc[200:200+WINDOW_SIZE][["timestamp", "cpu_percent", "memory_percent", "request_rate"]]
    window = last_10.to_dict("records")

    predicted_load, upper_bound, anomaly_flag = predict_load(window)
    print(f"🧠 [ProximaScale] predict_load() -> predicted_load={predicted_load:.2f}, "
          f"upper_bound={upper_bound:.2f}, is_anomaly={anomaly_flag}")

    assert isinstance(predicted_load, float), "predicted_load must be a native float"
    assert isinstance(upper_bound, float), "upper_bound must be a native float"
    assert isinstance(anomaly_flag, bool), "is_anomaly must be a native bool"

    print("🧠 [ProximaScale] Type checks passed.")

    fallback_result = predict_load(window[:5])
    print(f"🧠 [ProximaScale] Fallback test (short window): {fallback_result}")
    assert fallback_result[2] is True, "Fallback should always report is_anomaly=True"

    print("🧠 [ProximaScale] predict_load() self-test PASSED.")