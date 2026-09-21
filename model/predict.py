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

NOTE (model loading): load_model() comes from tensorflow.keras.models. An
earlier tf_keras / TF_USE_LEGACY_KERAS approach was reverted because it
cannot read Keras-3-saved models. MODEL_PATH (model/lstm_model.py) must point
at the residual-stacking LSTM artifact; see model/saved/README.md.
"""

import logging
import os
import sys
import pickle
from pathlib import Path

import pandas as pd

# Bare imports below match the rest of model/*.py -- resolve them relative
# to this file regardless of what cwd the caller is running from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import scale_features, load_scaler, WINDOW_SIZE
from lstm_model import MODEL_PATH, HORIZON
from prophet_model import load_prophet_model, get_prophet_fitted
from uncertainty import predict_with_uncertainty
from anomaly import is_anomaly
from counterfactual import apply_counterfactual_correction

logger = logging.getLogger(__name__)

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
        print(f"[ProximaScale] LSTM model loaded from {MODEL_PATH}")
    if _scaler is None:
        _scaler = load_scaler()
        print("[ProximaScale] Feature scaler loaded")
    if _residual_scaler is None:
        with open(RESIDUAL_SCALER_PATH, "rb") as f:
            _residual_scaler = pickle.load(f)
        print("[ProximaScale] Residual scaler loaded")
    if _prophet_model is None:
        _prophet_model = load_prophet_model()
        print("[ProximaScale] Prophet model loaded")


def prepare_scaled_window(window):
    """Shared window preprocessing: counterfactual correction (cpu_percent
    only) + feature scaling with the trained scaler. predict_load() uses this
    to build its LSTM input, and main.py's SHAP wiring uses it to rebuild the
    EXACT same input for attribution -- one source of truth for both.

    window: dicts (or MetricRecord objects with .to_dict()), oldest -> newest,
    length == WINDOW_SIZE. Returns (scaled, corrected_cpu):
        scaled        : np.ndarray shape (WINDOW_SIZE, 3) -- the LSTM input
        corrected_cpu : np.ndarray of the counterfactual-corrected cpu series
                        (predict_load's anomaly check consumes this)
    Requires _load_artifacts() to have run (needs the fitted feature scaler).
    """
    if window and hasattr(window[0], "to_dict"):
        window = [record.to_dict() for record in window]

    df = pd.DataFrame(window)
    if "timestamp" not in df.columns:
        raise ValueError("window dicts must include 'timestamp' for Phase 12's Prophet lookup")
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False

    # Counterfactual correction FIRST (cpu_percent only), matching how
    # the model was trained.
    df = apply_counterfactual_correction(df)
    df["cpu_percent"] = df["cpu_percent_corrected"]

    # Preprocess with the SAME feature scaler fit during training.
    return scale_features(df, _scaler), df["cpu_percent"].values


def get_shap_artifacts():
    """(keras_model, feature_scaler) after ensuring artifacts are loaded.
    Lets main.py's SHAP wiring reuse the cached model instead of loading the
    .h5 a second time."""
    _load_artifacts()
    return _model, _scaler


def predict_load(window, n_passes=None):
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

    Both load values are clamped to the physical 0–100 CPU% range.

    Never raises -- on any internal failure, falls back to
    (current_cpu * 1.15, current_cpu * 1.3, True) so Person D's main loop
    never dies because of us.
    """
   # Accept MetricRecord objects from monitoring.storage
# and dictionaries from tests/direct callers.
    if window and hasattr(window[0], "to_dict"):
        window = [record.to_dict() for record in window]

    current_cpu = float(window[-1].get("cpu_percent", 0.0)) if window else 0.0

    try:
        _load_artifacts()

        if len(window) != WINDOW_SIZE:
            raise ValueError(f"window must have exactly {WINDOW_SIZE} readings, got {len(window)}")

        # Prophet forecast, anchored to THIS window's own last timestamp --
        # not to wherever Prophet's training data happened to end.
        last_ts = pd.Timestamp(window[-1]["timestamp"])
        step = pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS)
        future_timestamps = [last_ts + step * i for i in range(1, HORIZON + 1)]
        prophet_forecast = get_prophet_fitted(_prophet_model, future_timestamps)

        # Counterfactual correction + feature scaling (shared helper so the
        # SHAP attribution in main.py sees the identical LSTM input).
        scaled, corrected_cpu = prepare_scaled_window(window)
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
        final_mean = [p + r for p, r in zip(prophet_forecast, uncertainty["mean"])]
        final_upper_bound = [p + r for p, r in zip(prophet_forecast, uncertainty["upper_bound"])]

        predicted_load = final_mean[-1]            # farthest step, ~90s ahead
        upper_bound = final_upper_bound[-1]

        # CPU% is physically bounded at 0–100. Prophet + residual + 2σ can
        # overshoot (observed live: upper bounds of 102–156 while the
        # container was pegged at 100); clamping keeps the decision engine's
        # thresholds meaningful. The clamp preserves ORDER, so a bound above
        # 100 still reads as "very high risk" to the engine.
        predicted_load = min(100.0, max(0.0, predicted_load))
        upper_bound = min(100.0, max(0.0, upper_bound))

        anomaly_flag = is_anomaly(corrected_cpu)

        return float(predicted_load), float(upper_bound), bool(anomaly_flag)

    except Exception as e:
        # Loud on purpose: a broken artifact/dependency makes EVERY call fall
        # back, which reports is_anomaly=True and so forces scale-ups.
        logger.error("predict_load FALLBACK (%s): %s", type(e).__name__, e)
        print(f"[ProximaScale] ERROR in predict_load, using fallback: {e}")
        fallback_mean = min(100.0, max(0.0, current_cpu * 1.15))
        fallback_upper = min(100.0, max(0.0, current_cpu * 1.3))
        return float(fallback_mean), float(fallback_upper), True


if __name__ == "__main__":
    from preprocessing import load_csv

    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)

    last_10 = df.tail(WINDOW_SIZE)[["timestamp", "cpu_percent", "memory_percent", "request_rate"]]
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
