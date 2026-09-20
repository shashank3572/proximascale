"""
ProximaScale - Phase 3: Prophet Model
----------------------------------------
Univariate complement to the LSTM: Facebook Prophet forecasting cpu_percent
3 steps ahead, to be blended with the LSTM output in Phase 4
(0.7 * LSTM + 0.3 * Prophet).

Prophet requires a dataframe with columns 'ds' (datetime) and 'y' (target) --
this module handles the conversion from our timestamp/cpu_percent columns.
"""

import pickle
from pathlib import Path

import pandas as pd
from prophet import Prophet

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
PROPHET_PATH = SAVED_DIR / "proximascale_prophet.pkl"

TARGET_COL = "cpu_percent"
TIMESTAMP_COL = "timestamp"
DEFAULT_PERIODS = 3
DEFAULT_FREQ = "30s"   # matches the 30-second sampling interval from Phase 0


def _to_prophet_format(df, target_col=TARGET_COL, timestamp_col=TIMESTAMP_COL):
    """Prophet requires exactly two columns: 'ds' and 'y'."""
    prophet_df = df[[timestamp_col, target_col]].rename(
        columns={timestamp_col: "ds", target_col: "y"}
    )
    return prophet_df


def train_prophet(train_df, target_col=TARGET_COL, timestamp_col=TIMESTAMP_COL):
    """
    Fits Prophet on the TRAINING split only (same rule as the LSTM scaler --
    never let Prophet see test-period rows before evaluation).
    train_df: dataframe with at least [timestamp_col, target_col].
    Returns the fitted Prophet model.
    """
    try:
        prophet_df = _to_prophet_format(train_df, target_col, timestamp_col)
        model = Prophet()
        model.fit(prophet_df)
        print(f"🧠 [ProximaScale] Prophet trained on {len(prophet_df)} rows")
        return model
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR training Prophet: {e}")
        raise


def predict_prophet(model, periods=DEFAULT_PERIODS, freq=DEFAULT_FREQ):
    """
    Forecasts `periods` steps ahead of the model's training data.
    Returns a plain Python list of floats, length == periods, ordered
    earliest -> latest (matches the LSTM's Dense(3) output ordering).
    """
    try:
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        future_only = forecast.tail(periods)
        predictions = [float(v) for v in future_only["yhat"].values]
        return predictions
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR predicting with Prophet: {e}")
        raise


def get_prophet_fitted(model, timestamps):
    """
    Returns Prophet's fitted trend+seasonality+holiday value (g(t)+s(t)+h(t)
    from Equation 1 in the reference paper) for ARBITRARY timestamps --
    past (to compute training residuals in Phase 12's train.py) or future
    (for live forecasting in predict.py). Unlike predict_prophet(), which
    only walks forward from the end of training data via
    make_future_dataframe(), this works for any timestamp because Prophet's
    own model.predict() just needs a 'ds' column -- it doesn't care whether
    those dates are in the future relative to training.

    timestamps: any iterable of datetime-like values (pd.Timestamp, str, etc).
    Returns a plain Python list of floats, same length/order as timestamps.
    Note: 'yhat' IS g(t)+s(t)+h(t) -- Prophet doesn't (and can't) predict
    the noise term ε_t, so yhat already is exactly the value we want here.
    """
    try:
        future_df = pd.DataFrame({"ds": pd.to_datetime(list(timestamps))})
        forecast = model.predict(future_df)
        return [float(v) for v in forecast["yhat"].values]
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in get_prophet_fitted: {e}")
        raise

def save_prophet_model(model, path=PROPHET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"🧠 [ProximaScale] Prophet model saved -> {path}")


def load_prophet_model(path=PROPHET_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from preprocessing import load_csv, chronological_split

    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    df = load_csv(csv_path)
    train_df, test_df = chronological_split(df)

    model = train_prophet(train_df)
    predictions = predict_prophet(model, periods=DEFAULT_PERIODS)

    print(f"🧠 [ProximaScale] Predicted next {DEFAULT_PERIODS} CPU values: "
          f"{[round(p, 2) for p in predictions]}")

    assert len(predictions) == DEFAULT_PERIODS, "predict_prophet returned wrong length"
    assert all(isinstance(p, float) for p in predictions), "predictions must be native floats"

    save_prophet_model(model)

    # Round-trip check: reload from disk, predict again
    reloaded = load_prophet_model()
    reloaded_predictions = predict_prophet(reloaded, periods=DEFAULT_PERIODS)
    assert len(reloaded_predictions) == DEFAULT_PERIODS, "Reloaded model prediction length mismatch"

    # Phase 12: verify get_prophet_fitted() works for arbitrary timestamps --
    # both historical (used to compute training residuals) and future.
    historical_fitted = get_prophet_fitted(model, train_df["timestamp"].iloc[:5])
    assert len(historical_fitted) == 5, "get_prophet_fitted length mismatch on historical timestamps"
    print(f"🧠 [ProximaScale] Prophet fitted (first 5 historical rows): "
          f"{[round(v, 2) for v in historical_fitted]}")

    print("🧠 [ProximaScale] Prophet self-test PASSED.")