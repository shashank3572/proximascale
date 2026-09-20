"""
ProximaScale - Phase 4: Hybrid Ensemble  [DEPRECATED - historical record]
--------------------------------------------------------------------------
NOT USED by the live pipeline. ProximaScale moved from this fixed-weight
blend to RESIDUAL STACKING (Phase 12):

    old (this file):  final = w_lstm * LSTM_forecast + w_prophet * Prophet_forecast
    now (predict.py): final = Prophet_forecast + LSTM_residual_forecast

The blend had a hard ceiling -- it could never beat the LSTM alone (see the
train.py docstring). We keep this module, and its tests, to document that a
weighted ensemble was tried and evaluated before residual stacking replaced it.
Nothing in model/predict.py, main.py or train.py imports it
(tests/test_ensemble_deprecated.py enforces that).

Original behaviour: combines the LSTM's 3-step CPU forecast with Prophet's
3-step forecast via a fixed weighted average. The weights below are the last
tuned values (the first version used 0.7 / 0.3).

Both forecasts must already be in real CPU % (not scaled) and the same
length/order (t+1, t+2, t+3) before being passed in here.
"""

LSTM_WEIGHT = 0.93
PROPHET_WEIGHT = 0.07


def ensemble_predict(lstm_forecast, prophet_forecast,
                      lstm_weight=LSTM_WEIGHT, prophet_weight=PROPHET_WEIGHT):
    """
    lstm_forecast, prophet_forecast: sequences of the same length (typically 3),
    both already in real CPU % units, same t+1..t+N ordering.
    Returns a plain Python list of floats: the weighted-average forecast.
    """
    try:
        if len(lstm_forecast) != len(prophet_forecast):
            raise ValueError(
                f"Forecast length mismatch: LSTM has {len(lstm_forecast)}, "
                f"Prophet has {len(prophet_forecast)}"
            )
        blended = [
            float(lstm_weight * l + prophet_weight * p)
            for l, p in zip(lstm_forecast, prophet_forecast)
        ]
        return blended
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in ensemble_predict: {e}")
        raise


if __name__ == "__main__":
    # End-to-end plumbing test: chain Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
    # together. NOTE: the LSTM is UNTRAINED at this point (random weights) --
    # its numbers are meaningless until Phase 8. This test only proves the
    # pieces fit together and ensemble_predict() blends correctly.
    from pathlib import Path

    from preprocessing import (
        load_csv, chronological_split, fit_scaler, scale_features,
        inverse_transform_cpu, WINDOW_SIZE,
    )
    from lstm_model import build_model
    from prophet_model import train_prophet, predict_prophet

    THIS_DIR = Path(__file__).parent
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"

    df = load_csv(csv_path)
    train_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)
    train_scaled = scale_features(train_df, scaler)

    # LSTM forecast from the most recent 10-step window (model is UNTRAINED)
    lstm_model = build_model()
    last_window = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)
    lstm_scaled_forecast = lstm_model.predict(last_window, verbose=0)[0]
    lstm_forecast = [float(v) for v in inverse_transform_cpu(scaler, lstm_scaled_forecast)]

    # Prophet forecast
    prophet_model = train_prophet(train_df)
    prophet_forecast = predict_prophet(prophet_model, periods=3)

    print(f"🧠 [ProximaScale] LSTM forecast (UNTRAINED, random weights -- "
          f"ignore the actual numbers): {[round(v, 2) for v in lstm_forecast]}")
    print(f"🧠 [ProximaScale] Prophet forecast: {[round(v, 2) for v in prophet_forecast]}")

    blended = ensemble_predict(lstm_forecast, prophet_forecast)
    print(f"🧠 [ProximaScale] Ensemble forecast ({LSTM_WEIGHT}*LSTM + {PROPHET_WEIGHT}*Prophet): "
          f"{[round(v, 2) for v in blended]}")

    assert len(blended) == 3, "Ensemble output should have 3 values"
    for l, p, b in zip(lstm_forecast, prophet_forecast, blended):
        expected = LSTM_WEIGHT * l + PROPHET_WEIGHT * p
        assert abs(b - expected) < 1e-6, "Weighted average math is wrong"

    print("🧠 [ProximaScale] Ensemble self-test PASSED.")