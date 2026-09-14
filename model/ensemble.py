"""
ProximaScale - Phase 4: Hybrid Ensemble
-------------------------------------------
Combines the LSTM's 3-step CPU forecast with Prophet's 3-step CPU forecast
via a fixed weighted average: 0.7 * LSTM + 0.3 * Prophet.

Both forecasts must already be in real CPU % (not scaled) and the same
length/order (t+1, t+2, t+3) before being passed in here.
"""

LSTM_WEIGHT = 0.7
PROPHET_WEIGHT = 0.3


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
    print(f"🧠 [ProximaScale] Ensemble forecast (0.7*LSTM + 0.3*Prophet): "
          f"{[round(v, 2) for v in blended]}")

    assert len(blended) == 3, "Ensemble output should have 3 values"
    for l, p, b in zip(lstm_forecast, prophet_forecast, blended):
        expected = 0.7 * l + 0.3 * p
        assert abs(b - expected) < 1e-6, "Weighted average math is wrong"

    print("🧠 [ProximaScale] Ensemble self-test PASSED.")