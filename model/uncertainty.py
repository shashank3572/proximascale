"""
ProximaScale - Phase 5: MC Dropout Uncertainty
----------------------------------------------------
THE PRIMARY NOVELTY. A standard LSTM point forecast gives no sense of how
confident the model actually is. Monte Carlo Dropout keeps the Dropout
layers ACTIVE at inference time (training=True) and runs many stochastic
forward passes over the same input -- the spread across those passes becomes
an uncertainty estimate, without needing a separate probabilistic model or an
ensemble of models.

predict_with_uncertainty() runs N_PASSES=30 forward passes, then computes:
    mean        -- average forecast across passes (real CPU %)
    std         -- how much the passes disagree (real CPU %)
    upper_bound -- mean + 2*std, the value autoscaling should actually act on
"""

import numpy as np
from preprocessing import inverse_transform_cpu, TARGET_COL_IDX

N_PASSES = 30
STD_MULTIPLIER = 2.0


def predict_with_uncertainty(model, window, scaler,
                              n_passes=N_PASSES, std_multiplier=STD_MULTIPLIER,
                              target_col_idx=TARGET_COL_IDX, inverse_transform_fn=None):
    """
    model:  a compiled Keras LSTM (ideally trained -- Phase 8).
    window: numpy array, shape (1, 10, 3) -- one scaled input window.
    scaler: the fitted scaler used to build `window` (3-feature scaler for
            the raw-CPU LSTM, or the 1-feature residual scaler for Phase 12's
            residual LSTM).
    inverse_transform_fn: optional. If None (default, unchanged behavior),
            uses inverse_transform_cpu(scaler, values) -- the original
            3-feature padding trick. Phase 12 passes a plain
            `residual_scaler.inverse_transform` here instead, since the
            residual scaler only has 1 column and doesn't need padding.

    Returns a dict with 'mean', 'std', 'upper_bound' -- each a list of
    `horizon` native Python floats, in the SAME units as whatever
    inverse_transform_fn produces (real CPU % normally, or real residual
    units when called from the Phase 12 residual path).
    """
    try:
        if inverse_transform_fn is None:
            inverse_transform_fn = lambda values: inverse_transform_cpu(scaler, values)

        # training=True keeps Dropout ACTIVE -- this is what makes it MC Dropout
        # instead of one deterministic forward pass.
        passes = np.stack([
            model(window, training=True).numpy()[0] for _ in range(n_passes)
        ])  # shape: (n_passes, horizon), still in SCALED units

        mean_scaled = passes.mean(axis=0)
        std_scaled = passes.std(axis=0)

        mean_cpu = inverse_transform_fn(mean_scaled)
        # std is a spread, not a position -- undo the scaler's linear factor
        # but NOT its offset, or the bound would be shifted off.
        std_cpu = std_scaled / scaler.scale_[target_col_idx]

        upper_bound_cpu = mean_cpu + std_multiplier * std_cpu
        return {
            "mean": [float(v) for v in mean_cpu],
            "std": [float(v) for v in std_cpu],
            "upper_bound": [float(v) for v in upper_bound_cpu],
        }
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in predict_with_uncertainty: {e}")
        raise


if __name__ == "__main__":
    from pathlib import Path
    from preprocessing import load_csv, chronological_split, fit_scaler, scale_features, WINDOW_SIZE
    from lstm_model import build_model

    THIS_DIR = Path(__file__).parent
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"

    df = load_csv(csv_path)
    train_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)
    train_scaled = scale_features(train_df, scaler)

    model = build_model()  # UNTRAINED -- this test only proves the mechanism works
    last_window = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)

    result = predict_with_uncertainty(model, last_window, scaler)

    print(f"🧠 [ProximaScale] Mean forecast:  {[round(v, 2) for v in result['mean']]}")
    print(f"🧠 [ProximaScale] Std:            {[round(v, 2) for v in result['std']]}")
    print(f"🧠 [ProximaScale] Upper bound:    {[round(v, 2) for v in result['upper_bound']]}")

    assert len(result["mean"]) == 3, "mean should have 3 values"
    assert len(result["std"]) == 3, "std should have 3 values"
    assert len(result["upper_bound"]) == 3, "upper_bound should have 3 values"
    assert all(s >= 0 for s in result["std"]), "std can't be negative"
    for m, s, u in zip(result["mean"], result["std"], result["upper_bound"]):
        assert abs(u - (m + 2 * s)) < 1e-6, "upper_bound should equal mean + 2*std"

    # With an UNTRAINED model, std should still be non-zero -- if it's exactly
    # 0.0 across the board, Dropout isn't actually firing.
    if all(s == 0.0 for s in result["std"]):
        print("🧠 [ProximaScale] WARNING: std is exactly 0 -- Dropout may not be active. Check training=True.")
    else:
        print("🧠 [ProximaScale] Non-zero std confirms Dropout is active across passes (as expected).")

    print("🧠 [ProximaScale] MC Dropout uncertainty self-test PASSED.")