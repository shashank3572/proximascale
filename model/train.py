"""
ProximaScale - Phase 8/12: Training Pipeline (Residual-Stacking)
------------------------------------------------------------------
REWORKED for Phase 12's residual-stacking hybrid (reference paper: Guruge &
Priyadarshana 2025, Section 3.1, Equation 1: y_t = g(t)+s(t)+h(t)+ε_t).

Old (Phase 8): LSTM trained directly on raw cpu_percent targets, later
blended with Prophet via a fixed weighted average (ensemble.py). That had a
hard ceiling -- the blend could never beat the LSTM alone.

New (Phase 12):
    1. Prophet fits g(t)+s(t)+h(t) on the corrected training series.
    2. residual = corrected_actual - Prophet_fitted, for every training row.
    3. The LSTM is trained to predict FUTURE RESIDUALS (3 steps ahead), not
       raw cpu_percent -- it only has to learn what Prophet gets wrong.
    4. At inference (predict.py): final = Prophet_forecast + LSTM_residual_forecast.

REQUIRES Prophet to already be trained and saved. Run
`python model/prophet_model.py` FIRST, every time, before this file, or
loading proximascale_prophet.pkl will fail.

ALSO FIXES A REAL BUG found while building this: the OLD prepare_training_data()
never applied counterfactual correction (Phase 7) before training, despite
predict.py's docstring claiming it did -- live inference was correcting its
input while the model was actually trained on raw, uncorrected data. This
version corrects both train and test splits before anything else.
"""

import pickle
from pathlib import Path

import numpy as np

from preprocessing import (
    load_csv, chronological_split, fit_scaler, save_scaler, scale_features,
    FEATURE_COLS, WINDOW_SIZE, HORIZON,
)
from lstm_model import build_model, MODEL_PATH
from prophet_model import load_prophet_model, get_prophet_fitted
from counterfactual import apply_counterfactual_correction

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
DEFAULT_CSV = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
RESIDUAL_SCALER_PATH = SAVED_DIR / "scaler_residual.pkl"

EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
TRAIN_FRAC = 0.8   # must match prophet_model.py's chronological_split() call, or
                   # Prophet and the LSTM will be trained on mismatched splits


def _prep_corrected(df):
    """Same pattern used everywhere else in the project: default
    post_scaling, then apply Phase 7's correction."""
    df = df.copy()
    if "post_scaling" not in df.columns:
        df["post_scaling"] = False
    return apply_counterfactual_correction(df)


def _create_residual_windows(feature_scaled, residual_scaled, window_size=WINDOW_SIZE, horizon=HORIZON):
    """
    Same sliding-window idea as preprocessing.create_sliding_windows(), but
    X comes from the 3-feature scaled array (unchanged) while y comes from
    a SEPARATE 1-D residual array with its own scaler -- the original
    function assumed X and y lived in the same array, which no longer
    holds once the residual has a different scale/distribution than CPU%.
    """
    n_rows = feature_scaled.shape[0]
    X, y = [], []
    last_start = n_rows - window_size - horizon
    for i in range(last_start + 1):
        X.append(feature_scaled[i: i + window_size])
        y.append(residual_scaled[i + window_size: i + window_size + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(csv_path=DEFAULT_CSV, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 model_path=MODEL_PATH, train_frac=TRAIN_FRAC):
    """
    Runs the residual-stacking training pipeline and saves the trained LSTM
    to disk. Returns (model, history, X_test, y_test, feature_scaler,
    residual_scaler, train_df, test_df) -- evaluate.py (Phase 12) needs the
    raw train/test dataframes too, not just arrays, since Prophet needs
    real timestamps to compute its fitted values.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler

        df = load_csv(csv_path)
        train_df, test_df = chronological_split(df, train_frac)

        train_df = _prep_corrected(train_df)
        test_df = _prep_corrected(test_df)
        train_df["cpu_percent"] = train_df["cpu_percent_corrected"]
        test_df["cpu_percent"] = test_df["cpu_percent_corrected"]

        prophet_model = load_prophet_model()
        print("🧠 [ProximaScale] Loaded pre-trained Prophet model for residual computation")

        train_fitted = get_prophet_fitted(prophet_model, train_df["timestamp"])
        test_fitted = get_prophet_fitted(prophet_model, test_df["timestamp"])
        train_residual = train_df["cpu_percent"].values - np.array(train_fitted)
        test_residual = test_df["cpu_percent"].values - np.array(test_fitted)

        feature_scaler = fit_scaler(train_df, FEATURE_COLS)
        save_scaler(feature_scaler)
        train_feature_scaled = scale_features(train_df, feature_scaler)
        test_feature_scaled = scale_features(test_df, feature_scaler)

        residual_scaler = MinMaxScaler(feature_range=(0, 1))
        train_residual_scaled = residual_scaler.fit_transform(train_residual.reshape(-1, 1)).flatten()
        test_residual_scaled = residual_scaler.transform(test_residual.reshape(-1, 1)).flatten()
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESIDUAL_SCALER_PATH, "wb") as f:
            pickle.dump(residual_scaler, f)
        print(f"🧠 [ProximaScale] Residual scaler saved -> {RESIDUAL_SCALER_PATH}")

        X_train, y_train = _create_residual_windows(train_feature_scaled, train_residual_scaled)
        X_test, y_test = _create_residual_windows(test_feature_scaled, test_residual_scaled)

        model = build_model()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        print(f"🧠 [ProximaScale] Training on {X_train.shape[0]} windows, "
              f"validating on {X_test.shape[0]} windows (RESIDUAL targets)...")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"🧠 [ProximaScale] Model saved -> {model_path}")

        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        print(f"🧠 [ProximaScale] Final train loss (MSE, scaled residual): {final_train_loss:.5f}")
        print(f"🧠 [ProximaScale] Final val loss (MSE, scaled residual):   {final_val_loss:.5f}")

        return model, history, X_test, y_test, feature_scaler, residual_scaler, train_df, test_df
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in train_model: {e}")
        raise


if __name__ == "__main__":
    (model, history, X_test, y_test, feature_scaler, residual_scaler,
     train_df, test_df) = train_model()

    assert MODEL_PATH.exists(), f"Expected saved model at {MODEL_PATH}, but it's missing"
    assert RESIDUAL_SCALER_PATH.exists(), f"Expected residual scaler at {RESIDUAL_SCALER_PATH}, but it's missing"
    print("🧠 [ProximaScale] Confirmed model + residual scaler exist on disk")

    final_val_loss = history.history["val_loss"][-1]
    assert final_val_loss == final_val_loss, "val_loss is NaN -- training diverged"
    print(f"🧠 [ProximaScale] val_loss is finite ({final_val_loss:.5f}) -- training did not diverge")

    print("🧠 [ProximaScale] Training pipeline (residual-stacking) self-test PASSED.")