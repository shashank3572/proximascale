"""
ProximaScale - Phase 8: Training Pipeline
------------------------------------------
Ties Phase 1 (preprocessing) and Phase 2 (LSTM architecture) together into
an actual training run: load CSV -> preprocess -> train -> save model.

Produces model/saved/proximascale_lstm.h5, the file Phase 9's predict_load()
will lazy-load.
"""

from pathlib import Path

from preprocessing import prepare_training_data
from lstm_model import build_model, MODEL_PATH

THIS_DIR = Path(__file__).parent
DEFAULT_CSV = THIS_DIR.parent / "data" / "collected" / "metrics.csv"

EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5


def train_model(csv_path=DEFAULT_CSV, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 model_path=MODEL_PATH):
    """
    Runs the full training pipeline and saves the trained model to disk.
    Returns (model, history, X_test, y_test, scaler) so evaluate.py
    (Phase 10) can reuse the exact same test split without recomputing it.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping

        X_train, y_train, X_test, y_test, scaler = prepare_training_data(csv_path)

        model = build_model()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        print(f"🧠 [ProximaScale] Training on {X_train.shape[0]} windows, "
              f"validating on {X_test.shape[0]} windows...")

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
        print(f"🧠 [ProximaScale] Final train loss (MSE, scaled): {final_train_loss:.5f}")
        print(f"🧠 [ProximaScale] Final val loss (MSE, scaled):   {final_val_loss:.5f}")

        return model, history, X_test, y_test, scaler
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in train_model: {e}")
        raise


if __name__ == "__main__":
    model, history, X_test, y_test, scaler = train_model()

    assert MODEL_PATH.exists(), f"Expected saved model at {MODEL_PATH}, but it's missing"
    print(f"🧠 [ProximaScale] Confirmed model file exists on disk: {MODEL_PATH}")

    # Sanity: val_loss should be a real, finite number -- catches silent NaN
    # blowups (common with LSTMs on unscaled or badly-shaped data).
    final_val_loss = history.history["val_loss"][-1]
    assert final_val_loss == final_val_loss, "val_loss is NaN -- training diverged"  # NaN != NaN
    print(f"🧠 [ProximaScale] val_loss is finite ({final_val_loss:.5f}) -- training did not diverge")

    print("🧠 [ProximaScale] Training pipeline self-test PASSED.")