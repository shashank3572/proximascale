"""
ProximaScale - Phase 2: LSTM Architecture
--------------------------------------------
Defines the LSTM backbone used by the hybrid ensemble:
    LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(3)

Input:  (batch, 10, 3)  -- 10 timesteps of [cpu_percent, memory_percent, request_rate]
Output: (batch, 3)      -- next 3 cpu_percent values (t+1, t+2, t+3)

NOTE: the Dropout layers must stay active at inference time for Phase 5
(MC Dropout uncertainty) by calling the model with training=True. Don't swap
them for a different regularizer or set them non-trainable later.
"""

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
MODEL_PATH = SAVED_DIR / "proximascale_lstm.keras"

WINDOW_SIZE = 10
N_FEATURES = 3
HORIZON = 3


def build_model(window_size=WINDOW_SIZE, n_features=N_FEATURES, horizon=HORIZON,
                 dropout_rate=0.2, learning_rate=0.001):
    """Builds and compiles the ProximaScale LSTM."""
    try:
        model = keras.Sequential([
            layers.Input(shape=(window_size, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(32),
            layers.Dropout(dropout_rate),
            layers.Dense(horizon),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR building LSTM model: {e}")
        raise


if __name__ == "__main__":
    import numpy as np

    model = build_model()
    model.summary()

    dummy_input = np.random.rand(1, WINDOW_SIZE, N_FEATURES).astype("float32")
    output = model.predict(dummy_input, verbose=0)

    print(f"🧠 [ProximaScale] Input shape:  {dummy_input.shape}")
    print(f"🧠 [ProximaScale] Output shape: {output.shape}")

    assert output.shape == (1, HORIZON), f"Expected output shape (1, {HORIZON}), got {output.shape}"
    print("🧠 [ProximaScale] LSTM architecture self-test PASSED: (1, 10, 3) -> (1, 3)")