"""
ProximaScale - Phase 1: Preprocessing
---------------------------------------
Turns raw data/collected/metrics.csv into normalized sliding windows for the LSTM:
    X: (batch, 10, 3)  -> 10 timesteps of [cpu_percent, memory_percent, request_rate]
    y: (batch, 3)      -> next 3 cpu_percent values (t+1, t+2, t+3)

The scaler is fit ONLY on the training split. The split is chronological
(earliest rows = train, latest rows = test) -- never shuffle time series data,
or you leak future information into training.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

THIS_DIR = Path(__file__).parent
SAVED_DIR = THIS_DIR / "saved"
SCALER_PATH = SAVED_DIR / "scaler.pkl"

FEATURE_COLS = ["cpu_percent", "memory_percent", "request_rate"]
TARGET_COL = "cpu_percent"
TARGET_COL_IDX = FEATURE_COLS.index(TARGET_COL)  # 0

WINDOW_SIZE = 10
HORIZON = 3


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def chronological_split(df, train_frac=0.8):
    """Time-ordered split -- NEVER shuffle. Train = earliest rows, test = latest."""
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def fit_scaler(train_df, feature_cols=FEATURE_COLS):
    """Fit MinMaxScaler on the TRAINING split only. Never fit on test data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols].values)
    return scaler


def save_scaler(scaler, path=SCALER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"🧠 [ProximaScale] Scaler saved -> {path}")


def load_scaler(path=SCALER_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def scale_features(df, scaler, feature_cols=FEATURE_COLS):
    return scaler.transform(df[feature_cols].values)


def inverse_transform_cpu(scaler, cpu_values):
    """Turns scaled cpu_percent predictions back into real CPU %. Reused by
    uncertainty.py in Phase 5 for the MC Dropout outputs."""
    cpu_values = np.asarray(cpu_values, dtype=float).reshape(-1)
    padded = np.zeros((len(cpu_values), len(FEATURE_COLS)))
    padded[:, TARGET_COL_IDX] = cpu_values
    inverted = scaler.inverse_transform(padded)
    return inverted[:, TARGET_COL_IDX]


def create_sliding_windows(scaled_data, window_size=WINDOW_SIZE, horizon=HORIZON,
                            target_col_idx=TARGET_COL_IDX):
    n_rows, n_features = scaled_data.shape
    X, y = [], []
    last_start = n_rows - window_size - horizon
    for i in range(last_start + 1):
        X.append(scaled_data[i: i + window_size])
        y.append(scaled_data[i + window_size: i + window_size + horizon, target_col_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def prepare_training_data(csv_path, train_frac=0.8, window_size=WINDOW_SIZE, horizon=HORIZON):
    try:
        df = load_csv(csv_path)
        train_df, test_df = chronological_split(df, train_frac)

        scaler = fit_scaler(train_df)
        save_scaler(scaler)

        train_scaled = scale_features(train_df, scaler)
        test_scaled = scale_features(test_df, scaler)

        X_train, y_train = create_sliding_windows(train_scaled, window_size, horizon)
        X_test, y_test = create_sliding_windows(test_scaled, window_size, horizon)

        print(f"🧠 [ProximaScale] Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print(f"🧠 [ProximaScale] X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"🧠 [ProximaScale] X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        print(f"🧠 [ProximaScale] ERROR in prepare_training_data: {e}")
        raise


if __name__ == "__main__":
    csv_path = THIS_DIR.parent / "data" / "collected" / "metrics.csv"
    X_train, y_train, X_test, y_test, scaler = prepare_training_data(csv_path)

    assert X_train.shape[1:] == (WINDOW_SIZE, len(FEATURE_COLS)), "X window shape mismatch"
    assert y_train.shape[1] == HORIZON, "y horizon shape mismatch"
    print(f"🧠 [ProximaScale] Shape check passed: X is (batch, {WINDOW_SIZE}, {len(FEATURE_COLS)}), "
          f"y is (batch, {HORIZON})")

    raw_df = load_csv(csv_path)
    original_cpu = raw_df[TARGET_COL].iloc[0]
    scaled_row = scaler.transform(raw_df[FEATURE_COLS].iloc[[0]].values)
    recovered_cpu = inverse_transform_cpu(scaler, [scaled_row[0, TARGET_COL_IDX]])[0]
    print(f"🧠 [ProximaScale] Round-trip check: original={original_cpu:.2f}, recovered={recovered_cpu:.2f}")
    assert abs(original_cpu - recovered_cpu) < 0.01, "Inverse transform round-trip failed"

    print("🧠 [ProximaScale] Preprocessing self-test PASSED.")