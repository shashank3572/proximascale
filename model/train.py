import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
import sys
import joblib

# Fix bare imports — works whether called from model/ or project root
sys.path.insert(0, os.path.dirname(__file__))

from lstm_model import build_model
from preprocessing import scale_data, create_windows, train_test_split_data

def generate_synthetic_data(n=500):
    """
    Generates synthetic data that looks like real server metrics.
    We use this to prototype before Person A gives us real data.
    """
    np.random.seed(42)
    t = np.linspace(0, 50, n)

    cpu_percent    = 50 + 40 * np.sin(t) + np.random.normal(0, 8, n)   # NEW: max ~100
    memory_percent = 50 + 10 * np.sin(t + 1) + np.random.normal(0, 3, n)
    request_rate   = 400 + 350 * np.sin(t + 2) + np.random.normal(0, 60, n) # NEW: max ~900

    cpu_percent    = np.clip(cpu_percent, 0, 100)
    memory_percent = np.clip(memory_percent, 0, 100)
    request_rate   = np.clip(request_rate, 0, 1000)

    timestamps = pd.date_range(start='2024-01-15', periods=n, freq='1min')

    df = pd.DataFrame({
        'timestamp'     : timestamps,
        'cpu_percent'   : cpu_percent,
        'memory_percent': memory_percent,
        'request_rate'  : request_rate
    })
    return df

def train(data_path=None):
    if data_path:
        from preprocessing import load_data
        print(f"Loading real data from {data_path}...")
        df = load_data(data_path)
    else:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n=500)

    print("Scaling data...")
    scaled, scaler = scale_data(df)

    print("Creating windows...")
    X, y = create_windows(scaled)
    print(f"Dataset size — X: {X.shape}, y: {y.shape}")

    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # FIX: Use absolute path based on this file's location,
    # so train.py works whether called from project root or model/ dir.
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
    os.makedirs(SAVE_DIR, exist_ok=True)

    scaler_path = os.path.join(SAVE_DIR, 'scaler.pkl')
    model_path  = os.path.join(SAVE_DIR, 'proximascale_lstm.h5')

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='Path to real CSV from Person A (optional)')
    args = parser.parse_args()
    train(data_path=args.data)