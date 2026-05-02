import numpy as np
import pandas as pd
import os
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

    cpu_percent    = 40 + 20 * np.sin(t) + np.random.normal(0, 5, n)
    memory_percent = 50 + 10 * np.sin(t + 1) + np.random.normal(0, 3, n)
    request_rate   = 100 + 50 * np.sin(t + 2) + np.random.normal(0, 10, n)

    cpu_percent    = np.clip(cpu_percent, 0, 100)
    memory_percent = np.clip(memory_percent, 0, 100)
    request_rate   = np.clip(request_rate, 0, 500)

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

    os.makedirs('saved', exist_ok=True)

    # FIX B3: Save the scaler — this is the critical missing piece
    joblib.dump(scaler, 'saved/scaler.pkl')
    print("Scaler saved to saved/scaler.pkl")

    model.save('saved/proximascale_lstm.keras')
    # FIX B1: print is now INSIDE train(), after model.save()
    print("Model saved to saved/proximascale_lstm.keras")

if __name__ == "__main__":
    train()