import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import pandas as pd
import joblib
from tf_keras.models import load_model

# Fix bare imports
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import FEATURES, INPUT_STEPS
from anomaly import detect_anomaly

# Load model and scaler ONCE at import time (not per-call)
_DIR        = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_DIR, 'saved', 'proximascale_lstm.h5')
SCALER_PATH = os.path.join(_DIR, 'saved', 'scaler.pkl')

model  = load_model(MODEL_PATH)
# FIX B4: Load the TRAINING scaler, not a new one
scaler = joblib.load(SCALER_PATH)

def predict(records):
    """
    Main prediction function called by teammates.

    Input:
        records — list of 10 dicts, each with keys:
                  timestamp, cpu_percent, memory_percent, request_rate

    Output:
        {
            "predicted_cpu": [val1, val2, val3],  # next 3 minutes
            "anomaly": True/False
        }
    """
    df = pd.DataFrame(records)

    # FIX B4: Use the training scaler — do NOT re-fit
    scaled = scaler.transform(df[FEATURES])

    # Reshape for LSTM: (1, INPUT_STEPS, features)
    X = scaled.reshape(1, INPUT_STEPS, len(FEATURES))

    # Run prediction
    prediction_scaled = model.predict(X, verbose=0)

    # Inverse transform — only the cpu_percent column (index 0)
    dummy = np.zeros((3, len(FEATURES)))
    dummy[:, 0] = prediction_scaled[0]
    predicted_cpu = scaler.inverse_transform(dummy)[:, 0].tolist()

    # Anomaly detection — delegated to anomaly.py
    cpu_values = df['cpu_percent'].values
    anomaly    = detect_anomaly(cpu_values)

    return {
        "predicted_cpu": predicted_cpu,
        "anomaly"      : anomaly
    }

if __name__ == "__main__":
    test_records = [
        {"timestamp": f"2024-01-15T14:{i:02d}:00",
         "cpu_percent": 50 + i,
         "memory_percent": 40 + i,
         "request_rate": 100 + i * 2}
        for i in range(10)
    ]
    result = predict(test_records)
    print("Predicted CPU for next 3 minutes:", result['predicted_cpu'])
    print("Anomaly detected               :", result['anomaly'])