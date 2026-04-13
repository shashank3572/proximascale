import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from preprocessing import FEATURES, INPUT_STEPS, scale_data

# Load the trained model once when this file is imported
MODEL_PATH = 'saved/proximascale_lstm.keras'
model = load_model(MODEL_PATH)

def detect_anomaly(values, threshold=2.0):
    """
    Detects if the latest CPU value is anomalous using Z-score.
    Returns True if anomaly detected, False otherwise.
    """
    mean = np.mean(values)
    std  = np.std(values)
    if std == 0:
        return False
    z_score = abs(values[-1] - mean) / std
    return bool(z_score > threshold)

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
    # Convert input records to DataFrame
    df = pd.DataFrame(records)

    # Scale the data
    scaled, scaler = scale_data(df)

    # Reshape for LSTM input: (1, INPUT_STEPS, features)
    X = scaled.reshape(1, INPUT_STEPS, len(FEATURES))

    # Run prediction
    prediction_scaled = model.predict(X, verbose=0)

    # Inverse transform to get real CPU% values
    dummy = np.zeros((3, len(FEATURES)))
    dummy[:, 0] = prediction_scaled[0]
    predicted_cpu = scaler.inverse_transform(dummy)[:, 0].tolist()

    # Detect anomaly on the last 10 CPU values
    cpu_values = df['cpu_percent'].values
    anomaly    = detect_anomaly(cpu_values)

    return {
        "predicted_cpu" : predicted_cpu,
        "anomaly"       : anomaly
    }

if __name__ == "__main__":
    # Test with fake data
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