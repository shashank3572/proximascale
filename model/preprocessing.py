import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# These are the 3 features our model uses
FEATURES = ['cpu_percent', 'memory_percent', 'request_rate']

# How many past timesteps the model looks at
INPUT_STEPS = 10

# How many future timesteps the model predicts
OUTPUT_STEPS = 3

def load_data(filepath):
    """
    Loads CSV data from filepath.
    CSV must have columns: timestamp, cpu_percent, memory_percent, request_rate
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def scale_data(df):
    """
    Scales all features to range 0-1.
    Returns scaled data and the scaler object.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    return scaled, scaler

def create_windows(scaled_data):
    """
    Creates sliding windows for LSTM input.
    X shape: (samples, INPUT_STEPS, features)
    y shape: (samples, OUTPUT_STEPS)
    """
    X, y = [], []
    for i in range(len(scaled_data) - INPUT_STEPS - OUTPUT_STEPS + 1):
        X.append(scaled_data[i : i + INPUT_STEPS])
        y.append(scaled_data[i + INPUT_STEPS : i + INPUT_STEPS + OUTPUT_STEPS, 0])
    return np.array(X), np.array(y)

def train_test_split_data(X, y, split=0.8):
    """
    Splits data into 80% train and 20% test.
    """
    split_idx = int(len(X) * split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("INPUT_STEPS  :", INPUT_STEPS)
    print("OUTPUT_STEPS :", OUTPUT_STEPS)
    print("FEATURES     :", FEATURES)
    print("Preprocessing module loaded successfully!")