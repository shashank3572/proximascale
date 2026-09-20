"""
Diagnostic: is the LSTM over-smoothing on the 20K dataset?
Run: python model/lstm_test.py
"""
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

from preprocessing import load_csv, chronological_split, WINDOW_SIZE, HORIZON
from predict import predict_load

THIS_DIR = Path(__file__).parent
MODEL_PATH = THIS_DIR / "saved" / "proximascale_lstm.h5"
CSV_PATH = THIS_DIR.parent / "data" / "collected" / "metrics.csv"

model = load_model(MODEL_PATH, compile=False)
df = load_csv(CSV_PATH)
train_df, test_df = chronological_split(df)

test_cpu = test_df["cpu_percent"].values

# Find the first spike in test data
above_threshold = np.where(test_cpu > 75)[0]
if len(above_threshold) == 0:
    print("No spikes >75% in test data. Try lowering threshold or checking data.")
    spike_idx = len(test_cpu) // 2
else:
    spike_idx = int(above_threshold[0])
    print(f"First spike at test index {spike_idx}")
    print(f"Actual CPU at spike: {test_cpu[spike_idx]:.1f}%")

print()
print(f"CPU values 5 steps before spike: {[round(v, 1) for v in test_cpu[max(0, spike_idx-5):spike_idx]]}")
print()

# Build the window leading UP TO the spike
window_start = max(0, spike_idx - WINDOW_SIZE)
window = test_df.iloc[window_start:window_start + WINDOW_SIZE][
    ["timestamp", "cpu_percent", "memory_percent", "request_rate"]
].to_dict("records")

if len(window) < WINDOW_SIZE:
    print(f"Not enough rows before spike (only {len(window)}). Skipping.")
else:
    last_window_cpu = window[-1]["cpu_percent"]
    print(f"Window last CPU (t-1): {last_window_cpu:.1f}%")

    mean, upper, anom = predict_load(window)
    print(f"LSTM predicted (mean):   {mean:.2f}%")
    print(f"LSTM predicted (upper):  {upper:.2f}%")
    print(f"Anomaly flag:            {anom}")

    # What actually happened at t+3?
    future_idx = spike_idx + HORIZON - 1
    if future_idx < len(test_cpu):
        actual_future = test_cpu[future_idx]
        print(f"Actual CPU at t+{HORIZON}:   {actual_future:.2f}%")
        print()
        print(f"Error (mean vs actual):  {abs(mean - actual_future):.2f}")
        print(f"Error (upper vs actual): {abs(upper - actual_future):.2f}")
        print()
        if abs(mean - last_window_cpu) < 3.0:
            print(">>> DIAGNOSIS: LSTM is OVER-SMOOTHING (predicting near last value)")
        elif abs(mean - actual_future) < 5.0:
            print(">>> DIAGNOSIS: LSTM is predicting well at this spike")
        else:
            print(">>> DIAGNOSIS: LSTM is under-predicting (spike missed)")