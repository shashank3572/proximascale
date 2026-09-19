"""Trace where NaN enters the pipeline."""
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import load_csv, chronological_split, WINDOW_SIZE
from predict import predict_load

THIS_DIR = Path(__file__).parent
CSV_PATH = THIS_DIR.parent / "data" / "collected" / "metrics.csv"

df = load_csv(CSV_PATH)
train_df, test_df = chronological_split(df)

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"Test NaN count: {test_df.isna().sum().sum()}")
print()

# Take one window and trace
window = test_df.iloc[100:100+WINDOW_SIZE][
    ["timestamp", "cpu_percent", "memory_percent", "request_rate"]
].to_dict("records")

print(f"Window length: {len(window)}")
print(f"Window last CPU: {window[-1]['cpu_percent']}")
print()

result = predict_load(window)
print(f"predict_load output: {result}")
print(f"  predicted_load: {result[0]}  (isnan? {np.isnan(result[0])})")
print(f"  upper_bound:    {result[1]}  (isnan? {np.isnan(result[1])})")
print(f"  anomaly_flag:   {result[2]}")
print()

# Also test the actuals from eval loop
n_windows = len(test_df) - WINDOW_SIZE - 3 + 1
actuals = []
for i in range(0, n_windows, max(1, n_windows // 20)):
    t = i + WINDOW_SIZE - 1
    target_idx = t + 3
    val = test_df.loc[target_idx, "cpu_percent"]
    actuals.append(val)

actuals = np.array(actuals)
print(f"Sampled actuals: {len(actuals)} values")
print(f"NaN count: {np.isnan(actuals).sum()}")
print(f"Min/Max: {np.nanmin(actuals):.2f} / {np.nanmax(actuals):.2f}")