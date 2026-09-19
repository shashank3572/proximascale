# Save as model/debug_residuals.py
import numpy as np
from pathlib import Path
from preprocessing import load_csv, chronological_split, WINDOW_SIZE
from prophet_model import train_prophet, get_prophet_fitted

df = load_csv(Path(__file__).parent.parent / "data" / "collected" / "metrics.csv")
train_df, test_df = chronological_split(df)

prophet = train_prophet(train_df)

# Get Prophet's fitted values on the TRAINING set
fitted = np.array(get_prophet_fitted(prophet, train_df["timestamp"]))
actual = train_df["cpu_percent"].values

residuals = actual - fitted
print(f"Prophet fitted range: {fitted.min():.1f} to {fitted.max():.1f}")
print(f"Actual CPU range:    {actual.min():.1f} to {actual.max():.1f}")
print(f"Residual mean:       {residuals.mean():.2f}")
print(f"Residual std:        {residuals.std():.2f}")
print(f"Residual min/max:    {residuals.min():.1f} / {residuals.max():.1f}")

# If residual std is close to actual CPU std, Prophet isn't capturing anything
ratio = residuals.std() / actual.std()
print(f"\nResidual/Actual std ratio: {ratio:.2f}")
print("  <0.3 = Prophet is useful")
print("  0.3-0.7 = Prophet marginally useful")
print("  >0.7 = Prophet is basically useless (this is you)")