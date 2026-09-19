import pandas as pd
from prophet_model import load_prophet_model, get_prophet_fitted
m = load_prophet_model()

# Sample the training history's hourly pattern
hist = m.history
hist['hour'] = hist['ds'].dt.hour
hourly = hist.groupby('hour')['y'].mean()
print('Prophet training data hourly pattern:')
print(hourly.round(1))
print()
print(f'Peak hour: {hourly.idxmax()} ({hourly.max():.1f}%)')
print(f'Trough hour: {hourly.idxmin()} ({hourly.min():.1f}%)')
