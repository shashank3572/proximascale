import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

INPUT = "data/processed/training_data.csv"
OUTPUT = "ml/proxima_model.pkl"

df = pd.read_csv(INPUT)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second

features = ["cpu_percent", "memory_percent", "request_rate", "hour", "minute", "second"]

df["target_cpu"] = df["cpu_percent"].shift(-1)
df = df.dropna(subset=features + ["target_cpu"])

X = df[features]
y = df["target_cpu"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

os.makedirs("ml", exist_ok=True)
joblib.dump(model, OUTPUT)

print("Model trained successfully.")
print("Training rows:", len(df))
print("Model saved:", OUTPUT)
