import pandas as pd
import joblib

MODEL = "ml/proxima_model.pkl"
INPUT = "data/processed/training_data.csv"

SCALE_UP_THRESHOLD = 70
SCALE_DOWN_THRESHOLD = 30

model = joblib.load(MODEL)

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

latest = df.iloc[-1]

features = pd.DataFrame([{
    "cpu_percent": float(latest["cpu_percent"]),
    "memory_percent": float(latest["memory_percent"]),
    "request_rate": float(latest["request_rate"]),
    "hour": latest["timestamp"].hour,
    "minute": latest["timestamp"].minute,
    "second": latest["timestamp"].second
}])

prediction = model.predict(features)[0]

print("Current CPU:", round(float(latest["cpu_percent"]), 2), "%")
print("Current Memory:", round(float(latest["memory_percent"]), 2), "%")
print("Request Rate:", round(float(latest["request_rate"]), 2))
print("Predicted Next CPU:", round(float(prediction), 2), "%")

if prediction >= SCALE_UP_THRESHOLD:
    decision = "SCALE UP"
elif prediction <= SCALE_DOWN_THRESHOLD:
    decision = "SCALE DOWN"
else:
    decision = "NO CHANGE"

print("Scaling Decision:", decision)
