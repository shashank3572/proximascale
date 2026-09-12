import pandas as pd
import joblib

MODEL = "ml/proxima_model.pkl"
INPUT = "data/processed/training_data.csv"

model = joblib.load(MODEL)

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

latest = df.iloc[-1]

features = [[
    float(latest["cpu_percent"]),
    float(latest["memory_percent"]),
    float(latest["request_rate"]),
    latest["timestamp"].hour,
    latest["timestamp"].minute,
    latest["timestamp"].second
]]

prediction = model.predict(features)[0]

print("Latest CPU:", float(latest["cpu_percent"]))
print("Latest Memory:", float(latest["memory_percent"]))
print("Latest Request Rate:", float(latest["request_rate"]))
print("Predicted Next CPU:", round(prediction, 2), "%")
