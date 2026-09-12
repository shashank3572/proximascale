import pandas as pd

INPUT = "data/processed/metrics_clean.csv"
OUTPUT = "data/processed/training_data.csv"

df = pd.read_csv(INPUT)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

for col in ["cpu_percent", "memory_percent", "request_rate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["cpu_percent", "memory_percent", "request_rate"])

df.to_csv(OUTPUT, index=False)

print("Training data prepared successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Output:", OUTPUT)
