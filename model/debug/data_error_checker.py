import pandas as pd
df = pd.read_csv("data/collected/metrics.csv")
print(df.isna().sum())
print(df["cpu_percent"].describe())