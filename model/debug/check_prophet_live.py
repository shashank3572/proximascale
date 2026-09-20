import pickle
from pathlib import Path

MODEL_PATH = Path("model/saved/proximascale_prophet.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Call the forecast twice in a row, exactly the way live code would --
# no new data given to it either time.
future1 = model.make_future_dataframe(periods=3, freq="30s")
forecast1 = model.predict(future1).tail(3)["yhat"].values

future2 = model.make_future_dataframe(periods=3, freq="30s")
forecast2 = model.predict(future2).tail(3)["yhat"].values

print("Call 1:", forecast1)
print("Call 2:", forecast2)
print("Identical:", (forecast1 == forecast2).all())