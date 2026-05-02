import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe on all machines
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tf_keras.models import load_model

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import scale_data, create_windows, train_test_split_data, load_data
from train import generate_synthetic_data

def evaluate(data_path=None):
    """
    Evaluates the LSTM model.
    If data_path is provided, loads real CSV from Person A.
    Otherwise falls back to synthetic data.
    """
    if data_path:
        print(f"Loading real data from {data_path}...")
        df = load_data(data_path)
    else:
        print("No CSV provided — using synthetic data...")
        df = generate_synthetic_data(n=500)

    print("Preprocessing...")
    scaled, scaler = scale_data(df)
    X, y = create_windows(scaled)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    _DIR = os.path.dirname(__file__)
    model = load_model(os.path.join(_DIR, 'saved', 'proximascale_lstm.keras'))

    print("Running predictions...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    def inverse_first_feature(arr):
        dummy = np.zeros((arr.shape[0] * arr.shape[1], scaled.shape[1]))
        dummy[:, 0] = arr.flatten()
        return scaler.inverse_transform(dummy)[:, 0].reshape(arr.shape)

    y_test_actual = inverse_first_feature(y_test)
    y_pred_actual = inverse_first_feature(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), y_pred_actual.flatten()))
    mae  = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())

    print(f"\n--- Evaluation Results ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test_actual[:, 0], label='Actual CPU%',    color='blue')
    plt.plot(y_pred_actual[:, 0], label='Predicted CPU%', color='red', linestyle='--')
    plt.title('ProximaScale — LSTM Prediction vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU %')
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join(_DIR, 'saved', 'evaluation_chart.png')
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")

    return {"rmse": rmse, "mae": mae}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='Path to real CSV from Person A (optional)')
    args = parser.parse_args()
    evaluate(data_path=args.data)