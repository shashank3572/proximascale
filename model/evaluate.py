import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from preprocessing import scale_data, create_windows, train_test_split_data
from train import generate_synthetic_data

def evaluate():
    print("Loading data...")
    df = generate_synthetic_data(n=500)

    print("Preprocessing...")
    scaled, scaler = scale_data(df)
    X, y = create_windows(scaled)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("Loading model...")
    model = load_model('saved/proximascale_lstm.keras')

    print("Running predictions...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions and actual values
    def inverse_first_feature(arr):
        dummy = np.zeros((arr.shape[0] * arr.shape[1], scaled.shape[1]))
        dummy[:, 0] = arr.flatten()
        return scaler.inverse_transform(dummy)[:, 0].reshape(arr.shape)

    y_test_actual = inverse_first_feature(y_test)
    y_pred_actual = inverse_first_feature(y_pred_scaled)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), y_pred_actual.flatten()))
    mae  = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())

    print(f"\n--- Evaluation Results ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_actual[:, 0], label='Actual CPU%',    color='blue')
    plt.plot(y_pred_actual[:, 0], label='Predicted CPU%', color='red', linestyle='--')
    plt.title('ProximaScale — LSTM Prediction vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU %')
    plt.legend()
    plt.tight_layout()
    plt.savefig('saved/evaluation_chart.png')
    print("Chart saved to saved/evaluation_chart.png")
    plt.show()

if __name__ == "__main__":
    evaluate()