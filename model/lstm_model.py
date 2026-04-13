from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_steps=10, features=3, output_steps=3):
    """
    Builds the LSTM model for ProximaScale.
    
    input_steps  : number of past timesteps to look at (10 minutes)
    features     : number of input features (cpu%, memory%, request_rate)
    output_steps : number of future timesteps to predict (3 minutes ahead)
    """
    model = Sequential([
        LSTM(64, input_shape=(input_steps, features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(output_steps)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()