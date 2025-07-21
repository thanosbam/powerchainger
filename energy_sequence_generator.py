import numpy as np
import pandas as pd

def create_lstm_sequences(data, lookback=24, target_column='co2_factor'):

    X, y = [], []
    data_values = data.values
    target_index = data.columns.get_loc(target_column)

    for i in range(len(data_values) - lookback):
        X.append(data_values[i:i+lookback])
        y.append(data_values[i+lookback, target_index])

    return np.array(X), np.array(y)

if __name__ == "__main__":
    df_scaled = pd.read_csv("/home3/s6018130/data/scaled_energy_data.csv", index_col=0, parse_dates=True)

    # Sequences
    lookback = 24
    X, y = create_lstm_sequences(df_scaled, lookback, target_column='co2_factor')

    # Save arrays
    np.save("/home3/s6018130/data/X_lstm.npy", X)
    np.save("/home3/s6018130/data/y_lstm.npy", y)

    print(f"Saved: X.shape = {X.shape}, y.shape = {y.shape}")
