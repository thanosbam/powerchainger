import numpy as np

def create_multistep_sequences(df, seq_len=168, horizon=168):
    vals = df.values
    X, y = [], []
    for i in range(len(df) - seq_len - horizon + 1):
        X.append(vals[i : i+seq_len])
        y.append(vals[i+seq_len : i+seq_len+horizon, 0])  # target = col 0
    return np.stack(X), np.stack(y)
