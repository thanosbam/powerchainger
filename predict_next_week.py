import torch
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
from lstm_model_168h_train import LSTM168
import common_config as cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_last_week():
    df = pd.read_csv(cfg.SCALED_FILE, index_col="datetime", parse_dates=True)
    last_hist = df.iloc[-cfg.SEQ_LEN:]
    x = torch.tensor(last_hist.values[np.newaxis, ...],
                     dtype=torch.float32).to(DEVICE)

    model = LSTM168(input_size=df.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_OUT, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        y_scaled = model(x).cpu().numpy().flatten()

    scaler = load(Path(cfg.DATA_DIR) / "scaler_updated.pkl")

    if scaler.n_features_in_ == 1:
        y_real = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    else:
        dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
        target_idx = 0
        dummy[:, target_idx] = y_scaled
        y_real = scaler.inverse_transform(dummy)[:, target_idx]

    start_ts = last_hist.index[-1] + pd.Timedelta(hours=1)
    pred_index = pd.date_range(start_ts, periods=cfg.HORIZON, freq="1H")

    pred_series = pd.Series(
        y_real,
        index=pred_index,
        name="co2_factor_pred_kg_per_kWh"
    )

    out_path = Path(cfg.DATA_DIR) / "next_week_pred_real.csv"
    pred_series.to_csv(out_path)
    print("Saved 168-hour forecast ", out_path)
    print(pred_series.head())

if __name__ == "__main__":
    predict_last_week()
