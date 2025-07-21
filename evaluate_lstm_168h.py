import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from energy_sequence_generator_multistep import create_multistep_sequences
from lstm_model_168h_train import LSTM168
import common_config as cfg
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_utc(idx):
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

def rolling_weekly_eval():
    # 1. Load data and normalise timezone
    df = pd.read_csv(cfg.SCALED_FILE, index_col="datetime", parse_dates=True)
    df.index = ensure_utc(df.index)


    model = LSTM168(input_size=df.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_OUT, map_location=DEVICE))
    model.eval()

    preds, trues = [], []

    weekly_starts = pd.date_range(cfg.CUTOFF_DATE, periods=365, freq="24h", tz="UTC")

    for start in weekly_starts:
        hist = df.loc[start - pd.Timedelta(hours=cfg.SEQ_LEN) : start - pd.Timedelta(hours=1)]
        future = df.loc[start : start + pd.Timedelta(hours=cfg.HORIZON - 1)]

        if len(hist) == cfg.SEQ_LEN and len(future) == cfg.HORIZON:
            x = torch.tensor(hist.values[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                y_pred = model(x).cpu().numpy().flatten()
            preds.append(y_pred)
            trues.append(future["co2_factor"].values)

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    horizon_mae = np.mean(np.abs(preds - trues), axis=0)

    # 3. Plot horizon error curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, cfg.HORIZON + 1), horizon_mae)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("MAE (scaled)")
    plt.title("Horizon MAE curve")
    plt.tight_layout()
    out_plot = Path(cfg.MODEL_DIR) / "horizon_mae_curve.png"
    plt.savefig(out_plot, dpi=300)
    print("Saved horizon-error plot ", out_plot)
    print(f"Overall MAE (avg over 168 steps): {horizon_mae.mean():.4f}")

if __name__ == "__main__":
    rolling_weekly_eval()
