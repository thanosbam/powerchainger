from pathlib import Path
DATA_PATH    = Path("/home3/s6018130/data/scaled_energy_data_updated.csv")
RAW_PATH     = Path("/home3/s6018130/data/merged_2021_2024_wind_solar_mix_energy_data.csv")
MODEL_PATH   = Path("/home3/s6018130/models/lstm_energy_forecast_updated.pt")
SCALER_PATH  = Path("/home3/s6018130/data/scaler_updated.pkl")
OUT_CSV_PATH = Path("/home3/s6018130/data/val_co2_actual_vs_pred_updated.csv")
PLOT_PATH   = Path("/home3/s6018130/data/val_co2_actual_vs_pred_UPDATED.png")

CUTOFF_DATE    = "2024-01-01 00:00:00"
TARGET_COL_RAW = "emissionfactor (kg CO2/kWh)_mix"
TARGET_COL_SCALED = "co2_factor"

SEQUENCE_LEN = 24
BATCH_SIZE   = 64


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
import matplotlib.pyplot as plt

from energy_sequence_generator import create_lstm_sequences
from energy_lstm_model import LSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading scaled dataset …")
df_scaled = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
df_scaled.index = (
    df_scaled.index.tz_localize("UTC") if df_scaled.index.tz is None
    else df_scaled.index.tz_convert("UTC")
)

val_df = df_scaled[df_scaled.index > CUTOFF_DATE]
print(f"Validation rows: {len(val_df):,}")


X_val, y_val = create_lstm_sequences(val_df, SEQUENCE_LEN)
val_dl = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                  torch.tensor(y_val, dtype=torch.float32)),
    batch_size=BATCH_SIZE
)


model = LSTMModel(input_size=X_val.shape[2]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


preds, trgs = [], []
with torch.no_grad():
    for xb, yb in val_dl:
        preds.append(model(xb.to(DEVICE)).squeeze(1).cpu().numpy())
        trgs.append(yb.numpy())

y_pred_scaled = np.concatenate(preds)
y_true_scaled = np.concatenate(trgs)


if SCALER_PATH.exists():
    scaler = load(SCALER_PATH)
else:
    raw_tmp = pd.read_csv(RAW_PATH, usecols=["validfrom (UTC)", TARGET_COL_RAW])
    raw_tmp["validfrom (UTC)"] = pd.to_datetime(raw_tmp["validfrom (UTC)"], utc=True)
    train_mask = raw_tmp["validfrom (UTC)"] <= pd.Timestamp(CUTOFF_DATE, tz="UTC")
    scaler = MinMaxScaler().fit(
        raw_tmp.loc[train_mask, TARGET_COL_RAW].values.reshape(-1, 1)
    )
    dump(scaler, SCALER_PATH)

# dummy matrix so scaler.inverse_transform works
dummy = np.zeros((len(y_pred_scaled), df_scaled.shape[1]))
t_idx = df_scaled.columns.get_loc(TARGET_COL_SCALED)
dummy[:, t_idx] = y_pred_scaled
y_pred_real = scaler.inverse_transform(dummy)[:, t_idx]

dummy[:, t_idx] = y_true_scaled
y_true_real = scaler.inverse_transform(dummy)[:, t_idx]

# Align raw and ground truth
raw = pd.read_csv(RAW_PATH, usecols=["validfrom (UTC)", TARGET_COL_RAW])
raw["validfrom (UTC)"] = pd.to_datetime(raw["validfrom (UTC)"], utc=True)
raw = raw.set_index("validfrom (UTC)").sort_index()

pred_index    = val_df.index[SEQUENCE_LEN:]
y_actual_real = raw.reindex(pred_index)[TARGET_COL_RAW].values

# Metrics
scaled_mae  = mean_absolute_error(y_true_scaled, y_pred_scaled)
scaled_rmse = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
scaled_r2   = r2_score(y_true_scaled, y_pred_scaled)

real_mae  = mean_absolute_error(y_actual_real, y_pred_real)
real_rmse = np.sqrt(mean_squared_error(y_actual_real, y_pred_real))
real_r2   = r2_score(y_actual_real, y_pred_real)

print("\n--- Scaled space ---")
print(f"MAE {scaled_mae:.4f}  RMSE {scaled_rmse:.4f}  R² {scaled_r2:.4f}")
print("\n--- Real units (kgCO₂/kWh) ---")
print(f"MAE {real_mae:.4f}  RMSE {real_rmse:.4f}  R² {real_r2:.4f}")


comparison = pd.DataFrame({
    "co2_actual": y_actual_real,
    "co2_pred":   y_pred_real
}, index=pred_index)

comparison.to_csv(OUT_CSV_PATH)
print(f"Saved comparison CSV → {OUT_CSV_PATH}")

# plotting is not very telling
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 7),
    gridspec_kw={"height_ratios": [3, 2]},
)

# time‑series overlay, too crumbed up, might delete
ax1.plot(pred_index, y_actual_real, label="CO₂ actual", lw=1.0)
ax1.plot(pred_index, y_pred_real,   label="CO₂ predicted", lw=1.0, alpha=0.6)
ax1.set_ylabel("kgCO₂/kWh")
ax1.set_title("Validation period – forecast vs actual")
ax1.legend(loc="upper right")

# scatter (actual vs predicted)
ax2.scatter(
    y_actual_real,
    y_pred_real,
    s=6,
    alpha=0.35,
    edgecolors="none"
)

# 45‑degree reference
lims = [0, max(y_actual_real.max(), y_pred_real.max())]
ax2.plot(lims, lims, "k--", lw=1)

ax2.set_xlabel("Actual  (kgCO₂/kWh)")
ax2.set_ylabel("Predicted  (kgCO₂/kWh)")
ax2.set_aspect("equal")  # makes squares look like squares

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()
print(f"Saved plot  {PLOT_PATH}")

print("Evaluation complete.")
