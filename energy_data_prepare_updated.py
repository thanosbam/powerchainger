
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# ----------------------------------------------------------------------
RAW_FILE     = Path("/home3/s6018130/data/merged_2021_2024_wind_solar_mix_energy_data.csv")
SCALED_FILE  = Path("/home3/s6018130/data/scaled_energy_data_updated.csv")
SCALER_FILE  = Path("/home3/s6018130/data/scaler_updated.pkl")
# ----------------------------------------------------------------------

print("Loading merged dataset …")
df = pd.read_csv(RAW_FILE)

df["datetime"] = pd.to_datetime(df["validfrom (UTC)"], utc=True)
df = df.sort_values("datetime")

df_model = df[[
    "datetime",
    "emissionfactor (kg CO2/kWh)_mix",
    "volume (kWh)_solar",
    "volume (kWh)_wind"
]].copy()

df_model.columns = [
    "datetime",
    "co2_factor",
    "solar_volume",
    "wind_volume"
]

for col in ["co2_factor", "solar_volume", "wind_volume"]:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna()

df_model = df_model.set_index("datetime")

# Min‑Max scaler
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_model)
df_scaled = pd.DataFrame(
    scaled_values,
    index=df_model.index,
    columns=df_model.columns
)


print(f"Saving scaled data  {SCALED_FILE}")
df_scaled.to_csv(SCALED_FILE)

print(f"Saving fitted scaler  {SCALER_FILE}")
dump(scaler, SCALER_FILE)

print("Pre‑processing complete.")
