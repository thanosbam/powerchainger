
import pandas as pd
from pathlib import Path


# ----------------------------------------------------------------------
PATH_MIX   = Path("total electricity production/electriciteitsmix-2021-2024.csv")
PATH_SOLAR = Path("solar energy/zon-2021-2024-uur-data.csv")
PATH_WIND  = Path("onshore wind energy/wind-2021-2024-uur-data.csv")

OUT_PATH   = Path("merged_2021_2024_wind_solar_mix_energy_data.csv")
# ----------------------------------------------------------------------

def load_and_prepare(csv_path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["validfrom (UTC)"] = pd.to_datetime(df["validfrom (UTC)"], utc=True)

    validto = df.pop("validto (UTC)") if "validto (UTC)" in df.columns else None

    df = df.rename(columns={
        col: f"{col}_{suffix}" for col in df.columns if col != "validfrom (UTC)"
    })

    if validto is not None:
        df[f"validto (UTC)_{suffix}"] = validto

    return df

def main() -> None:
    print("Loading and preparing individual datasets …")
    df_mix   = load_and_prepare(PATH_MIX,   "mix")
    df_solar = load_and_prepare(PATH_SOLAR, "solar")
    df_wind  = load_and_prepare(PATH_WIND,  "wind")

    print("Merging on validfrom (UTC) …")
    merged = (
        df_mix
        .merge(df_solar, on="validfrom (UTC)", how="inner")
        .merge(df_wind,  on="validfrom (UTC)", how="inner")
        .sort_values("validfrom (UTC)")
    )

    print(f"Writing merged file → {OUT_PATH.resolve()}")
    merged.to_csv(OUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
