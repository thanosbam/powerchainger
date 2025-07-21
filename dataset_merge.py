import pandas as pd

files = [
    #"total electricity production/electriciteitsmix-2021-uur-data.csv",
    #"total electricity production/electriciteitsmix-2022-uur-data.csv",
    #"total electricity production/electriciteitsmix-2023-uur-data.csv",
    #"total electricity production/electriciteitsmix-2024-uur-data.csv"
    #"solar energy/zon-2017-uur-data.csv",
    #"solar energy/zon-2018-uur-data.csv",
    #"solar energy/zon-2019-uur-data.csv",
    #"solar energy/zon-2020-uur-data.csv",
    #"solar energy/zon-2021-uur-data.csv",
    #"solar energy/zon-2022-uur-data.csv",
    #"solar energy/zon-2023-uur-data.csv",
    #"solar energy/zon-2024-uur-data.csv"
    "onshore wind energy/wind-2021-uur-data.csv",
    "onshore wind energy/wind-2022-uur-data.csv",
    "onshore wind energy/wind-2023-uur-data.csv",
    "onshore wind energy/wind-2024-uur-data.csv"
]

combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Convert datetime
combined_df["validfrom (UTC)"] = pd.to_datetime(combined_df["validfrom (UTC)"], errors="coerce")

combined_df.shape, sorted(combined_df["validfrom (UTC)"].dt.year.dropna().unique())

output_path = "onshore wind energy/wind-2021-2024-uur-data.csv"
combined_df.to_csv(output_path, index=False)

output_path

