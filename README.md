# powerchainger

CO2‑FACTOR FORECASTING PIPELINE
===============================

This repository contains two end‑to‑end pipelines:

1. 1‑hour‑ahead forecaster (baseline)
2. 168‑hour (7‑day) forecaster

Both pipelines work off the same three‑feature data set:
   * co2_factor (target)
   * solar_volume
   * wind_volume


--------------------------------------------------------------------
1. DATA ASSEMBLY
--------------------------------------------------------------------
1) dataset_merge.py  – merges the four yearly wind, solar, mix files
   (wind‑2021‑uur‑data … wind‑2024‑uur‑data) into
   wind‑2021‑2024‑uur‑data.csv etc.

2) dataset_merge_solar_wind_mix.py
   • merge raw electricity‑mix, solar and wind CSVs on UTC timestamp  
   • add suffixes: _mix, _solar, _wind  
   → merged_2021_2024_wind_solar_mix_energy_data.csv

3) energy_data_prepare_updated.py
   • keep co2_factor, solar_volume, wind volume  
   • drop NaNs, sort by datetime, set index  
   • fit MinMaxScaler(0‑1) on all three columns  
   → scaled_energy_data_updated.csv  
   → scaler_updated.pkl


--------------------------------------------------------------------
2. 1‑HOUR‑AHEAD FORECASTER
--------------------------------------------------------------------

*Sequence builder*        : energy_sequence_generator.py (24‑hour window)
*Model & training script* : energy_lstm_model_updated.py  
*Evaluation script*       : evaluate_lstm_energy_updated.py  

Outputs
-------
lstm_energy_forecast_updated.pt  
loss_curve_updated.png  
val_co2_actual_vs_pred_updated.csv  
val_co2_actual_vs_pred_updated.png  

2024 validation metrics (real kg CO2/kWh)
-----------------------------------------
MAE = 0.0071  
RMSE = 0.0098  
R²  = 0.9913


--------------------------------------------------------------------
3. 168‑HOUR (7‑DAY) FORECASTER
--------------------------------------------------------------------

*Shared config*           : common_config.py  
*Sequence builder*        : energy_sequence_generator_multistep.py  
*Model & training script* : lstm_model_168h_train.py  
*Evaluation script*       : evaluate_lstm_168h.py  
*Operational forecast*    : predict_next_week.py  

Outputs
-------
lstm_168h_forecast.pt  
loss_curve_168h.png  
horizon_mae_curve.png      (MAE vs lead time 1‑168 h)  
next_week_pred_real.csv    (168 future hours, real units)

Typical horizon MAE (scaled space)
----------------------------------
t+1 h  : ~0.016  
t+24 h : ~0.018  
t+168 h: ~0.030  
Overall mean (1‑168) ≈ 0.022

### Scaling & de‑scaling

Both pipelines work in two spaces:

1. **Scaled space** – every feature is transformed to [0‑1] by the  
   Min‑Max scaler saved as `scaler_updated.pkl`.  
   *Training* and *validation* losses are computed here.

2. **Real space** – before reporting metrics or writing CSVs, each
   script calls

       scaler.inverse_transform(...)

   to convert the scaled CO₂ predictions back to real
   **kg CO₂ / kWh**. 
--------------------------------------------------------------------
4. HOW TO RE‑RUN EVERYTHING
--------------------------------------------------------------------

# 1‑hour model
python energy_data_prepare_updated.py
python energy_lstm_model_updated.py
python evaluate_lstm_energy_updated.py

# 7‑day model
python lstm_model_168h_train.py
python evaluate_lstm_168h.py
python predict_next_week.py    # writes next_week_pred_real.csv

All paths can be changed in common_config.py or the CONFIG blocks
inside each script.


--------------------------------------------------------------------
5. NOTES AND NEXT STEPS
--------------------------------------------------------------------

* Adding wind_volume to the original mix+solar features reduced MAE
  by ~3 percent.
* Error doubles by day 7, from what I read it's typical for direct LSTM output.
* Further gains likely require exogenous weather forecasts or a
  seq2seq/transformer decoder, also I was ment to use offshore wind energy that is availiable to ned.nl.

----------------------------------------------------------------------------------------------------------
pip.list
----------------------------------------------------------------------------------------------------------
(script_env) [s6018130@l40sgpu1 ~]$ pip list
Package                  Version
------------------------ -----------
accelerate               1.7.0
certifi                  2025.4.26
charset-normalizer       3.4.2
click                    8.2.1
contourpy                1.3.2
cycler                   0.12.1
filelock                 3.18.0
fonttools                4.58.0
fsspec                   2025.5.0
hf-xet                   1.1.3
huggingface-hub          0.32.4
idna                     3.10
Jinja2                   3.1.6
jiwer                    3.1.0
joblib                   1.5.1
kiwisolver               1.4.8
MarkupSafe               3.0.2
matplotlib               3.10.3
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.6
nvidia-cublas-cu12       12.6.4.1
nvidia-cuda-cupti-cu12   12.6.80
nvidia-cuda-nvrtc-cu12   12.6.77
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12        9.5.1.17
nvidia-cufft-cu12        11.3.0.4
nvidia-cufile-cu12       1.11.1.6
nvidia-curand-cu12       10.3.7.77
nvidia-cusolver-cu12     11.7.1.2
nvidia-cusparse-cu12     12.5.4.2
nvidia-cusparselt-cu12   0.6.3
nvidia-nccl-cu12         2.26.2
nvidia-nvjitlink-cu12    12.6.85
nvidia-nvtx-cu12         12.6.77
packaging                25.0
pandas                   2.3.1
pillow                   11.2.1
pip                      25.1.1
psutil                   7.0.0
pycocotools              2.0.8
pyparsing                3.2.3
python-dateutil          2.9.0.post0
pytz                     2025.2
PyYAML                   6.0.2
RapidFuzz                3.13.0
regex                    2024.11.6
requests                 2.32.3
safetensors              0.5.3
scikit-learn             1.7.1
scipy                    1.15.3
setuptools               63.2.0
six                      1.17.0
sympy                    1.14.0
threadpoolctl            3.6.0
tokenizers               0.21.1
torch                    2.7.0
torchvision              0.22.0
tqdm                     4.67.1
transformers             4.52.4
triton                   3.3.0
typing_extensions        4.13.2
tzdata                   2025.2
urllib3                  2.4.0
wheel                    0.45.1

