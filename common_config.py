DATA_DIR   = "/home3/s6018130/data"
MODEL_DIR  = "/home3/s6018130/models"

#SCALED_TWO   = f"{DATA_DIR}/scaled_energy_data.csv"

SCALED_THREE = f"{DATA_DIR}/scaled_energy_data_updated.csv"

USE_THREE_FEATURES = True

SCALED_FILE = SCALED_THREE if USE_THREE_FEATURES else SCALED_TWO
MODEL_OUT   = f"{MODEL_DIR}/lstm_168h_forecast.pt"
LOSS_PLOT   = f"{MODEL_DIR}/loss_curve_168h.png"

CUTOFF_DATE = "2024-01-01 00:00:00"
SEQ_LEN     = 168      # look‑back window
HORIZON     = 168
BATCH       = 64
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 8
