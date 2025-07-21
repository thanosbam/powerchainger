import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from energy_sequence_generator import create_lstm_sequences



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# Training
if __name__ == "__main__":

    DATA_PATH       = "/home3/s6018130/data/scaled_energy_data_updated.csv"
    LOSS_PLOT_PATH  = "/home3/s6018130/models/loss_curve_updated.png"


    MODEL_SAVE_PATH = "/home3/s6018130/models/lstm_energy_forecast_updated.pt"
    CUTOFF_DATE     = "2024-01-01 00:00:00"

    SEQUENCE_LENGTH = 24
    BATCH_SIZE      = 64
    EPOCHS          = 100
    LEARNING_RATE   = 0.001
    PATIENCE        = 10
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load
    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
    train_df = df[df.index <= CUTOFF_DATE]
    val_df   = df[df.index >  CUTOFF_DATE]

    # Sequences
    X_train, y_train = create_lstm_sequences(train_df, SEQUENCE_LENGTH)
    X_val,   y_val   = create_lstm_sequences(val_df,   SEQUENCE_LENGTH)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE
    )

    # Model
    model     = LSTMModel(input_size=X_train.shape[2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03}/{EPOCHS} - Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  improved – model saved (epoch {epoch+1})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  early stopping at epoch {epoch+1}")
                break


    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train", lw=2)
    plt.plot(val_losses, label="Validation", lw=2)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    print(f"Loss curve saved → {LOSS_PLOT_PATH}")
