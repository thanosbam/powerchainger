import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from energy_sequence_generator_multistep import create_multistep_sequences
import common_config as cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM168(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, horizon=cfg.HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True,
                            dropout=0.1 if layers>1 else 0.0)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def main():
    print("Reading scaled data:", cfg.SCALED_FILE)
    df = pd.read_csv(cfg.SCALED_FILE, index_col="datetime", parse_dates=True)

    train_df = df[df.index <= cfg.CUTOFF_DATE]
    val_df   = df[df.index >  cfg.CUTOFF_DATE]

    Xtr, ytr = create_multistep_sequences(train_df, cfg.SEQ_LEN, cfg.HORIZON)
    Xval, yval = create_multistep_sequences(val_df, cfg.SEQ_LEN, cfg.HORIZON)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.float32)),
        batch_size=cfg.BATCH, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(Xval, dtype=torch.float32),
                      torch.tensor(yval, dtype=torch.float32)),
        batch_size=cfg.BATCH
    )

    model = LSTM168(input_size=Xtr.shape[2]).to(DEVICE)
    crit  = nn.MSELoss(reduction="none")
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best, wait = float("inf"), 0
    tr_losses, val_losses = [], []

    for epoch in range(1, cfg.EPOCHS+1):
        model.train(); run = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb).mean()   # average over horizon & batch
            loss.backward(); opt.step(); run += loss.item()
        tr_loss = run/len(train_dl)

        model.eval(); vrun = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vrun += crit(model(xb), yb).mean().item()
        val_loss = vrun/len(val_dl)

        tr_losses.append(tr_loss); val_losses.append(val_loss)
        print(f"Ep {epoch:03} | Train {tr_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), cfg.MODEL_OUT)
            print("  saved.")
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                print("  early stop.")
                break

    plt.figure(figsize=(7,4))
    plt.plot(tr_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.legend(); plt.tight_layout(); plt.savefig(cfg.LOSS_PLOT, dpi=300)

if __name__ == "__main__":
    main()
