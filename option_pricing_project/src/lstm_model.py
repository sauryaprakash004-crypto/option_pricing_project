"""A very small LSTM that forecasts the next volatility value from the previous 20.

Architecture: 1 LSTM layer (hidden size 16) -> Linear(16, 1). Loss = MSE, optimiser = Adam.
"""

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 20   # number of past volatility values used as input
HIDDEN = 16    # LSTM hidden size
EPOCHS = 15
BATCH_SIZE = 32
LR = 0.01


class VolLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):                # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)            # out: (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])    # use the last time step -> (batch, 1)


def make_windows(series, seq_len=SEQ_LEN):
    """Turn a 1-D series into (previous seq_len values -> next value) pairs."""
    X = np.array([series[i:i + seq_len] for i in range(len(series) - seq_len)])
    y = series[seq_len:]
    return X[:, :, None], y              # X shape: (N, seq_len, 1)


def train_and_predict(vol_series, seq_len=SEQ_LEN, epochs=EPOCHS,
                      train_frac=0.8, seed=42):
    """Train on the first 80% of windows (time order kept), predict the last 20%.

    Returns: actual test vols, predicted test vols, index in the original series
    of the first test target, and the training loss per epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    vol_series = np.asarray(vol_series, dtype=np.float32)
    X, y = make_windows(vol_series, seq_len)
    n_train = int(len(X) * train_frac)

    # Standardise using training data only (avoids leaking test information)
    train_part = vol_series[:n_train + seq_len]
    mean, std = train_part.mean(), train_part.std()
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32)
    yn = torch.tensor(((y - mean) / std)[:, None], dtype=torch.float32)

    model = VolLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(Xn[idx]), yn[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n_train)
        print(f"  epoch {epoch + 1:2d}/{epochs}  train MSE (standardised) = {losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xn[n_train:]).squeeze(1).numpy() * std + mean   # back to vol units

    return y[n_train:], pred, n_train + seq_len, losses
