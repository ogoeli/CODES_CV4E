# =====================================================
# LSTM for Sentinel-2 Time Series Pixel Classification
# =====================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# =====================================================
# CONFIG
# =====================================================

TRAIN_TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DFB_TRAIN_CLIPPED\DFB_TRAIN_CLIPPED.tif"
]

TEST_TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DFB_TEST_CLIPPED\DFB_TEST_CLIPPED.tif"
]

N_BANDS = 12
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# DATA UTILITIES
# =====================================================

def load_multiband_raster(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data[data == -1] = np.nan
    return data


def extract_sequences(data):
    """
    Returns:
        X: (N, T, 12)
        y: (N,)
    """
    bands, rows, cols = data.shape
    step = N_BANDS + 1
    T = bands // step

    data = data.reshape(T, step, rows, cols)
    X = data[:, :N_BANDS, :, :]
    labels = data[:, -1, :, :]

    # reshape pixels
    X = X.transpose(2, 3, 0, 1).reshape(-1, T, N_BANDS)
    labels = labels.transpose(1, 2, 0).reshape(-1, T)

    # --- label majority vote ---
    y = np.zeros(labels.shape[0]) * np.nan

    for i in range(labels.shape[0]):
        vals = labels[i]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            y[i] = np.bincount(vals.astype(int)).argmax()

    # --- masking ---
    bad_X = np.isnan(X).any(axis=(1, 2))
    bad_y = np.isnan(y)

    mask = ~bad_X & ~bad_y

    X = X[mask]
    y = y[mask].astype(int)

    return X, y


def load_and_stack(files, tag="data"):
    X_all, y_all = [], []

    print(f"\nLoading {tag} data...")
    for f in files:
        print(" ", os.path.basename(f))
        data = load_multiband_raster(f)
        X, y = extract_sequences(data)
        X_all.append(X)
        y_all.append(y)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    # --- remap labels ---
    uniq = np.unique(y)
    print("Original labels:", uniq)

    label_map = {lbl: i for i, lbl in enumerate(uniq)}
    y = np.vectorize(label_map.get)(y)

    print("Mapped labels:", np.unique(y))
    return X, y


# =====================================================
# DATASET
# =====================================================

class PixelSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================
# MODEL
# =====================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# =====================================================
# TRAINING
# =====================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            preds = model(X).argmax(1).cpu()
            y_true.append(y)
            y_pred.append(preds)

    return torch.cat(y_true), torch.cat(y_pred)


# =====================================================
# MAIN
# =====================================================

def main():
    X_train, y_train = load_and_stack(TRAIN_TIFF_FILES, "training")
    X_test, y_test = load_and_stack(TEST_TIFF_FILES, "testing")

    train_ds = PixelSequenceDataset(X_train, y_train)
    test_ds = PixelSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LSTMClassifier(
        input_dim=N_BANDS,
        hidden_dim=64,
        num_classes=len(np.unique(y_train))
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader)
    print("\nTest performance:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
