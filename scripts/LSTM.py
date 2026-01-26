import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import rasterio
from scipy.stats import mode

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

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
LABEL_OFFSET = 12
GROUND_CLASS = 3
CLASS_NAMES = ["Healthy", "DFB", "Drought"]
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# =====================================================
# DATA LOADER
# =====================================================

class PixelTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================================================
# LSTM MODEL
# =====================================================

class PixelLSTM(nn.Module):
    def __init__(self, n_bands, hidden_size=64, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_bands, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, n_classes)
        
    def forward(self, x):
        # x: batch x time x bands
        _, (h_n, _) = self.lstm(x)
        # bidirectional → concat forward/backward
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        out = self.fc(h)
        return out

# =====================================================
# FEATURE EXTRACTION (KEEP TIME ORDER)
# =====================================================

def extract_sequence_features(data):
    """
    Returns:
        X: n_pixels x time x bands
        y: n_pixels
    """
    bands, rows, cols = data.shape
    label_idx = np.arange(LABEL_OFFSET, bands, N_BANDS+1)
    
    # modal label across time
    labels_all = data[label_idx].reshape(len(label_idx), -1)
    y = mode(labels_all, axis=0, keepdims=False).mode.astype(int)
    
    # remove label bands
    feature_data = np.delete(data, label_idx, axis=0)
    n_time = len(label_idx)
    
    # reshape to (pixels, time, bands)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2,3,0,1) # rows,cols,time,bands
    X = feature_data.reshape(-1, n_time, N_BANDS)
    
    # mask
    has_nan = np.isnan(X).any(axis=2)
    is_ground = (y == GROUND_CLASS)
    mask = ~has_nan & ~is_ground
    
    return X[mask], y[mask]

def load_and_stack(tiff_list, tag="dataset"):
    X_all, y_all = [], []
    print(f"\nLoading {tag} data...")
    for path in tiff_list:
        print(f"  {os.path.basename(path)}")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data[data==-1] = np.nan
        X, y = extract_sequence_features(data)
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

# =====================================================
# MAIN
# =====================================================

def main():
    X_train, y_train = load_and_stack(TRAIN_TIFF_FILES, "training")
    X_test, y_test = load_and_stack(TEST_TIFF_FILES, "test")
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    train_dataset = PixelTimeSeriesDataset(X_train, y_train)
    test_dataset = PixelTimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PixelLSTM(N_BANDS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_dataset):.4f}")
    
    # Evaluation
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(y_batch.numpy())
    
    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES, digits=4))
    
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("\nConfusion Matrix:")
    print(cm)

if __name__=="__main__":
    main()
