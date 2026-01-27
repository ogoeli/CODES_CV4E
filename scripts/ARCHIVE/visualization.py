import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import mode

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# CONFIGURATION
# =====================================================

N_BANDS = 12          # Sentinel-2 bands per timestep
LABEL_OFFSET = 12    # 13th band
GROUND_CLASS = 3     # ground / ignore class
RANDOM_STATE = 42

CLASS_NAMES = ["Healthy", "DFB", "Drought"]

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_temporal_features(data):
    """
    data shape: (bands, rows, cols)
    """
    bands, rows, cols = data.shape

    # Identify label bands
    label_idx = np.arange(LABEL_OFFSET, bands, N_BANDS + 1)

    labels_all = data[label_idx].reshape(len(label_idx), -1)

    # --- FIX: modal label across time ---
    y = mode(labels_all, axis=0, keepdims=False).mode.astype(int)

    drift = np.mean(np.any(labels_all != labels_all[0], axis=0))
    print(f"    Label drift fraction: {drift:.4f}")

    # Remove label bands
    feature_data = np.delete(data, label_idx, axis=0)
    n_time = len(label_idx)

    # Reshape → (pixels, bands, time)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Mask invalid pixels
    has_nan = np.isnan(feature_data).any(axis=(1, 2))
    is_ground = (y == GROUND_CLASS)
    mask = ~has_nan & ~is_ground

    ts = feature_data[mask]
    y = y[mask]

    # =========================
    # TEMPORAL FEATURES
    # =========================
    feats = []

    feats.append(ts.mean(axis=2))
    feats.append(ts.std(axis=2))
    feats.append(ts.min(axis=2))
    feats.append(ts.max(axis=2))
    feats.append(ts.max(axis=2) - ts.min(axis=2))  # amplitude

    # Trend (robust linear proxy)
    t = np.arange(ts.shape[2])
    t = (t - t.mean()) / t.std()
    feats.append(np.mean(ts * t, axis=2))

    # NDVI features
    red = ts[:, 3, :]
    nir = ts[:, 7, :]
    ndvi = (nir - red) / (nir + red + 1e-6)

    feats.append(ndvi.mean(axis=1, keepdims=True))
    feats.append(ndvi.std(axis=1, keepdims=True))

    X = np.concatenate(feats, axis=1)

    return X, y


# =====================================================
# LOAD MULTIPLE RASTERS
# =====================================================

path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif"
    # r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DFB_TRAIN_CLIPPED\DFB_TRAIN_CLIPPED.tif"
    # r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif",
    # r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DFB_TEST_CLIPPED\DFB_TEST_CLIPPED.tif"

name = os.path.basename(path)
print(f"  {name}")

with rasterio.open(path) as src:
    data = src.read().astype(np.float32)
    data[data == -1] = np.nan


bands, nrows, ncols = data.shape

label_img = data[LABEL_OFFSET]

classes = np.unique(label_img)

# Indices of label bands
label_idx = np.arange(LABEL_OFFSET, bands, N_BANDS + 1)

# Boolean mask for all non-label bands
band_mask = np.ones(bands, dtype=bool)
band_mask[label_idx] = False

ini, fin = 0, bands - len(label_idx)
# ini, fin = 2000, 2100

plt.figure(figsize=(12, 6))

for cls in classes:
    if np.isnan(cls):
        continue

    # Get all pixels of this class
    rows, cols = np.where(np.isclose(label_img, cls))
    if rows.size == 0:
        continue

    # Extract time series for all pixels in this class (non-label bands)
    ts_class = data[band_mask][:, rows, cols]  # shape: (time, num_pixels)
    ts_class = ts_class[ini:fin]

    # Compute mean and standard deviation across pixels
    ts_mean = np.mean(ts_class, axis=1)
    ts_std = np.std(ts_class, axis=1)

    # Plot mean
    plt.plot(ts_mean, label=f'Class {int(cls)}')

    # Plot deviation bands (mean ± std)
    plt.fill_between(
        np.arange(len(ts_mean)),
        ts_mean - ts_std,
        ts_mean + ts_std,
        alpha=0.3
    )

plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Average time series per class ± std deviation")
plt.show()
