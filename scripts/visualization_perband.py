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

N_BANDS = 4          # Sentinel-2 bands per timestep
LABEL_OFFSET = 4     # index of the first label band
GROUND_CLASS = 3     # class to ignore
RANDOM_STATE = 42

CLASS_NAMES = ["Healthy", "DFB", "Drought"]

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_temporal_features(data):
    """
    Extract temporal features from multi-band raster data, ignoring ground class.
    
    data shape: (bands, rows, cols)
    """
    bands, rows, cols = data.shape

    # Identify label bands
    label_idx = np.arange(LABEL_OFFSET, bands, N_BANDS + 1)

    # Extract labels for all timesteps
    labels_all = data[label_idx].reshape(len(label_idx), -1)

    # Compute modal label per pixel
    y = mode(labels_all, axis=0, keepdims=False).mode.astype(int)

    drift = np.mean(np.any(labels_all != labels_all[0], axis=0))
    print(f"    Label drift fraction: {drift:.4f}")

    # Remove label bands from feature data
    feature_data = np.delete(data, label_idx, axis=0)
    n_time = len(label_idx)

    # Reshape → (pixels, bands, time)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Mask invalid pixels and ground
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
    red = ts[:, 2, :]
    nir = ts[:, 3, :]
    ndvi = (nir - red) / (nir + red + 1e-6)

    feats.append(ndvi.mean(axis=1, keepdims=True))
    feats.append(ndvi.std(axis=1, keepdims=True))

    X = np.concatenate(feats, axis=1)

    return X, y


# =====================================================
# LOAD MULTIPLE RASTERS
# =====================================================

# Example: pick one file
# TEST 

#path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\MERGED\DROUGHT_TEST.tif" 
#path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\MERGED\DFB_TEST.tif" 
# # TRAIN 
# 
#path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\MERGED\DFB_TRAIN.tif" 
path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\MERGED\DROUGHT_TRAIN.tif"


name = os.path.basename(path)
print(f"  Loading: {name}")

with rasterio.open(path) as src:
    data = src.read().astype(np.float32)
    data[data == -1] = np.nan

bands, nrows, ncols = data.shape
label_img = data[LABEL_OFFSET]

# Remove ground from classes
classes = np.unique(label_img)
classes = classes[classes != GROUND_CLASS]

# Indices of label bands
label_idx = np.arange(LABEL_OFFSET, bands, N_BANDS + 1)

# Boolean mask for all non-label bands
band_mask = np.ones(bands, dtype=bool)
band_mask[label_idx] = False

# Extract time series (all non-label bands)
ts_all = data[band_mask]  # shape: (time, rows, cols)
n_time = ts_all.shape[0]

# Prepare storage: mean & std per class
class_ts_mean = {}
class_ts_std = {}

for cls in classes:
    # Pixels of this class
    rows, cols = np.where(np.isclose(label_img, cls))
    if rows.size == 0:
        continue

    ts_class = ts_all[:, rows, cols]  # shape: (time, num_pixels)

    # Compute mean and std across pixels
    ts_mean = np.mean(ts_class, axis=1)
    ts_std = np.std(ts_class, axis=1)

    # Reshape into bands
    ts_mean_band = ts_mean.reshape(-1, N_BANDS).T
    ts_std_band = ts_std.reshape(-1, N_BANDS).T

    class_ts_mean[cls] = ts_mean_band
    class_ts_std[cls] = ts_std_band

# =====================================================
# PLOTTING
# =====================================================

for i_band in range(N_BANDS):
    plt.figure(figsize=(12, 6))

    for cls in classes:
        if cls not in class_ts_mean:
            continue

        ts_mean = class_ts_mean[cls][i_band]
        ts_std = class_ts_std[cls][i_band]

        plt.plot(ts_mean, label=f'Class {int(cls)}')
        plt.fill_between(
            np.arange(len(ts_mean)),
            ts_mean - ts_std,
            ts_mean + ts_std,
            alpha=0.3
        )

    plt.xlabel("Time within band")
    plt.ylabel("Value")
    plt.title(f"Band {i_band + 1}: Average time series per class ± std deviation")
    plt.legend()
plt.show()

