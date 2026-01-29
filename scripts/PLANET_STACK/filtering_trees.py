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

SELECTED_BAND = 2          # Red band
SELECTED_TIME_POSITION = 1  # Second time point

N_BANDS = 4          # Sentinel-2 bands per timestep
LABEL_OFFSET = 4     # index of the first label band
GROUND_CLASS = 3     # class to ignore
RANDOM_STATE = 42

CLASS_NAMES = ["Healthy", "DFB", "Drought"]
TARGET_CLASS = 2          # Drought
HEALTHY_CLASS = 0
OVERLAP_CLASS = 4         # new label ID



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

# First reshape into (time, band, H, W)
ts_tb = ts_all.reshape(N_BANDS, -1, ts_all.shape[1], ts_all.shape[2])

# Then swap axes to get (band, time, H, W)
ts_bt = ts_tb.transpose(1, 0, 2, 3)

print(ts_bt.shape)
# (4, 4, 340, 669)

in_img = ts_bt[SELECTED_BAND][SELECTED_TIME_POSITION]


# =========================
# BOXPLOTS BY CLASS
# =========================

box_data = []
box_labels = []

for cls in classes:
    mask = (label_img == cls) & ~np.isnan(in_img)
    values = in_img[mask]

    if values.size > 0:
        box_data.append(values)
        box_labels.append(CLASS_NAMES[int(cls)])

plt.figure(figsize=(8, 5))
plt.boxplot(box_data, labels=box_labels, showfliers=False)
plt.ylabel("Pixel value")
plt.title("Pixel value distribution by class")
plt.grid(alpha=0.3)
plt.tight_layout()



# =========================
# Extract drought pixels
# ========================
drought_mask = (label_img == TARGET_CLASS) & ~np.isnan(in_img)
drought_vals = in_img[drought_mask]

mu = drought_vals.mean()
sigma = drought_vals.std()

low = mu - 2 * sigma
high = mu + 2 * sigma

print(f"Drought range: [{low:.3f}, {high:.3f}]")

# =========================
# Identify overlapping healthy pixels and create new label image
# =========================

healthy_mask = (label_img == HEALTHY_CLASS) & ~np.isnan(in_img)
overlap_mask = healthy_mask & (in_img >= low) & (in_img <= high)
new_label_img = label_img.copy()
new_label_img[overlap_mask] = OVERLAP_CLASS
print("Healthy total:", healthy_mask.sum())
print("Overlapping Healthy:", overlap_mask.sum())

# =========================
# Visualization
# =========================

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap = ListedColormap([
    "#2ca02c",  # 0 Healthy (green)
    "#ff7f0e",  # 1 DFB (orange)
    "#d62728",  # 2 Drought (red)
    "#7f7f7f",  # 3 Ground (gray)
    "#1f77b4",  # 4 Healthy-overlap (blue)
])

legend_elements = [
    Patch(facecolor="#2ca02c", label="Healthy"),
    Patch(facecolor="#1f77b4", label="Healthy–Overlap"),
    Patch(facecolor="#ff7f0e", label="DFB"),
    Patch(facecolor="#d62728", label="Drought"),
]


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original image
im0 = axes[0].imshow(in_img, cmap="gray")
axes[0].set_title("Input image")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# New label image
axes[1].imshow(new_label_img, cmap=cmap, vmin=0, vmax=4)
axes[1].set_title("Labels with overlapping Healthy class")
axes[1].axis("off")
axes[1].legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()

