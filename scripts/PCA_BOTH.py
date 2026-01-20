import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= USER SETTINGS =================
TIFF_FILES = [
    # TEST
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DFB_TEST_CLIPPED\DFB_TEST_CLIPPED.tif",
    # TRAIN
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DFB_TRAIN_CLIPPED\DFB_TRAIN_CLIPPED.tif"
]

CLASS_BAND_IDX = 12        # 0-based
BANDS_PER_DATE = 13
SPECTRAL_BANDS = 12
N_COMPONENTS = 5
NODATA_VALUE = -1

# =================================================

all_pixels = []
all_classes = []
all_split = []   # 0 = train, 1 = test

# ================= PROCESS TIFFS =================
for path in TIFF_FILES:

    split = 0 if "TRAIN" in path.upper() else 1  # infer split from filename

    with rasterio.open(path) as src:
        img = src.read().astype("float32")
        bands, rows, cols = img.shape

        # -------- Extract class band (NO temporal aggregation!) --------
        class_band = img[CLASS_BAND_IDX].reshape(-1)

        # -------- Extract spectral bands only --------
        spectral_idx = [i for i in range(bands) if (i + 1) % BANDS_PER_DATE != 0]
        spectral = img[spectral_idx]
        spectral[spectral == NODATA_VALUE] = np.nan

        # -------- Temporal median on spectral bands ONLY --------
        n_dates = spectral.shape[0] // SPECTRAL_BANDS
        spectral = spectral.reshape(n_dates, SPECTRAL_BANDS, rows, cols)
        spectral_median = np.nanmedian(spectral, axis=0)

        # -------- Flatten --------
        X = spectral_median.reshape(SPECTRAL_BANDS, -1).T

        # -------- Valid pixels --------
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid_mask]
        class_valid = class_band[valid_mask]

        # -------- Keep only known classes --------
        keep = np.isin(class_valid, [0, 1, 2])
        all_pixels.append(X_valid[keep])
        all_classes.append(class_valid[keep])
        all_split.append(np.full(np.sum(keep), split))

# ================= STACK =================
X = np.vstack(all_pixels)
y_class = np.hstack(all_classes)   # 0 healthy, 1 dfb, 2 drought
y_split = np.hstack(all_split)     # 0 train, 1 test

# ================= PCA =================
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

print("PCA completed")
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

# ================= SANITY CHECK =================
print("Class counts:")
unique, counts = np.unique(y_class, return_counts=True)
print(dict(zip(unique, counts)))

# ================= PLOT PC1 vs PC2 =================
class_colors = {0: "green", 1: "red", 2: "orange"}
class_names  = {0: "Healthy", 1: "DFB", 2: "Drought"}
split_markers = {0: "o", 1: "^"}   # train/test

plt.figure(figsize=(9, 7))

for cls in [0, 1, 2]:
    for split in [0, 1]:
        mask = (y_class == cls) & (y_split == split)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=class_colors[cls],
            marker=split_markers[split],
            s=6,
            alpha=0.6,
            label=f"{class_names[cls]} - {'Train' if split == 0 else 'Test'}"
        )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: Healthy vs DFB vs Drought (Train/Test)")
plt.legend(markerscale=2)
plt.grid(True)
plt.show()
