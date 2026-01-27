import os

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= USER SETTINGS =================
TIFF_FILES = [
    # TEST
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\MERGED\DROUGHT_TEST.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\MERGED\DFB_TEST.tif",

    # TRAIN
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\MERGED\DFB_TRAIN.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\MERGED\DROUGHT_TRAIN.tif"
]

N_COMPONENTS = 5
NODATA_VALUE = -1
VALID_CLASSES = [0, 1, 2]   # healthy, dfb, drought

# =================================================

all_pixels = []
all_classes = []
all_split = []   # 0 = train, 1 = test

# ================= PROCESS TIFFS =================
for path in TIFF_FILES:

    split = 0 if "TRAIN" in path.upper() else 1

    with rasterio.open(path) as src:
        img = src.read().astype("float32")
        bands, rows, cols = img.shape

        # -------- Identify raster type --------
        if "DFB" in path.upper():
            bands_per_date = 9
            spectral_per_date = 8
            class_band_idx = 8  # 0-based
        elif "DROUGHT" in path.upper():
            bands_per_date = 5
            spectral_per_date = 4
            class_band_idx = 4
        else:
            raise ValueError(f"Unknown raster type: {path}")
        print("\nFILE:", os.path.basename(path))
        print("Total bands:", bands)
        print("Expected bands per date:", bands_per_date)
        print("bands % bands_per_date =", bands % bands_per_date)


        # -------- Infer number of dates --------
        if bands % bands_per_date != 0:
            raise ValueError("Band count is not divisible by bands_per_date")

        n_dates = bands // bands_per_date

        # -------- Reshape to (time, bands, rows, cols) --------
        img = img.reshape(n_dates, bands_per_date, rows, cols)

        # -------- Separate spectral and class --------
        spectral = img[:, :spectral_per_date, :, :]
        classes = img[:, class_band_idx, :, :]

        spectral[spectral == NODATA_VALUE] = np.nan

        # -------- Temporal aggregation (spectral only) --------
        spectral_median = np.nanmedian(spectral, axis=0)
        X = spectral_median.reshape(spectral_per_date, -1).T

        # -------- Class handling (first valid class per pixel) --------
        classes_flat = classes.reshape(n_dates, -1)

        y = np.full(classes_flat.shape[1], NODATA_VALUE, dtype=np.int16)
        for i in range(classes_flat.shape[1]):
            vals = classes_flat[:, i]
            vals = vals[vals != NODATA_VALUE]
            if len(vals) > 0:
                y[i] = vals[0]

        # -------- Mask invalid pixels --------
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        # -------- Keep only known classes --------
        keep = np.isin(y, VALID_CLASSES)

        all_pixels.append(X[keep])
        all_classes.append(y[keep])
        all_split.append(np.full(np.sum(keep), split))

# ================= STACK =================
X = np.vstack(all_pixels)
y_class = np.hstack(all_classes)
y_split = np.hstack(all_split)

print("Final feature matrix shape:", X.shape)
print("Class distribution:", dict(zip(*np.unique(y_class, return_counts=True))))

# ================= PCA =================
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

print("PCA completed")
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

# ================= PLOT =================
class_colors = {0: "green", 1: "red", 2: "orange"}
class_names  = {0: "Healthy", 1: "DFB", 2: "Drought"}
split_markers = {0: "o", 1: "^"}   # train/test

plt.figure(figsize=(9, 7))

for cls in VALID_CLASSES:
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
