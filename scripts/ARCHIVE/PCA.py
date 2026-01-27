import rasterio
import numpy as np

# Load raster
src = rasterio.open(r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif")
img = src.read()  # shape: (bands, rows, cols)
print(f"Image: {img}")

# Mask all -1 values
img_masked = np.where(img == -1, np.nan, img)
print(f"Masked Image: {img_masked}")








import rasterio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DFB_TRAIN_CLIPPED\DFB_TRAIN_CLIPPED.tif"
]

OUTPUT_DIR = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CODES_CV4E\outputs\PCA_OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_COMPONENTS = 5
SPECTRAL_BANDS = 12
BANDS_PER_DATE = 13
NODATA_VALUE = -1

# --------------------------------------------------
# STORAGE
# --------------------------------------------------
all_pixels = []
image_shapes = []
valid_masks = []

# --------------------------------------------------
# STEP 1: LOAD + TEMPORAL MEDIAN
# --------------------------------------------------
for idx, tiff_path in enumerate(TIFF_FILES):
    with rasterio.open(tiff_path) as src:
        img = src.read().astype("float32")  # (bands, rows, cols)
        bands, rows, cols = img.shape

        print(f"\nTIFF {idx+1}: {bands} bands, {rows} x {cols}")

        # Identify spectral band indices (exclude class band)
        spectral_idx = [i for i in range(bands) if (i + 1) % BANDS_PER_DATE != 0]
        spectral = img[spectral_idx]

        spectral[spectral == NODATA_VALUE] = np.nan

        # Infer number of dates
        n_dates = spectral.shape[0] // SPECTRAL_BANDS
        spectral = spectral.reshape(
            n_dates, SPECTRAL_BANDS, rows, cols
        )

        # Temporal median
        spectral_median = np.nanmedian(spectral, axis=0)  # (12, rows, cols)

        # Flatten
        spectral_2d = spectral_median.reshape(SPECTRAL_BANDS, -1).T

        valid_mask = ~np.any(np.isnan(spectral_2d), axis=1)
        spectral_valid = spectral_2d[valid_mask]

        print(f"  → Dates: {n_dates}")
        print(f"  → Valid pixels: {spectral_valid.shape[0]}")

        all_pixels.append(spectral_valid)
        image_shapes.append((rows, cols))
        valid_masks.append(valid_mask)

# --------------------------------------------------
# STEP 2: STACK FOR PCA
# --------------------------------------------------
X = np.vstack(all_pixels)
print(f"\nTotal PCA samples: {X.shape}")

# --------------------------------------------------
# STEP 3: STANDARDIZE
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# STEP 4: PCA
# --------------------------------------------------
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA completed")
print("Explained variance ratio:")
print(np.round(pca.explained_variance_ratio_, 4))

# --------------------------------------------------
# STEP 5: MAP PCA BACK TO EACH TIFF
# --------------------------------------------------
start = 0

for idx, tiff_path in enumerate(TIFF_FILES):
    rows, cols = image_shapes[idx]
    valid_mask = valid_masks[idx]
    n_valid = np.sum(valid_mask)

    pca_subset = X_pca[start:start + n_valid]
    start += n_valid

    full_pca = np.full((N_COMPONENTS, rows * cols), np.nan, dtype="float32")
    full_pca[:, valid_mask] = pca_subset.T
    full_pca = full_pca.reshape(N_COMPONENTS, rows, cols)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"PCA_{os.path.basename(tiff_path)}"
    )

    with rasterio.open(tiff_path) as src:
        profile = src.profile
        profile.update(
            count=N_COMPONENTS,
            dtype="float32",
            nodata=np.nan
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(full_pca)

    print(f"Saved: {out_path}")

print("\n✅ PCA PIPELINE COMPLETED SUCCESSFULLY")




################### PCA + Class Visualization Script

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# INPUT FILES
# --------------------------------------------------
PCA_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CODES_CV4E\outputs\PCA_OUTPUT\PCA_DROUGHT_TEST_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CODES_CV4E\outputs\PCA_OUTPUT\PCA_DFB_TEST_CLIPPED.tif"
]

CLASS_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DFB_TEST_CLIPPED\DFB_TEST_CLIPPED.tif"
]

N_COMPONENTS = 5
CLASS_BAND_IDX = 12  # 0-based: last band per date = class
BANDS_PER_DATE = 13

# --------------------------------------------------
# STEP 1: EXTRACT VALID PIXELS + CLASS
# --------------------------------------------------
all_pc = []
all_class = []

for pca_file, class_file in zip(PCA_FILES, CLASS_FILES):
    with rasterio.open(pca_file) as src_pca:
        pca_img = src_pca.read()  # shape: (PCs, rows, cols)
        rows, cols = pca_img.shape[1], pca_img.shape[2]

    with rasterio.open(class_file) as src_cls:
        cls_img = src_cls.read(CLASS_BAND_IDX+1)  # select last band
        cls_img[cls_img == -1] = np.nan

    # Flatten
    pca_2d = pca_img[:2].reshape(2, -1).T  # PC1 and PC2
    cls_flat = cls_img.flatten()

    # Mask invalid pixels
    valid_mask = ~np.any(np.isnan(pca_2d), axis=1) & ~np.isnan(cls_flat)
    pca_valid = pca_2d[valid_mask]
    cls_valid = cls_flat[valid_mask]

    all_pc.append(pca_valid)
    all_class.append(cls_valid)

# --------------------------------------------------
# STEP 2: COMBINE RASTERS
# --------------------------------------------------
pc_all = np.vstack(all_pc)
class_all = np.hstack(all_class)

print(f"PC shape: {pc_all.shape}, Class shape: {class_all.shape}")

# --------------------------------------------------
# STEP 3: PLOT PC1 vs PC2
# --------------------------------------------------
plt.figure(figsize=(8,6))
classes = np.unique(class_all)
colors = ['green', 'red', 'blue']  # adjust for your class codes

for cls, color in zip(classes, colors):
    mask = class_all == cls
    plt.scatter(pc_all[mask,0], pc_all[mask,1], c=color, label=f'class {int(cls)}', s=5, alpha=0.6)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 vs PC2 Scatter Plot by Class')
plt.legend()
plt.grid(True)
plt.show()



