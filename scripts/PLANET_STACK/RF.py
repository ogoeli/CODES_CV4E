import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
from imblearn.over_sampling import RandomOverSampler  # <-- added for oversampling

# =====================================================
# 1. FILE PATHS
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
NODATA_VALUE = 255

LABELS = [0, 1, 2]  # Ignore class 3 (ground)
CLASS_NAMES = ["Healthy", "DFB", "Drought"]

# compute minimum feature-band count across all TIFFs so FFT input length is fixed
min_feature_bands = None
for tiff in TRAIN_TIFF_FILES + TEST_TIFF_FILES:
    with rasterio.open(tiff) as src:
        bands = src.count
        label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)
        feature_bands = bands - len(label_indices)
        if min_feature_bands is None or feature_bands < min_feature_bands:
            min_feature_bands = feature_bands
if min_feature_bands is None:
    raise RuntimeError("No training TIFFs found or unable to compute feature bands.")

# =====================================================
# 2. LOAD TRAINING DATA (IGNORE CLASS 3)
# =====================================================

X_list, y_list = [], []

for tiff in TRAIN_TIFF_FILES:
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan

        bands = data.shape[0]
        label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)
        y = data[label_indices[0]].reshape(-1)
        feature_data = np.delete(data, label_indices, axis=0)

        if feature_data.shape[0] < min_feature_bands:
            raise RuntimeError(
                f"Training TIFF {tiff} has fewer feature bands ({feature_data.shape[0]}) than "
                f"training min ({min_feature_bands})."
            )
        if feature_data.shape[0] > min_feature_bands:
            feature_data = feature_data[:min_feature_bands]

        X = feature_data.reshape(feature_data.shape[0], -1).T
        mask = (~np.isnan(X).any(axis=1)) & (y != 3)
        Xmasked = X[mask]
        Ymasked = y[mask].astype(int)

        # FFT magnitude per row
        X_fft = np.fft.rfft(Xmasked, n=min_feature_bands, axis=1)
        X_fft_mag = np.abs(X_fft).astype(np.float32)

        # L2 normalize each sample
        X_fft_norm = normalize(X_fft_mag, norm='l2', axis=1)

        X_list.append(X_fft_norm)
        y_list.append(Ymasked)

# stack training data
X_train = np.vstack(X_list)
y_train = np.concatenate(y_list).astype(int)

print(f"Training samples (no ground): {X_train.shape[0]}  FFT bins: {X_train.shape[1]}")
print("Training label distribution:", np.bincount(y_train))

# =====================================================
# 2b. OVERSAMPLE MINORITY CLASSES
# =====================================================

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

print("After oversampling label distribution:", np.bincount(y_train_bal))

# =====================================================
# 3. TRAIN RANDOM FOREST
# =====================================================

clf = RandomForestClassifier(
    n_estimators=500,  # more trees for stability
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=5
)

clf.fit(X_train_bal, y_train_bal)

# =====================================================
# 4. EVALUATE ON INDEPENDENT TEST DATA (IGNORE CLASS 3)
# =====================================================

y_true_all, y_pred_all = [], []

for tiff in TEST_TIFF_FILES:
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan

        bands = data.shape[0]
        label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)
        y = data[label_indices[0]].reshape(-1)
        feature_data = np.delete(data, label_indices, axis=0)

        if feature_data.shape[0] < min_feature_bands:
            raise RuntimeError(
                f"Test TIFF has fewer feature bands ({feature_data.shape[0]}) than "
                f"training min ({min_feature_bands})."
            )
        if feature_data.shape[0] > min_feature_bands:
            feature_data = feature_data[:min_feature_bands]

        X = feature_data.reshape(feature_data.shape[0], -1).T
        mask = (~np.isnan(X).any(axis=1)) & (y != 3)
        X_masked = X[mask]
        y_masked = y[mask].astype(int)

        X_fft = np.fft.rfft(X_masked, n=min_feature_bands, axis=1)
        X_fft_mag = np.abs(X_fft).astype(np.float32)
        X_fft_norm = normalize(X_fft_mag, norm='l2', axis=1)

        preds = clf.predict(X_fft_norm).astype(np.uint8)

        y_true_all.append(y_masked)
        y_pred_all.append(preds)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("\nClassification Report (Independent Test Data):")
print(classification_report(
    y_true_all,
    y_pred_all,
    labels=LABELS,
    target_names=CLASS_NAMES,
    zero_division=0  # avoids warnings
))

# =====================================================
# 5. CONFUSION MATRICES
# =====================================================

cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)
cm_norm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS, normalize="true")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax[0], cmap="Blues", values_format="d")
ax[0].set_title("Confusion Matrix")

ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(ax=ax[1], cmap="Blues", values_format=".2f")
ax[1].set_title("Normalized Confusion Matrix (Recall)")

plt.tight_layout()
plt.show()

# =====================================================
# 6. SAVE CLASSIFIED RASTERS (GROUND → NODATA)
# =====================================================

def classify_and_save(tiff, output_path):
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan

        bands, rows, cols = data.shape
        label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)
        label_band = data[label_indices[0]].reshape(-1)
        feature_data = np.delete(data, label_indices, axis=0)

        if feature_data.shape[0] < min_feature_bands:
            raise RuntimeError(
                f"Raster has fewer feature bands ({feature_data.shape[0]}) than "
                f"training min ({min_feature_bands})."
            )
        if feature_data.shape[0] > min_feature_bands:
            feature_data = feature_data[:min_feature_bands]

        X = feature_data.reshape(feature_data.shape[0], -1).T
        mask = (~np.isnan(X).any(axis=1)) & (label_band != 3)

        X_fft = np.fft.rfft(X[mask], n=min_feature_bands, axis=1)
        X_fft_mag = np.abs(X_fft).astype(np.float32)
        X_fft_norm = normalize(X_fft_mag, norm='l2', axis=1)

        preds = clf.predict(X_fft_norm).astype(np.uint8)

        out = np.full(rows * cols, NODATA_VALUE, dtype=np.uint8)
        out[mask] = preds
        out = out.reshape(rows, cols)

        profile = src.profile
        profile.update(count=1, dtype=rasterio.uint8, nodata=NODATA_VALUE, compress="lzw")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out, 1)

# Save outputs
for i, tiff in enumerate(TRAIN_TIFF_FILES):
    classify_and_save(tiff, f"C:\\Users\\ope4\\Desktop\\RF_TRAIN_CLASSIFIED_{i}.tif")

for i, tiff in enumerate(TEST_TIFF_FILES):
    classify_and_save(tiff, f"C:\\Users\\ope4\\Desktop\\RF_TEST_CLASSIFIED_{i}.tif")

print("\n✅ Classification complete. Ground ignored, rasters saved.")
