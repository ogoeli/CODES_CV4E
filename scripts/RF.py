import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler  # for oversampling minority classes

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

N_FEATURES = 12
NODATA_VALUE = 255

# =====================================================
# 2. LOAD TRAINING DATA
# =====================================================

X_list, y_list = [], []

for tiff in TRAIN_TIFF_FILES:
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan  # Replace invalid pixels with NaN

        X = data[:N_FEATURES].reshape(N_FEATURES, -1).T
        y = data[N_FEATURES].reshape(-1)

        mask = ~np.isnan(X).any(axis=1)
        X_list.append(X[mask])
        y_list.append(y[mask])

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)

print(f"Original training samples: {X_train.shape[0]}")
print("Original training label distribution:", np.bincount(y_train.astype(int)))

# =====================================================
# 3. OVERSAMPLE MINORITY CLASSES
# =====================================================

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train.astype(int))

print(f"\nAfter oversampling: {X_train_res.shape[0]} samples")
print("New training label distribution:", np.bincount(y_train_res))

# =====================================================
# 4. TRAIN RANDOM FOREST
# =====================================================

clf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train_res, y_train_res)

# =====================================================
# 5. EVALUATE ON INDEPENDENT TEST TIFFs
# =====================================================

y_true_all, y_pred_all = [], []

for tiff in TEST_TIFF_FILES:
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan

        X = data[:N_FEATURES].reshape(N_FEATURES, -1).T
        y = data[N_FEATURES].reshape(-1)

        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        y_pred = clf.predict(X)

        y_true_all.append(y)
        y_pred_all.append(y_pred)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("\nClassification Report (Independent Test Data):")
print(classification_report(y_true_all, y_pred_all))

# =====================================================
# 6. CONFUSION MATRICES
# =====================================================

class_names = ["Healthy", "DFB", "Drought"]

cm = confusion_matrix(y_true_all, y_pred_all)
cm_norm = confusion_matrix(y_true_all, y_pred_all, normalize="true")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax[0], cmap="Blues", values_format="d"
)
ax[0].set_title("Confusion Matrix")

ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
    ax=ax[1], cmap="Blues", values_format=".2f"
)
ax[1].set_title("Normalized Confusion Matrix (Recall)")

plt.tight_layout()
plt.show()

# =====================================================
# 7. SAVE CLASSIFIED RASTERS
# =====================================================

def classify_and_save(tiff, output_path):
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan

        bands, rows, cols = data.shape
        X = data[:N_FEATURES].reshape(N_FEATURES, -1).T
        mask = ~np.isnan(X).any(axis=1)

        X_valid = X[mask]
        preds = clf.predict(X_valid).astype(np.uint8)

        out = np.full(rows * cols, NODATA_VALUE, dtype=np.uint8)
        out[mask] = preds
        out = out.reshape(rows, cols)

        profile = src.profile
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=NODATA_VALUE,
            compress="lzw"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out, 1)

# Save outputs
for i, tiff in enumerate(TRAIN_TIFF_FILES):
    classify_and_save(
        tiff,
        f"C:\\Users\\ope4\\Desktop\\RF_TRAIN_CLASSIFIED_{i}.tif"
    )

for i, tiff in enumerate(TEST_TIFF_FILES):
    classify_and_save(
        tiff,
        f"C:\\Users\\ope4\\Desktop\\RF_TEST_CLASSIFIED_{i}.tif"
    )

print("\n✅ Classification complete. All rasters saved.")
