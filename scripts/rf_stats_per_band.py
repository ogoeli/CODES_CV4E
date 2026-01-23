import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# =====================================================
# 1. CONFIGURATION
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
LABELS = [0, 1, 2]
CLASS_NAMES = ["Healthy", "DFB", "Drought"]

# =====================================================
# 2. FEATURE EXTRACTION: TEMPORAL STATISTICS
# =====================================================

def extract_temporal_stats(data):
    """
    Extract per-band temporal statistics for all pixels.
    
    Returns:
        X: np.ndarray (n_valid_pixels, n_features)
        y: np.ndarray (n_valid_pixels,)
        mask: boolean mask for valid pixels
    """
    bands, rows, cols = data.shape
    label_idx = np.arange(N_BANDS, bands, N_BANDS + 1)
    y = data[label_idx[0]].reshape(-1)

    # Remove label bands
    feature_data = np.delete(data, label_idx, axis=0)
    n_time = len(label_idx)

    # Reshape to (pixels, bands, time)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)  # rows,cols,bands,time
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Mask invalid pixels (NaN or ground class)
    has_nan = np.isnan(feature_data).any(axis=(1, 2))
    is_ground = (y == 3)
    mask = ~has_nan & ~is_ground
    ts = feature_data[mask]
    y = y[mask].astype(int)

    # Compute statistics per band
    feats = []
    feats.append(ts.mean(axis=2))
    feats.append(ts.std(axis=2))
    feats.append(ts.min(axis=2))
    feats.append(ts.max(axis=2))
    # slope per band
    t = np.arange(ts.shape[2])
    slope = np.apply_along_axis(lambda x: np.polyfit(t, x, 1)[0], 2, ts)
    feats.append(slope)
    # amplitude
    feats.append(ts.max(axis=2) - ts.min(axis=2))

    X = np.concatenate(feats, axis=1)
    return X, y, mask

# =====================================================
# 3. LOAD TRAINING DATA
# =====================================================

X_list, y_list = [], []

print("\nLoading training data...")
for tiff in TRAIN_TIFF_FILES:
    print(f"Processing: {tiff.split('\\')[-2]}")
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan
        X, y, _ = extract_temporal_stats(data)
        X_list.append(X)
        y_list.append(y)

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)

print("\nTraining distribution before SMOTE:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(y_train == i).sum()}")

# =====================================================
# 4. SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =====================================================
# 5. BALANCE CLASSES WITH SMOTE
# =====================================================

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\nTraining distribution after SMOTE:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(y_train_bal == i).sum()}")

# =====================================================
# 6. TRAIN RANDOM FOREST
# =====================================================

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight=None,  # already balanced
    n_jobs=-1,
    random_state=42,
    verbose=1
)
clf.fit(X_train_bal, y_train_bal)

# =====================================================
# 7. TRAIN PERFORMANCE
# =====================================================

print("\nTraining Performance:")
y_train_pred = clf.predict(X_train_scaled)
print(classification_report(
    y_train, y_train_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

# =====================================================
# 8. EVALUATE ON TEST DATA
# =====================================================

y_true_all, y_pred_all = [], []

print("\nEvaluating test data...")
for tiff in TEST_TIFF_FILES:
    print(f"Processing: {tiff.split('\\')[-2]}")
    with rasterio.open(tiff) as src:
        data = src.read().astype(np.float32)
        data[data == -1] = np.nan
        X, y, _ = extract_temporal_stats(data)
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        y_true_all.append(y)
        y_pred_all.append(preds)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("\nTest Performance:")
print(classification_report(
    y_true_all, y_pred_all,
    target_names=CLASS_NAMES,
    zero_division=0
))

# =====================================================
# 9. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_true_all, y_pred_all, normalize="true")
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    cmap="Blues", values_format=".2f"
)
plt.title("Normalized Confusion Matrix (Recall)")
plt.show()
