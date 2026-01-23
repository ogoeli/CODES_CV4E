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

LABELS_BINARY = [0, 1]
CLASS_NAMES_BINARY = ["Healthy", "DFB_or_Drought"]

LABELS_3CLASS = [0, 1, 2]
CLASS_NAMES_3CLASS = ["Healthy", "DFB_or_Drought", "Undetermined"]

# =====================================================
# 2. FEATURE EXTRACTION
# =====================================================

def extract_temporal_stats(data):
    bands, rows, cols = data.shape
    label_idx = np.arange(N_BANDS, bands, N_BANDS + 1)
    y = data[label_idx[0]].reshape(-1)

    # Merge DFB and Drought into one class
    y = np.where((y == 1) | (y == 2), 1, 0)

    # Remove label bands
    feature_data = np.delete(data, label_idx, axis=0)
    n_time = len(label_idx)

    # Reshape to (pixels, bands, time)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)  # rows,cols,bands,time
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Mask invalid pixels
    has_nan = np.isnan(feature_data).any(axis=(1, 2))
    mask = ~has_nan
    ts = feature_data[mask]
    y = y[mask].astype(int)

    # Compute features: mean, std, min, max, slope, amplitude
    feats = [
        ts.mean(axis=2),
        ts.std(axis=2),
        ts.min(axis=2),
        ts.max(axis=2),
        np.apply_along_axis(lambda x: np.polyfit(np.arange(ts.shape[2]), x, 1)[0], 2, ts),
        ts.max(axis=2) - ts.min(axis=2)
    ]

    X = np.concatenate(feats, axis=1)
    return X, y, mask

# =====================================================
# 3. LOAD TRAINING DATA
# =====================================================

X_list, y_list = [], []

print("Loading training data...")
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

# =====================================================
# 4. SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =====================================================
# 5. ITERATIVE TRAINING WITH MAX ITERATIONS
# =====================================================

y_train_3class = y_train.copy()
iteration = 0
MAX_ITER = 20  # maximum number of iterations

while True:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    
    # Balance classes with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_3class)
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    clf.fit(X_train_bal, y_train_bal)
    
    # Predict on training set
    y_train_pred = clf.predict(X_train_scaled)
    
    # Identify misclassified pixels (excluding already Undetermined)
    mis_idx = np.where((y_train_pred != y_train_3class) & (y_train_3class != 2))[0]
    
    print(f"Misclassified samples in training (excluding 'Undetermined'): {len(mis_idx)}")
    
    if len(mis_idx) == 0:
        print("No more misclassified pixels. Stopping iteration.")
        break
    
    # Stop if maximum iterations reached
    if iteration >= MAX_ITER:
        print(f"Reached maximum iterations ({MAX_ITER}). Continuing with current classifier.")
        break
    
    # Reclassify misclassified pixels as Undetermined
    y_train_3class[mis_idx] = 2


# =====================================================
# 6. TRAIN CONFUSION MATRIX ON TRAINING SET
# =====================================================

y_train_pred_final = clf.predict(X_train_scaled)
print("\nFinal Training Performance:")
print(classification_report(y_train_3class, y_train_pred_final, target_names=CLASS_NAMES_3CLASS, zero_division=0))

cm_train_final = confusion_matrix(y_train_3class, y_train_pred_final, labels=[0,1,2])
print("Final Training Confusion Matrix:")
print(cm_train_final)

# =====================================================
# 7. TEST EVALUATION
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

# Test set sizes per class
print("\nTest set size per class (original binary labels):")
for i, name in enumerate(CLASS_NAMES_BINARY):
    print(f"  {name}: {(y_true_all==i).sum()}")

# =====================================================
# 8. TEST CONFUSION MATRIX (INCLUDING UNDETERMINED)
# =====================================================

print("\nTest Performance (predictions may include 'Undetermined'):")
print(classification_report(
    y_true_all,
    y_pred_all,
    labels=[0,1,2],
    target_names=CLASS_NAMES_3CLASS,
    zero_division=0
))

cm_test_final = confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2])
print("Test Confusion Matrix (Raw Counts):")
print(cm_test_final)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ConfusionMatrixDisplay(cm_train_final, display_labels=CLASS_NAMES_3CLASS).plot(ax=ax[0], cmap="Blues", values_format="d")
ax[0].set_title("Final Training Confusion Matrix")
ConfusionMatrixDisplay(cm_test_final, display_labels=CLASS_NAMES_3CLASS).plot(ax=ax[1], cmap="Blues", values_format="d")
ax[1].set_title("Test Confusion Matrix (predictions may include 'Undetermined')")
plt.tight_layout()
plt.show()
