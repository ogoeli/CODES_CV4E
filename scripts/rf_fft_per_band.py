import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
N_FFT_COMPONENTS = 512
FFT_INPUT_SIZE = (N_FFT_COMPONENTS - 1) * 2  # = 1022
NODATA_VALUE = 255

LABELS = [0, 1, 2]
CLASS_NAMES = ["Healthy", "DFB", "Drought"]

TOTAL_FEATURES = N_BANDS * N_FFT_COMPONENTS
print(f"{N_BANDS} bands × {N_FFT_COMPONENTS} FFT = {TOTAL_FEATURES} features")

# =====================================================
# 2. FFT FEATURE EXTRACTION (CORRECT & SAFE)
# =====================================================

def extract_fft_features(data):
    """
    Returns:
        X : (n_valid_pixels, N_BANDS * N_FFT_COMPONENTS)
        y : (n_valid_pixels,)
        mask : boolean mask of valid pixels
    """
    bands, rows, cols = data.shape

    # Label bands occur every (N_BANDS + 1)
    label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)

    # Extract labels (all identical across time)
    y = data[label_indices[0]].reshape(-1)

    # Remove label bands
    feature_data = np.delete(data, label_indices, axis=0)
    n_time = len(label_indices)

    # Reshape → (pixels, bands, time)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Mask invalid pixels
    has_nan = np.isnan(feature_data).any(axis=(1, 2))
    is_ground = (y == 3)
    mask = ~has_nan & ~is_ground

    feature_data = feature_data[mask]
    y = y[mask].astype(int)

    n_valid = feature_data.shape[0]
    print(f"  Valid pixels: {n_valid}")

    X_fft = np.zeros((n_valid, N_BANDS, N_FFT_COMPONENTS), dtype=np.float32)

    for b in range(N_BANDS):
        ts = feature_data[:, b, :]  # (pixels, time)

        # 🔑 FORCE CORRECT FFT INPUT LENGTH
        if ts.shape[1] < FFT_INPUT_SIZE:
            ts = np.pad(
                ts,
                ((0, 0), (0, FFT_INPUT_SIZE - ts.shape[1])),
                mode="constant"
            )
        else:
            ts = ts[:, :FFT_INPUT_SIZE]

        fft_mag = np.abs(np.fft.rfft(ts, axis=1)).astype(np.float32)

        # 🔑 PER-BAND NORMALIZATION (CRITICAL)
        fft_mag /= (np.linalg.norm(fft_mag, axis=1, keepdims=True) + 1e-8)

        # Safety check
        assert fft_mag.shape[1] == N_FFT_COMPONENTS

        X_fft[:, b, :] = fft_mag

    X = X_fft.reshape(n_valid, -1)
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

        X, y, _ = extract_fft_features(data)
        X_list.append(X)
        y_list.append(y)

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)

print("\nTraining distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(y_train == i).sum()}")

# =====================================================
# 4. FEATURE SCALING
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =====================================================
# 5. RANDOM FOREST (NO SMOTE)
# =====================================================

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

clf.fit(X_train_scaled, y_train)

# =====================================================
# 6. TRAINING SANITY CHECK
# =====================================================

print("\nTraining performance:")
print(classification_report(
    y_train,
    clf.predict(X_train_scaled),
    target_names=CLASS_NAMES,
    zero_division=0
))

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

        X, y, _ = extract_fft_features(data)
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)

        y_true_all.append(y)
        y_pred_all.append(preds)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("\nIndependent test performance:")
print(classification_report(
    y_true_all,
    y_pred_all,
    target_names=CLASS_NAMES,
    zero_division=0
))

# =====================================================
# 8. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_true_all, y_pred_all, normalize="true")
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    cmap="Blues", values_format=".2f"
)
plt.title("Normalized Confusion Matrix (Recall)")
plt.show()
