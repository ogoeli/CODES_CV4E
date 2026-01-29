import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import mode
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# CONFIGURATION
# =====================================================

N_BANDS = 4          # Spectral bands per timestep
GROUND_CLASS = 3     # Ground / ignore class
RANDOM_STATE = 42

TRAIN_TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\MERGED\DFB_TRAIN.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\MERGED\DROUGHT_TRAIN.tif"
]

TEST_TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\MERGED\DROUGHT_TEST.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\MERGED\DFB_TEST.tif",
]

CLASS_NAMES = ["Healthy", "DFB", "Drought", "Ground"]
VALID_CLASSES = [0, 1, 2, 3]

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_temporal_features(data):
    bands, rows, cols = data.shape
    n_time = bands // (N_BANDS + 1)  # 4 spectral + 1 label per date
    label_idx = np.arange(N_BANDS, bands, N_BANDS + 1)
    labels_all = data[label_idx].reshape(len(label_idx), -1)

    # Modal label across time
    y = mode(labels_all, axis=0, keepdims=False).mode.astype(int)
    drift = np.mean(np.any(labels_all != labels_all[0], axis=0))
    print(f"    Label drift fraction: {drift:.4f}")

    feature_data = np.delete(data, label_idx, axis=0)
    feature_data = feature_data.reshape(n_time, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)
    feature_data = feature_data.reshape(-1, N_BANDS, n_time)

    # Only mask NaN pixels, include all classes
    has_nan = np.isnan(feature_data).any(axis=(1, 2))
    mask = ~has_nan

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

    t = np.arange(ts.shape[2])
    t = (t - t.mean()) / t.std()
    feats.append(np.mean(ts * t, axis=2))

    red = ts[:, 2, :]
    nir = ts[:, 3, :]
    ndvi = (nir - red) / (nir + red + 1e-6)
    feats.append(ndvi.mean(axis=1, keepdims=True))
    feats.append(ndvi.std(axis=1, keepdims=True))

    X = np.concatenate(feats, axis=1)
    return X, y

# =====================================================
# LOAD AND STACK MULTIPLE RASTERS
# =====================================================

def load_and_stack(tiff_list, tag="dataset"):
    X_all, y_all = [], []
    print(f"\nLoading {tag} data...")
    for path in tiff_list:
        name = os.path.basename(path)
        print(f"  {name}")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data[data == -1] = np.nan
        X, y = extract_temporal_features(data)
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

# =====================================================
# MAIN
# =====================================================

def main():
    X_train, y_train = load_and_stack(TRAIN_TIFF_FILES, "training")
    X_test, y_test = load_and_stack(TEST_TIFF_FILES, "test")

    print(f"\nTrain samples: {X_train.shape}")
    print(f"Test samples:  {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=VALID_CLASSES)

    print("\nConfusion Matrix (numeric):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        labels=VALID_CLASSES,
        digits=4
    ))

    # =========================
    # VISUALIZE CONFUSION MATRIX
    # =========================
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()