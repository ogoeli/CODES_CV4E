import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# =====================================================
# 1. CONFIGURATION
# =====================================================

TRAIN_TIFF_FILES = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TRAIN\CLIPPED\DFB_TRAIN_CLIPPED\DFB_TRAIN_CLIPPED.tif"
]

N_BANDS = 12
N_FFT_COMPONENTS = 512  # Fixed number of FFT components per band

LABELS = [0, 1, 2]  # Ignore class 3 (ground)
CLASS_NAMES = ["Healthy", "DFB", "Drought"]
CLASS_COLORS = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange

# Total features per pixel: N_BANDS × N_FFT_COMPONENTS
TOTAL_FEATURES = N_BANDS * N_FFT_COMPONENTS

print(f"Configuration: {N_BANDS} bands × {N_FFT_COMPONENTS} FFT components = {TOTAL_FEATURES} features per pixel")

# =====================================================
# 2. HELPER FUNCTION: EXTRACT FFT FEATURES PER BAND
# =====================================================

def extract_fft_features(data):
    """
    Extract FFT features per band from multi-temporal data.
    
    Args:
        data: numpy array of shape (bands, rows, cols) where bands follow the pattern:
              t1_B1, t1_B2, ..., t1_B12, t1_label, t2_B1, ..., t2_B12, t2_label, ...
    
    Returns:
        X: Feature matrix of shape (n_valid_pixels, N_BANDS × N_FFT_COMPONENTS)
        y: Label vector of shape (n_valid_pixels,)
        mask: Boolean mask indicating valid pixels
    """
    bands, rows, cols = data.shape
    
    # Identify label band indices (every N_BANDS+1 band)
    label_indices = np.arange(N_BANDS, bands, N_BANDS + 1)
    
    # Extract labels (using first label band, all should be the same)
    y = data[label_indices[0]].reshape(-1)
    
    # Remove label bands to get only feature bands
    feature_data = np.delete(data, label_indices, axis=0)
    
    # Calculate number of time steps
    n_time_steps = len(label_indices)
    
    print(f"  Time steps: {n_time_steps}, Feature bands: {feature_data.shape[0]}")
    
    # Reshape to (n_pixels, n_time_steps, N_BANDS)
    # feature_data shape: (n_time_steps * N_BANDS, rows, cols)
    feature_data = feature_data.reshape(n_time_steps, N_BANDS, rows, cols)
    feature_data = feature_data.transpose(2, 3, 1, 0)  # (rows, cols, N_BANDS, n_time_steps)
    feature_data = feature_data.reshape(-1, N_BANDS, n_time_steps)  # (n_pixels, N_BANDS, n_time_steps)
    
    n_pixels = feature_data.shape[0]
    
    # Create mask for valid pixels (no NaN values and not ground class)
    # Check for NaN across all bands and time steps
    has_nan = np.isnan(feature_data).any(axis=(1, 2))  # (n_pixels,)
    is_ground = (y == 3)
    mask = ~has_nan & ~is_ground
    
    # Filter to valid pixels only
    feature_data = feature_data[mask]
    y = y[mask]
    
    n_valid = feature_data.shape[0]
    print(f"  Valid pixels (no NaN, no ground): {n_valid} / {n_pixels}")
    
    # Prepare output array for FFT features
    X_fft = np.zeros((n_valid, N_BANDS, N_FFT_COMPONENTS), dtype=np.float32)
    
    # Process each band independently
    for band_idx in range(N_BANDS):
        # Extract time series for this band across all pixels
        time_series = feature_data[:, band_idx, :]  # (n_valid, n_time_steps)
        
        # Determine FFT input size to get exactly N_FFT_COMPONENTS output
        # For rfft: output size = (input_size // 2) + 1
        # So we need: input_size = (N_FFT_COMPONENTS - 1) * 2
        fft_input_size = (N_FFT_COMPONENTS - 1) * 2
        
        # Pad or trim to target length for FFT
        if n_time_steps < fft_input_size:
            # Pad with zeros
            padded = np.pad(time_series, ((0, 0), (0, fft_input_size - n_time_steps)), 
                          mode='constant', constant_values=0)
        else:
            # Trim to target length
            padded = time_series[:, :fft_input_size]
        
        # Apply FFT (rfft returns only positive frequencies)
        fft_result = np.fft.rfft(padded, axis=1)
        fft_mag = np.abs(fft_result).astype(np.float32)
        
        # L2 normalize per band
        fft_norm = normalize(fft_mag, norm='l2', axis=1)
        
        X_fft[:, band_idx, :] = fft_norm
    
    # Concatenate all bands: (n_valid, N_BANDS * N_FFT_COMPONENTS)
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
        y_list.append(y.astype(int))

# Stack training data
X_train = np.vstack(X_list)
y_train = np.concatenate(y_list).astype(int)

print(f"\nTotal training samples: {X_train.shape[0]}")
print(f"Feature dimensions: {X_train.shape[1]}")
print("Training label distribution:", {CLASS_NAMES[i]: count for i, count in enumerate(np.bincount(y_train))})

# =====================================================
# 4. APPLY PCA TO REDUCE TO 2 DIMENSIONS
# =====================================================

print("\n" + "="*60)
print("APPLYING PCA")
print("="*60)

# Initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)

# Fit and transform the training data
print("Fitting PCA on training data...")
X_pca = pca.fit_transform(X_train)

# Print explained variance
print(f"\nExplained variance ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# =====================================================
# 5. VISUALIZE PCA RESULTS
# =====================================================

print("\nGenerating PCA visualization...")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: All points with transparency
ax1 = axes[0]
for label_idx, label_name in enumerate(CLASS_NAMES):
    mask = y_train == label_idx
    ax1.scatter(
        X_pca[mask, 0], 
        X_pca[mask, 1],
        c=CLASS_COLORS[label_idx],
        label=f'{label_name} (n={mask.sum()})',
        alpha=0.4,
        s=10,
        edgecolors='none'
    )

ax1.set_xlabel('PC1 (First Principal Component)', fontsize=12)
ax1.set_ylabel('PC2 (Second Principal Component)', fontsize=12)
ax1.set_title('PCA of FFT Features - All Points', fontsize=14, fontweight='bold')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Downsampled for clarity (max 2000 points per class)
ax2 = axes[1]
max_points_per_class = 2000

for label_idx, label_name in enumerate(CLASS_NAMES):
    mask = y_train == label_idx
    indices = np.where(mask)[0]
    
    # Downsample if needed
    if len(indices) > max_points_per_class:
        indices = np.random.choice(indices, max_points_per_class, replace=False)
    
    ax2.scatter(
        X_pca[indices, 0], 
        X_pca[indices, 1],
        c=CLASS_COLORS[label_idx],
        label=f'{label_name} (n={len(indices)})',
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

ax2.set_xlabel('PC1 (First Principal Component)', fontsize=12)
ax2.set_ylabel('PC2 (Second Principal Component)', fontsize=12)
ax2.set_title('PCA of FFT Features - Downsampled View', fontsize=14, fontweight='bold')
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\ope4\\Desktop\\PCA_FFT_Visualization.png', dpi=300, bbox_inches='tight')
print("Saved: C:\\Users\\ope4\\Desktop\\PCA_FFT_Visualization.png")
plt.show()

# =====================================================
# 6. ADDITIONAL ANALYSIS
# =====================================================

print("\n" + "="*60)
print("ADDITIONAL PCA STATISTICS")
print("="*60)

# Compute per-class statistics in PCA space
for label_idx, label_name in enumerate(CLASS_NAMES):
    mask = y_train == label_idx
    pca_subset = X_pca[mask]
    
    print(f"\n{label_name}:")
    print(f"  Count: {mask.sum()}")
    print(f"  PC1 - Mean: {pca_subset[:, 0].mean():.4f}, Std: {pca_subset[:, 0].std():.4f}")
    print(f"  PC2 - Mean: {pca_subset[:, 1].mean():.4f}, Std: {pca_subset[:, 1].std():.4f}")

# Save PCA-transformed data
output_file = 'C:\\Users\\ope4\\Desktop\\PCA_FFT_Features.npz'
np.savez(
    output_file,
    X_pca=X_pca,
    y_train=y_train,
    explained_variance_ratio=pca.explained_variance_ratio_,
    class_names=CLASS_NAMES
)
print(f"\n✅ PCA-transformed data saved to: {output_file}")
