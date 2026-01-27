import rasterio
import numpy as np
import os
from rasterio.enums import Resampling
import rasterio.warp

# ------------------------------
# Planet images (order: 4 -> 3 -> 2 -> 1)
# ------------------------------
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_4_psscene_analytic_8b_sr_udm2\DFB_TRAIN_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_3_psscene_analytic_8b_sr_udm2\DFB_TRAIN_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_2_psscene_analytic_8b_sr_udm2\DFB_TRAIN_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_1_psscene_analytic_8b_sr_udm2\DFB_TRAIN_1.tif"
]

# ------------------------------
# NAIP classification raster (0-3)
# ------------------------------
naip_file = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_3M\DFB_TRAIN_3M.tif"

# ------------------------------
# Output 36-band raster
# ------------------------------
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\MERGED\DFB_TRAIN.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# Delete existing output file if present
if os.path.exists(output_stack):
    os.remove(output_stack)

# ------------------------------
# Read NAIP as float32 to preserve classes 0-3
# ------------------------------
with rasterio.open(naip_file) as src:
    naip_data = src.read(1).astype(np.float32)
    naip_meta = src.meta

all_bands = []
band_names = []

# ------------------------------
# Loop over Planet images
# ------------------------------
for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        # Planet bands as float32
        planet_data = src.read().astype(np.float32)

        # Resample NAIP if needed
        if planet_data.shape[1:] != naip_data.shape:
            naip_resampled = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.float32)
            rasterio.warp.reproject(
                source=naip_data,
                destination=naip_resampled,
                src_transform=naip_meta['transform'],
                src_crs=naip_meta['crs'],
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            naip_band = naip_resampled
        else:
            naip_band = naip_data

        # Stack Planet 8 bands + NAIP as 9th band
        combined = np.vstack([planet_data, naip_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 9):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_NAIP")

# ------------------------------
# Stack all 4 dates → 36 bands
# ------------------------------
stacked_data = np.vstack(all_bands)

# ------------------------------
# Metadata: use first Planet image
# ------------------------------
with rasterio.open(planet_files[0]) as src:
    meta = src.meta

meta.update(count=stacked_data.shape[0], dtype=np.float32)  # important: float32

# ------------------------------
# Save raster
# ------------------------------
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)

    # Add band descriptions
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved 36-band raster with NAIP preserved 0-3: {output_stack}")





#####---------------------------------------------------------------------------------------------------
# DFB TEST
#------------------------------------------------------------------------------------------------------

# ------------------------------
# Planet images (order: 4 -> 3 -> 2 -> 1)
# ------------------------------
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_4_psscene_analytic_8b_sr_udm2\DFB_TEST_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_3_psscene_analytic_8b_sr_udm2\DFB_TEST_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_2_psscene_analytic_8b_sr_udm2\DFB_TEST_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_1_psscene_analytic_8b_sr_udm2\DFB_TEST_1.tif"
]

# ------------------------------
# NAIP classification raster (0-3)
# ------------------------------
naip_file = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_3M\DFB_TEST_3M.tif"

# ------------------------------
# Output 36-band raster
# ------------------------------
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\MERGED\DFB_TEST.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# Delete existing output file if present
if os.path.exists(output_stack):
    os.remove(output_stack)

# ------------------------------
# Read NAIP as float32 to preserve classes 0-3
# ------------------------------
with rasterio.open(naip_file) as src:
    naip_data = src.read(1).astype(np.float32)
    naip_meta = src.meta

all_bands = []
band_names = []

# ------------------------------
# Loop over Planet images
# ------------------------------
for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        # Planet bands as float32
        planet_data = src.read().astype(np.float32)

        # Resample NAIP if needed
        if planet_data.shape[1:] != naip_data.shape:
            naip_resampled = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.float32)
            rasterio.warp.reproject(
                source=naip_data,
                destination=naip_resampled,
                src_transform=naip_meta['transform'],
                src_crs=naip_meta['crs'],
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            naip_band = naip_resampled
        else:
            naip_band = naip_data

        # Stack Planet 8 bands + NAIP as 9th band
        combined = np.vstack([planet_data, naip_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 9):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_NAIP")

# ------------------------------
# Stack all 4 dates → 36 bands
# ------------------------------
stacked_data = np.vstack(all_bands)

# ------------------------------
# Metadata: use first Planet image
# ------------------------------
with rasterio.open(planet_files[0]) as src:
    meta = src.meta

meta.update(count=stacked_data.shape[0], dtype=np.float32)  # important: float32

# ------------------------------
# Save raster
# ------------------------------
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)

    # Add band descriptions
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved 36-band raster with NAIP preserved 0-3: {output_stack}")



##--------------------------------------------------------------------------------------------------------
# DROUGHT TRAIN
#--------------------------------------------------------------------------------------------------------

# ------------------------------
# Planet images (order: 4 -> 3 -> 2 -> 1)
# ------------------------------
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_4_psscene_analytic_sr_udm2\DROUGHT_TRAIN_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_3_psscene_analytic_sr_udm2\DROUGHT_TRAIN_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_2_psscene_analytic_8b_sr_udm2\DROUGHT_TRAIN_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_1_psscene_analytic_8b_sr_udm2\DROUGHT_TRAIN_1.tif"
]

# ------------------------------
# NAIP classification raster (0-3)
# ------------------------------
naip_file = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_3M\DROUGHT_TRAIN_3M.tif"

# ------------------------------
# Output 36-band raster
# ------------------------------
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\MERGED\DROUGHT_TRAIN.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# Delete existing output file if present
if os.path.exists(output_stack):
    os.remove(output_stack)

# ------------------------------
# Read NAIP as float32 to preserve classes 0-3
# ------------------------------
with rasterio.open(naip_file) as src:
    naip_data = src.read(1).astype(np.float32)
    naip_meta = src.meta

all_bands = []
band_names = []

# ------------------------------
# Loop over Planet images
# ------------------------------
for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        # Planet bands as float32
        planet_data = src.read().astype(np.float32)

        # Resample NAIP if needed
        if planet_data.shape[1:] != naip_data.shape:
            naip_resampled = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.float32)
            rasterio.warp.reproject(
                source=naip_data,
                destination=naip_resampled,
                src_transform=naip_meta['transform'],
                src_crs=naip_meta['crs'],
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            naip_band = naip_resampled
        else:
            naip_band = naip_data

        # Stack Planet 8 bands + NAIP as 9th band
        combined = np.vstack([planet_data, naip_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_NAIP")

# ------------------------------
# Stack all 4 dates → 36 bands
# ------------------------------
stacked_data = np.vstack(all_bands)

# ------------------------------
# Metadata: use first Planet image
# ------------------------------
with rasterio.open(planet_files[0]) as src:
    meta = src.meta

meta.update(count=stacked_data.shape[0], dtype=np.float32)  # important: float32

# ------------------------------
# Save raster
# ------------------------------
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)

    # Add band descriptions
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved 36-band raster with NAIP preserved 0-3: {output_stack}")




##--------------------------------------------------------------------------------------------------------
# DROUGHT TEST
#--------------------------------------------------------------------------------------------------------

# ------------------------------
# Planet images (order: 4 -> 3 -> 2 -> 1)
# ------------------------------
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_4_psscene_analytic_sr_udm2\DROUGHT_TEST_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_3_psscene_analytic_sr_udm2\DROUGHT_TEST_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_2_psscene_analytic_sr_udm2\DROUGHT_TEST_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_1_psscene_analytic_sr_udm2\DROUGHT_TEST_1.tif"
]

# ------------------------------
# NAIP classification raster (0-3)
# ------------------------------
naip_file = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_3M\DROUGHT_TEST_3M.tif"

# ------------------------------
# Output 36-band raster
# ------------------------------
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\MERGED\DROUGHT_TEST.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# Delete existing output file if present
if os.path.exists(output_stack):
    os.remove(output_stack)

# ------------------------------
# Read NAIP as float32 to preserve classes 0-3
# ------------------------------
with rasterio.open(naip_file) as src:
    naip_data = src.read(1).astype(np.float32)
    naip_meta = src.meta

all_bands = []
band_names = []

# ------------------------------
# Loop over Planet images
# ------------------------------
for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        # Planet bands as float32
        planet_data = src.read().astype(np.float32)

        # Resample NAIP if needed
        if planet_data.shape[1:] != naip_data.shape:
            naip_resampled = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.float32)
            rasterio.warp.reproject(
                source=naip_data,
                destination=naip_resampled,
                src_transform=naip_meta['transform'],
                src_crs=naip_meta['crs'],
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            naip_band = naip_resampled
        else:
            naip_band = naip_data

        # Stack Planet 8 bands + NAIP as 9th band
        combined = np.vstack([planet_data, naip_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_NAIP")

# ------------------------------
# Stack all 4 dates → 36 bands
# ------------------------------
stacked_data = np.vstack(all_bands)

# ------------------------------
# Metadata: use first Planet image
# ------------------------------
with rasterio.open(planet_files[0]) as src:
    meta = src.meta

meta.update(count=stacked_data.shape[0], dtype=np.float32)  # important: float32

# ------------------------------
# Save raster
# ------------------------------
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)

    # Add band descriptions
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved 36-band raster with NAIP preserved 0-3: {output_stack}")

