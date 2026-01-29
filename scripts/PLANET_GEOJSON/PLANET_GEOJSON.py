import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import os

# ===============================
# CONFIGURATION
# ===============================

# Planet image paths (4 dates)
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_4_psscene_analytic_sr_udm2\DFB_TEST_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_3_psscene_analytic_sr_udm2\DFB_TEST_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_2_psscene_analytic_sr_udm2\DFB_TEST_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_1_psscene_analytic_sr_udm2\DFB_TEST_1.tif"
]

# GeoJSON files with class mapping: (filepath, class_id)
geojson_files = [
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TEST.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TRAIN.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TEST.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TRAIN.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TEST.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TRAIN.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TEST.geojson", 3),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TRAIN.geojson", 3),
]

# Output file
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\MERGED\DFB_TEST.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# ===============================
# STEP 1: Reference Planet image
# ===============================
with rasterio.open(planet_files[0]) as src:
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    meta = src.meta.copy()

# ===============================
# STEP 2: Rasterize GeoJSON annotations
# ===============================
all_shapes = []
for file, class_id in geojson_files:
    gdf = gpd.read_file(file).to_crs(crs)  # match CRS to Planet
    shapes = [(geom, class_id) for geom in gdf.geometry]
    all_shapes.extend(shapes)

label_raster = rasterize(
    all_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# ===============================
# STEP 3: Read Planet images and stack bands + repeated label
# ===============================
all_bands = []
band_names = []

for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        planet_data = src.read([1, 2, 3, 4]).astype(np.float32)

        # Resample label raster if needed
        if (planet_data.shape[1], planet_data.shape[2]) != label_raster.shape:
            resampled_label = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.uint8)
            rasterio.warp.reproject(
                source=label_raster,
                destination=resampled_label,
                src_transform=transform,
                src_crs=crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            label_band = resampled_label
        else:
            label_band = label_raster

        # Stack Planet bands + label as 5th band
        combined = np.vstack([planet_data, label_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_Label")

# ===============================
# STEP 4: Stack all dates
# ===============================
stacked_data = np.vstack(all_bands)
meta.update(count=stacked_data.shape[0], dtype=np.float32)

# ===============================
# STEP 5: Save raster
# ===============================
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved multi-date Planet stack with repeated label bands: {output_stack}")






#-----------------------------------------------------------------------------
# DFB Train
#-----------------------------------------------------------------------

# ===============================
# CONFIGURATION
# ===============================

# Planet image paths (4 dates)
planet_files = [
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_4_psscene_analytic_sr_udm2\DFB_TRAIN_4.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_3_psscene_analytic_sr_udm2\DFB_TRAIN_3.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_2_psscene_analytic_sr_udm2\DFB_TRAIN_2.tif",
    r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_1_psscene_analytic_sr_udm2\DFB_TRAIN_1.tif"
]

# GeoJSON files with class mapping: (filepath, class_id)
geojson_files = [
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TEST.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TRAIN.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TEST.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TRAIN.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TEST.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TRAIN.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TEST.geojson", 3),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TRAIN.geojson", 3),
]

# Output file
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\MERGED\DFB_TRAIN.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# ===============================
# STEP 1: Reference Planet image
# ===============================
with rasterio.open(planet_files[0]) as src:
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    meta = src.meta.copy()

# ===============================
# STEP 2: Rasterize GeoJSON annotations
# ===============================
all_shapes = []
for file, class_id in geojson_files:
    gdf = gpd.read_file(file).to_crs(crs)  # match CRS to Planet
    shapes = [(geom, class_id) for geom in gdf.geometry]
    all_shapes.extend(shapes)

label_raster = rasterize(
    all_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# ===============================
# STEP 3: Read Planet images and stack bands + repeated label
# ===============================
all_bands = []
band_names = []

for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        planet_data = src.read([1, 2, 3, 4]).astype(np.float32)

        # Resample label raster if needed
        if (planet_data.shape[1], planet_data.shape[2]) != label_raster.shape:
            resampled_label = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.uint8)
            rasterio.warp.reproject(
                source=label_raster,
                destination=resampled_label,
                src_transform=transform,
                src_crs=crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            label_band = resampled_label
        else:
            label_band = label_raster

        # Stack Planet bands + label as 5th band
        combined = np.vstack([planet_data, label_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_Label")

# ===============================
# STEP 4: Stack all dates
# ===============================
stacked_data = np.vstack(all_bands)
meta.update(count=stacked_data.shape[0], dtype=np.float32)

# ===============================
# STEP 5: Save raster
# ===============================
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved multi-date Planet stack with repeated label bands: {output_stack}")




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


# GeoJSON files with class mapping: (filepath, class_id)
geojson_files = [
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TEST.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TRAIN.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TEST.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TRAIN.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TEST.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TRAIN.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TEST.geojson", 3),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TRAIN.geojson", 3),
]

# Output file
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\MERGED\DROUGHT_TRAIN.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# ===============================
# STEP 1: Reference Planet image
# ===============================
with rasterio.open(planet_files[0]) as src:
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    meta = src.meta.copy()

# ===============================
# STEP 2: Rasterize GeoJSON annotations
# ===============================
all_shapes = []
for file, class_id in geojson_files:
    gdf = gpd.read_file(file).to_crs(crs)  # match CRS to Planet
    shapes = [(geom, class_id) for geom in gdf.geometry]
    all_shapes.extend(shapes)

label_raster = rasterize(
    all_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# ===============================
# STEP 3: Read Planet images and stack bands + repeated label
# ===============================
all_bands = []
band_names = []

for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        planet_data = src.read([1, 2, 3, 4]).astype(np.float32)

        # Resample label raster if needed
        if (planet_data.shape[1], planet_data.shape[2]) != label_raster.shape:
            resampled_label = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.uint8)
            rasterio.warp.reproject(
                source=label_raster,
                destination=resampled_label,
                src_transform=transform,
                src_crs=crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            label_band = resampled_label
        else:
            label_band = label_raster

        # Stack Planet bands + label as 5th band
        combined = np.vstack([planet_data, label_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_Label")

# ===============================
# STEP 4: Stack all dates
# ===============================
stacked_data = np.vstack(all_bands)
meta.update(count=stacked_data.shape[0], dtype=np.float32)

# ===============================
# STEP 5: Save raster
# ===============================
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved multi-date Planet stack with repeated label bands: {output_stack}")





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


# GeoJSON files with class mapping: (filepath, class_id)
geojson_files = [
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TEST.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\HEALTHY_TRAIN.geojson", 0),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TEST.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DFB_TRAIN.geojson", 1),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TEST.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\DROUGHT_TRAIN.geojson", 2),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TEST.geojson", 3),
    (r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\CV4E_GeoJson\GROUND_TRAIN.geojson", 3),
]

# Output file
output_stack = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\MERGED\DROUGHT_TEST.tif"
os.makedirs(os.path.dirname(output_stack), exist_ok=True)

# ===============================
# STEP 1: Reference Planet image
# ===============================
with rasterio.open(planet_files[0]) as src:
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    meta = src.meta.copy()

# ===============================
# STEP 2: Rasterize GeoJSON annotations
# ===============================
all_shapes = []
for file, class_id in geojson_files:
    gdf = gpd.read_file(file).to_crs(crs)  # match CRS to Planet
    shapes = [(geom, class_id) for geom in gdf.geometry]
    all_shapes.extend(shapes)

label_raster = rasterize(
    all_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

# ===============================
# STEP 3: Read Planet images and stack bands + repeated label
# ===============================
all_bands = []
band_names = []

for idx, f in enumerate(planet_files, start=1):
    with rasterio.open(f) as src:
        planet_data = src.read([1, 2, 3, 4]).astype(np.float32)

        # Resample label raster if needed
        if (planet_data.shape[1], planet_data.shape[2]) != label_raster.shape:
            resampled_label = np.empty((planet_data.shape[1], planet_data.shape[2]), dtype=np.uint8)
            rasterio.warp.reproject(
                source=label_raster,
                destination=resampled_label,
                src_transform=transform,
                src_crs=crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            label_band = resampled_label
        else:
            label_band = label_raster

        # Stack Planet bands + label as 5th band
        combined = np.vstack([planet_data, label_band[np.newaxis, :, :]])
        all_bands.append(combined)

        # Band names
        for b in range(1, 5):
            band_names.append(f"Date{idx}_B{b}")
        band_names.append(f"Date{idx}_Label")

# ===============================
# STEP 4: Stack all dates
# ===============================
stacked_data = np.vstack(all_bands)
meta.update(count=stacked_data.shape[0], dtype=np.float32)

# ===============================
# STEP 5: Save raster
# ===============================
with rasterio.open(output_stack, 'w', **meta) as dst:
    dst.write(stacked_data)
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print(f"✅ Saved multi-date Planet stack with repeated label bands: {output_stack}")




