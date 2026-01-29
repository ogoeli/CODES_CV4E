import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import os

# ===============================
# CONFIGURATION
# ===============================

# Each region gets a list of 4 Planet images (dates)
planet_regions = {
    "DFB_TEST": [
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_4_psscene_analytic_sr_udm2\DFB_TEST_4.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_3_psscene_analytic_sr_udm2\DFB_TEST_3.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_2_psscene_analytic_sr_udm2\DFB_TEST_2.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DFB_TEST\DFB_TEST_1_psscene_analytic_sr_udm2\DFB_TEST_1.tif"
    ],
    "DFB_TRAIN": [
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_4_psscene_analytic_sr_udm2\DFB_TRAIN_4.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_3_psscene_analytic_sr_udm2\DFB_TRAIN_3.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_2_psscene_analytic_sr_udm2\DFB_TRAIN_2.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DFB_TRAIN\DFB_TRAIN_1_psscene_analytic_sr_udm2\DFB_TRAIN_1.tif"
    ],
    "DROUGHT_TEST": [
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_4_psscene_analytic_sr_udm2\DROUGHT_TEST_4.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_3_psscene_analytic_sr_udm2\DROUGHT_TEST_3.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_2_psscene_analytic_sr_udm2\DROUGHT_TEST_2.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TEST\DROUGHT_TEST\DROUGHT_TEST_1_psscene_analytic_sr_udm2\DROUGHT_TEST_1.tif"
    ],
    "DROUGHT_TRAIN": [
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_4_psscene_analytic_sr_udm2\DROUGHT_TRAIN_4.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_3_psscene_analytic_sr_udm2\DROUGHT_TRAIN_3.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_2_psscene_analytic_8b_sr_udm2\DROUGHT_TRAIN_2.tif",
        r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\TRAIN\DROUGHT_TRAIN\DROUGHT_TRAIN_1_psscene_analytic_8b_sr_udm2\DROUGHT_TRAIN_1.tif"
    ]
}

# GeoJSONs for all classes (0=Healthy, 1=DFB, 2=Drought, 3=Ground)
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

output_dir = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\PLANET\STACKED_OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# PROCESS EACH REGION
# ===============================
for region_name, planet_files in planet_regions.items():
    print(f"\nProcessing region: {region_name}")

    # Reference metadata from first Planet image
    with rasterio.open(planet_files[0]) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        meta = src.meta.copy()

    # Rasterize labels **once per region**
    all_shapes = []
    for geojson_file, class_id in geojson_files:
        gdf = gpd.read_file(geojson_file).to_crs(crs)
        shapes = [(geom, class_id) for geom in gdf.geometry]
        all_shapes.extend(shapes)

    label_raster = rasterize(
        all_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Stack Planet bands + repeated label for each date
    all_bands = []
    band_names = []

    for idx, planet_file in enumerate(planet_files, start=1):
        with rasterio.open(planet_file) as src:
            planet_data = src.read([1,2,3,4]).astype(np.float32)
            combined = np.vstack([planet_data, label_raster[np.newaxis, :, :]])
            all_bands.append(combined)

            for b in range(1, 5):
                band_names.append(f"Date{idx}_B{b}")
            band_names.append(f"Date{idx}_Label")

    # Stack all dates
    stacked_data = np.vstack(all_bands)
    meta.update(count=stacked_data.shape[0], dtype=np.float32)

    # Save multi-date stack
    output_file = os.path.join(output_dir, f"{region_name}_stack.tif")
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(stacked_data)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)

    print(f"✅ Saved multi-date Planet stack: {output_file}")
