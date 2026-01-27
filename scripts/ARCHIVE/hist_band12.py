#!/usr/bin/env python3
import sys, os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# -------- settings --------
TIF = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\Train\CLIPPED\DROUGHT_TRAIN_CLIPPED\DROUGHT_TRAIN_CLIPPED.tif"
BAND = 13
BINS = 256
OUTDIR = "outputs"
# --------------------------

def main(fp=TIF, band=BAND, bins=BINS):
    with rasterio.open(fp) as src:
        if src.count < band:
            raise ValueError(f"Only {src.count} bands in file")

        data = src.read(band).astype("float32")
        nodata = src.nodatavals[band-1]

        if nodata is not None:
            data[data == nodata] = np.nan

        valid = data[np.isfinite(data)]
        if valid.size == 0:
            raise ValueError("No valid pixels")

        print(f"Band {band}")
        print(f"  Mean   : {valid.mean():.4f}")
        print(f"  Median : {np.median(valid):.4f}")
        print(f"  Std    : {valid.std():.4f}")
        print(f"  Pixels : {valid.size}")

        vals, counts = np.unique(valid.astype(int), return_counts=True)

        plt.figure(figsize=(7,4))
        if vals.size <= 50:
            plt.bar(vals.astype(str), counts)
            plt.xlabel("Class value")
        else:
            plt.hist(valid, bins=bins)
            plt.xlabel("Value")

        plt.ylabel("Pixel count")
        plt.title(f"{os.path.basename(fp)} – Band {band}")
        os.makedirs(OUTDIR, exist_ok=True)
        out = f"{OUTDIR}/band{band}_hist_droughtTrain.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
