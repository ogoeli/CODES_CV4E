#!/usr/bin/env python3
import os
import sys

try:
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    sys.stderr.write("Missing dependencies. Install with: pip install rasterio numpy matplotlib\n")
    sys.exit(2)

SEARCH_DIR = os.path.join("GitIgnore", "SENTINEL_TIME_SERIES")

def find_tifs(path):
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith('.tif')]

def main():
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    tifs = find_tifs(SEARCH_DIR)
    if not tifs and not fp:
        sys.stderr.write(f"No .tif files found in {SEARCH_DIR}\n")
        sys.exit(1)

    if not fp:
        # pick first file that has >=12 bands, otherwise first file
        chosen = None
        for c in tifs:
            try:
                with rasterio.open(c) as src:
                    if src.count >= 12:
                        chosen = c
                        break
            except Exception:
                continue
        if chosen is None:
            chosen = tifs[0]
            print(f"No file with >=12 bands found; using first file: {chosen}")
        fp = chosen

    with rasterio.open(fp) as src:
        print(f"Opened {fp} (bands={src.count})")
        band_idx = 12
        if src.count < band_idx:
            sys.stderr.write(f"File has only {src.count} bands; cannot read band {band_idx}\n")
            sys.exit(1)
        data = src.read(band_idx)
        nodata = None
        if src.nodatavals:
            nodata = src.nodatavals[band_idx-1]
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        data = data.astype('float32')
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            sys.stderr.write("No valid pixels in band\n")
            sys.exit(1)

        plt.figure(figsize=(8,4))
        plt.hist(valid.flatten(), bins=256)
        plt.title(f"Histogram - {os.path.basename(fp)} - Band {band_idx}")
        plt.xlabel("DN / Reflectance")
        plt.ylabel("Pixel count")
        outdir = os.path.join("outputs")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "band12_hist.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        print(f"Saved histogram to {outpath}")

if __name__ == '__main__':
    main()
