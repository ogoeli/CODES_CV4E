#file = open("DFB_TEST_Sentinel_2_NETCDF.tif")


# ...existing code...
import rasterio
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\ope4\OneDrive - Northern Arizona University\Desktop\RESEARCH\PRO_DEVE\CV4E\GitIgnore\SENTINEL_TIME_SERIES\ROI\TEST\CLIPPED\DROUGHT_TEST_CLIPPED\DROUGHT_TEST_CLIPPED.tif"

with rasterio.open(path) as src:
    print(src.meta)            # basic metadata
    arr = src.read()           # shape: (bands, rows, cols)
    band1 = src.read(1)        # first band


# Time series for one pixel from one band across multiple dates
n_bands = 13
band=1
i, j = 18, 19  # pixel coordinates
time_series = arr[band::n_bands, i, j]  # shape: (bands,)

plt.plot(time_series, marker='o')
plt.title(f"Time Series for Pixel ({i}, {j}) - Band {band}")
plt.xlabel("Time Point")
plt.ylabel("Reflectance")
plt.show()

##for loop to loop through the pixels cordinate's band 1 - 13

n_bands = 13
i, j = 18, 19  # pixel coordinates

plt.figure()
for band in range(n_bands-1):
    time_series = arr[band::n_bands, i, j]
    plt.plot(time_series, marker='o', label=f"Band {band}")
    plt.title(f"Band")

##check the class of the cordinates. 
band = 12
time_series = arr[band::n_bands, i, j]
time_series = 4000*time_series
plt.plot(time_series, marker='o', label=f"Band {band}")

plt.title(f"Time Series for Pixel ({i}, {j}) – All Bands")
plt.xlabel("Time Point")
plt.ylabel("Reflectance")
plt.legend(ncol=2, fontsize=8)
plt.show()
