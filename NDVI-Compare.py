import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parse the MTL file to remove the specific variables we need, in this case reflectance
def parse_mtl(mtl_path):
    coeffs = {}
    with open(mtl_path) as f:
        for line in f:
            if ("REFLECTANCE_MULT_BAND" in line or
                "REFLECTANCE_ADD_BAND" in line or
                "SUN_ELEVATION" in line):
                key, val = line.strip().split(" = ")
                coeffs[key] = float(val)
    return coeffs

# Function to align imagery.
def align_to_reference(src_path,ref_path):
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height,ref.width)
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        aligned = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source = src_data,
            destination = aligned,
            src_transform=src.transform,
            src_crs = src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
    return aligned

# Function to calculate the NDVI given the following variables
# red_path: The path to the Band 4 TIF.
# nir_path: The path to the Band 5 TIF.
# mtl_path: The path to the MTL file.
# ref_path: Reference path for proper overlay.

def compute_ndvi(red_path, nir_path, mtl_path, ref_path = None):
    mtl = parse_mtl(mtl_path)
    M4, A4 = mtl["REFLECTANCE_MULT_BAND_4"], mtl["REFLECTANCE_ADD_BAND_4"]
    M5, A5 = mtl["REFLECTANCE_MULT_BAND_5"], mtl["REFLECTANCE_ADD_BAND_5"]

    if ref_path:
        red_dn = align_to_reference(red_path, ref_path)
        nir_dn = align_to_reference(nir_path, ref_path)
    else:
        with rasterio.open(red_path) as src:
            red_dn = src.read(1).astype(np.float32)
        with rasterio.open(nir_path) as src:
            nir_dn = src.read(1).astype(np.float32)

    nodata_mask = (red_dn == 0) | (nir_dn == 0)
    red = M4 * red_dn + A4
    nir = M5 * nir_dn + A5

    red[nodata_mask] = np.nan
    nir[nodata_mask] = np.nan
    return np.divide(
        nir - red,
        nir + red,
        out=np.full_like(nir, np.nan, dtype=np.float32),
        where=(nir + red) !=0
    )
# RP1 holds the newer LANDSAT imagery
RP1 = "./SET1/LC08_L2SP_026029_20260524_20260603_02_T1_SR_B4.TIF"
NIR1 = "./SET1/LC08_L2SP_026029_20260524_20260603_02_T1_SR_B5.TIF"
mtlPath1 = "./SET1/LC08_L2SP_026029_20260524_20260603_02_T1_MTL.txt"

# RP2 holds the older LANDSAT imagery
RP2 = "./SET2/LC08_L2SP_026029_20250708_20250715_02_T1_SR_B4.TIF"
NIR2 = "./SET2/LC08_L2SP_026029_20250708_20250715_02_T1_SR_B5.TIF"
mtlPath2 = "./SET2/LC08_L2SP_026029_20250708_20250715_02_T1_MTL.txt"

ndvi_new = compute_ndvi(RP1, NIR1, mtlPath1)
ndvi_old = compute_ndvi(RP2, NIR2, mtlPath2, ref_path=RP1)

ndvi_percent_change = np.divide(
    ndvi_new - ndvi_old,
    ndvi_old,
    out = np.full_like(ndvi_new, np.nan, dtype=np.float32),
    where=np.abs(ndvi_old) > 0.10
) * 100


ndvi_compare = (ndvi_new - ndvi_old)
ndviVmin, ndviVmax = np.nanpercentile(ndvi_compare, (2, 98))
pctVmin, pctVmax = np.nanpercentile(ndvi_percent_change, (2,98))

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

im0 = axes[0].imshow(ndvi_compare, cmap='RdYlGn', vmin=ndviVmin, vmax=ndviVmax)
axes[0].set_title('NDVI Raw Difference')
axes[0].axis('off')
fig.colorbar(im0,fraction=0.046, pad=0.04)

im1 = axes[1].imshow(ndvi_percent_change, cmap='RdYlGn', vmin=pctVmin, vmax=pctVmax)
axes[1].set_title('NDVI Percentage Change')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("ndvi_sbs.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
im = plt.imshow(ndvi_compare, cmap="RdYlGn", vmin=ndviVmin, vmax=ndviVmax)
cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
plt.title("NDVI")
plt.savefig("ndvi_compare.png", dpi=300)

