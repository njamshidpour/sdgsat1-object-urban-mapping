import os
import numpy as np
import rasterio
from pathlib import Path
ntl_folder = Path(__file__).resolve().parent / "output_corrected_bands"
optic_folder = Path(__file__).resolve().parent / "optic_data"
input_paths = {
    'Pan':  os.path.join(ntl_folder, 'Pan_HDR_Radiance.tif'),
    'R':  os.path.join(ntl_folder,'Red_Radiance_salt_pepper_10m.tif'),
    'G':  os.path.join(ntl_folder,'Green_Radiance_salt_pepper_10m.tif'),
    'B':  os.path.join(ntl_folder,'Blue_Radiance_salt_pepper_10m.tif'),
    'LST':  os.path.join(optic_folder,"Tehran_LST_2023_10m.tif"),
    'NDVI': os.path.join(optic_folder,"Tehran_NDBI_2023_Int16_compressed.tif"),
    'NDWI': os.path.join(optic_folder,"Tehran_NDVI_2023_Int16_compressed.tif"),
    'NDBI': os.path.join(optic_folder,"Tehran_NDWI_2023_Int16_compressed.tif")
}

output_dir = str(Path(__file__).resolve().parent / 'NTL_indices')
os.makedirs(output_dir, exist_ok=True)

def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        meta = src.meta
    return data, meta

def save_raster(data, filename, metadata):
    data = np.where(np.isinf(data), np.nan, data)
    filepath = os.path.join(output_dir, filename)
    with rasterio.open(filepath, 'w', **metadata) as dst:
        dst.write(data, 1)

print("Loading data...")
Pan, meta = read_raster(input_paths['Pan'])
R, _ = read_raster(input_paths['R'])
G, _ = read_raster(input_paths['G'])
B, _ = read_raster(input_paths['B'])
LST, _ = read_raster(input_paths['LST'])
NDVI, _ = read_raster(input_paths['NDVI'])
NDWI, _ = read_raster(input_paths['NDWI'])
NDBI, _ = read_raster(input_paths['NDBI'])

NDVI /= 10000
NDWI /= 10000
NDBI /= 10000
print("NDI:", NDVI.shape)
print("LST:", LST.shape)
print("Pan:", Pan.shape)
print("R", R.shape)

meta.update(dtype=rasterio.float32, nodata=np.nan)

print("Calculating indices...")
with np.errstate(divide='ignore', invalid='ignore'):
    
    BR = 0.2989 * R + 0.5870 * G + 0.1140 * B
    save_raster(BR, 'BR_10m.tif', meta)
    print("- BR saved.")

    NDIGR = (G - R) / (G + R)
    save_raster(NDIGR, 'NDIGR.tif', meta)
    print("- NDIGR saved.")

    NDIBG = (B - G) / (B + G)
    save_raster(NDIBG, 'NDIBG.tif', meta)
    print("- NDIBG saved.")

    SONDI = NDIGR + NDIBG
    save_raster(SONDI, 'SONDI.tif', meta)
    print("- SONDI saved.")

    VANUI = BR * (1 - NDVI)
    save_raster(VANUI, 'VANUI.tif', meta)
    print("- VANUI saved.")

    TVANUI = np.arctan(LST / NDVI) * (2 / np.pi) * BR
    save_raster(TVANUI, 'TVANUI.tif', meta)
    print("- TVANUI saved.")

    ENUI = BR * (1 - NDVI) * (1 - NDWI) * NDBI
    save_raster(ENUI, 'ENUI.tif', meta)
    print("- ENUI saved.")

    NBDI = (Pan - BR) / (Pan + BR)
    save_raster(NBDI, 'NBDI.tif', meta)
    print("- NBDI saved.")

    max_rgb = np.maximum(np.maximum(R, G), B)
    min_rgb = np.minimum(np.minimum(R, G), B)
    NLCSI = (max_rgb - min_rgb) / max_rgb
    save_raster(NLCSI, 'NLCSI.tif', meta)
    print("- NLCSI saved.")

print(f"All indices successfully calculated and saved in the '{output_dir}' directory.")
