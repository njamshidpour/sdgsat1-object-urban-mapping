import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.decomposition import PCA
import cv2
import os
from pathlib import Path
import argparse

def create_synthetic_pan_sentinel2(b2_path, b3_path, b4_path, b8_path):
    """
    Generate synthetic panchromatic band from Sentinel-2 10m bands using PCA
    """
    with rasterio.open(b2_path) as src:
        profile = src.profile
        shape = src.shape
        b2 = src.read(1)

    with rasterio.open(b3_path) as src:
        b3 = src.read(1)
    with rasterio.open(b4_path) as src:
        b4 = src.read(1)
    with rasterio.open(b8_path) as src:
        b8 = src.read(1)

    # Stack 10m bands
    stacked_bands = np.stack([b2, b3, b4, b8], axis=-1)
    h, w, c = stacked_bands.shape
    flat_bands = stacked_bands.reshape(-1, c)

    # Mask zero or valid pixels
    mask = np.all(flat_bands > 0, axis=1)
    pc1_flat = np.zeros(flat_bands.shape[0], dtype=np.float32)

    # Apply PCA on valid pixels
    pca = PCA(n_components=1)
    pc1_valid = pca.fit_transform(flat_bands[mask])
    pc1_flat[mask] = pc1_valid.squeeze()

    # Reshape back to image dimensions
    synthetic_pan = pc1_flat.reshape(h, w)
    
    # Normalize to 8-bit range for processing
    synthetic_pan_norm = cv2.normalize(synthetic_pan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return synthetic_pan_norm, profile

def align_sdgsat_to_sentinel2(sdgsat_pan_path, s2_profile, output_path):
    """
    Reproject and align SDGSAT-1 Pan to Sentinel-2 CRS and transform
    """
    with rasterio.open(sdgsat_pan_path) as src:
        sdgsat_data = src.read(1)
        sdgsat_crs = src.crs
        sdgsat_transform = src.transform

    resampled_sdgsat = np.zeros((s2_profile['height'], s2_profile['width']), dtype=np.float32)

    reproject(
        source=sdgsat_data,
        destination=resampled_sdgsat,
        src_transform=sdgsat_transform,
        src_crs=sdgsat_crs,
        dst_transform=s2_profile['transform'],
        dst_crs=s2_profile['crs'],
        resampling=Resampling.bilinear
    )

    out_profile = s2_profile.copy()
    out_profile.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1
    })

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(resampled_sdgsat, 1)

    print(f"Initial georeferenced image saved at: {output_path}")
    return resampled_sdgsat

def fine_coregistration(ref_img, sensed_img):
    """
    Perform sub-pixel coregistration using Phase Correlation in OpenCV
    """
    ref_32 = ref_img.astype(np.float32)
    sensed_32 = sensed_img.astype(np.float32)

    (dx, dy), response = cv2.phaseCorrelate(ref_32, sensed_32)
    print(f"Detected offset: Shift X = {dx:.3f} px, Shift Y = {dy:.3f} px")

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned_img = cv2.warpAffine(sensed_img, M, (sensed_img.shape[1], sensed_img.shape[0]))

    return aligned_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-register one SDGSAT-1 PAN radiance band to Sentinel-2.")
    parser.add_argument("--band", choices=("PL", "PH"), default="PL")
    args = parser.parse_args()
    # Sentinel-2 10m bands
    optic_folder = Path(__file__).resolve().parent / "optic_data"
    b2_path = os.path.join(optic_folder,"Tehran_Raw_B2_10m.tif")
    b3_path = os.path.join(optic_folder,"Tehran_Raw_B3_10m.tif")
    b4_path = os.path.join(optic_folder,"Tehran_Raw_B4_10m.tif")
    b8_path = os.path.join(optic_folder,"Tehran_Raw_B8_10m.tif")

    # SDGSAT-1 Pan band
    ntl_folder = Path(__file__).resolve().parent / "output_corrected_bands"
    sdgsat_pan_path = os.path.join(ntl_folder, f'{args.band}_Radiance_salt_pepper.tif')
    output_aligned_path = os.path.join(ntl_folder, f'{args.band}_Radiance_salt_pepper_aligned.tif')

    # Step 1: Create synthetic Pan band from Sentinel-2
    s2_synthetic_pan, s2_profile = create_synthetic_pan_sentinel2(b2_path, b3_path, b4_path, b8_path)

    # Step 2: Reproject and resample SDGSAT-1 to Sentinel-2 grid
    resampled_sdgsat = align_sdgsat_to_sentinel2(sdgsat_pan_path, s2_profile, output_aligned_path)

    # Step 3: Fine coregistration using Phase Correlation
    aligned_sdgsat = fine_coregistration(s2_synthetic_pan, resampled_sdgsat)

    # Save final output
    final_output_path = os.path.join(ntl_folder, f'{args.band}_Radiance_salt_pepper_coregistered.tif')
    out_profile = s2_profile.copy()
    out_profile.update({'driver': 'GTiff', 'dtype': 'float32', 'count': 1})
    
    with rasterio.open(final_output_path, 'w', **out_profile) as dst:
        dst.write(aligned_sdgsat.astype(np.float32), 1)

    print("Coregistration and reprojection successfully completed.")
