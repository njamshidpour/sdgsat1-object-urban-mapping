import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import match_histograms
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

def resample_single_band(band_path, target_profile):
    """
    Resample a single 40m band to match the 10m Pan target grid and profile.
    """
    with rasterio.open(band_path) as src:
        data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

    resampled_data = np.zeros(
        (target_profile['height'], target_profile['width']), 
        dtype=np.float32
    )

    reproject(
        source=data,
        destination=resampled_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_profile['transform'],
        dst_crs=target_profile['crs'],
        resampling=Resampling.bilinear
    )

    return resampled_data

def pansharpen_hsv(red_band, green_band, blue_band, pan_array):
    """
    Perform HSV-based pan-sharpening on separate R, G, B bands with histogram matching.
    """
    # Stack individual bands into an RGB array (H, W, 3)
    rgb_array = np.stack([red_band, green_band, blue_band], axis=-1)
    
    # Normalize RGB to 0-1 range for skimage processing
    max_val = np.nanmax(rgb_array)
    if max_val == 0: 
        max_val = 1
    rgb_normalized = np.clip(rgb_array / max_val, 0, 1)
    
    # Convert RGB to HSV space
    hsv_image = rgb2hsv(rgb_normalized)
    
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]
    
    # Normalize Pan band to 0-1 scale
    pan_max = np.nanmax(pan_array)
    if pan_max == 0: 
        pan_max = 1
    pan_normalized = np.clip(pan_array / pan_max, 0, 1)
    
    # Match Pan histogram to the original Intensity/Value channel
    pan_matched = match_histograms(pan_normalized, value)
    
    # Replace Value channel with histogram-matched Pan band
    hsv_pansharpened = np.stack([hue, saturation, pan_matched], axis=-1)
    
    # Convert back to RGB space
    rgb_pansharpened = hsv2rgb(hsv_pansharpened)
    
    # Scale back to original radiance/DN values
    rgb_final = (rgb_pansharpened * max_val).astype(np.float32)
    
    # Transpose to (Bands, Height, Width) format for Rasterio saving
    rgb_final_transposed = np.moveaxis(rgb_final, -1, 0)
    
    return rgb_final_transposed

if __name__ == "__main__":
    # Inputs: Aligned 10m Pan band and individual 40m Red, Green, Blue bands
    ntl_folder = Path(__file__).resolve().parent / "output_corrected_bands"
    aligned_pan_path = os.path.join(ntl_folder,'Pan_HDR_Radiance.tif')

    red_40m_path = os.path.join(ntl_folder,'Red_Radiance_salt_pepper.tif')
    green_40m_path = os.path.join(ntl_folder,'Green_Radiance_salt_pepper.tif')
    blue_40m_path = os.path.join(ntl_folder,'Blue_Radiance_salt_pepper.tif')

    out_red_10m_path = os.path.join(ntl_folder,'Red_Radiance_salt_pepper_10m.tif')
    out_green_10m_path = os.path.join(ntl_folder,'Green_Radiance_salt_pepper_10m.tif')
    out_blue_10m_path = os.path.join(ntl_folder,'Blue_Radiance_salt_pepper_10m.tif')
    
    final_pansharpened_path = os.path.join(ntl_folder, "SDGSAT1_RGB_Pansharpened_10m.tif")

    # Step 1: Read Pan band and target metadata
    with rasterio.open(aligned_pan_path) as pan_src:
        pan_data = pan_src.read(1)
        pan_profile = pan_src.profile


    # Step 2: Resample each 40m band individually to match 10m Pan grid
    print("Resampling 40m Red band to 10m grid...")
    red_10m = resample_single_band(red_40m_path, pan_profile)

    print("Resampling 40m Green band to 10m grid...")
    green_10m = resample_single_band(green_40m_path, pan_profile)

    print("Resampling 40m Blue band to 10m grid...")
    blue_10m = resample_single_band(blue_40m_path, pan_profile)

    # Step 3: Perform HSV Pan-sharpening
    print("Performing HSV Pan-sharpening...")
    pansharpened_rgb = pansharpen_hsv(red_10m, green_10m, blue_10m, pan_data)

    # Step 4: Save final 3-band output raster
    print("Saving final 10m 3-band Pansharpened image...")
    out_profile = pan_profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'float32',
    })

    with rasterio.open(out_red_10m_path, 'w', **out_profile) as dst:
        dst.write(pansharpened_rgb[0], 1)

    with rasterio.open(out_green_10m_path, 'w', **out_profile) as dst:
        dst.write(pansharpened_rgb[1], 1)

    with rasterio.open(out_blue_10m_path, 'w', **out_profile) as dst:
        dst.write(pansharpened_rgb[2], 1)

    print("Individual 10m bands saved successfully.")
