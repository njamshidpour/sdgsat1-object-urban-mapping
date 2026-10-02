import numpy as np
import rasterio
import os
from pathlib import Path
import argparse

def create_pan_hdr(ph_path, pl_path, output_hdr_path):
    """
    Synthesize High Dynamic Range (HDR) Panchromatic band by 50% weighted average of PH and PL.
    """
    with rasterio.open(ph_path) as ph_src, rasterio.open(pl_path) as pl_src:
        ph_data = ph_src.read(1).astype(np.float32)
        pl_data = pl_src.read(1).astype(np.float32)
        
        # Extract profile from PH (identical to PL)
        profile = ph_src.profile.copy()
        
        # Handle NoData values if present
        nodata_val = ph_src.nodata
        if nodata_val is not None:

            ph_data[ph_data == nodata_val] = np.nan
            pl_data[pl_data == nodata_val] = np.nan

    # Calculate 50% weighted average
    hdr_data = 0.5 * ph_data + 0.5 * pl_data

    # Update output profile metadata
    profile.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan if nodata_val is None else nodata_val
    })

    # Save synthesized Pan HDR band
    with rasterio.open(output_hdr_path, 'w', **profile) as dst:
        dst.write(hdr_data, 1)

    print(f"Pan HDR band successfully created at: {output_hdr_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average PH and PL radiance with equal weights.")
    ntl_folder = Path(__file__).resolve().parent / "output_corrected_bands"
    parser.add_argument("--ph", type=Path, default=ntl_folder / "PH_Radiance_salt_pepper_coregistered.tif")
    parser.add_argument("--pl", type=Path, default=ntl_folder / "PL_Radiance_salt_pepper_coregistered.tif")
    parser.add_argument("--output", type=Path, default=ntl_folder / "Pan_HDR_Radiance.tif")
    args = parser.parse_args()
    ph_path = args.ph
    pl_path = args.pl
    hdr_output_path = args.output

    create_pan_hdr(ph_path, pl_path, hdr_output_path)
