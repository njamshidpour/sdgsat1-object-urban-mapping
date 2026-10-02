import os
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from skyfield.api import load, wgs84
import pandas as pd
from skyfield.almanac import fraction_illuminated

def parse_sdgsat1_calibration(xml_path):
    """
    Parses the SDGSAT-1 calibration XML file to extract Gain and Bias.
    Returns a dictionary with band names as keys.
    """
            

    calib_params = {
        'Red': {'gain': 0.00001354, 'bias': 0.0000136754}, 
        'Green': {'gain': 0.00000507, 'bias': 0.000006084},
        'Blue': {'gain': 0.0000099253, 'bias': 0.0000099253},
        'PL': {'gain': 0.00008832, 'bias': 0.0000167808},
        'PH': {'gain': 0.00008757, 'bias':0.0000183897}
    }
  
    return calib_params
def read_moon_data_from_csv(csv_path):
    """
    Reads the locally saved moon parameters from a CSV file.
    Returns the illumination fraction and moon altitude.
    """
    df = pd.read_csv(csv_path)
    # Extract the first row's data
    illumination = df['illumination_fraction'].iloc[0]
    altitude = df['altitude_degrees'].iloc[0]
    return illumination, altitude

def get_moon_parameters(time_utc_str, lat, lon):
    """
    Calculates the lunar illumination fraction and altitude using Skyfield.
    time_utc_str format: 'YYYY-MM-DD HH:MM:SS'
    """
    eph = load('de421.bsp')
    earth, moon = eph['earth'], eph['moon']
    ts = load.timescale()
    
    dt_obj = datetime.strptime(time_utc_str, '%Y-%m-%d %H:%M:%S')
    t = ts.utc(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.second)
     
    illumination = fraction_illuminated(eph, 'moon', t)
    
    observer = earth + wgs84.latlon(lat, lon)
    astrometric = observer.at(t).observe(moon)
    alt, _, _ = astrometric.apparent().altaz()
    
    return illumination, alt.degrees

def align_albedo_to_band(albedo_path, reference_meta):
    """
    Resamples the 10m Albedo raster to match the target band's spatial resolution (e.g., 40m for RGB).
    """
    with rasterio.open(albedo_path) as src:
        albedo_data = src.read(
            1,
            out_shape=(reference_meta['height'], reference_meta['width']),
            resampling=Resampling.bilinear
        )
    return albedo_data.astype(np.float32)

def process_sdgsat1_band(band_path, out_path, gain, bias, albedo_path, moon_fraction, moon_alt, max_lunar_irradiance):
    """
    Applies absolute radiometric calibration and empirical moonlight correction.
    """
    with rasterio.open(band_path) as src:
        meta = src.meta.copy()
        dn_data = src.read(1).astype(np.float32)
        
        radiance = (dn_data * gain) + bias
        
            
        meta.update(dtype=rasterio.float32, nodata=0)
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(radiance, 1)

if __name__ == "__main__":
    ntl_folder = Path(__file__).resolve().parent / "output_corrected_bands"
    optic_folder = Path(__file__).resolve().parent / "NTL_data"

    CALIB_XML_PATH = os.path.join(ntl_folder,'KX10_GIU_calib.xml')
    ALBEDO_RASTER_PATH = os.path.join(optic_folder,'Albedo_10m.tif')
    csv_path = os.path.join(optic_folder,'moon_data.csv')
    
    BANDS = {
        "Red": os.path.join(ntl_folder,'RED_DN_salt_pepper.tif'),
        "Green": os.path.join(ntl_folder,'GREEN_DN_salt_pepper.tif'),
        "Blue": os.path.join(ntl_folder,'BLUE_DN_salt_pepper.tif'),
        "PL": os.path.join(ntl_folder,'PL_DN_salt_pepper.tif'),
        "PH": os.path.join(ntl_folder, 'PH_DN_salt_pepper.tif')
    }
    
    OUTPUT_DIR = str(ntl_folder)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    IMAGE_ACQ_TIME_UTC = "2023-05-12 17:51:16" 
    SCENE_CENTER_LAT = 35.694389 
    SCENE_CENTER_LON = 51.048211
    
    MAX_LUNAR_RAD = 1.5e-4 
    
    try:
        print("1. Parsing SDGSAT-1 Calibration XML...")
        calib_params = parse_sdgsat1_calibration(CALIB_XML_PATH)
        print(f"Extracted Calibration Parameters: {calib_params}\n")
        
        print("2. Calculating Lunar Parameters using Skyfield...")
        moon_frac, moon_alt = read_moon_data_from_csv(csv_path)
        print(f"Moon Illumination Fraction: {moon_frac:.4f}")
        print(f"Moon Altitude: {moon_alt:.2f} degrees\n")
        
        if moon_alt > 0:
            print("Moon is above horizon. Applying Albedo-based Moonlight Correction.")
        else:
            print("Moon is below horizon. Only Radiometric Calibration will be applied.")

        print("\n3. Processing Spectral Bands...")
        for band_name, band_path in BANDS.items():
            if not os.path.exists(band_path):
                print(f"Skipping {band_name}: File not found ({band_path})")
                continue
                
            out_file = os.path.join(OUTPUT_DIR, f"{band_name}_Radiance_salt_pepper.tif")
            
            b_gain = calib_params.get(band_name, calib_params.get('PL'))['gain']
            b_bias = calib_params.get(band_name, calib_params.get('PL'))['bias']
            
            print(f" -> Processing {band_name} (Gain: {b_gain}, Bias: {b_bias})...")
            
            process_sdgsat1_band(
                band_path=band_path,
                out_path=out_file,
                gain=b_gain,
                bias=b_bias,
                albedo_path=ALBEDO_RASTER_PATH,
                moon_fraction=moon_frac,
                moon_alt=moon_alt,
                max_lunar_irradiance=MAX_LUNAR_RAD
            )
            print(f"    Saved to: {out_file}")
            
        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline encountered an error: {e}")
