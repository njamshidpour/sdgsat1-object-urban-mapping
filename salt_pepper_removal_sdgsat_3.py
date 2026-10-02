import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import median_filter
import os
from pathlib import Path
import argparse

def remove_noise_and_plot_single_axis(image_path, output_path, min_pixel_size, num_bins=60):
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        data = src.read(1).astype(np.float32)
        
        # 1. Detect Noise using skimage
        threshold = np.percentile(data, 5)
        min_value = np.min(data)
        print(f'Minimun:{min_value:.20f}')
        print(f'Threshold:{threshold:.20f}')
        binary_mask = (data > threshold)
        labeled_mask = label(binary_mask, connectivity=2)

        large_objects_mask = remove_small_objects(labeled_mask, min_size=min_pixel_size) > 0
        
        noise_mask = binary_mask & (~large_objects_mask)

        local_med = median_filter(data, size=3)
        cleaned_data = data.copy()
        cleaned_data[noise_mask] = local_med[noise_mask]
        
        # Save output raster
        meta.update(dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(cleaned_data, 1)
            
        removed_intensities = data[noise_mask]
        removed_pixels_count = len(removed_intensities)

        if removed_pixels_count == 0:
            print("No noise detected.")
            return

        # 2. Define Bins purely based on Noise Min and Max Range
        noise_min = np.min(removed_intensities)
        noise_max = np.max(removed_intensities)
        bins = np.linspace(noise_min, noise_max, num_bins)

        # Filter original image pixels in this exact range
        image_pixels_in_noise_range = data[(data >= noise_min) & (data <= noise_max)]

        # 3. Plot BOTH on a SINGLE Y-Axis
        plt.figure(figsize=(10, 6))

        # Original image pixels (Blue background)
        plt.hist(image_pixels_in_noise_range, bins=bins, color='navy', alpha=0.4, 
                 label='Original Image Pixels (In Noise Range)', edgecolor='blue')

        # Removed noise pixels (Red bars in front)
        plt.hist(removed_intensities, bins=bins, color='crimson', alpha=0.75, 
                 label='Removed Noise Pixels', edgecolor='black')

        # Formatting
        plt.xlabel('Radiance Value', fontsize=11, fontweight='regular')
        plt.ylabel('Pixel Count', fontsize=11, fontweight='regular')
        plt.title(f"Direct Pixel Count Comparison (Range: {noise_min:.4f} to {noise_max:.4f})", 
                  fontsize=12, fontweight='regular', pad=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', framealpha=0.9)
        total_pixels = data.size


        stats_text = (f"Total Noise Pixels Removed: {removed_pixels_count}\n"
                      f"Total Image Pixels: {total_pixels}\n"
                      f"Noise Mean Radiance: {np.mean(removed_intensities):.4f}\n"
                      f"Noise Max Radiance: {noise_max:.4f}")
        
        plt.gca().text(0.985, 0.85, stats_text, transform=plt.gca().transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the executed percentile/component/median noise filter to one SDGSAT-1 band.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pixel-size", type=int, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remove_noise_and_plot_single_axis(args.input, args.output, min_pixel_size=args.min_pixel_size)
