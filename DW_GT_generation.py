"""Create a three-class Dynamic World reference with data-driven entropy filtering.

The original four input probabilities are preserved. Bare and scrub are merged
pixel-by-pixel, then the resulting Urban, Vegetation, and Bare/Sparse values are
normalized. Processing is block-wise so the full Tehran rasters need not be held
in memory. Use either --pilot or --full-run explicitly.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Sequence

# The project venv contains a newer PROJ database than the installed QGIS copy.
_proj = Path(sys.executable).resolve().parent.parent / "Lib/site-packages/rasterio/proj_data"
if _proj.exists():
    os.environ["PROJ_DATA"] = str(_proj)
    os.environ["PROJ_LIB"] = str(_proj)
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import mapping

# Rasterio 1.4.4 emits this harmless warning with newer NumPy releases.
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)


HMAX = math.log2(3.0)
CLASSES = {1: "Urban / Built-up", 2: "Vegetation", 3: "Bare / Sparsely Vegetated Land"}
SHORT = {1: "Urban", 2: "Vegetation", 3: "Bare_Sparse"}
COLORS = {0: (125, 125, 125), 1: (220, 45, 38), 2: (42, 145, 62), 3: (222, 184, 94)}


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class Progress:
    """A clean, single-line progress bar."""

    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(1, int(total))
        self.started, self.previous_length = time.perf_counter(), 0

    def update(self, completed: int) -> None:
        completed = min(int(completed), self.total)
        fraction = completed / self.total
        elapsed = time.perf_counter() - self.started
        eta = (self.total - completed) / (completed / elapsed) if completed and elapsed else float("nan")
        eta_text = duration(eta) if np.isfinite(eta) else "--:--"
        width, filled = 26, int(round(26 * fraction))
        message = (f"\r{self.label:<30} [{'#' * filled}{'-' * (width-filled)}] "
                   f"{100*fraction:6.2f}% ({completed:,}/{self.total:,}) ETA {eta_text}")
        print(message + " " * max(0, self.previous_length - len(message)), end="", flush=True)
        self.previous_length = len(message)
        if completed == self.total:
            print()


def calculate_shannon_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Base-2 Shannon entropy; zero probabilities contribute exactly zero."""
    terms = np.zeros_like(probabilities, dtype=np.float32)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
    return -np.sum(terms, axis=0, dtype=np.float32)


def validate_alignment(sources: Sequence[rasterio.io.DatasetReader]) -> dict[str, object]:
    base = sources[0]
    for source in sources[1:]:
        if (source.width, source.height) != (base.width, base.height):
            raise ValueError(f"Raster dimensions differ: {source.name}")
        if source.crs != base.crs:
            raise ValueError(f"Raster CRS differs: {source.name}")
        if not source.transform.almost_equals(base.transform):
            raise ValueError(f"Raster affine transform differs: {source.name}")
    return {"width": base.width, "height": base.height, "crs": str(base.crs),
            "transform": tuple(base.transform)}


def processing_windows(area: Window, block_size: int) -> Iterable[tuple[Window, Window]]:
    """Yield source windows and matching zero-based output windows."""
    for local_row in range(0, int(area.height), block_size):
        height = min(block_size, int(area.height) - local_row)
        for local_col in range(0, int(area.width), block_size):
            width = min(block_size, int(area.width) - local_col)
            yield (Window(int(area.col_off) + local_col, int(area.row_off) + local_row, width, height),
                   Window(local_col, local_row, width, height))


def number_of_windows(area: Window, block_size: int) -> int:
    return math.ceil(area.width / block_size) * math.ceil(area.height / block_size)


def pilot_area(base, size: int, explicit: Sequence[int] | None) -> Window:
    if explicit:
        col, row, width, height = map(int, explicit)
        col, row = max(0, col), max(0, row)
        width, height = min(width, base.width-col), min(height, base.height-row)
    else:
        width, height = min(size, base.width), min(size, base.height)
        col, row = (base.width-width)//2, (base.height-height)//2
    if width <= 0 or height <= 0:
        raise ValueError("Pilot window does not intersect the input rasters.")
    return Window(col, row, width, height)


def valid_source(raw: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(raw)
    if nodata is not None:
        valid &= ~np.isnan(raw) if np.isnan(nodata) else raw != nodata
    return valid


def province_geometry(shapefile: Path, raster_crs):
    if not shapefile.is_file():
        raise FileNotFoundError(f"Province shapefile not found: {shapefile}")
    province = gpd.read_file(shapefile)
    if province.empty or province.crs is None or province.geometry.isna().any():
        raise ValueError("Province shapefile must have nonempty geometries and a defined CRS.")
    if not province.geometry.is_valid.all():
        raise ValueError("Province shapefile contains invalid geometry; repair it before running.")
    geometry = province.to_crs(raster_crs).geometry.union_all()
    if geometry.is_empty:
        raise ValueError("Province geometry is empty after reprojection.")
    return mapping(geometry), str(province.crs)


def study_mask(geometry, base, window: Window) -> np.ndarray:
    # Pixel-centre inclusion; the native raster grid and square pixels are unchanged.
    return geometry_mask(
        [geometry], out_shape=(int(window.height), int(window.width)),
        transform=base.window_transform(window), invert=True, all_touched=False,
    )


def read_four(sources, window: Window, scale_factor: float, in_province: np.ndarray):
    arrays, ranges = [], []
    joint_valid = in_province.copy()
    raw_sum = np.zeros(joint_valid.shape, dtype=np.float64)
    for source in sources:
        raw = source.read(1, window=window)
        valid = valid_source(raw, source.nodata)
        joint_valid &= valid
        # The input rasters are integer-scaled. Summing their original values
        # avoids rejecting pixels whose true scaled sum is exactly 0.70.
        raw_sum += raw.astype(np.float64)
        raw = raw.astype(np.float32, copy=False)
        probability = raw / np.float32(scale_factor)
        values = probability[valid & in_province]
        ranges.append((float(values.min()), float(values.max())) if values.size else (np.nan, np.nan))
        arrays.append(probability)
    return np.stack(arrays, axis=0), joint_valid, ranges, raw_sum / scale_factor


def merge_and_normalize(
    four: np.ndarray, joint_valid: np.ndarray,
    exact_probability_sum: np.ndarray, minimum_probability_sum: float,
):
    valid = (
        joint_valid & np.isfinite(exact_probability_sum)
        & (exact_probability_sum >= minimum_probability_sum)
    )
    # Source order is Urban, Vegetation, Bare, Scrub. Only the valid mask is
    # normalized; S-filtered pixels never contribute to entropy or class GT.
    three = np.stack((four[0], four[1], four[2] + four[3]), axis=0).astype(np.float32)
    normalization_sum = np.sum(three, axis=0, dtype=np.float32)
    valid &= np.isfinite(normalization_sum) & (normalization_sum > 0)
    normalized = np.zeros_like(three)
    normalized[:, valid] = three[:, valid] / normalization_sum[valid]
    return normalized, exact_probability_sum, valid


def output_profile(base, area: Window, dtype: str, nodata):
    profile = base.profile.copy()
    profile.update(driver="GTiff", width=int(area.width), height=int(area.height),
                   transform=base.window_transform(area), count=1, dtype=dtype,
                   nodata=nodata, compress="deflate", predictor=3 if dtype == "float32" else 2,
                   BIGTIFF="IF_SAFER")
    if area.width >= 256 and area.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        for key in ("tiled", "blockxsize", "blockysize"):
            profile.pop(key, None)
    return profile


def update_range(target: list[float], low: float, high: float):
    if np.isfinite(low):
        target[0], target[1] = min(target[0], low), max(target[1], high)


def hist_quantile(counts: np.ndarray, scale_factor: float, q: float) -> float:
    total = int(counts.sum())
    if not total:
        return np.nan
    rank = max(1, int(math.ceil(q * total)))
    return float(np.searchsorted(np.cumsum(counts, dtype=np.uint64), rank)) / scale_factor


def write_csv(path: Path, rows: Sequence[dict[str, object]]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_threshold_rows(
    thresholds, bin_count, bin_entropy, bin_classes, total_valid, confidence_weight
):
    cumulative_count = np.cumsum(bin_count, dtype=np.uint64)
    cumulative_entropy = np.cumsum(bin_entropy, dtype=np.float64)
    cumulative_classes = np.cumsum(bin_classes, axis=1, dtype=np.uint64)
    rows = []
    for i, threshold in enumerate(thresholds):
        retained = int(cumulative_count[i]); rejected = total_valid-retained
        fraction = retained/total_valid
        mean_h = float(cumulative_entropy[i]/retained) if retained else np.nan
        confidence = 1-mean_h/HMAX if retained else np.nan
        coverage_weight = round(1.0 - confidence_weight, 12)
        score = (
            1.0 / (coverage_weight / fraction + confidence_weight / confidence)
            if retained and fraction > 0 and confidence > 0
            else np.nan
        )
        row = {"Entropy_Threshold": float(threshold), "Retained_Pixels": retained,
               "Retained_Percentage": 100*fraction, "Rejected_Pixels": rejected,
               "Rejected_Percentage": 100*rejected/total_valid,
               "Mean_Retained_Entropy": mean_h,
               "Mean_Normalized_Confidence": confidence,
               "Coverage_Weight": coverage_weight,
               "Confidence_Weight": confidence_weight,
               "Weighted_Harmonic_Score": score}
        for code in CLASSES:
            count = int(cumulative_classes[code-1, i])
            row[f"{SHORT[code]}_Pixels"] = count
            row[f"{SHORT[code]}_Pct_of_Retained"] = 100*count/retained if retained else np.nan
        rows.append(row)
    return rows


def constrained_choice(rows, target):
    eligible = [r for r in rows if float(r["Mean_Normalized_Confidence"]) >= target]
    return max(eligible, key=lambda r: (int(r["Retained_Pixels"]), float(r["Entropy_Threshold"]))) if eligible else None


def entropy_figure(entropy_path, counts, edges, cutoff, observed_minimum, png, pdf):
    """Publication-style spatial entropy map and actual entropy histogram."""
    with rasterio.open(entropy_path) as source:
        scale = min(1.0, 1400.0 / max(source.width, source.height))
        shape = (max(1, int(source.height * scale)), max(1, int(source.width * scale)))
        data = source.read(1, out_shape=shape, masked=True, resampling=Resampling.average)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, (map_ax, hist_ax) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    mapped = map_ax.imshow(data, cmap="viridis", vmin=observed_minimum, vmax=HMAX)
    map_ax.set_title("Spatial distribution of three-class Shannon entropy", pad=10)
    map_ax.set_axis_off()
    colorbar = fig.colorbar(mapped, ax=map_ax, orientation="horizontal", pad=0.04, fraction=0.055)
    colorbar.set_label("Shannon entropy (bits)")
    colorbar_ticks = np.linspace(observed_minimum, HMAX, 4)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(
        [f"min={observed_minimum:.3f}"]
        + [f"{value:.3f}" for value in colorbar_ticks[1:-1]]
        + [f"log2(3)={HMAX:.3f}"]
    )

    hist_ax.stairs(counts, edges, fill=True, color="#238b7e", alpha=0.9, linewidth=0.8)
    hist_ax.axvline(cutoff, color="#c62828", linestyle="--", linewidth=1.8,
                    label=f"Selected cutoff = {cutoff:.2f} bits")
    hist_ax.set_title(
        f"Distribution of valid-pixel entropy\nTheoretical maximum: log2(3) = {HMAX:.3f} bits",
        pad=10,
    )
    hist_ax.set_xlabel("Shannon entropy (bits)")
    hist_ax.set_ylabel("Pixel frequency")
    hist_ax.set_xlim(observed_minimum, HMAX)
    entropy_ticks = np.linspace(observed_minimum, HMAX, 6)
    hist_ax.set_xticks(entropy_ticks)
    hist_ax.set_xticklabels([f"{value:.3f}" for value in entropy_ticks])
    hist_ax.set_yscale("log")
    hist_ax.grid(axis="y", which="major", color="0.88", linewidth=0.7)
    hist_ax.spines[["top", "right"]].set_visible(False)
    # The upper-left entropy range is empty in practice, keeping this compact
    # legend away from both the distribution and the subplot title.
    hist_ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                   handlelength=2.5, borderaxespad=0.7)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagnostic_figure(rows, cutoff, observed_minimum, png, pdf):
    """Coverage and confidence curves for the threshold grid search."""
    x = np.array([r["Entropy_Threshold"] for r in rows], dtype=float)
    retained = np.array([r["Retained_Percentage"] for r in rows], dtype=float)
    confidence = np.array([r["Mean_Normalized_Confidence"] for r in rows], dtype=float)
    fig, axes_ = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    specifications = [
        (axes_[0], retained, "Retained reference coverage", "Retained valid pixels (%)", (0, 100)),
        (axes_[1], confidence, "Confidence of retained pixels", "Mean normalized confidence", (0, 1)),
    ]
    for axis, y, title, ylabel, limits in specifications:
        axis.plot(x, y, color="#2166ac", linewidth=2)
        axis.axvline(cutoff, color="#c62828", linestyle="--", linewidth=1.5,
                     label=f"Selected cutoff = {cutoff:.2f} bits")
        axis.set(title=title, xlabel="Entropy threshold (bits)", ylabel=ylabel,
                 xlim=(observed_minimum, HMAX), ylim=limits)
        entropy_ticks = np.linspace(observed_minimum, HMAX, 6)
        axis.set_xticks(entropy_ticks)
        axis.set_xticklabels([f"{value:.3f}" for value in entropy_ticks])
        axis.grid(color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("Entropy-threshold grid-search diagnostics", fontsize=14)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def final_map(gt_path, png, pdf):
    """Display final labels with uncertain pixels as a separate category."""
    with rasterio.open(gt_path) as source:
        scale = min(1.0, 1600.0 / max(source.width, source.height))
        shape = (max(1, int(source.height * scale)), max(1, int(source.width * scale)))
        labels = source.read(1, out_shape=shape, resampling=Resampling.nearest)
    cmap = ListedColormap([np.array(COLORS[code]) / 255 for code in range(4)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axis = plt.subplots(figsize=(11, 7.2), constrained_layout=True)
    axis.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_title("Final three-class entropy-filtered reference", fontsize=13, pad=10)
    axis.set_axis_off()
    legend = [Patch(facecolor=np.array(COLORS[0]) / 255, edgecolor="0.25", label="Uncertain / NoData")]
    legend.extend(Patch(facecolor=np.array(COLORS[code]) / 255, edgecolor="0.25", label=name)
                  for code, name in CLASSES.items())
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.015),
                ncol=4, frameon=False, fontsize=8.5, columnspacing=1.2,
                handletextpad=0.5)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(rows):
    lines=["| Metric | Value |","|---|---:|"]
    for name,value in rows:
        if isinstance(value,float): text=f"{value:.8g}" if np.isfinite(value) else "NA"
        elif isinstance(value,(int,np.integer)): text=f"{int(value):,}"
        else: text=str(value)
        lines.append(f"| {name} | {text} |")
    return "\n".join(lines)


def process_separate_probabilities_uncertainty(
    urban_prob_path: Path, veg_prob_path: Path, bare_prob_path: Path,
    scrub_prob_path: Path, output_dir: Path, province_shapefile: Path,
    previous_statistics: Path | None = None, scale_factor: float=10000.0,
    threshold_start: float=0.05, threshold_stop: float=1.55,
    threshold_step: float=0.01, confidence_weight: float=0.90,
    minimum_probability_sum: float=0.70,
    block_size: int=1024, pilot: bool=False,
    pilot_size: int=4096, explicit_pilot_window: Sequence[int]|None=None,
):
    started=time.perf_counter(); output_dir.mkdir(parents=True,exist_ok=True)
    suffix="_PILOT" if pilot else ""
    out={
        "Final three-class GeoTIFF":output_dir/f"Tehran_DW_ThreeClass_GT_2023{suffix}.tif",
        "Float32 entropy GeoTIFF":output_dir/f"Tehran_DW_ThreeClass_ShannonEntropy_2023{suffix}.tif",
        "Entropy figure PNG":output_dir/f"Tehran_DW_ThreeClass_Entropy_Map_Histogram_2023{suffix}.png",
        "Entropy figure PDF":output_dir/f"Tehran_DW_ThreeClass_Entropy_Map_Histogram_2023{suffix}.pdf",
        "Grid-search figure PNG":output_dir/f"Tehran_DW_ThreeClass_Threshold_Diagnostics_2023{suffix}.png",
        "Grid-search figure PDF":output_dir/f"Tehran_DW_ThreeClass_Threshold_Diagnostics_2023{suffix}.pdf",
        "Grid-search CSV":output_dir/f"Tehran_DW_ThreeClass_Entropy_Grid_Search_2023{suffix}.csv",
        "Class-distribution CSV":output_dir/f"Tehran_DW_ThreeClass_Class_Distribution_2023{suffix}.csv",
        "Statistics CSV":output_dir/f"Tehran_DW_ThreeClass_Statistics_2023{suffix}.csv",
        "Final visualization PNG":output_dir/f"Tehran_DW_ThreeClass_GT_Visualization_2023{suffix}.png",
        "Final visualization PDF":output_dir/f"Tehran_DW_ThreeClass_GT_Visualization_2023{suffix}.pdf",
        "Markdown report":output_dir/f"DW_GT_generation_ThreeClass_Report{suffix}.md",
    }
    input_paths=[urban_prob_path,veg_prob_path,bare_prob_path,scrub_prob_path]
    missing=[p for p in input_paths+[province_shapefile] if not p.exists()]
    if missing: raise FileNotFoundError("Missing input(s): "+", ".join(map(str,missing)))
    thresholds=np.round(np.arange(threshold_start,threshold_stop+threshold_step*.5,threshold_step),10)
    if scale_factor<=0 or not len(thresholds) or np.any(np.diff(thresholds)<=0): raise ValueError("Invalid scale factor or threshold grid.")
    if not 0 < confidence_weight < 1:
        raise ValueError("confidence_weight must be strictly between 0 and 1.")
    if not 0 < minimum_probability_sum <= 1:
        raise ValueError("minimum_probability_sum must be in (0, 1].")
    sources=[rasterio.open(p) for p in input_paths]
    try:
        metadata=validate_alignment(sources); base=sources[0]
        geometry, province_crs=province_geometry(province_shapefile,base.crs)
        area=pilot_area(base,pilot_size,explicit_pilot_window) if pilot else Window(0,0,base.width,base.height)
        total_blocks=number_of_windows(area,block_size); n=len(thresholds)
        print(f"Mode: {'PILOT' if pilot else 'FULL DATASET'} | area: {int(area.width):,} x {int(area.height):,} pixels")
        print(f"Study area: {province_shapefile.resolve()} (pixel centres inside polygon)")
        print(f"Input-probability sum validity filter: S >= {minimum_probability_sum:.2f}")
        bins=np.zeros(n,np.uint64); bin_h=np.zeros(n,np.float64); bin_classes=np.zeros((3,n),np.uint64)
        h_edges=np.linspace(0,HMAX,257); h_hist=np.zeros(256,np.uint64)
        sum_hist=np.zeros(int(math.ceil(4*scale_factor))+1,np.uint64)
        source_ranges=[[np.inf,-np.inf] for _ in range(4)]; class_ranges=[[np.inf,-np.inf] for _ in range(3)]
        study_total=0; source_complete_total=0; sum_filter_removed=0; valid_total=0
        h_total=0.; h_min=np.inf; h_max=-np.inf; sum_total=0.; sum_sq_total=0.; post_error=0.
        progress=Progress("Pass 1/3: entropy + search",total_blocks)
        with rasterio.open(out["Float32 entropy GeoTIFF"],"w",**output_profile(base,area,"float32",-9999.0)) as destination:
            for k,(source_window,output_window) in enumerate(processing_windows(area,block_size),1):
                in_province=study_mask(geometry,base,source_window)
                study_total+=int(np.count_nonzero(in_province))
                four,joint,ranges,exact_sum=read_four(sources,source_window,scale_factor,in_province)
                for i,(low,high) in enumerate(ranges):
                    update_range(source_ranges[i],low,high)
                    if np.isfinite(low) and (low < -1e-6 or high > 1+1e-6):
                        raise ValueError(f"Scaled probability outside [0,1] in {sources[i].name}: min={low}, max={high}. Check --scale-factor.")
                normalized,pre_sum,valid=merge_and_normalize(
                    four,joint,exact_sum,minimum_probability_sum
                )
                source_complete=joint & np.isfinite(pre_sum)
                source_complete_total+=int(np.count_nonzero(source_complete))
                sum_filter_removed+=int(np.count_nonzero(
                    source_complete & (pre_sum < minimum_probability_sum)
                ))
                if source_complete.any():
                    # Sum diagnostics are pre-filter, so the low-S tail remains
                    # visible in the end-of-run report.
                    sums=pre_sum[source_complete]
                    sum_total+=float(sums.sum(dtype=np.float64))
                    sum_sq_total+=float(np.square(sums).sum(dtype=np.float64))
                    sum_indices=np.clip(
                        np.rint(sums*scale_factor).astype(np.int64),0,len(sum_hist)-1
                    )
                    sum_hist+=np.bincount(sum_indices,minlength=len(sum_hist)).astype(np.uint64)
                entropy_out=np.full(valid.shape,-9999,np.float32)
                if valid.any():
                    values=normalized[:,valid]; post=np.sum(values,axis=0)
                    if not np.all(np.isfinite(values)): raise ValueError("NaN/Inf after normalization.")
                    post_error=max(post_error,float(np.max(np.abs(post-1))))
                    entropy_values=calculate_shannon_entropy(normalized)[valid]
                    if not np.all(np.isfinite(entropy_values)): raise ValueError("NaN/Inf in valid entropy.")
                    entropy_out[valid]=entropy_values; dominant=np.argmax(values,axis=0)+1
                    valid_total+=entropy_values.size; h_total+=float(entropy_values.sum(dtype=np.float64))
                    h_min=min(h_min,float(entropy_values.min())); h_max=max(h_max,float(entropy_values.max()))
                    h_hist+=np.histogram(entropy_values,bins=h_edges)[0].astype(np.uint64)
                    indices=np.searchsorted(thresholds,entropy_values,side="left"); inside=indices<n
                    bins+=np.bincount(indices[inside],minlength=n).astype(np.uint64)
                    bin_h+=np.bincount(indices[inside],weights=entropy_values[inside].astype(np.float64),minlength=n)
                    for code in CLASSES: bin_classes[code-1]+=np.bincount(indices[inside & (dominant==code)],minlength=n).astype(np.uint64)
                    for i in range(3): update_range(class_ranges[i],float(values[i].min()),float(values[i].max()))
                destination.write(entropy_out,1,window=output_window); progress.update(k)
    finally:
        for source in sources: source.close()
    if not study_total: raise RuntimeError("No raster pixel centres fall inside the province study area.")
    if source_complete_total-sum_filter_removed != valid_total:
        raise RuntimeError("Post-S valid pixel count does not match the source-valid minus S-filter counts.")
    if not valid_total: raise RuntimeError("No pixels passed the input-probability sum filter.")
    if post_error>1e-5: raise RuntimeError(f"Normalized probabilities failed sum-to-one check: {post_error}")
    rows=make_threshold_rows(
        thresholds, bins, bin_h, bin_classes, valid_total, confidence_weight
    )
    selected=max(rows,key=lambda r: r["Weighted_Harmonic_Score"] if np.isfinite(r["Weighted_Harmonic_Score"]) else -np.inf)
    cutoff=float(selected["Entropy_Threshold"]); alternatives={x:constrained_choice(rows,x) for x in (.70,.75,.80)}
    write_csv(out["Grid-search CSV"],rows)
    print(
        f"Selected cutoff: {cutoff:.2f} bits | weighted harmonic score: "
        f"{selected['Weighted_Harmonic_Score']:.6f} "
        f"(confidence weight={confidence_weight:.2f})"
    )

    sources=[rasterio.open(p) for p in input_paths]; winning_sum=0.; generated=set(); progress=Progress("Pass 2/3: final labels",total_blocks)
    try:
        with rasterio.open(out["Float32 entropy GeoTIFF"]) as entropy_source, rasterio.open(out["Final three-class GeoTIFF"],"w",**output_profile(sources[0],area,"uint8",0)) as destination:
            for k,(source_window,output_window) in enumerate(processing_windows(area,block_size),1):
                in_province=study_mask(geometry,sources[0],source_window)
                four,joint,_,exact_sum=read_four(sources,source_window,scale_factor,in_province)
                normalized,_,valid=merge_and_normalize(
                    four,joint,exact_sum,minimum_probability_sum
                )
                entropy=entropy_source.read(1,window=output_window)
                # Compare in float64, matching the float64 threshold grid used
                # in Pass 1. NumPy may otherwise round a Python cutoff such as
                # 0.99 to float32 and include pixels just above that cutoff.
                confident=valid & (entropy.astype(np.float64)<=cutoff)
                labels=np.zeros(valid.shape,np.uint8)
                if confident.any():
                    dominant=np.argmax(normalized,axis=0)+1; labels[confident]=dominant[confident]
                    winning_sum+=float(np.max(normalized,axis=0)[confident].sum(dtype=np.float64))
                generated.update(map(int,np.unique(labels))); destination.write(labels,1,window=output_window); progress.update(k)
    finally:
        for source in sources: source.close()
    observed=set(); persisted_counts=np.zeros(4,dtype=np.uint64)
    progress=Progress("Pass 3/3: reload check",total_blocks)
    with rasterio.open(out["Final three-class GeoTIFF"]) as check, rasterio.open(out["Float32 entropy GeoTIFF"]) as entropy_check:
        if check.dtypes[0]!="uint8" or check.nodata!=0: raise RuntimeError("Final raster dtype/NoData validation failed.")
        if entropy_check.dtypes[0]!="float32" or entropy_check.nodata!=-9999.0:
            raise RuntimeError("Entropy raster dtype/NoData validation failed.")
        for k,(source_window,window) in enumerate(processing_windows(area,block_size),1):
            saved_labels=check.read(1,window=window)
            saved_entropy=entropy_check.read(1,window=window)
            in_province=study_mask(geometry,base,source_window)
            if np.any(saved_labels[~in_province]!=0) or np.any(saved_entropy[~in_province]!=-9999.0):
                raise RuntimeError("Non-NoData output detected outside the Tehran Province mask.")
            observed.update(map(int,np.unique(saved_labels)))
            persisted_counts+=np.bincount(saved_labels.ravel(),minlength=4).astype(np.uint64)
            progress.update(k)
    if not observed.issubset({0,1,2,3}): raise RuntimeError(f"Unexpected final labels: {sorted(observed)}")
    expected_counts=np.array(
        [int(selected[f"{SHORT[code]}_Pixels"]) for code in CLASSES],
        dtype=np.uint64,
    )
    if not np.array_equal(persisted_counts[1:],expected_counts):
        raise RuntimeError(
            "Saved class counts differ from the selected grid-search row: "
            f"saved={persisted_counts[1:].tolist()}, expected={expected_counts.tolist()}"
        )

    retained=int(selected["Retained_Pixels"]); rejected=valid_total-retained
    raster_total=int(area.width)*int(area.height)
    source_invalid_total=study_total-source_complete_total
    if int(persisted_counts[0]) != raster_total-retained:
        raise RuntimeError("Reloaded label-0 count does not match raster extent minus retained pixels.")
    sum_mean=sum_total/source_complete_total
    sum_std=math.sqrt(max(0,sum_sq_total/source_complete_total-sum_mean**2))
    mean_win=winning_sum/retained if retained else np.nan
    stats={"Run_Mode":"PILOT" if pilot else "FULL_DATASET", "Total_Valid_Input_Pixels":valid_total,
           "Study_Area_Pixels":study_total, "Outside_Study_Area_Pixels":raster_total-study_total,
           "Raster_Extent_Pixels":raster_total, "Province_Shapefile":str(province_shapefile.resolve()),
           "Province_CRS":province_crs, "Raster_CRS":metadata["crs"],
           "Source_Valid_Pixels_Before_S_Filter":source_complete_total,
           "Source_Invalid_Pixels_Inside_Study_Area":source_invalid_total,
           "Minimum_Probability_Sum":minimum_probability_sum,
           "S_Filter_Removed_Pixels":sum_filter_removed,
           "S_Filter_Removed_Pct_of_Source_Valid":100*sum_filter_removed/source_complete_total,
           "S_Filter_Removed_Pct_of_Study_Area":100*sum_filter_removed/study_total,
           "Remaining_Valid_Pixels_After_S_Filter":valid_total,
           "Label_0_Pixels_Inside_Study_Area":study_total-retained,
           "Retained_Pct_of_Study_Area":100*retained/study_total,
           "Selected_Entropy_Cutoff_Bits":cutoff,
           "Selection_Criterion":"Maximum confidence-weighted harmonic mean of retained fraction and mean normalized confidence",
           "Confidence_Weight":confidence_weight,
           "Coverage_Weight":round(1.0-confidence_weight, 12),
           "Maximum_Weighted_Harmonic_Score":selected["Weighted_Harmonic_Score"], "Confident_Pixels_Retained":retained,
           "Confident_Percentage_of_Valid":100*retained/valid_total, "Uncertain_Excluded_Pixels":rejected,
           "Uncertain_Percentage_of_Valid":100*rejected/valid_total, "Mean_Entropy_Before_Filtering":h_total/valid_total,
           "Mean_Entropy_After_Filtering":selected["Mean_Retained_Entropy"],
           "Mean_Normalized_Confidence_After_Filtering":selected["Mean_Normalized_Confidence"],
           "Mean_Winning_Class_Probability_After_Filtering":mean_win,
           "Minimum_Observed_Entropy":h_min, "Maximum_Observed_Entropy":h_max, "Theoretical_Maximum_Entropy":HMAX,
           "Pre_Normalization_Sum_Min":hist_quantile(sum_hist,scale_factor,0), "Pre_Normalization_Sum_P01":hist_quantile(sum_hist,scale_factor,.01),
           "Pre_Normalization_Sum_P05":hist_quantile(sum_hist,scale_factor,.05), "Pre_Normalization_Sum_P25":hist_quantile(sum_hist,scale_factor,.25),
           "Pre_Normalization_Sum_P50":hist_quantile(sum_hist,scale_factor,.5), "Pre_Normalization_Sum_P75":hist_quantile(sum_hist,scale_factor,.75),
           "Pre_Normalization_Sum_P95":hist_quantile(sum_hist,scale_factor,.95), "Pre_Normalization_Sum_P99":hist_quantile(sum_hist,scale_factor,.99),
           "Pre_Normalization_Sum_Max":hist_quantile(sum_hist,scale_factor,1), "Pre_Normalization_Sum_Mean":sum_mean,
           "Pre_Normalization_Sum_Std":sum_std, "Post_Normalization_Max_Absolute_Sum_Error":post_error}
    for code in CLASSES:
        name=SHORT[code]; count=int(selected[f"{name}_Pixels"])
        stats.update({f"{name}_Pixel_Count":count, f"{name}_Pct_of_All_Valid":100*count/valid_total,
                      f"{name}_Pct_of_Retained":100*count/retained if retained else np.nan,
                      f"{name}_Normalized_Probability_Min":class_ranges[code-1][0], f"{name}_Normalized_Probability_Max":class_ranges[code-1][1]})
    for name,value_range in zip(("Urban_Source","Vegetation_Source","Bare_Source","Scrub_Source"),source_ranges):
        stats[f"{name}_Scaled_Probability_Min"],stats[f"{name}_Scaled_Probability_Max"]=value_range
    for target,row in alternatives.items():
        prefix=f"Confidence_{int(target*100)}pct"
        stats[f"{prefix}_Coverage_Maximizing_Threshold"]=row["Entropy_Threshold"] if row else "NA"
        stats[f"{prefix}_Retained_Pixels"]=row["Retained_Pixels"] if row else "NA"
        stats[f"{prefix}_Retained_Percentage"]=row["Retained_Percentage"] if row else "NA"
        stats[f"{prefix}_Mean_Confidence"]=row["Mean_Normalized_Confidence"] if row else "NA"
    stats["Saved_Final_Labels"]=", ".join(map(str,sorted(observed))); stats["Runtime_Seconds"]=time.perf_counter()-started
    write_csv(out["Statistics CSV"],[stats])
    class_distribution=[]
    for code in CLASSES:
        count=int(selected[f"{SHORT[code]}_Pixels"])
        class_distribution.append({
            "Class_Code":code, "Class_Name":CLASSES[code], "Pixel_Count":count,
            "Pct_of_Study_Area":100*count/study_total,
            "Pct_of_Post_S_Valid":100*count/valid_total,
            "Pct_of_Retained":100*count/retained if retained else np.nan,
        })
    write_csv(out["Class-distribution CSV"],class_distribution)
    print("Creating publication figures...")
    entropy_figure(
        out["Float32 entropy GeoTIFF"], h_hist, h_edges, cutoff, h_min,
        out["Entropy figure PNG"], out["Entropy figure PDF"],
    )
    diagnostic_figure(
        rows, cutoff, h_min,
        out["Grid-search figure PNG"], out["Grid-search figure PDF"],
    )
    final_map(out["Final three-class GeoTIFF"],out["Final visualization PNG"],out["Final visualization PDF"])

    summary=[
        ("Province study-area pixels",study_total),
        ("Outside-province pixels excluded",raster_total-study_total),
        ("Source-invalid pixels inside province",source_invalid_total),
        ("Source-valid pixels before S filter",source_complete_total),
        (f"Pixels removed by S < {minimum_probability_sum:.2f}",sum_filter_removed),
        ("S-filter removed % of source-valid pixels",100*sum_filter_removed/source_complete_total),
        ("S-filter removed % of province pixels",100*sum_filter_removed/study_total),
        ("Remaining valid pixels after S filter",valid_total),
        ("Total valid input pixels",valid_total),
        ("Selected entropy cutoff (bits)",cutoff),
        ("Confident pixels retained",retained),
        ("Confident pixels retained % of province",100*retained/study_total),
        ("Confident percentage of post-S valid pixels",100*retained/valid_total),
        ("Entropy-rejected pixels after S filter",rejected),
        ("Entropy-rejected % of post-S valid pixels",100*rejected/valid_total),
        ("NoData/uncertain label-0 pixels inside province",study_total-retained),
    ]
    for code in CLASSES:
        count=int(selected[f"{SHORT[code]}_Pixels"]); summary.extend([(f"{CLASSES[code]} pixels",count),(f"{CLASSES[code]} % of province",100*count/study_total),(f"{CLASSES[code]} % of all valid",100*count/valid_total),(f"{CLASSES[code]} % of retained",100*count/retained if retained else np.nan)])
    summary.extend([("Mean entropy before filtering",h_total/valid_total),("Mean entropy after filtering",selected["Mean_Retained_Entropy"]),
                    ("Mean normalized confidence after filtering",selected["Mean_Normalized_Confidence"]),
                    ("Mean winning-class probability after filtering",mean_win),("Maximum observed entropy",h_max)])
    sum_distribution=[("Minimum",stats["Pre_Normalization_Sum_Min"]),("P1",stats["Pre_Normalization_Sum_P01"]),
                      ("P5",stats["Pre_Normalization_Sum_P05"]),("P25",stats["Pre_Normalization_Sum_P25"]),
                      ("P50",stats["Pre_Normalization_Sum_P50"]),("P75",stats["Pre_Normalization_Sum_P75"]),
                      ("P95",stats["Pre_Normalization_Sum_P95"]),("P99",stats["Pre_Normalization_Sum_P99"]),
                      ("Maximum",stats["Pre_Normalization_Sum_Max"]),("Mean",sum_mean),("Standard deviation",sum_std)]
    alternative_lines=[]
    for target,row in alternatives.items():
        alternative_lines.append(f"- Mean confidence >= {target:.2f}: no qualifying threshold." if row is None else
                                 f"- Mean confidence >= {target:.2f}: threshold `{row['Entropy_Threshold']:.2f}` bits; `{int(row['Retained_Pixels']):,}` retained pixels (`{row['Retained_Percentage']:.4f}%`).")
    validation=["PASS: all four sources have identical dimensions, CRS, and affine transform.",
                "PASS: province shapefile was reprojected to the probability-raster grid; only pixel centres within the province contributed to statistics and threshold search.",
                "PASS: source probabilities are within [0, 1] after scaling.",
                "PASS: source-valid minus S-filtered pixels equals the post-S valid pixel count.",
                f"PASS: maximum normalized sum error is {post_error:.3g}.",
                "PASS: no NaN/Inf entered valid output data.",
                f"PASS: the reloaded final raster contains only allowed labels ({', '.join(map(str,sorted(observed)))}).",
                "PASS: reloaded final class counts exactly match the selected grid-search row.",
                "PASS: reloaded GT and entropy rasters contain only NoData outside the province.",
                "PASS: entropy is float32 and final labels are uint8."]
    limitation=("This is a pilot crop; its selected threshold and statistics are diagnostic only." if pilot else
                "No independent ground truth was used; the cutoff is an operational coverage-confidence compromise, not an accuracy-optimal threshold.")
    probability_ranges=[("Urban minimum",class_ranges[0][0]),("Urban maximum",class_ranges[0][1]),
                        ("Vegetation minimum",class_ranges[1][0]),("Vegetation maximum",class_ranges[1][1]),
                        ("Bare/Sparse minimum",class_ranges[2][0]),("Bare/Sparse maximum",class_ranges[2][1])]
    comparison_text="A directly comparable previous full-run statistics file was not available."
    if pilot:
        comparison_text="Pilot-only run; compare full-run results after processing the entire province."
    elif previous_statistics is not None and previous_statistics.is_file():
        with previous_statistics.open(newline="",encoding="utf-8-sig") as handle:
            prior=next(csv.DictReader(handle),None)
        if (prior and prior.get("Run_Mode")=="FULL_DATASET"
                and math.isclose(float(prior["Confidence_Weight"]),confidence_weight)
                and math.isclose(float(prior["Minimum_Probability_Sum"]),minimum_probability_sum)):
            comparisons=[
                ("Analysis pixels",int(prior["Total_Raster_Pixels"]),study_total),
                ("Selected cutoff (bits)",float(prior["Selected_Entropy_Cutoff_Bits"]),cutoff),
                ("Pixels removed by S filter",int(prior["S_Filter_Removed_Pixels"]),sum_filter_removed),
                ("Post-S valid pixels",int(prior["Total_Valid_Input_Pixels"]),valid_total),
                ("Retained pixels",int(prior["Confident_Pixels_Retained"]),retained),
                ("Retained % of post-S valid",float(prior["Confident_Percentage_of_Valid"]),100*retained/valid_total),
                ("Entropy-rejected pixels",int(prior["Uncertain_Excluded_Pixels"]),rejected),
            ]
            comparisons.extend((f"{CLASSES[code]} pixels",int(prior[f"{SHORT[code]}_Pixel_Count"]),
                                int(selected[f"{SHORT[code]}_Pixels"])) for code in CLASSES)
            comparison_text=(
                f"Previous statistics: `{previous_statistics.resolve()}`. The previous analysis used "
                "the entire rectangular raster extent; the new analysis uses only the province mask. "
                "Counts and retained fractions therefore have different spatial denominators, and "
                "the cutoff was reoptimized. These changes should not be interpreted as an accuracy gain.\n\n"
                "| Metric | Previous: whole raster | Updated: Tehran Province |\n|---|---:|---:|\n"+
                "\n".join(f"| {name} | {old:,.4f} | {new:,.4f} |" if isinstance(old,float) else
                          f"| {name} | {old:,} | {new:,} |" for name,old,new in comparisons)
            )
        else:
            comparison_text=(f"Previous file `{previous_statistics.resolve()}` exists but does not "
                             "match this full-run confidence weight and minimum-S threshold; no numeric comparison is made.")
    outputs_text="\n".join(f"- `{path.resolve()}`" for path in out.values())
    report=f"""# Three-Class Dynamic World Ground-Truth Generation Report

## Inputs and scope

- Mode: **{'PILOT' if pilot else 'FULL DATASET'}**
- CRS: `{metadata['crs']}`
- Province shapefile: `{province_shapefile.resolve()}` (source CRS `{province_crs}`)
- Study-area rule: pixel centre inside the province polygon (`all_touched=False`); output keeps the source raster grid, with NoData outside the province.
- Processed dimensions: `{int(area.width):,} x {int(area.height):,}` pixels
- Scale factor: `{scale_factor:g}`
- Minimum input-probability sum: `S >= {minimum_probability_sum:.2f}`
- Urban: `{urban_prob_path.resolve()}`
- Vegetation: `{veg_prob_path.resolve()}`
- Bare: `{bare_prob_path.resolve()}`
- Scrub: `{scrub_prob_path.resolve()}`

## Probability merging and normalization

The integer values were divided by `{scale_factor:g}`. Before normalization and entropy, the four-class sum `S = P_urban + P_vegetation + P_bare + P_scrub` was tested in high precision. Pixels with `S < {minimum_probability_sum:.2f}` were excluded from the valid data, cutoff optimization, and class assignment; their entropy output is NoData and their final class label is 0. Only pixels with `S >= {minimum_probability_sum:.2f}` proceeded. The final stack was formed as `Urban = P_urban`, `Vegetation = P_vegetation`, and `Bare/Sparse = P_bare + P_scrub`; scrub was not discarded. The three probabilities were normalized pixel-wise only for these valid pixels.

Pre-normalization distribution of `P_urban + P_vegetation + P_bare + P_scrub` over **all source-valid pixels, before the S filter**:

{markdown_table(sum_distribution)}

Normalized probability ranges:

{markdown_table(probability_ranges)}

## Entropy threshold

Base-2 Shannon entropy was evaluated **only among pixels passing the S filter**, from `{thresholds[0]:.2f}` to `{thresholds[-1]:.2f}` bits in `{threshold_step:.2f}`-bit steps. Its theoretical three-class maximum is `log2(3) = {HMAX:.8f}`. The selected cutoff is **{cutoff:.2f} bits**, maximizing the weighted harmonic score `1 / [(1-w)/R + w/C]`, where `R` is retained fraction among post-S valid pixels, `C` is mean normalized confidence, and `w = {confidence_weight:.2f}` is the confidence weight. Thus confidence receives `{confidence_weight:.0%}` of the weighting and coverage receives `{1-confidence_weight:.0%}`; no minimum or target retained fraction is imposed. The maximum score is `{selected['Weighted_Harmonic_Score']:.8f}`. It is not claimed to be accuracy-optimal.

Sensitivity thresholds:

{chr(10).join(alternative_lines)}

## Final statistics

{markdown_table(summary)}

## Comparison with previous results

{comparison_text}

## Validation

{chr(10).join('- '+item for item in validation)}

## Limitations or deviations

- {limitation}
- Inside the province, label 0 combines S-filtered pixels, entropy-rejected post-S valid pixels, and source-invalid pixels. Outside the province, label 0 means out of study area. All reported analysis percentages use province-only denominators; entropy-rejection percentages use post-S valid pixels.
- Figure overviews are resampled only for display; output rasters and statistics use native pixels.

## Output files

{outputs_text}
"""
    out["Markdown report"].write_text(report,encoding="utf-8")
    print(f"Completed in {duration(time.perf_counter()-started)}")
    print(f"Report: {out['Markdown report'].resolve()}")
    return out


def parse_args():
    root=Path(__file__).resolve().parent; gt=root/"GTs"
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot",action="store_true",help="Process only a native-resolution central crop.")
    mode.add_argument("--full-run",action="store_true",help="Process the complete raster extent.")
    parser.add_argument("--urban",type=Path,default=gt/"Tehran_Built_Probability_2023_Int16_compressed.tif")
    parser.add_argument("--vegetation",type=Path,default=gt/"Tehran_Veg_Probability_2023_Int16_compressed.tif")
    parser.add_argument("--bare",type=Path,default=gt/"Tehran_Bare_Probability_2023_Int16_compressed.tif")
    parser.add_argument("--scrub",type=Path,default=gt/"Tehran_Scrub_Probability_2023_Int16_compressed.tif")
    parser.add_argument("--province-shapefile",type=Path,default=root/"Shapefile"/"Tehran_True_Province.shp",
                        help="Tehran Province polygon used to mask statistics and outputs on the source raster grid.")
    parser.add_argument("--previous-statistics",type=Path,default=None,
                        help="Previous full-extent statistics CSV for the report comparison.")
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--scale-factor",type=float,default=10000.0)
    parser.add_argument("--threshold-start",type=float,default=.05); parser.add_argument("--threshold-stop",type=float,default=1.55)
    parser.add_argument("--threshold-step",type=float,default=.01)
    parser.add_argument(
        "--confidence-weight", type=float, default=.90,
        help="Weight assigned to retained-pixel confidence in the weighted harmonic score; coverage receives 1-weight.",
    )
    parser.add_argument(
        "--minimum-probability-sum", type=float, default=.70,
        help="Keep only pixels whose original four-class probability sum S meets this minimum.",
    )
    parser.add_argument("--block-size",type=int,default=1024)
    parser.add_argument("--pilot-size",type=int,default=4096)
    parser.add_argument("--pilot-window",type=int,nargs=4,metavar=("COL","ROW","WIDTH","HEIGHT"))
    args=parser.parse_args()
    if args.block_size<=0 or args.pilot_size<=0: parser.error("Block and pilot sizes must be positive.")
    if args.output_dir is None:
        weight_tag=int(round(args.confidence_weight*100))
        sum_tag=int(round(args.minimum_probability_sum*100))
        folder=f"DW_GT_ThreeClass_2023_weight_{weight_tag:03d}_Sge{sum_tag:03d}_TehranProvince"
        args.output_dir=gt/(folder+"_pilot" if args.pilot else folder)
    if args.previous_statistics is None:
        weight_tag=int(round(args.confidence_weight*100))
        sum_tag=int(round(args.minimum_probability_sum*100))
        old_folder=f"DW_GT_ThreeClass_2023_weight_{weight_tag:03d}_Sge{sum_tag:03d}"
        args.previous_statistics=gt/old_folder/"Tehran_DW_ThreeClass_Statistics_2023.csv"
    return args


def main():
    args=parse_args()
    process_separate_probabilities_uncertainty(
        urban_prob_path=args.urban,
        veg_prob_path=args.vegetation,
        bare_prob_path=args.bare,
        scrub_prob_path=args.scrub,
        output_dir=args.output_dir,
        province_shapefile=args.province_shapefile,
        previous_statistics=args.previous_statistics,
        scale_factor=args.scale_factor,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
        confidence_weight=args.confidence_weight,
        minimum_probability_sum=args.minimum_probability_sum,
        block_size=args.block_size,
        pilot=args.pilot,
        pilot_size=args.pilot_size,
        explicit_pilot_window=args.pilot_window,
    )


if __name__ == "__main__":
    main()
