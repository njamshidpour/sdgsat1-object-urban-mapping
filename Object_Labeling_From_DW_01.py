"""Create high-confidence object labels from the adopted three-class DW raster.

Only reference pixels are used to derive labels. Object features and model
predictions are never read. Use --pilot for a representative validation run or
--full-run for all segmentation objects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

# Project inputs/outputs are deliberately explicit and visible at file start.
ROOT = Path(__file__).resolve().parent
SEGMENTATION_PATH = ROOT / "Object_based_fusion" / "Segmentation_output" / "eCog_segments.shp"
REFERENCE_RASTER_PATH = (ROOT / "GTs" / "DW_GT_ThreeClass_2023_weight_090_Sge070_TehranProvince"
                         / "Tehran_DW_ThreeClass_GT_2023.tif")
OUTPUT_DIRECTORY = ROOT / "Object_based_fusion" / "Segmentation_output" / "Object_Labeling_From_DW_01"
INDEPENDENT_TEST_POINTS_PATH = ROOT / "GTs" / "Test_Points_1000.shp"
ROOK_GRAPH_PATH = (ROOT / "Object_based_fusion" / "Segmentation_output"
                   / "rook_adjacency_graph_v01.npz")

site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
proj_data = site_packages / "rasterio" / "proj_data"
if proj_data.is_dir():
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import box, mapping
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*",
                        category=DeprecationWarning)

ID_FIELD = "Seg_id"
CLASS_NAMES = {1: "Urban", 2: "Vegetation", 3: "BareScrub"}
PRIMARY_PURITY = 0.90
PRIMARY_COVERAGE = 0.50
MIN_LABELED_PIXELS = 5
PURITY_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
COVERAGE_THRESHOLDS = (0.30, 0.50, 0.70)


class Progress:
    """Single-line progress display suitable for large processing runs."""
    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(1, int(total))
        self.start, self.previous = time.perf_counter(), 0

    @staticmethod
    def clock(seconds):
        if not np.isfinite(seconds):
            return "--:--"
        minutes, seconds = divmod(max(0, int(seconds)), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update(self, completed: int):
        completed = min(completed, self.total)
        elapsed = time.perf_counter() - self.start
        fraction = completed / self.total
        eta = elapsed / completed * (self.total-completed) if completed else np.nan
        filled = round(24*fraction)
        message = (f"\r{self.label:<28} [{'#'*filled}{'-'*(24-filled)}] "
                   f"{100*fraction:6.2f}% ETA {self.clock(eta)}")
        print(message + " "*max(0, self.previous-len(message)), end="", flush=True)
        self.previous = len(message)
        if completed == self.total:
            print()


def polygonal_part(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms
                 if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty]
        return unary_union(parts) if parts else None
    return None


def output_paths(directory: Path, pilot: bool):
    suffix = "_PILOT" if pilot else ""
    return {
        "master": directory / f"Object_Reference_Labels_DW_01{suffix}.csv",
        "training": directory / f"Object_Training_Labels_DW_HighConfidence_01{suffix}.csv",
        "sensitivity": directory / f"Object_Label_Threshold_Sensitivity{suffix}.csv",
        "class_summary": directory / f"Object_Labeling_Class_Summary{suffix}.csv",
        "rejection_summary": directory / f"Object_Labeling_Rejection_Summary{suffix}.csv",
        "purity_figure": directory / f"Object_Label_Purity_Histogram{suffix}.png",
        "coverage_figure": directory / f"Object_Label_Coverage_Histogram{suffix}.png",
        "scatter_figure": directory / f"Object_Purity_vs_Coverage{suffix}.png",
        "vector": directory / f"Objects_DW_Reference_Labels_01{suffix}.gpkg",
        "diagnostics": directory / f"Object_Labeling_From_DW_01_Diagnostics{suffix}.json",
        "report": directory / f"Object_Labeling_From_DW_01_Report{suffix}.md",
    }


def load_segmentation(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    original = gpd.read_file(path)
    if original.crs is None or ID_FIELD not in original:
        raise ValueError(f"Segmentation needs CRS metadata and field {ID_FIELD!r}.")
    if original[ID_FIELD].isna().any() or not original[ID_FIELD].is_unique:
        raise ValueError(f"{ID_FIELD} must be non-null and unique; field ID is intentionally ignored.")
    ids = original[ID_FIELD].to_numpy(dtype=np.int64)
    if not np.all(original[ID_FIELD].to_numpy() == ids):
        raise ValueError(f"{ID_FIELD} cannot be represented exactly as int64.")
    processing = original[[ID_FIELD, "geometry"]].copy()
    invalid = ~processing.geometry.is_valid
    repaired_count = int(invalid.sum())
    if repaired_count:
        processing.loc[invalid, "geometry"] = [polygonal_part(make_valid(g))
                                                for g in processing.loc[invalid, "geometry"]]
    unusable = processing.geometry.isna() | processing.geometry.is_empty | ~processing.geometry.is_valid
    if unusable.any():
        raise ValueError(f"{int(unusable.sum())} geometries remain unusable after make_valid.")
    return original, processing, repaired_count


def systematic_pilot(frame: gpd.GeoDataFrame, size: int):
    if size >= len(frame):
        return frame.copy()
    order = np.argsort(frame[ID_FIELD].to_numpy(dtype=np.int64), kind="stable")
    positions = np.unique(np.linspace(0, len(frame)-1, size).round().astype(int))
    return frame.iloc[order[positions]].copy()


def blocks(width: int, height: int, size: int):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def pixel_counts(polygons: gpd.GeoDataFrame, raster_path: Path, block_size: int):
    """Blockwise center-based rasterization; avoids a dense whole-scene object raster."""
    n = len(polygons)
    counts = np.zeros((n, 4), dtype=np.int64)  # columns: geometric total, class 1, 2, 3
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.count != 1:
            raise ValueError("Reference raster must be single-band with a defined CRS.")
        if polygons.crs != src.crs:
            polygons = polygons.to_crs(src.crs)
        if not box(*src.bounds).intersects(box(*polygons.total_bounds)):
            raise ValueError("Segmentation and reference-raster extents do not overlap.")
        possible_codes = set()
        index = polygons.sindex
        windows = list(blocks(src.width, src.height, block_size))
        progress = Progress("Pixel-to-object counts", len(windows))
        for k, window in enumerate(windows, 1):
            bounds = src.window_bounds(window)
            candidates = np.asarray(index.query(box(*bounds), predicate="intersects"), dtype=np.int64)
            if candidates.size:
                candidates.sort()
                shapes = [(mapping(polygons.geometry.iloc[i]), int(i)+1) for i in candidates]
                object_grid = rasterize(
                    shapes, out_shape=(int(window.height), int(window.width)),
                    transform=src.window_transform(window), fill=0,
                    all_touched=False, dtype="int32",
                )
                inside = object_grid > 0
                if inside.any():
                    labels = object_grid[inside]
                    values = src.read(1, window=window)[inside]
                    possible_codes.update(map(int, np.unique(values)))
                    counts[:, 0] += np.bincount(labels, minlength=n+1)[1:n+1]
                    for code in CLASS_NAMES:
                        counts[:, code] += np.bincount(
                            labels[values == code], minlength=n+1
                        )[1:n+1]
            progress.update(k)
        unexpected = possible_codes.difference({0, 1, 2, 3})
        if unexpected:
            raise ValueError(f"Unexpected reference codes inside segmentation: {sorted(unexpected)}")
        metadata = {
            "crs": str(src.crs), "shape": [src.height, src.width],
            "transform": tuple(src.transform), "resolution": list(src.res),
            "bounds": list(src.bounds), "nodata": src.nodata,
            "observed_codes_inside_objects": sorted(possible_codes),
        }
    return counts, polygons, metadata


def find_test_objects(test_path: Path, polygons_all: gpd.GeoDataFrame):
    info = {"available": False, "path": str(test_path.resolve()), "total_points": 0,
            "labeled_points": 0, "matched_points": 0, "unmatched_points": 0,
            "unique_test_objects": 0, "duplicate_points_within_objects": 0,
            "conflicting_test_objects": 0, "boundary_ambiguous_points": 0}
    if not test_path.is_file():
        return set(), set(), info
    points = gpd.read_file(test_path)
    if points.crs is None or "Class_ID" not in points or not points.geom_type.eq("Point").all():
        raise ValueError("Independent-test layer must contain Point geometry, CRS, and Class_ID.")
    info["available"], info["total_points"] = True, len(points)
    points = points.dropna(subset=["Class_ID"]).copy()
    info["labeled_points"] = len(points)
    raw_classes = points["Class_ID"].astype(int)
    unexpected = set(raw_classes.unique()).difference({1,2,3,4})
    if unexpected:
        raise ValueError(f"Unexpected manual test Class_ID values: {sorted(unexpected)}")
    # The established manual test layer uses the older four-class convention:
    # 1 Urban, 2 Vegetation, 3 Scrub, 4 Bare. For conflict detection only,
    # codes 3 and 4 are the same adopted three-class BareScrub label.
    points["Adopted_Test_Class"] = raw_classes.replace({4:3})
    info["class_mapping_for_conflict_check"] = {
        "1":"Urban", "2":"Vegetation", "3":"BareScrub", "4":"BareScrub"
    }
    if points.crs != polygons_all.crs:
        points = points.to_crs(polygons_all.crs)
    joined = gpd.sjoin(points, polygons_all[[ID_FIELD, "geometry"]], how="left", predicate="within")
    matched = joined.dropna(subset=[ID_FIELD]).copy()
    info["matched_points"] = len(matched)
    info["unmatched_points"] = len(points)-len(matched)
    # A strict within join should produce one row per point for a non-overlapping segmentation.
    per_point = matched.groupby(matched.index)[ID_FIELD].nunique()
    info["boundary_ambiguous_points"] = int((per_point > 1).sum())
    if info["boundary_ambiguous_points"]:
        raise ValueError("Some independent test points map within multiple segmentation polygons.")
    grouped = matched.groupby(ID_FIELD)["Adopted_Test_Class"]
    test_ids = set(map(int, grouped.size().index))
    conflicts = set(map(int, grouped.nunique()[lambda value: value > 1].index))
    info["unique_test_objects"] = len(test_ids)
    info["duplicate_points_within_objects"] = len(matched)-len(test_ids)
    info["conflicting_test_objects"] = len(conflicts)
    return test_ids, conflicts, info


def rook_neighbors(graph_path: Path, test_ids: set[int], expected_ids: np.ndarray):
    if not graph_path.is_file() or not test_ids:
        return set(), {"available": graph_path.is_file(), "path": str(graph_path.resolve()),
                       "neighbor_count": 0}
    with np.load(graph_path, allow_pickle=False) as graph:
        required = {"src_object_id", "dst_object_id", "object_id", "neighbor_count"}
        if not required.issubset(graph.files):
            raise ValueError(f"Rook graph lacks arrays: {sorted(required.difference(graph.files))}")
        src, dst = graph["src_object_id"], graph["dst_object_id"]
        object_ids = graph["object_id"]
        if len(object_ids) != len(expected_ids) or set(object_ids) != set(expected_ids):
            raise ValueError("Rook graph object IDs do not match the segmentation Seg_id values.")
        test = np.fromiter(test_ids, dtype=np.int64)
        neighbors = set(map(int, np.concatenate((dst[np.isin(src, test)], src[np.isin(dst, test)]))))
        neighbors.difference_update(test_ids)
    return neighbors, {"available": True, "path": str(graph_path.resolve()),
                       "edge_count": len(src), "neighbor_count": len(neighbors)}


def make_label_table(polygons, counts, areas, test_ids, conflicts, neighbors):
    total = counts[:, 0]
    class_counts = counts[:, 1:]
    labeled = class_counts.sum(axis=1)
    proportions = np.divide(class_counts, labeled[:, None],
                            out=np.full(class_counts.shape, np.nan, dtype=float),
                            where=labeled[:, None] > 0)
    dominant_index = np.argmax(np.nan_to_num(proportions, nan=-1), axis=1)
    dominant_code = np.where(labeled > 0, dominant_index+1, 0)
    purity = np.where(labeled > 0, np.max(np.nan_to_num(proportions, nan=-1), axis=1), np.nan)
    coverage = np.divide(labeled, total, out=np.full(len(total), np.nan, dtype=float), where=total > 0)
    high = ((purity >= PRIMARY_PURITY) & (coverage >= PRIMARY_COVERAGE)
            & (labeled >= MIN_LABELED_PIXELS))
    object_ids = polygons[ID_FIELD].to_numpy(dtype=np.int64)
    is_test = np.isin(object_ids, np.fromiter(test_ids, dtype=np.int64)) if test_ids else np.zeros(len(total), bool)
    is_conflict = np.isin(object_ids, np.fromiter(conflicts, dtype=np.int64)) if conflicts else np.zeros(len(total), bool)
    is_neighbor = np.isin(object_ids, np.fromiter(neighbors, dtype=np.int64)) if neighbors else np.zeros(len(total), bool)
    eligible = high & ~is_test & ~is_neighbor
    table = pd.DataFrame({
        "Object_ID": object_ids, "N_valid_pixels": total, "N_labeled_pixels": labeled,
        "N_Urban": class_counts[:, 0], "N_Vegetation": class_counts[:, 1],
        "N_BareScrub": class_counts[:, 2], "P_Urban": proportions[:, 0],
        "P_Vegetation": proportions[:, 1], "P_BareScrub": proportions[:, 2],
        "DominantClass": [CLASS_NAMES.get(int(code), np.nan) for code in dominant_code],
        "DominantClassCode": pd.array(np.where(dominant_code > 0, dominant_code, None), dtype="Int64"),
        "Purity": purity, "Coverage": coverage, "HighConfidence": high.astype(np.uint8),
        "FinalTrainingLabel": pd.array(np.where(high, dominant_code, None), dtype="Int64"),
        "FinalTrainingClass": [CLASS_NAMES.get(int(code), np.nan) if accepted else np.nan
                               for code, accepted in zip(dominant_code, high)],
        "IsIndependentTestObject": is_test.astype(np.uint8),
        "ConflictingTestLabels": is_conflict.astype(np.uint8),
        "IsTestNeighbor": is_neighbor.astype(np.uint8),
        "EligibleForTraining": eligible.astype(np.uint8), "Object_Area_m2": areas,
    })
    return table


def sensitivity_table(table):
    rows = []
    for purity in PURITY_THRESHOLDS:
        for coverage in COVERAGE_THRESHOLDS:
            accepted = ((table.Purity >= purity) & (table.Coverage >= coverage)
                        & (table.N_labeled_pixels >= MIN_LABELED_PIXELS))
            row = {"PurityThreshold": purity, "CoverageThreshold": coverage,
                   "MinLabeledPixels": MIN_LABELED_PIXELS, "TotalAccepted": int(accepted.sum())}
            for code, name in CLASS_NAMES.items():
                row[f"{name}_Accepted"] = int((accepted & table.DominantClassCode.eq(code).fillna(False)).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def class_summary(table):
    rows = []
    for code, name in CLASS_NAMES.items():
        values = table[(table.HighConfidence == 1) & table.DominantClassCode.eq(code)].copy()
        row = {"Class_Code": code, "Class_Name": name, "Accepted_Objects": len(values),
               "Pct_of_All_Accepted": 100*len(values)/max(1, int(table.HighConfidence.sum())),
               "Eligible_Training_Objects": int(values.EligibleForTraining.sum())}
        for column in ("Purity", "Coverage", "N_labeled_pixels", "Object_Area_m2"):
            series = values[column].dropna()
            for stat, value in (("Min", series.min()), ("Q25", series.quantile(.25)),
                                ("Median", series.median()), ("Mean", series.mean()),
                                ("Q75", series.quantile(.75)), ("Max", series.max())):
                row[f"{column}_{stat}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def rejection_summary(table):
    fail_purity = table.N_labeled_pixels.gt(0) & table.Purity.lt(PRIMARY_PURITY)
    fail_coverage = table.N_valid_pixels.eq(0) | table.Coverage.lt(PRIMARY_COVERAGE)
    fail_minimum = table.N_labeled_pixels.lt(MIN_LABELED_PIXELS)
    rejected = table.HighConfidence.eq(0)
    rows = [
        {"Rejection_Category": "A: Purity < 0.90", "Definition": "inclusive; may overlap B/C",
         "Object_Count": int((rejected & fail_purity).sum())},
        {"Rejection_Category": "B: Coverage < 0.50 or undefined", "Definition": "inclusive; may overlap A/C",
         "Object_Count": int((rejected & fail_coverage).sum())},
        {"Rejection_Category": "C: N_labeled_pixels < 5", "Definition": "inclusive; may overlap A/B",
         "Object_Count": int((rejected & fail_minimum).sum())},
    ]
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                mask = rejected & fail_purity.eq(a) & fail_coverage.eq(b) & fail_minimum.eq(c)
                if mask.any():
                    labels = [label for flag,label in zip((a,b,c),("A","B","C")) if flag]
                    rows.append({"Rejection_Category": "Exclusive combination: "+("+".join(labels) or "other/undefined"),
                                 "Definition": "mutually exclusive", "Object_Count": int(mask.sum())})
    rows.extend([
        {"Rejection_Category": "High purity but low coverage",
         "Definition": "Purity >= 0.90 and Coverage < 0.50",
         "Object_Count": int((table.Purity.ge(.90) & table.Coverage.lt(.50)).sum())},
        {"Rejection_Category": "High coverage but low purity",
         "Definition": "Coverage >= 0.50 and Purity < 0.90",
         "Object_Count": int((table.Coverage.ge(.50) & table.Purity.lt(.90)).sum())},
    ])
    return pd.DataFrame(rows)


def figures(table, paths):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    labeled = table[table.N_labeled_pixels > 0]
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    ax.hist(labeled.Purity, bins=np.linspace(1/3,1,55), color="#237a8b", edgecolor="none")
    ax.axvline(PRIMARY_PURITY, color="#c62828", linestyle="--", linewidth=1.8,
               label="Primary threshold = 0.90")
    ax.set(title="Object-label purity", xlabel="Purity", ylabel="Number of objects", xlim=(1/3,1))
    ax.legend(frameon=False); ax.grid(axis="y", color=".9"); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(paths["purity_figure"], dpi=250, bbox_inches="tight"); plt.close(fig)

    valid = table[table.N_valid_pixels > 0]
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    ax.hist(valid.Coverage, bins=np.linspace(0,1,55), color="#3a923a", edgecolor="none")
    ax.axvline(PRIMARY_COVERAGE, color="#c62828", linestyle="--", linewidth=1.8,
               label="Primary threshold = 0.50")
    ax.set(title="Object reference coverage", xlabel="Coverage", ylabel="Number of objects", xlim=(0,1))
    ax.legend(frameon=False); ax.grid(axis="y", color=".9"); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(paths["coverage_figure"], dpi=250, bbox_inches="tight"); plt.close(fig)

    plot = labeled.dropna(subset=["Purity","Coverage"])
    if len(plot) > 100000:
        plot = plot.iloc[np.linspace(0,len(plot)-1,100000).round().astype(int)]
    accepted = plot.HighConfidence.eq(1)
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.scatter(plot.loc[~accepted,"Coverage"], plot.loc[~accepted,"Purity"], s=3,
               alpha=.18, color="#777777", label="Not high confidence", rasterized=True)
    ax.scatter(plot.loc[accepted,"Coverage"], plot.loc[accepted,"Purity"], s=4,
               alpha=.28, color="#2166ac", label="Passes all three conditions", rasterized=True)
    ax.axvline(.50,color="#c62828",linestyle="--",linewidth=1.2); ax.axhline(.90,color="#c62828",linestyle="--",linewidth=1.2)
    ax.set(title="Purity versus reference coverage", xlabel="Coverage", ylabel="Purity",
           xlim=(0,1), ylim=(1/3,1)); ax.legend(frameon=False, markerscale=3)
    ax.grid(color=".92"); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(paths["scatter_figure"], dpi=250, bbox_inches="tight"); plt.close(fig)


def markdown_table(frame, columns=None, digits=4):
    frame = frame if columns is None else frame[columns]
    headers = list(frame.columns)
    lines = ["| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |"]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if pd.isna(value): rendered.append("NA")
            elif isinstance(value, (float,np.floating)): rendered.append(f"{value:.{digits}f}")
            elif isinstance(value, (int,np.integer)): rendered.append(f"{int(value):,}")
            else: rendered.append(str(value).replace("|","\\|"))
        lines.append("| "+" | ".join(rendered)+" |")
    return "\n".join(lines)


def write_report(args, paths, raster_meta, segmentation_crs, repaired_count, table,
                 sensitivity, classes, rejection, test_info, graph_info, validation, runtime):
    accepted = int(table.HighConfidence.sum()); eligible = int(table.EligibleForTraining.sum())
    class_compact = classes[["Class_Code","Class_Name","Accepted_Objects",
                             "Pct_of_All_Accepted","Eligible_Training_Objects"]]
    primary_row = sensitivity[(sensitivity.PurityThreshold.eq(.90))
                              & sensitivity.CoverageThreshold.eq(.50)]
    loose = sensitivity[(sensitivity.PurityThreshold.eq(.80)) & sensitivity.CoverageThreshold.eq(.30)]
    strict = sensitivity[(sensitivity.PurityThreshold.eq(.95)) & sensitivity.CoverageThreshold.eq(.70)]
    outputs = "\n".join(f"- {name}: `{path.resolve()}`" for name,path in paths.items())
    mode = "PILOT" if args.pilot else "FULL DATASET"
    if args.pilot:
        conclusion = "Class balance and adequacy must be judged from the full run; this pilot validates the method only."
    else:
        eligible_counts = classes.Eligible_Training_Objects.to_numpy(dtype=int)
        ratio = eligible_counts.max()/max(1,eligible_counts.min())
        size_text = ("sufficiently large for subsequent modeling diagnostics" if eligible_counts.min() >= 1000
                     else "limited for at least one class and requiring careful downstream evaluation")
        balance_text = ("reasonably balanced" if ratio <= 2 else
                        f"class-imbalanced (largest/smallest eligible count ratio {ratio:.2f})")
        conclusion = (f"The eligible confidence-filtered set is {size_text}, but is {balance_text}. "
                      "This script did not balance classes, and thresholds should not be weakened solely to increase counts.")
    text = f"""# Object Labeling From Dynamic World 01 — {mode}

## A. Inputs

- Segmentation: `{args.segmentation.resolve()}`
- Pixel reference: `{args.reference.resolve()}`
- Output directory: `{args.output_dir.resolve()}`
- Segmentation CRS: `{segmentation_crs}`; raster CRS: `{raster_meta['crs']}`
- Raster shape/resolution: `{raster_meta['shape']}` / `{raster_meta['resolution']}`; transform `{raster_meta['transform']}`
- Raster NoData metadata: `{raster_meta['nodata']}`; adopted codes: `0=uncertain/NoData`, `1=Urban`, `2=Vegetation`, `3=Bare soil + Scrub`.
- Object ID: source `Seg_id`, copied exactly to `Object_ID`; source field `ID` was not used.
- Mode: **{mode}**; processed objects `{len(table):,}`. Full segmentation contains `{validation['full_object_count']:,}` objects.

## B. Pixel-to-object labeling method

Polygons were transformed to the reference CRS only when necessary; the classification raster was never resampled or modified. Blockwise rasterization used pixel-center inclusion (`all_touched=False`). Invalid processing copies were repaired with `make_valid` and polygonal-part retention ({repaired_count:,} in the full layer); output vector geometries remain the original geometries.

For object *i* and class *c*, `n_ic` is the number of class-*c* pixel centers inside the object, `N_labeled_i = Σ_c n_ic`, `p_ic = n_ic / N_labeled_i`, `c*_i = argmax_c(p_ic)`, `Purity_i = max_c(p_ic)`, and `Coverage_i = N_labeled_i / N_valid_i`.

Because the adopted GT stores both uncertain and NoData as code 0, they cannot be separated from this file. `N_valid_i` is therefore the number of reference-grid pixel centers geometrically inside the object/raster extent; code 0 contributes to this denominator but never to a class count or `N_labeled_i`. This avoids the scientifically invalid result `Coverage=1` for every labeled object.

## C. Primary acceptance rule

`HighConfidence = 1` only when **Purity ≥ 0.90**, **Coverage ≥ 0.50**, and **N_labeled_pixels ≥ 5**. Thresholds were not tuned or relaxed. Objects failing any condition retain diagnostic dominance but `FinalTrainingLabel` is null.

## D. Dataset statistics

| Metric | Count / percent |
|---|---:|
| Processed objects | {len(table):,} |
| Zero geometric raster pixels | {int(table.N_valid_pixels.eq(0).sum()):,} |
| Valid geometry pixels but zero accepted labels | {int((table.N_valid_pixels.gt(0)&table.N_labeled_pixels.eq(0)).sum()):,} |
| High-confidence accepted | {accepted:,} ({100*accepted/len(table):.3f}%) |
| Rejected | {len(table)-accepted:,} ({100*(len(table)-accepted)/len(table):.3f}%) |
| Eligible after test/test-neighbor exclusion | {eligible:,} |

{markdown_table(class_compact)}

The class-summary CSV additionally contains min/Q25/median/mean/Q75/max for Purity, Coverage, labeled-pixel count, and metric object area.

## E. Rejection analysis

{markdown_table(rejection)}

Inclusive A/B/C rows overlap; rows explicitly named “Exclusive combination” are mutually exclusive. Undefined purity for zero labeled pixels is not falsely counted as low purity.

## F. Threshold sensitivity

The complete 4×3 table is saved separately. Diagnostic endpoints and the unchanged primary rule are:

{markdown_table(pd.concat([loose,primary_row,strict], ignore_index=True))}

This table is diagnostic only; no alternative was selected to maximize sample count.

## G. Independent-test exclusion

- Test layer available: `{test_info['available']}` at `{test_info['path']}`.
- Total/labeled/matched/unmatched test points: `{test_info['total_points']:,}` / `{test_info['labeled_points']:,}` / `{test_info['matched_points']:,}` / `{test_info['unmatched_points']:,}`.
- Unique test objects: `{test_info['unique_test_objects']:,}`; duplicate test points within objects: `{test_info['duplicate_points_within_objects']:,}`; conflicting test objects: `{test_info['conflicting_test_objects']:,}`.
- Saved Rook graph used: `{graph_info['available']}` at `{graph_info['path']}`; first-order non-test neighboring objects: `{graph_info['neighbor_count']:,}`.
- `EligibleForTraining=1` requires HighConfidence and exclusion of test objects and first-order Rook neighbors. Nothing was deleted from the master table.

`Test_Points_1000.shp` was recognized because the project uses it in `manual_test_import.py`; `New_Test_Points.shp` was not used because its 1,000 Class_ID values are all zero/unlabeled. For conflict detection, the older manual codes 3=Scrub and 4=Bare were both mapped to the adopted three-class BareScrub class. Manual test classes were used only to detect within-object conflicts, never to derive Dynamic World training labels.

## H. Validation

{chr(10).join('- '+item for item in validation['messages'])}

## I. Deviations and limitations

- The merged raster code 0 prevents separate counts for physical raster NoData versus uncertainty; the center-based geometric denominator described above is used for Coverage.
- This {'pilot is a systematic Object_ID sample and is not a final class-balance estimate' if args.pilot else 'full run processed every segmentation object'}.
- Strict `within` was used for point-to-object assignment; unmatched points were not forced into a nearest object.

## J. Final summary

{conclusion} Labels were derived solely from the adopted pixel reference. No feature table, classifier prediction, class balancing, undersampling, or threshold optimization was used.

## Outputs

{outputs}

Runtime: `{runtime:.2f}` seconds.
"""
    paths["report"].write_text(text, encoding="utf-8")


def run(args):
    started = time.perf_counter()
    print("Resolved paths:")
    print(f"  SEGMENTATION_PATH    = {args.segmentation.resolve()}")
    print(f"  REFERENCE_RASTER_PATH= {args.reference.resolve()}")
    print(f"  OUTPUT_DIRECTORY     = {args.output_dir.resolve()}")
    for path in (args.segmentation,args.reference):
        if not path.is_file(): raise FileNotFoundError(path)
    paths = output_paths(args.output_dir,args.pilot)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output(s) exist; use --overwrite: "+", ".join(map(str,existing)))
    args.output_dir.mkdir(parents=True,exist_ok=True)

    print("Loading segmentation and validating Seg_id...")
    original_all, processing_all, repaired_count = load_segmentation(args.segmentation)
    full_ids = original_all[ID_FIELD].to_numpy(dtype=np.int64)
    test_ids, conflicts, test_info = find_test_objects(args.test_points, original_all)
    neighbors, graph_info = rook_neighbors(args.rook_graph,test_ids,full_ids)
    if args.pilot:
        chosen = systematic_pilot(processing_all,args.pilot_size).index
        processing = processing_all.loc[chosen].reset_index(drop=True)
        original = original_all.loc[chosen].reset_index(drop=True)
    else:
        processing = processing_all.reset_index(drop=True)
        original = original_all.reset_index(drop=True)
    counts, processing_raster_crs, raster_meta = pixel_counts(processing,args.reference,args.block_size)
    areas = processing.to_crs(args.area_crs).geometry.area.to_numpy(dtype=float)
    table = make_label_table(processing,counts,areas,test_ids,conflicts,neighbors)

    # Independent mathematical validations.
    duplicate_ids = int(table.Object_ID.duplicated().sum())
    labeled_mask = table.N_labeled_pixels.gt(0)
    proportion_sum_error = float(np.nanmax(np.abs(table.loc[labeled_mask,
        ["P_Urban","P_Vegetation","P_BareScrub"]].sum(axis=1)-1))) if labeled_mask.any() else 0.0
    purity_error = float(np.nanmax(np.abs(table.loc[labeled_mask,"Purity"]-
        table.loc[labeled_mask,["P_Urban","P_Vegetation","P_BareScrub"]].max(axis=1)))) if labeled_mask.any() else 0.0
    expected_dominant = (table.loc[labeled_mask,["P_Urban","P_Vegetation","P_BareScrub"]]
                         .to_numpy().argmax(axis=1)+1)
    dominant_failures = int(np.count_nonzero(expected_dominant !=
                            table.loc[labeled_mask,"DominantClassCode"].to_numpy(dtype=int)))
    high = table.HighConfidence.eq(1)
    expected_high = (table.Purity.ge(.90) & table.Coverage.ge(.50)
                     & table.N_labeled_pixels.ge(5))
    rule_failures = int((high != expected_high).sum())
    if duplicate_ids or proportion_sum_error > 1e-12 or purity_error > 1e-12 or dominant_failures or rule_failures:
        raise RuntimeError("Object-label mathematical validation failed.")

    sensitivity = sensitivity_table(table)
    classes = class_summary(table)
    rejection = rejection_summary(table)
    table.to_csv(paths["master"],index=False)
    training = table[table.EligibleForTraining.eq(1)][
        ["Object_ID","FinalTrainingLabel","FinalTrainingClass","Purity","Coverage","N_labeled_pixels"]]
    training.to_csv(paths["training"],index=False)
    sensitivity.to_csv(paths["sensitivity"],index=False)
    classes.to_csv(paths["class_summary"],index=False)
    rejection.to_csv(paths["rejection_summary"],index=False)
    figures(table,paths)

    vector = original.copy()
    vector = vector.rename(columns={ID_FIELD:"Object_ID"})
    vector = vector.merge(table.drop(columns="Object_Area_m2"),on="Object_ID",validate="one_to_one")
    if paths["vector"].exists(): paths["vector"].unlink()
    vector.to_file(paths["vector"],layer="object_reference_labels",driver="GPKG")
    saved_vector = gpd.read_file(paths["vector"])
    expected_vector = vector.sort_values("Object_ID").reset_index(drop=True)
    saved_vector = saved_vector.sort_values("Object_ID").reset_index(drop=True)
    vector_reload_ok = (
        len(saved_vector) == len(expected_vector)
        and saved_vector.crs == expected_vector.crs
        and saved_vector.Object_ID.equals(expected_vector.Object_ID)
        and all(left.equals(right) for left,right in
                zip(saved_vector.geometry,expected_vector.geometry))
    )
    if not vector_reload_ok:
        raise RuntimeError("Reloaded output vector does not preserve input geometry/ID alignment.")

    validation = {"full_object_count":len(original_all), "processed_object_count":len(table),
                  "duplicate_ids":duplicate_ids,"proportion_sum_max_error":proportion_sum_error,
                  "purity_max_error":purity_error,"dominant_failures":dominant_failures,
                  "rule_failures":rule_failures,
                  "vector_reload_geometry_and_id_check":vector_reload_ok,
                  "messages":[
                      f"PASS: Seg_id is non-null and unique; duplicate output Object_ID count = {duplicate_ids}.",
                      f"PASS: maximum class-proportion sum error = {proportion_sum_error:.3g}.",
                      f"PASS: maximum Purity-versus-max-proportion error = {purity_error:.3g}.",
                      f"PASS: dominant-class argmax failures = {dominant_failures}.",
                      f"PASS: primary-rule reproduction failures = {rule_failures}.",
                      "PASS: raster CRS/extent and polygon CRS/extent overlap; raster was read-only and not resampled.",
                      "PASS: reloaded output vector retains source geometry/CRS and attaches attributes one-to-one by Object_ID.",
                  ]}
    diagnostics={"mode":"PILOT" if args.pilot else "FULL_DATASET","inputs":{
        "segmentation":str(args.segmentation.resolve()),"reference":str(args.reference.resolve()),
        "test_points":str(args.test_points.resolve()),"rook_graph":str(args.rook_graph.resolve())},
        "raster":raster_meta,"geometry_repaired_processing_copy":repaired_count,
        "test":test_info,"rook":graph_info,"validation":validation,
        "output_counts":{"high_confidence":int(table.HighConfidence.sum()),
                         "eligible_training":len(training)}}
    paths["diagnostics"].write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    write_report(args,paths,raster_meta,str(original_all.crs),repaired_count,table,
                 sensitivity,classes,rejection,test_info,graph_info,validation,
                 time.perf_counter()-started)
    print(f"Completed: {len(table):,} objects; high confidence {int(table.HighConfidence.sum()):,}; "
          f"training eligible {len(training):,}.")
    print(f"Report: {paths['report'].resolve()}")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot",action="store_true",help="Run a systematic representative object sample.")
    mode.add_argument("--full-run",action="store_true",help="Process all 336,109 objects.")
    parser.add_argument("--segmentation",type=Path,default=SEGMENTATION_PATH)
    parser.add_argument("--reference",type=Path,default=REFERENCE_RASTER_PATH)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--test-points",type=Path,default=INDEPENDENT_TEST_POINTS_PATH)
    parser.add_argument("--rook-graph",type=Path,default=ROOK_GRAPH_PATH)
    parser.add_argument("--area-crs",default="EPSG:32639")
    parser.add_argument("--pilot-size",type=int,default=5000)
    parser.add_argument("--block-size",type=int,default=1024)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if args.output_dir is None:
        args.output_dir=Path(str(OUTPUT_DIRECTORY)+( "_pilot" if args.pilot else ""))
    if args.pilot_size <= 0 or args.block_size <= 0:
        parser.error("pilot-size and block-size must be positive.")
    run(args)


if __name__ == "__main__":
    main()
