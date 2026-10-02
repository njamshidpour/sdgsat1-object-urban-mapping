"""Propose 5-km Tehran test blocks without creating test labels or changing the GT.

The selection is deterministic. All native-resolution reference statistics use
pixel-centre inclusion inside both the province and each unmodified square.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
proj_data = site_packages / "rasterio" / "proj_data"
if proj_data.is_dir():
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping, shape

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*",
                        category=DeprecationWarning)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROVINCE = ROOT / "Shapefile" / "Tehran_True_Province.shp"
DEFAULT_REFERENCE = (ROOT / "GTs" / "DW_GT_ThreeClass_2023_weight_090_Sge070_TehranProvince"
                     / "Tehran_DW_ThreeClass_GT_2023.tif")
DEFAULT_OUTPUT = ROOT / "GTs" / "Tehran_ThreeClass_Test_Blocks_5km_2023"
CRS = "EPSG:32639"
CLASS = {1: "Urban", 2: "Vegetation", 3: "Bare_Sparse"}
CATEGORY_TARGETS = {"Urban-dominated": 2, "Vegetation-dominated": 2,
                    "Urban-transition": 3, "Mixed": 2, "Bare-dominated": 3}
CATEGORY_COLOR = {"Urban-dominated": "#d73027", "Vegetation-dominated": "#1a9850",
                  "Bare-dominated": "#c49a41", "Urban-transition": "#8e44ad",
                  "Mixed": "#2c7fb8", "Insufficient-reference": "#888888"}


def progress(label: str, done: int, total: int) -> None:
    if done == total or done == 1 or done % max(1, total // 20) == 0:
        print(f"\r{label}: {done:,}/{total:,} ({done/total:.0%})", end="" if done < total else "\n", flush=True)


def load_province(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = gpd.read_file(path)
    if frame.empty or frame.crs is None or frame.geometry.isna().any():
        raise ValueError("Province shapefile is empty or lacks valid geometry/CRS.")
    if not frame.geometry.is_valid.all():
        raise ValueError("Province boundary contains invalid geometry.")
    province = frame.to_crs(CRS).geometry.union_all()
    if province.is_empty:
        raise ValueError("Province geometry is empty after reprojection.")
    return province, str(frame.crs)


def make_grid(province, grid_size: int, overlap_min: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = province.bounds
    e0, e1 = math.floor(minx/grid_size), math.ceil(maxx/grid_size)
    n0, n1 = math.floor(miny/grid_size), math.ceil(maxy/grid_size)
    records = []
    for north in range(n0, n1):
        for east in range(e0, e1):
            square = box(east*grid_size, north*grid_size,
                         (east+1)*grid_size, (north+1)*grid_size)
            overlap = square.intersection(province).area / square.area
            if overlap + 1e-12 < overlap_min:
                continue
            records.append({"Grid_ID": f"G{grid_size//1000 if grid_size%1000 == 0 else str(grid_size)+'m'}_E{east:03d}_N{north:04d}",
                            "Grid_E": east, "Grid_N": north,
                            "Province_Overlap_Pct": 100*overlap,
                            "Center_E_m": square.centroid.x,
                            "Center_N_m": square.centroid.y, "geometry": square})
    if not records:
        raise ValueError("No grid cell meets the province-overlap threshold.")
    return gpd.GeoDataFrame(records, crs=CRS).sort_values("Grid_ID").reset_index(drop=True)


def region_codes(grid: gpd.GeoDataFrame, province) -> None:
    xmin, ymin, xmax, ymax = province.bounds
    ex = np.clip((3*(grid.Center_E_m-xmin)/(xmax-xmin)).astype(int), 0, 2)
    ny = np.clip((3*(grid.Center_N_m-ymin)/(ymax-ymin)).astype(int), 0, 2)
    grid["Region"] = [f"R{n}{e}" for e, n in zip(ex, ny)]


def masked_values(src, window: Window, geom, province_geom):
    values = src.read(1, window=window)
    transform = src.window_transform(window)
    in_cell = geometry_mask([geom], out_shape=values.shape, transform=transform,
                            invert=True, all_touched=False)
    in_province = geometry_mask([province_geom], out_shape=values.shape,
                                transform=transform, invert=True, all_touched=False)
    return values, in_cell & in_province


def province_counts(src, province_geom):
    counts = np.zeros(4, dtype=np.int64)
    outside = 0
    block = 2048
    rows = list(range(0, src.height, block))
    for k, row in enumerate(rows, 1):
        for col in range(0, src.width, block):
            window = Window(col, row, min(block, src.width-col), min(block, src.height-row))
            values = src.read(1, window=window)
            mask = geometry_mask([province_geom], out_shape=values.shape,
                                 transform=src.window_transform(window), invert=True,
                                 all_touched=False)
            if np.any(mask & (values > 3)):
                raise ValueError("Reference contains an unexpected class code inside the province.")
            counts += np.bincount(values[mask], minlength=4)[:4]
            outside += int((~mask).sum())
        progress("Province reference pool", k, len(rows))
    if np.any(counts[1:] == 0):
        raise ValueError(f"One or more confident classes have no province-wide pixels: {counts.tolist()}")
    return counts, outside


def cell_counts(src, grid, province_geom):
    data = []
    for k, row in enumerate(grid.itertuples(index=False), 1):
        # Densify square edges before reprojection so its raster-space bounds
        # safely contain every curved projected edge.
        polygon = shape(transform_geom(CRS, src.crs, mapping(row.geometry.segmentize(500))))
        left, bottom, right, top = polygon.bounds
        raw = from_bounds(left, bottom, right, top, src.transform)
        c0, r0 = max(0, math.floor(raw.col_off)), max(0, math.floor(raw.row_off))
        c1 = min(src.width, math.ceil(raw.col_off+raw.width))
        r1 = min(src.height, math.ceil(raw.row_off+raw.height))
        counts = np.zeros(4, dtype=np.int64)
        if c1 > c0 and r1 > r0:
            values, mask = masked_values(src, Window(c0, r0, c1-c0, r1-r0),
                                         mapping(polygon), province_geom)
            if np.any(mask & (values > 3)):
                raise ValueError(f"Unexpected class code in {row.Grid_ID}.")
            counts = np.bincount(values[mask], minlength=4)[:4]
        data.append(counts)
        progress("5-km grid statistics", k, len(grid))
    return np.asarray(data, dtype=np.int64)


def add_statistics(grid, cells, pool, minimum_confident):
    confident = cells[:, 1:].sum(axis=1)
    inside = cells.sum(axis=1)
    shares = np.divide(cells[:, 1:], confident[:, None],
                       out=np.zeros((len(grid), 3), dtype=float), where=confident[:, None] > 0)
    terms = np.zeros_like(shares)
    positive = shares > 0
    terms[positive] = -shares[positive]*np.log2(shares[positive])
    grid["Province_Pixels"] = inside
    grid["Confident_Pixels"] = confident
    grid["Uncertain_Pixels"] = cells[:, 0]
    grid["Confident_Pct"] = np.divide(100*confident, inside, out=np.zeros(len(grid)), where=inside > 0)
    grid["Class_Diversity"] = (shares > 0).sum(axis=1)
    grid["Shannon_Entropy"] = terms.sum(axis=1)
    for code, label in CLASS.items():
        grid[f"{label}_Count"] = cells[:, code]
        grid[f"{label}_Pct_of_Confident"] = 100*shares[:, code-1]
        grid[f"{label}_Pct_of_Province_Pool"] = 100*cells[:, code]/pool[code]
    grid["Dominant_Class"] = [CLASS[int(np.argmax(row))+1] if total >= minimum_confident
                              else "Insufficient-reference" for row, total in zip(shares, confident)]
    categories = []
    for total, (urban, vegetation, bare) in zip(confident, shares):
        if total < minimum_confident:
            category = "Insufficient-reference"
        elif urban >= .40:
            category = "Urban-dominated"
        elif urban >= .06 and urban < .40 and vegetation >= .05 and bare >= .05:
            category = "Urban-transition"
        elif vegetation >= .40:
            category = "Vegetation-dominated"
        elif bare >= .65:
            category = "Bare-dominated"
        else:
            category = "Mixed"
        categories.append(category)
    grid["Landscape_Category"] = categories
    grid["Selected"] = False
    grid["Selection_Rank"] = 0
    grid["Selection_Category"] = ""
    return grid


def select_blocks(grid, pool, target, min_count, max_loss, grid_size):
    """Greedy category-stratified maximin selection with cumulative budgets."""
    count_columns = [f"{name}_Count" for name in CLASS.values()]
    counts = grid[count_columns].to_numpy(dtype=np.int64)
    region_names = sorted(grid.Region.unique())
    region_totals = {name: counts[grid.Region.eq(name).to_numpy()].sum(axis=0)
                     for name in region_names}
    eligible = np.flatnonzero(grid.Confident_Pixels.to_numpy() >= min_count)
    coords = grid[["Center_E_m", "Center_N_m"]].to_numpy(dtype=float)
    requested = list(CATEGORY_TARGETS)

    def attempt(cap, distance_min):
        chosen = []
        lost = np.zeros(3, dtype=np.int64)
        regional_lost = {name: np.zeros(3, dtype=np.int64) for name in region_names}
        category_counts = {name: 0 for name in requested}
        chosen_regions = set()
        while len(chosen) < target:
            wanted = [name for name in requested
                      if category_counts[name] < CATEGORY_TARGETS[name]]
            # Fill missing strata from other eligible categories if necessary.
            phases = (wanted, requested) if wanted else (requested,)
            best = None
            for category_set in phases:
                for index in eligible:
                    if index in chosen or grid.at[index, "Landscape_Category"] not in category_set:
                        continue
                    prospective = lost + counts[index]
                    if np.any(prospective > cap*pool[1:] + 1e-9):
                        continue
                    region = grid.at[index, "Region"]
                    support = region_totals[region]
                    meaningful = support >= .005*pool[1:]
                    if np.any((regional_lost[region]+counts[index])[meaningful] > .25*support[meaningful]):
                        continue
                    distance = (min(np.linalg.norm(coords[index]-coords[chosen], axis=1))
                                if chosen else 60000.0)
                    if distance + 1e-8 < distance_min:
                        continue
                    category = grid.at[index, "Landscape_Category"]
                    deficit = max(0, CATEGORY_TARGETS[category]-category_counts[category])
                    reference_share = grid.at[index, "Confident_Pct"] / 100
                    diversity = grid.at[index, "Shannon_Entropy"] / math.log2(3)
                    increment = max(counts[index]/pool[1:]) / cap
                    # Stratification and distance dominate; reference quantity
                    # contributes only a small, capped adequacy term.
                    score = (3.0*(deficit > 0) + 1.7*(region not in chosen_regions)
                             + 1.5*min(distance/30000, 1) + .50*diversity
                             + .18*min(reference_share/.30, 1)
                             + .12*grid.at[index, "Province_Overlap_Pct"]/100
                             - .9*increment)
                    key = (score, -increment, grid.at[index, "Grid_ID"])
                    if best is None or key > best[0]:
                        best = (key, index)
                if best is not None:
                    break
            if best is None:
                break
            index = best[1]
            chosen.append(index)
            lost += counts[index]
            regional_lost[grid.at[index, "Region"]] += counts[index]
            category_counts[grid.at[index, "Landscape_Category"]] += 1
            chosen_regions.add(grid.at[index, "Region"])
        return chosen, lost, regional_lost

    for cap in (max_loss, .10) if max_loss < .10 else (max_loss,):
        for distance in (2*grid_size, grid_size*math.sqrt(2)+1):
            chosen, lost, regional_lost = attempt(cap, distance)
            if len(chosen) >= 10:
                return chosen, lost, regional_lost, cap, distance
    raise RuntimeError("Could not identify at least 10 blocks under the 90%-retention rule.")


def output_paths(directory: Path):
    return {
        "full_grid": directory / "Tehran_5km_Complete_Grid.gpkg",
        "selected_gpkg": directory / "Tehran_5km_Selected_Candidates.gpkg",
        "selected_shp": directory / "Tehran_5km_Selected_Candidates.shp",
        "all_csv": directory / "Tehran_5km_All_Grid_Statistics.csv",
        "selected_csv": directory / "Tehran_5km_Selected_Candidates.csv",
        "training_csv": directory / "Tehran_5km_Cumulative_Training_Reference.csv",
        "reference_map": directory / "Tehran_5km_Reference_Grid_Candidates.png",
        "inspection_map": directory / "Tehran_5km_Selected_Blocks_Inspection.png",
        "report": directory / "Tehran_5km_Test_Block_Candidates_Report.md",
    }


def save_maps(province, grid, selected, raster, paths):
    xmin, ymin, xmax, ymax = province.bounds
    margin = 12000
    extent = (xmin-margin, xmax+margin, ymin-margin, ymax+margin)
    width, height = 1900, 1100
    display = np.zeros((height, width), dtype=np.uint8)
    dst_transform = rasterio.transform.from_bounds(xmin-margin, ymin-margin,
                                                   xmax+margin, ymax+margin, width, height)
    with rasterio.open(raster) as src:
        reproject(rasterio.band(src, 1), display, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=dst_transform, dst_crs=CRS,
                  src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    province_display = geometry_mask([mapping(province)], out_shape=display.shape,
                                     transform=dst_transform, invert=True, all_touched=False)
    display[~province_display] = 0
    colors = ListedColormap(["#eeeeee", "#d73027", "#1a9850", "#c49a41"])
    norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5], colors.N)
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    ax.imshow(display, origin="upper", extent=extent, cmap=colors, norm=norm,
              interpolation="nearest", zorder=1)
    grid.boundary.plot(ax=ax, color="#606060", linewidth=.20, alpha=.45, zorder=2)
    gpd.GeoSeries([province], crs=CRS).boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)
    for category, color in CATEGORY_COLOR.items():
        subset = selected[selected.Selection_Category.eq(category)]
        if not subset.empty:
            subset.boundary.plot(ax=ax, color=color, linewidth=2.6, zorder=4)
    legend = [Patch(facecolor=colors(i), label=label) for i, label in
              enumerate(["Uncertain / outside", "Urban", "Vegetation", "Bare/Sparse"])]
    legend += [Patch(facecolor="none", edgecolor=color, linewidth=2, label=cat)
               for cat, color in CATEGORY_COLOR.items() if cat in set(selected.Selection_Category)]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,-.08),
              ncol=5, frameon=False, fontsize=9)
    ax.set(title="Tehran Province: three-class reference, 5-km grid, provisional test blocks",
           xlabel="UTM easting (m), EPSG:32639", ylabel="UTM northing (m), EPSG:32639")
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    ax.set_aspect("equal")
    fig.savefig(paths["reference_map"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)
    gpd.GeoSeries([province], crs=CRS).plot(ax=ax, facecolor="#f4f4f4",
                                               edgecolor="black", linewidth=1.2)
    for category, color in CATEGORY_COLOR.items():
        subset = selected[selected.Selection_Category.eq(category)]
        if not subset.empty:
            subset.plot(ax=ax, facecolor=color, edgecolor="black", linewidth=.8,
                        alpha=.78, label=category)
    for row in selected.itertuples(index=False):
        ax.annotate(row.Grid_ID, (row.Center_E_m, row.Center_N_m),
                    xytext=(5, 5), textcoords="offset points", fontsize=7.5,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.78,"pad":1.5})
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set(title="Provisional 5-km test blocks for manual review (grid IDs shown)",
           xlabel="UTM easting (m), EPSG:32639", ylabel="UTM northing (m), EPSG:32639")
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    ax.set_aspect("equal")
    fig.savefig(paths["inspection_map"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def report(args, paths, province_crs, raster_crs, grid, selected, pool, lost,
           cap, distance, elapsed):
    class_lines = []
    for code, name in CLASS.items():
        removed = int(lost[code-1]); total = int(pool[code]); remain = total-removed
        class_lines.append(f"| {name} | {total:,} | {removed:,} | {100*removed/total:.3f}% | "
                           f"{remain:,} | {100*remain/total:.3f}% |")
    category_counts = selected.Selection_Category.value_counts()
    category_text = ", ".join(f"{name}: {int(category_counts.get(name,0))}"
                              for name in CATEGORY_TARGETS)
    initial_regions = grid.groupby("Region")[[f"{name}_Count" for name in CLASS.values()]].sum()
    withdrawn_regions = selected.groupby("Region")[[f"{name}_Count" for name in CLASS.values()]].sum()
    remaining_regions = initial_regions.subtract(withdrawn_regions, fill_value=0)
    interior_coverage = ", ".join(
        f"{name} {100*grid[f'{name}_Count'].sum()/pool[code]:.2f}%"
        for code,name in CLASS.items()
    )
    spatial = []
    for name in CLASS.values():
        column = f"{name}_Count"
        baseline_supported = int((initial_regions[column] > 0).sum())
        final_supported = int((remaining_regions[column] > 0).sum())
        largest_before = 100*initial_regions[column].max()/initial_regions[column].sum()
        largest = 100*remaining_regions[column].max()/remaining_regions[column].sum()
        spatial.append(f"- {name}: reference remains in {final_supported}/{baseline_supported} "
                       f"supported 3×3 sectors; largest-sector share was {largest_before:.2f}% "
                       f"before selection and is {largest:.2f}% afterward (retained interior grid cells only).")
    selected_lines = [f"- {row.Grid_ID}: {row.Selection_Category}, sector {row.Region}; "
                      f"Urban {row.Urban_Pct_of_Confident:.1f}%, vegetation "
                      f"{row.Vegetation_Pct_of_Confident:.1f}%, bare/sparse "
                      f"{row.Bare_Sparse_Pct_of_Confident:.1f}%."
                      for row in selected.itertuples(index=False)]
    deviations = []
    if cap > args.max_loss_fraction:
        deviations.append(f"The 95%-retention target was not feasible for at least 10 blocks; "
                          f"the selection used the allowed {100*(1-cap):.0f}%-retention floor.")
    if distance < 2*args.grid_size_m:
        deviations.append("The two-cell spacing preference was relaxed, but queen-adjacent "
                          "grid cells remain excluded.")
    for name, target in CATEGORY_TARGETS.items():
        actual = int(category_counts.get(name,0))
        if actual < target:
            deviations.append(f"{name}: selected {actual} rather than target {target} because "
                              "of available cells and cumulative constraints.")
    if 100*remaining_regions["Vegetation_Count"].max()/remaining_regions["Vegetation_Count"].sum() > 60:
        deviations.append("Vegetation reference remains concentrated in one 3×3 sector; "
                          "this pre-existing concentration changed little under the 95%-retention constraint.")
    if not deviations:
        deviations.append("None against the stated selection and retention criteria.")
    output_text = "\n".join(f"- {key}: `{path.resolve()}`" for key,path in paths.items())
    text = f"""# Tehran 5-km provisional test-block candidates

## Inputs and method

- Province: `{args.province.resolve()}` (CRS `{province_crs}`)
- Final three-class reference: `{args.reference.resolve()}` (CRS `{raster_crs}`)
- Grid: {args.grid_size_m/1000:g} × {args.grid_size_m/1000:g} km in `{CRS}`; complete squares retained only when at least {100*args.minimum_overlap:.0f}% lies inside the province.
- {len(grid)} retained squares; {len(selected)} provisionally selected across {selected.Region.nunique()} of 9 geographic sectors, with stable IDs and no clipping.
- Class 0 is uncertain/NoData, never a class label. All native-resolution counts require the pixel centre to be inside the province and the square.

Selection is deterministic (no random draws). A category-stratified greedy maximin rule favors representation of urban, vegetation, bare/sparse, urban-transition and mixed landscapes, new 3×3 geographic sectors, and larger distances from previously selected blocks. Reference adequacy contributes only a small capped term. At every choice it enforces cumulative province-wide class-loss limits and protects at least 75% of reference in each meaningfully supported sector of the analyzed interior grid. The chosen minimum centre spacing was {distance/1000:.2f} km; candidate squares are not queen-adjacent.

## Selected grid IDs and available-reference composition

{chr(10).join(selected_lines)}

Landscape categories: {category_text}. Categories reflect only the incomplete, confident reference; they are not manually verified land-cover composition.

## Cumulative training-reference protection

All selected blocks are withheld **simultaneously** in this calculation. The actual per-class loss ceiling used was {cap:.0%} of the province-wide pool.

| Class | Province reference | Removed by all blocks | Removed % | Retained for training | Retained % |
|---|---:|---:|---:|---:|---:|
{chr(10).join(class_lines)}

Spatial-support check across the 3×3 sectors of retained ≥80%-interior grid cells (edge cells are not included in this diagnostic):

Those interior cells contain the following shares of each complete province-wide reference pool: {interior_coverage}.

{chr(10).join(spatial)}

## Validation and limitations

- PASS: all output statistics were derived from the source reference opened read-only; the raster was not modified.
- PASS: selected IDs are unique and are a subset of complete-grid IDs; all selected squares meet the province-overlap threshold.
- PASS: summed selected class counts match the cumulative-loss table, and each class satisfies the stated retention floor.
- PASS: all selected pairs satisfy the reported minimum centre spacing.
- This is a provisional spatial-split design, **not** independently labeled test ground truth. No test labels, test objects, model training, or classification were produced.
- The reference is spatially incomplete, so 0 may represent uncertainty rather than a land-cover class. Manual image interpretation must verify each candidate.
- The geographic-support diagnostic uses the retained interior grid; reference in omitted boundary cells remains available for training but is not included in that sector diagnostic.

Deviations: {" ".join(deviations)}

## Output files

{output_text}

Runtime: {elapsed:.1f} seconds.
"""
    paths["report"].write_text(text, encoding="utf-8")


def run(args):
    started = perf_counter()
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    province, province_crs = load_province(args.province)
    grid = make_grid(province, args.grid_size_m, args.minimum_overlap)
    region_codes(grid, province)
    with rasterio.open(args.reference) as src:
        if src.crs is None or src.count != 1:
            raise ValueError("Reference must be a single-band raster with a defined CRS.")
        raster_crs = str(src.crs)
        province_raster = transform_geom(CRS, src.crs, mapping(province))
        pool, outside = province_counts(src, province_raster)
        cells = cell_counts(src, grid, province_raster)
        if int(pool.sum())+outside != src.width*src.height:
            raise RuntimeError("Province and outside pixel counts do not cover the source raster.")
    add_statistics(grid, cells, pool, args.minimum_confident)
    chosen, lost, regional_lost, cap, distance = select_blocks(
        grid, pool, args.target_count, args.minimum_confident,
        args.max_loss_fraction, args.grid_size_m)
    if not 10 <= len(chosen) <= 15:
        raise RuntimeError("Selection did not produce 10–15 blocks.")
    grid.loc[chosen, "Selected"] = True
    grid.loc[chosen, "Selection_Rank"] = range(1,len(chosen)+1)
    grid.loc[chosen, "Selection_Category"] = grid.loc[chosen,"Landscape_Category"]
    selected = grid.loc[chosen].sort_values("Selection_Rank").copy()
    if not selected.Grid_ID.is_unique or not set(selected.Grid_ID).issubset(set(grid.Grid_ID)):
        raise RuntimeError("Selected-grid IDs are not unique/subset of the full grid.")
    if not np.array_equal(selected[[f"{name}_Count" for name in CLASS.values()]].sum().to_numpy(), lost):
        raise RuntimeError("Cumulative loss does not match the selected-grid class counts.")
    if np.any(lost > cap*pool[1:]+1e-9):
        raise RuntimeError("Cumulative class-retention threshold was violated.")
    coords = selected[["Center_E_m","Center_N_m"]].to_numpy()
    for i in range(len(coords)):
        for j in range(i):
            if np.linalg.norm(coords[i]-coords[j]) + 1e-8 < distance:
                raise RuntimeError("Selected blocks violate minimum separation.")
    paths = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Writing GeoPackages, shapefile, CSVs and maps...")
    grid.to_file(paths["full_grid"], layer="complete_grid", driver="GPKG")
    selected.to_file(paths["selected_gpkg"], layer="candidate_blocks", driver="GPKG")
    selected[["Grid_ID","Selection_Rank","Selection_Category","Region",
              "Province_Overlap_Pct","Urban_Count","Vegetation_Count",
              "Bare_Sparse_Count","geometry"]].rename(columns={
                  "Selection_Rank":"Sel_Rank", "Selection_Category":"Sel_Cat",
                  "Province_Overlap_Pct":"Prov_Pct", "Urban_Count":"Urb_Count",
                  "Vegetation_Count":"Veg_Count",
                  "Bare_Sparse_Count":"Bare_Count"}).to_file(paths["selected_shp"], driver="ESRI Shapefile")
    grid.drop(columns="geometry").to_csv(paths["all_csv"], index=False)
    selected.drop(columns="geometry").to_csv(paths["selected_csv"], index=False)
    pd.DataFrame([{"Class_Code":code,"Class_Name":name,"Province_Reference_Pixels":int(pool[code]),
                   "Withheld_Pixels_All_Selected":int(lost[code-1]),
                   "Removed_Pct":100*lost[code-1]/pool[code],
                   "Retained_Training_Pixels":int(pool[code]-lost[code-1]),
                   "Retained_Pct":100*(pool[code]-lost[code-1])/pool[code]}
                  for code,name in CLASS.items()]).to_csv(paths["training_csv"], index=False)
    save_maps(province, grid, selected, args.reference, paths)
    report(args, paths, province_crs, raster_crs, grid, selected, pool, lost,
           cap, distance, perf_counter()-started)
    print(f"Selected {len(selected)} blocks; cumulative losses: "
          + ", ".join(f"{CLASS[code]}={100*lost[code-1]/pool[code]:.2f}%" for code in CLASS))
    print(f"Report: {paths['report'].resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--province", type=Path, default=DEFAULT_PROVINCE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size-m", type=int, default=5000)
    parser.add_argument("--minimum-overlap", type=float, default=.80)
    parser.add_argument("--minimum-confident", type=int, default=1000)
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-loss-fraction", type=float, default=.05)
    args = parser.parse_args()
    if not 10 <= args.target_count <= 15 or args.grid_size_m <= 0 or not 0 < args.minimum_overlap <= 1:
        parser.error("Target must be 10–15, grid size positive, and overlap fraction in (0,1].")
    if not 0 < args.max_loss_fraction <= .10 or args.minimum_confident < 1:
        parser.error("Loss fraction must be in (0,0.10] and minimum confident pixels >=1.")
    run(args)


if __name__ == "__main__":
    main()
