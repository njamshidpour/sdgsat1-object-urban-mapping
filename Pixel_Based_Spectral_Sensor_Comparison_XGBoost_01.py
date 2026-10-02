#!/usr/bin/env python
"""Controlled pixel-based SDGSAT-1/Sentinel-2 Urban-vs-NonUrban experiment.

The default run is a bounded pilot.  The complete study-area scan, 50,000
samples per source class, 5-fold/24-configuration tuning, 2,000 bootstrap
replicates, and six full-scene rasters are enabled only by ``--full-run``.

Pixel predictors are direct raster values.  No object summaries, texture,
geometry, or spatial-context predictors enter any model.  Segmentation is
used solely to reproduce manual-test membership and the conservative
test-object plus first-order-Rook-neighbour training exclusion.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from pathlib import Path

# Prevent a system QGIS PROJ database from overriding the active conda env.
_conda_proj = Path(sys.prefix) / "Library" / "share" / "proj"
if _conda_proj.exists():
    os.environ["PROJ_LIB"] = str(_conda_proj)

import numpy as np
import pandas as pd


RANDOM_SEED = 20261001
THRESHOLD = 0.50
PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"
BINARY_DIR = SEGMENTATION_DIR / "Binary_Stratified_Object_Ablation_XGBoost_01"
OBJECT_SENSOR_DIR = SEGMENTATION_DIR / "Spectral_Sensor_Comparison_XGBoost_01"

MASTER_RASTER = PROJECT_ROOT / "optic_data_codex" / "Tehran_Raw_B2_10m.tif"
REFERENCE_RASTER = (
    PROJECT_ROOT / "GTs" / "DW_GT_ThreeClass_2023_weight_090_Sge070_TehranProvince"
    / "Tehran_DW_ThreeClass_GT_2023.tif"
)
PROVINCE_PATH = PROJECT_ROOT / "Shapefile" / "Tehran_True_Province.shp"
MANUAL_POINTS = PROJECT_ROOT / "GTs" / "Test_Points_1000.shp"
SEGMENTS_PATH = SEGMENTATION_DIR / "eCog_segments.shp"
ADJACENCY_PATH = SEGMENTATION_DIR / "rook_adjacency_graph_v01.npz"
PRIMARY_MEMBERSHIP = BINARY_DIR / "Binary_Primary_Stratified_Test_Set.csv"
FULL_MEMBERSHIP = BINARY_DIR / "Binary_Secondary_Full_Test_Set.csv"

PREDICTOR_PATHS = OrderedDict([
    ("SDGSAT_Pan", PROJECT_ROOT / "output_corrected_bands" / "Pan_HDR_Radiance.tif"),
    ("SDGSAT_R", PROJECT_ROOT / "output_corrected_bands" / "Red_Radiance_salt_pepper_10m.tif"),
    ("SDGSAT_G", PROJECT_ROOT / "output_corrected_bands" / "Green_Radiance_salt_pepper_10m.tif"),
    ("SDGSAT_B", PROJECT_ROOT / "output_corrected_bands" / "Blue_Radiance_salt_pepper_10m.tif"),
    ("S2_B2", PROJECT_ROOT / "optic_data_codex" / "Tehran_Raw_B2_10m.tif"),
    ("S2_B3", PROJECT_ROOT / "optic_data_codex" / "Tehran_Raw_B3_10m.tif"),
    ("S2_B4", PROJECT_ROOT / "optic_data_codex" / "Tehran_Raw_B4_10m.tif"),
    ("S2_B8", PROJECT_ROOT / "optic_data_codex" / "Tehran_Raw_B8_10m.tif"),
])
FEATURE_GROUPS = {
    "NTL": ["SDGSAT_Pan", "SDGSAT_R", "SDGSAT_G", "SDGSAT_B"],
    "S2": ["S2_B2", "S2_B3", "S2_B4", "S2_B8"],
}
MODEL_GROUPS = OrderedDict([
    ("P1", ("NTL",)),
    ("P2", ("S2",)),
    ("P3", ("NTL", "S2")),
])
MODEL_NAMES = {
    "P1": "NTL",
    "P2": "Sentinel-2",
    "P3": "NTL + Sentinel-2",
}
SOURCE_NAMES = {1: "Urban", 2: "Vegetation", 3: "BareScrub"}
BINARY_NAMES = {0: "NonUrban", 1: "Urban"}
COMPARISONS = [("P2", "P1"), ("P3", "P1"), ("P3", "P2")]
DIFFERENCE_METRICS = [
    "BalancedAccuracy", "MacroF1", "mIoU", "UrbanPrecision",
    "UrbanRecall", "UrbanF1", "UrbanIoU",
]
BOOTSTRAP_METRICS = ["BalancedAccuracy", "MacroF1", "mIoU", "UrbanF1", "UrbanIoU"]
PARAMETER_SPACE = OrderedDict([
    ("n_estimators", [300, 500, 800]),
    ("max_depth", [3, 5, 7]),
    ("learning_rate", [0.03, 0.05, 0.10]),
    ("min_child_weight", [1, 3, 5]),
    ("subsample", [0.8, 1.0]),
    ("colsample_bytree", [0.7, 0.9, 1.0]),
    ("reg_lambda", [1, 5, 10]),
    ("reg_alpha", [0, 0.1, 0.5]),
])


class Progress:
    """Single-line progress bar that does not flood ordinary terminals."""

    def __init__(self, total: int, label: str):
        self.total = max(1, int(total))
        self.label = label
        self.done = 0
        self.started = time.perf_counter()
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.last_bucket = -1

    def update(self, amount: int = 1):
        self.done = min(self.total, self.done + amount)
        fraction = self.done / self.total
        bucket = int(fraction * 20)
        if self.tty:
            width = 28
            filled = int(round(width * fraction))
            elapsed = time.perf_counter() - self.started
            rate = self.done / elapsed if elapsed else 0.0
            eta = (self.total - self.done) / rate if rate else math.nan
            print(
                f"\r{self.label}: [{'#' * filled}{'.' * (width-filled)}] "
                f"{100*fraction:6.2f}%  ETA {eta:7.1f}s",
                end="", flush=True,
            )
            if self.done == self.total:
                print()
        elif bucket != self.last_bucket or self.done == self.total:
            self.last_bucket = bucket
            print(f"{self.label}: {self.done}/{self.total} ({100*fraction:.1f}%)", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Run bounded pilot (default).")
    mode.add_argument("--full-run", action="store_true", help="Run complete experiment and maps.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--pilot-windows", type=int, default=32)
    parser.add_argument("--pilot-target-per-class", type=int, default=5000)
    parser.add_argument("--pilot-configurations", type=int, default=6)
    parser.add_argument("--pilot-bootstrap", type=int, default=200)
    parser.add_argument("--skip-maps", action="store_true", help="Full-run diagnostic override only.")
    return parser.parse_args()


def dependencies():
    mandatory = ["rasterio", "geopandas", "shapely", "pyproj", "xgboost", "sklearn", "matplotlib"]
    missing = [name for name in mandatory if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("STOP: missing mandatory package(s): " + ", ".join(missing))
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from pyproj import CRS, Geod, Transformer
    from rasterio.features import rasterize
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling
    from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
    from shapely.geometry import box
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, cohen_kappa_score,
        confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
    return locals()


def require_inputs():
    paths = list(PREDICTOR_PATHS.values()) + [
        REFERENCE_RASTER, PROVINCE_PATH, MANUAL_POINTS, SEGMENTS_PATH,
        ADJACENCY_PATH, PRIMARY_MEMBERSHIP, FULL_MEMBERSHIP,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("STOP: required input(s) missing:\n" + "\n".join(missing))


def same_grid(source, master) -> bool:
    return (
        source.crs == master.crs
        and source.width == master.width
        and source.height == master.height
        and source.transform.almost_equals(master.transform, precision=1e-12)
        and np.allclose(source.bounds, master.bounds, rtol=0, atol=1e-9)
    )


class RasterStack:
    def __init__(self, bundle, output_dir: Path):
        self.b = bundle
        self.output_dir = output_dir
        self.stack = ExitStack()
        self.master = self.stack.enter_context(bundle["rasterio"].open(MASTER_RASTER))
        if self.master.crs is None:
            raise SystemExit("STOP: master Sentinel-2 raster has no CRS.")
        self.readers = OrderedDict()
        self.native = OrderedDict()
        self.status = {}
        self.audit_rows = []
        for feature, path in PREDICTOR_PATHS.items():
            source = self.stack.enter_context(bundle["rasterio"].open(path))
            self.native[feature] = source
            aligned = same_grid(source, self.master)
            if aligned:
                reader = source
                status = "Already aligned; no resampling"
            else:
                reader = self.stack.enter_context(bundle["WarpedVRT"](
                    source, crs=self.master.crs, transform=self.master.transform,
                    width=self.master.width, height=self.master.height,
                    resampling=bundle["Resampling"].bilinear,
                ))
                status = "Aligned on read with bilinear interpolation"
            self.readers[feature] = reader
            self.status[feature] = status
            self.audit_rows.append(self._audit_row(feature, path, source, aligned, status, "Predictor"))
        ref_source = self.stack.enter_context(bundle["rasterio"].open(REFERENCE_RASTER))
        ref_aligned = same_grid(ref_source, self.master)
        if ref_aligned:
            self.reference = ref_source
            ref_status = "Already aligned; no resampling"
        else:
            self.reference = self.stack.enter_context(bundle["WarpedVRT"](
                ref_source, crs=self.master.crs, transform=self.master.transform,
                width=self.master.width, height=self.master.height,
                resampling=bundle["Resampling"].nearest,
            ))
            ref_status = "Aligned on read with nearest-neighbour interpolation"
        self.audit_rows.append(self._audit_row(
            "DW_Reference", REFERENCE_RASTER, ref_source, ref_aligned, ref_status, "Categorical reference"
        ))
        pd.DataFrame(self.audit_rows).to_csv(output_dir / "Pixel_Raster_Alignment_Audit.csv", index=False)
        # The resolved project data are expected to be exact already.  A VRT is
        # supported by design, but post-alignment equality is mandatory.
        for name, reader in list(self.readers.items()) + [("DW_Reference", self.reference)]:
            if not same_grid(reader, self.master):
                raise SystemExit(f"STOP: post-alignment grid mismatch remains for {name}.")

    def _audit_row(self, feature, path, source, aligned, status, role):
        return {
            "Layer": feature, "Role": role, "ResolvedPath": str(path.resolve()),
            "NativeCRS": str(source.crs), "NativeWidth": source.width,
            "NativeHeight": source.height, "NativeResolutionX": source.res[0],
            "NativeResolutionY": source.res[1], "NativeTransform": tuple(source.transform),
            "NativeBounds": tuple(source.bounds), "NativeDType": source.dtypes[0],
            "NativeNoData": source.nodata, "InitiallyExactMasterGrid": aligned,
            "ProcessingStatus": status, "FinalCRS": str(self.master.crs),
            "FinalWidth": self.master.width, "FinalHeight": self.master.height,
            "FinalResolutionX": self.master.res[0], "FinalResolutionY": self.master.res[1],
            "FinalTransform": tuple(self.master.transform), "PixelCenterAligned": True,
        }

    def close(self):
        self.stack.close()


def write_feature_manifest(stack: RasterStack, path: Path):
    rows = []
    for sensor, names in FEATURE_GROUPS.items():
        for feature in names:
            rows.append({
                "FeatureName": feature,
                "Sensor": "SDGSAT-1 NTL" if sensor == "NTL" else "Sentinel-2",
                "SourceRaster": str(PREDICTOR_PATHS[feature].resolve()),
                "ProcessingStatus": stack.status[feature],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def manual_to_source(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value_l = value.strip().lower().replace("_", "").replace("/", "")
        if "urban" in value_l or "built" in value_l:
            return 1
        if "veget" in value_l:
            return 2
        if "bare" in value_l or "scrub" in value_l:
            return 3
        try:
            value = float(value)
        except ValueError:
            return np.nan
    return {1: 1, 2: 2, 3: 3, 4: 3}.get(int(value), np.nan)


def prepare_test_points(bundle, stack: RasterStack):
    gpd = bundle["gpd"]
    segments = gpd.read_file(SEGMENTS_PATH)
    if segments.crs is None or "Seg_id" not in segments:
        raise SystemExit("STOP: segmentation lacks CRS or Seg_id.")
    if segments["Seg_id"].duplicated().any():
        raise SystemExit("STOP: segmentation Seg_id is not unique.")
    segments["Seg_id"] = segments["Seg_id"].astype("int64")
    points = gpd.read_file(MANUAL_POINTS)
    label_candidates = [c for c in ("Class_ID", "Class", "class_id", "class", "Label", "label") if c in points]
    if not label_candidates or points.crs is None:
        raise SystemExit("STOP: manual points lack a recognizable label field or CRS.")
    label_field = label_candidates[0]
    points = points.reset_index(names="SourcePointIndex").to_crs(segments.crs)
    keep_cols = ["SourcePointIndex", label_field, "geometry"]
    if "Point_ID" in points:
        keep_cols.insert(1, "Point_ID")
    joined = gpd.sjoin(points[keep_cols], segments[["Seg_id", "geometry"]], how="left", predicate="within")
    joined["SourceCode3"] = joined[label_field].map(manual_to_source)
    valid = joined.dropna(subset=["Seg_id", "SourceCode3"]).copy()
    valid["Object_ID"] = valid["Seg_id"].astype("int64")
    valid["SourceCode3"] = valid["SourceCode3"].astype("int64")
    valid["BinaryLabel"] = (valid["SourceCode3"] == 1).astype("int64")
    # All labeled manual-point objects, including any later conflict, drive exclusion.
    all_test_object_ids = np.sort(valid["Object_ID"].unique())
    conflict_ids = []
    chosen_rows = []
    for object_id, group in valid.groupby("Object_ID", sort=True):
        if group["BinaryLabel"].nunique() != 1:
            conflict_ids.append(int(object_id))
            continue
        sort_cols = ["Point_ID", "SourcePointIndex"] if "Point_ID" in group else ["SourcePointIndex"]
        row = group.sort_values(sort_cols).iloc[0].copy()
        row["ManualPointCount"] = len(group)
        chosen_rows.append(row)
    chosen = gpd.GeoDataFrame(chosen_rows, crs=segments.crs)
    if chosen.empty:
        raise SystemExit("STOP: no nonconflicting manual test object could be recovered.")
    chosen = chosen.set_index("Object_ID", drop=False)

    def recover(path: Path, name: str):
        membership = pd.read_csv(path)
        required = {"Object_ID", "BinaryLabel", "SourceClass3"}
        if not required.issubset(membership.columns):
            raise SystemExit(f"STOP: {name} membership lacks required columns.")
        membership["Object_ID"] = membership["Object_ID"].astype("int64")
        missing = pd.Index(membership["Object_ID"]).difference(chosen.index)
        if len(missing):
            raise SystemExit(f"STOP: {len(missing)} {name} Object_IDs have no recoverable manual point.")
        selected = chosen.loc[membership["Object_ID"].to_numpy()].copy().reset_index(drop=True)
        expected = membership.set_index("Object_ID").loc[selected["Object_ID"], "BinaryLabel"].to_numpy(dtype="int64")
        if not np.array_equal(expected, selected["BinaryLabel"].to_numpy(dtype="int64")):
            raise SystemExit(f"STOP: binary label disagreement while recovering {name} points.")
        selected = selected.to_crs(stack.master.crs)
        selected["X"] = selected.geometry.x
        selected["Y"] = selected.geometry.y
        metric = selected.to_crs("EPSG:32639")
        selected["X_32639"] = metric.geometry.x
        selected["Y_32639"] = metric.geometry.y
        rc = [stack.master.index(x, y) for x, y in zip(selected["X"], selected["Y"])]
        selected["Row"] = [v[0] for v in rc]
        selected["Col"] = [v[1] for v in rc]
        if "Point_ID" in selected:
            selected["Test_ID"] = selected["Point_ID"].astype("int64")
        else:
            selected["Test_ID"] = selected["SourcePointIndex"].astype("int64") + 1
        selected["SourceClass3"] = selected["SourceCode3"].map(SOURCE_NAMES)
        selected["TrueClass"] = selected["BinaryLabel"].map(BINARY_NAMES)
        return selected

    primary = recover(PRIMARY_MEMBERSHIP, "primary")
    full = recover(FULL_MEMBERSHIP, "full")
    diagnostics = {
        "manual_source_points": int(len(points)),
        "valid_point_object_assignments": int(len(valid)),
        "all_test_objects_used_for_exclusion": int(len(all_test_object_ids)),
        "binary_conflict_objects": int(len(conflict_ids)),
        "binary_conflict_ids": conflict_ids,
        "deterministic_rule": "smallest Point_ID, then source row index, within each nonconflicting Object_ID",
        "primary_count": int(len(primary)), "full_count": int(len(full)),
    }
    return segments, primary, full, all_test_object_ids, diagnostics


def rook_neighbors(test_ids):
    with np.load(ADJACENCY_PATH, allow_pickle=False) as graph:
        if not {"src_object_id", "dst_object_id"}.issubset(graph.files):
            raise SystemExit("STOP: canonical Rook graph lacks required ID edge arrays.")
        src = graph["src_object_id"].astype("int64", copy=False)
        dst = graph["dst_object_id"].astype("int64", copy=False)
        src_hit = np.isin(src, test_ids)
        dst_hit = np.isin(dst, test_ids)
        neighbors = np.unique(np.concatenate([dst[src_hit], src[dst_hit]]))
    return neighbors[~np.isin(neighbors, test_ids)]


class SpatialMasks:
    def __init__(self, bundle, stack: RasterStack, segments, test_ids):
        self.b = bundle
        self.stack = stack
        gpd = bundle["gpd"]
        province = gpd.read_file(PROVINCE_PATH)
        if province.crs is None:
            raise SystemExit("STOP: province boundary has no CRS.")
        self.province = province.to_crs(stack.master.crs).geometry
        self.province_union = self.province.union_all()
        neighbors = rook_neighbors(test_ids)
        self.neighbor_ids = neighbors
        excluded_ids = np.unique(np.concatenate([test_ids, neighbors])).astype("int64")
        excluded = segments[segments["Seg_id"].isin(excluded_ids)][["Seg_id", "geometry"]].copy()
        missing = pd.Index(excluded_ids).difference(excluded["Seg_id"])
        if len(missing):
            raise SystemExit(f"STOP: {len(missing)} exclusion Object_IDs are absent from segmentation.")
        self.excluded = excluded.to_crs(stack.master.crs)
        self.excluded_sindex = self.excluded.sindex
        self.excluded_ids = excluded_ids

    def study(self, window):
        return self.b["rasterize"](
            ((geom, 1) for geom in self.province if geom is not None and not geom.is_empty),
            out_shape=(int(window.height), int(window.width)),
            transform=self.b["window_transform"](window, self.stack.master.transform),
            fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)

    def exclusion(self, window):
        bounds = self.b["window_bounds"](window, self.stack.master.transform)
        candidates = list(self.excluded_sindex.intersection(bounds))
        if not candidates:
            return np.zeros((int(window.height), int(window.width)), dtype=bool)
        geoms = self.excluded.iloc[candidates].geometry
        return self.b["rasterize"](
            ((geom, 1) for geom in geoms if geom is not None and not geom.is_empty),
            out_shape=(int(window.height), int(window.width)),
            transform=self.b["window_transform"](window, self.stack.master.transform),
            fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)


def all_windows(stack: RasterStack, block_size: int):
    from rasterio.windows import Window
    result = []
    for row in range(0, stack.master.height, block_size):
        for col in range(0, stack.master.width, block_size):
            result.append(Window(
                col, row, min(block_size, stack.master.width-col),
                min(block_size, stack.master.height-row),
            ))
    return result


def relevant_windows(bundle, stack: RasterStack, masks: SpatialMasks, block_size: int):
    result = []
    province_bounds = masks.province_union.bounds
    pbox = bundle["box"](*province_bounds)
    for window in all_windows(stack, block_size):
        if bundle["box"](*bundle["window_bounds"](window, stack.master.transform)).intersects(pbox):
            if masks.study(window).any():
                result.append(window)
    return result


def choose_pilot_windows(bundle, stack, masks, windows, number):
    if number >= len(windows):
        return windows
    centers = np.asarray([
        ((b := bundle["window_bounds"](w, stack.master.transform))[0] + b[2], b[1] + b[3])
        for w in windows
    ], dtype="float64") / 2.0
    centroid = np.asarray([masks.province_union.centroid.x, masks.province_union.centroid.y])
    chosen = [int(np.argmin(np.sum((centers-centroid)**2, axis=1)))]
    minimum_distance = np.sum((centers-centers[chosen[0]])**2, axis=1)
    for _ in range(1, number):
        minimum_distance[chosen] = -1
        index = int(np.argmax(minimum_distance))
        chosen.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.sum((centers-centers[index])**2, axis=1)
        )
    return [windows[i] for i in sorted(chosen)]


def read_predictor_arrays(stack: RasterStack, window):
    arrays = []
    invalid_masks = []
    for reader in stack.readers.values():
        value = reader.read(1, window=window, masked=True)
        data = np.asarray(value.data, dtype="float32")
        invalid = np.ma.getmaskarray(value) | ~np.isfinite(data)
        arrays.append(data)
        invalid_masks.append(invalid)
    invalid_masks = np.stack(invalid_masks)
    return np.stack(arrays), np.logical_or.reduce(invalid_masks), invalid_masks


def pixel_centers(bundle, stack, rows, cols, transformer):
    t = stack.master.transform
    native_x = t.c + t.a * (cols + 0.5) + t.b * (rows + 0.5)
    native_y = t.f + t.d * (cols + 0.5) + t.e * (rows + 0.5)
    if transformer is not None:
        metric_x, metric_y = transformer.transform(native_x, native_y)
    else:
        metric_x, metric_y = native_x, native_y
    return (
        np.asarray(native_x), np.asarray(native_y),
        np.asarray(metric_x), np.asarray(metric_y),
    )


def first_pass(bundle, stack, masks, windows):
    feature_names = list(PREDICTOR_PATHS)
    stats = {
        name: {"valid": 0, "nodata": 0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}
        for name in feature_names
    }
    candidate_counts = defaultdict(int)
    class_counts = defaultdict(int)
    class_cells = defaultdict(set)
    study_pixels = 0
    transformer = None
    if not stack.master.crs.is_projected or stack.master.crs.to_epsg() != 32639:
        transformer = bundle["Transformer"].from_crs(stack.master.crs, "EPSG:32639", always_xy=True)
    progress = Progress(len(windows), "Candidate/audit scan")
    for window in windows:
        study = masks.study(window)
        study_pixels += int(study.sum())
        excluded = masks.exclusion(window)
        reference = stack.reference.read(1, window=window)
        predictors, any_invalid, layer_invalid = read_predictor_arrays(stack, window)
        for index, name in enumerate(feature_names):
            valid = study & ~layer_invalid[index]
            values = predictors[index][valid].astype("float64", copy=False)
            item = stats[name]
            item["valid"] += len(values)
            item["nodata"] += int(study.sum() - len(values))
            if len(values):
                item["sum"] += float(values.sum(dtype="float64"))
                item["sumsq"] += float(np.square(values, dtype="float64").sum(dtype="float64"))
                item["min"] = min(item["min"], float(values.min()))
                item["max"] = max(item["max"], float(values.max()))
        candidate = study & ~excluded & ~any_invalid & np.isin(reference, [1, 2, 3])
        local_rows, local_cols = np.nonzero(candidate)
        if len(local_rows):
            rows = local_rows.astype("int64") + int(window.row_off)
            cols = local_cols.astype("int64") + int(window.col_off)
            _, _, metric_x, metric_y = pixel_centers(bundle, stack, rows, cols, transformer)
            gx = np.floor(metric_x / 5000.0).astype("int64")
            gy = np.floor(metric_y / 5000.0).astype("int64")
            codes = reference[local_rows, local_cols].astype("int64")
            triples, counts = np.unique(np.column_stack([codes, gx, gy]), axis=0, return_counts=True)
            for (code, cell_x, cell_y), count in zip(triples, counts):
                key = (int(code), int(cell_x), int(cell_y))
                candidate_counts[key] += int(count)
                class_counts[int(code)] += int(count)
                class_cells[int(code)].add((int(cell_x), int(cell_y)))
        progress.update()
    if study_pixels == 0:
        raise SystemExit("STOP: no master-grid pixel center falls inside the province mask.")
    for name, item in stats.items():
        if item["valid"] == 0:
            raise SystemExit(f"STOP: predictor {name} has no valid study-area pixels.")
    return candidate_counts, class_counts, class_cells, stats, study_pixels, transformer


def even_quota(capacity: dict[tuple[int, int], int], total: int):
    if total > sum(capacity.values()):
        raise ValueError("quota target exceeds capacity")
    quota = {key: 0 for key in sorted(capacity)}
    remaining = total
    active = sorted(key for key, value in capacity.items() if value > 0)
    while remaining and active:
        level = min(capacity[key] - quota[key] for key in active)
        step = min(level, remaining // len(active))
        if step > 0:
            for key in active:
                quota[key] += step
            remaining -= step * len(active)
        if remaining:
            for key in active:
                if remaining == 0:
                    break
                if quota[key] < capacity[key]:
                    quota[key] += 1
                    remaining -= 1
        active = [key for key in active if quota[key] < capacity[key]]
    if remaining:
        raise RuntimeError("unable to complete even quota allocation")
    return quota


def quotas_from_counts(candidate_counts, class_counts, cap):
    target = min(cap, *(class_counts.get(code, 0) for code in (1, 2, 3)))
    if target <= 0:
        raise SystemExit("STOP: at least one source class has no training candidates.")
    quotas = {}
    for code in (1, 2, 3):
        capacity = {(gx, gy): n for (c, gx, gy), n in candidate_counts.items() if c == code}
        allocation = even_quota(capacity, target)
        quotas.update({(code, gx, gy): value for (gx, gy), value in allocation.items()})
    return target, quotas


def second_pass_sample(bundle, stack, masks, windows, candidate_counts, quotas, transformer):
    rng = np.random.default_rng(RANDOM_SEED)
    remaining_population = dict(candidate_counts)
    remaining_needed = dict(quotas)
    selected_parts = []
    feature_names = list(PREDICTOR_PATHS)
    progress = Progress(len(windows), "Spatially stratified pixel sampling")
    for window in windows:
        study = masks.study(window)
        reference = stack.reference.read(1, window=window)
        predictors, any_invalid, _ = read_predictor_arrays(stack, window)
        candidate = study & ~masks.exclusion(window) & ~any_invalid & np.isin(reference, [1, 2, 3])
        lr, lc = np.nonzero(candidate)
        if len(lr):
            rows = lr.astype("int64") + int(window.row_off)
            cols = lc.astype("int64") + int(window.col_off)
            native_x, native_y, metric_x, metric_y = pixel_centers(
                bundle, stack, rows, cols, transformer
            )
            gx = np.floor(metric_x / 5000.0).astype("int64")
            gy = np.floor(metric_y / 5000.0).astype("int64")
            codes = reference[lr, lc].astype("int64")
            triples = np.column_stack([codes, gx, gy])
            unique, inverse = np.unique(triples, axis=0, return_inverse=True)
            for group_number, (code, cell_x, cell_y) in enumerate(unique):
                key = (int(code), int(cell_x), int(cell_y))
                positions = np.flatnonzero(inverse == group_number)
                population = remaining_population[key]
                needed = remaining_needed.get(key, 0)
                batch = len(positions)
                if needed:
                    take = int(rng.hypergeometric(needed, population-needed, batch))
                    if take:
                        picked = rng.choice(positions, size=take, replace=False)
                        data = {
                            "Row": rows[picked], "Col": cols[picked],
                            "X": native_x[picked], "Y": native_y[picked],
                            "X_32639": metric_x[picked], "Y_32639": metric_y[picked],
                            "SourceCode3": np.full(take, int(code), dtype="int64"),
                            "SourceClass3": np.full(take, SOURCE_NAMES[int(code)], dtype=object),
                            "BinaryLabel": np.full(take, 1 if int(code) == 1 else 0, dtype="int64"),
                            "BinaryClass": np.full(take, "Urban" if int(code) == 1 else "NonUrban", dtype=object),
                            "SpatialCell": np.full(take, f"{int(cell_x)}_{int(cell_y)}", dtype=object),
                        }
                        for feature_index, feature in enumerate(feature_names):
                            data[feature] = predictors[feature_index, lr[picked], lc[picked]]
                        selected_parts.append(pd.DataFrame(data))
                    remaining_needed[key] = needed - take
                remaining_population[key] = population - batch
        progress.update()
    incomplete = {key: value for key, value in remaining_needed.items() if value != 0}
    if incomplete:
        raise SystemExit(f"STOP: streaming sampler left {sum(incomplete.values())} quotas unfilled.")
    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sort_values(["SourceCode3", "SpatialCell", "Row", "Col"]).reset_index(drop=True)
    if sample[["Row", "Col"]].duplicated().any():
        raise SystemExit("STOP: duplicate training pixel coordinates were sampled.")
    return sample


def verify_no_exclusion_leakage(bundle, sample, masks: SpatialMasks, stack):
    gpd = bundle["gpd"]
    points = gpd.GeoDataFrame(
        sample[["Row", "Col"]].copy(),
        geometry=gpd.points_from_xy(sample["X"], sample["Y"]), crs=stack.master.crs,
    )
    joined = gpd.sjoin(points, masks.excluded[["Seg_id", "geometry"]], how="left", predicate="within")
    leaked = int(joined["Seg_id"].notna().sum())
    if leaked:
        raise SystemExit(f"STOP: spatial leakage detected for {leaked} sampled pixels.")
    return leaked


def sampling_summary(sample, candidate_counts, class_counts, class_cells):
    rows = []
    for code in (1, 2, 3):
        selected = sample[sample["SourceCode3"] == code]
        rows.append({
            "Level": "SourceClass", "SourceClass3": SOURCE_NAMES[code], "SpatialCell": "ALL",
            "CandidateCount": class_counts[code], "SelectedCount": len(selected),
            "CandidateOccupiedCells": len(class_cells[code]),
            "SelectedOccupiedCells": selected["SpatialCell"].nunique(),
            "CandidateRetainedPercent": 100 * len(selected) / class_counts[code],
        })
        capacities = {(gx, gy): n for (c, gx, gy), n in candidate_counts.items() if c == code}
        selected_counts = selected["SpatialCell"].value_counts()
        for (gx, gy), number in sorted(capacities.items()):
            cell = f"{gx}_{gy}"
            count = int(selected_counts.get(cell, 0))
            rows.append({
                "Level": "SourceClassSpatialCell", "SourceClass3": SOURCE_NAMES[code],
                "SpatialCell": cell, "CandidateCount": number, "SelectedCount": count,
                "CandidateOccupiedCells": np.nan, "SelectedOccupiedCells": np.nan,
                "CandidateRetainedPercent": 100 * count / number,
            })
    rows.append({
        "Level": "BinaryClass", "SourceClass3": "Urban", "SpatialCell": "ALL",
        "CandidateCount": np.nan, "SelectedCount": int((sample["BinaryLabel"] == 1).sum()),
        "CandidateOccupiedCells": np.nan,
        "SelectedOccupiedCells": sample.loc[sample["BinaryLabel"] == 1, "SpatialCell"].nunique(),
        "CandidateRetainedPercent": np.nan,
    })
    rows.append({
        "Level": "BinaryClass", "SourceClass3": "NonUrban", "SpatialCell": "ALL",
        "CandidateCount": np.nan, "SelectedCount": int((sample["BinaryLabel"] == 0).sum()),
        "CandidateOccupiedCells": np.nan,
        "SelectedOccupiedCells": sample.loc[sample["BinaryLabel"] == 0, "SpatialCell"].nunique(),
        "CandidateRetainedPercent": np.nan,
    })
    return pd.DataFrame(rows)


def data_quality_table(sample, stats, study_pixels, scope):
    rows = []
    for feature in PREDICTOR_PATHS:
        item = stats[feature]
        mean = item["sum"] / item["valid"]
        variance = max(0.0, item["sumsq"] / item["valid"] - mean * mean)
        rows.append({
            "FeatureName": feature,
            "Sensor": "NTL" if feature in FEATURE_GROUPS["NTL"] else "S2",
            "AuditScope": scope, "StudyPixelsInScope": study_pixels,
            "ValidPixelCount": item["valid"], "NaN_NoDataCount": item["nodata"],
            "Minimum": item["min"], "Maximum": item["max"], "Mean": mean,
            "StandardDeviation": math.sqrt(variance),
            "TrainingSampleUniqueValues": int(sample[feature].nunique(dropna=True)),
            "ConstantRaster": bool(item["min"] == item["max"]),
        })
    table = pd.DataFrame(rows)
    constants = table.loc[table["ConstantRaster"], "FeatureName"].tolist()
    duplicate_pairs = []
    names = list(PREDICTOR_PATHS)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if np.array_equal(sample[first].to_numpy(), sample[second].to_numpy(), equal_nan=True):
                duplicate_pairs.append((first, second))
    if constants:
        raise SystemExit("STOP: constant predictor raster(s): " + ", ".join(constants))
    if duplicate_pairs:
        raise SystemExit("STOP: exact duplicate predictor columns in training sample: " + str(duplicate_pairs))
    for feature in FEATURE_GROUPS["S2"]:
        row = table.set_index("FeatureName").loc[feature]
        if row["Minimum"] < -1000 or row["Maximum"] > 65535:
            raise SystemExit(f"STOP: unreasonable optical DN range detected for {feature}.")
    return table, duplicate_pairs


def attach_point_predictors(stack, frame):
    coords = list(zip(frame["X"], frame["Y"]))
    result = frame.copy()
    for feature, reader in stack.readers.items():
        values = np.asarray([value[0] for value in reader.sample(coords, indexes=1, masked=False)], dtype="float32")
        result[feature] = values
        nodata = reader.nodata
        invalid = ~np.isfinite(values)
        if nodata is not None and np.isfinite(nodata):
            invalid |= values == nodata
        if invalid.any():
            raise SystemExit(f"STOP: {invalid.sum()} manual test points lack valid {feature} values.")
    return result


def binary_class_weights(y):
    y = np.asarray(y, dtype="int64")
    weights = {}
    for code in (0, 1):
        count = int((y == code).sum())
        if count == 0:
            raise SystemExit("STOP: a binary training class is absent.")
        weights[code] = len(y) / (2 * count)
    return weights, np.asarray([weights[int(value)] for value in y], dtype="float64")


def sampled_configurations(number, pilot):
    keys = list(PARAMETER_SPACE)
    combinations = list(itertools.product(*(PARAMETER_SPACE[key] for key in keys)))
    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(len(combinations), size=number, replace=False)
    configs = [dict(zip(keys, combinations[index])) for index in selected]
    if pilot:
        for config in configs:
            config["n_estimators"] = min(config["n_estimators"], 120)
    return configs


def estimator(bundle, parameters, n_jobs):
    return bundle["xgb"].XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=n_jobs, missing=np.nan, verbosity=0,
        **parameters,
    )


def tune_p3(bundle, sample, weights, configurations, folds, n_jobs):
    features = FEATURE_GROUPS["NTL"] + FEATURE_GROUPS["S2"]
    X = sample[features].to_numpy(dtype="float32")
    y = sample["BinaryLabel"].to_numpy(dtype="int64")
    groups = sample["SpatialCell"].to_numpy()
    splitter = bundle["GroupKFold"](n_splits=folds)
    splits = list(splitter.split(X, y, groups))
    rows = []
    progress = Progress(len(configurations) * folds, "P3 spatial CV tuning")
    for number, parameters in enumerate(configurations, start=1):
        macro, balanced = [], []
        for fold, (train_i, val_i) in enumerate(splits, start=1):
            if len(np.unique(y[train_i])) != 2 or len(np.unique(y[val_i])) != 2:
                raise SystemExit(f"STOP: binary class absent from spatial CV fold {fold}.")
            model = estimator(bundle, parameters, n_jobs)
            model.fit(X[train_i], y[train_i], sample_weight=weights[train_i])
            probability = model.predict_proba(X[val_i])[:, 1]
            prediction = (probability >= THRESHOLD).astype("int64")
            macro.append(bundle["f1_score"](y[val_i], prediction, average="macro", zero_division=0))
            balanced.append(bundle["balanced_accuracy_score"](y[val_i], prediction))
            progress.update()
        row = {"Configuration": number, **parameters}
        for fold, value in enumerate(macro, start=1):
            row[f"Fold_{fold}_MacroF1"] = value
        for fold, value in enumerate(balanced, start=1):
            row[f"Fold_{fold}_BalancedAccuracy"] = value
        row["Mean_MacroF1"] = float(np.mean(macro))
        row["Std_MacroF1"] = float(np.std(macro, ddof=1))
        row["Mean_BalancedAccuracy"] = float(np.mean(balanced))
        rows.append(row)
    results = pd.DataFrame(rows)
    best = results[results["Mean_MacroF1"] >= results["Mean_MacroF1"].max() - 1e-4].sort_values(
        ["max_depth", "n_estimators", "Mean_MacroF1"], ascending=[True, True, False]
    ).iloc[0]
    selected = {key: type(PARAMETER_SPACE[key][0])(best[key]) for key in PARAMETER_SPACE}
    fold_rows = []
    for fold, (train_i, val_i) in enumerate(splits, start=1):
        fold_rows.append({
            "Fold": fold, "TrainingPixels": len(train_i), "ValidationPixels": len(val_i),
            "TrainingGroups": len(np.unique(groups[train_i])),
            "ValidationGroups": len(np.unique(groups[val_i])),
            "TrainingUrban": int((y[train_i] == 1).sum()),
            "TrainingNonUrban": int((y[train_i] == 0).sum()),
            "ValidationUrban": int((y[val_i] == 1).sum()),
            "ValidationNonUrban": int((y[val_i] == 0).sum()),
        })
    return selected, float(best["Mean_MacroF1"]), float(best["Mean_BalancedAccuracy"]), results, pd.DataFrame(fold_rows)


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype="float64")
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, predicted, out=np.zeros(2), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(2), where=support > 0)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(2), where=precision+recall > 0)
        iou = np.divide(tp, support+predicted-tp, out=np.zeros(2), where=support+predicted-tp > 0)
    return {
        "OA": float(tp.sum()/cm.sum()), "BalancedAccuracy": float(recall.mean()),
        "MacroF1": float(f1.mean()), "mIoU": float(iou.mean()),
        "NonUrbanPrecision": precision[0], "NonUrbanRecall": recall[0],
        "NonUrbanF1": f1[0], "NonUrbanIoU": iou[0],
        "UrbanPrecision": precision[1], "UrbanRecall": recall[1],
        "UrbanF1": f1[1], "UrbanIoU": iou[1],
        "support": support.astype("int64"), "precision": precision, "recall": recall,
        "f1": f1, "iou": iou,
    }


def evaluate(bundle, truth, prediction, probability):
    cm = bundle["confusion_matrix"](truth, prediction, labels=[0, 1])
    result = metrics_from_confusion(cm)
    result.update({
        "Kappa": bundle["cohen_kappa_score"](truth, prediction),
        "MCC": bundle["matthews_corrcoef"](truth, prediction),
        "AUROC": bundle["roc_auc_score"](truth, probability),
        "AUPRC": bundle["average_precision_score"](truth, probability),
        "confusion": cm,
    })
    return result


def fit_and_evaluate(bundle, sample, primary, full, parameters, weights, n_jobs):
    y_train = sample["BinaryLabel"].to_numpy(dtype="int64")
    sample_weight = np.asarray([weights[int(value)] for value in y_train])
    predictions = {"primary": {}, "full": {}}
    probabilities = {"primary": {}, "full": {}}
    evaluations = {"primary": {}, "full": {}}
    models = {}
    training_seconds = prediction_seconds = 0.0
    progress = Progress(3, "Final P1-P3 models")
    for model_name, groups in MODEL_GROUPS.items():
        columns = [feature for group in groups for feature in FEATURE_GROUPS[group]]
        model = estimator(bundle, parameters, n_jobs)
        started = time.perf_counter()
        model.fit(sample[columns].to_numpy(dtype="float32"), y_train, sample_weight=sample_weight)
        training_seconds += time.perf_counter() - started
        models[model_name] = model
        started = time.perf_counter()
        for set_name, frame in (("primary", primary), ("full", full)):
            probability = model.predict_proba(frame[columns].to_numpy(dtype="float32"))[:, 1]
            prediction = (probability >= THRESHOLD).astype("int64")
            predictions[set_name][model_name] = prediction
            probabilities[set_name][model_name] = probability
            evaluations[set_name][model_name] = evaluate(
                bundle, frame["BinaryLabel"].to_numpy(dtype="int64"), prediction, probability
            )
        prediction_seconds += time.perf_counter() - started
        progress.update()
    return models, predictions, probabilities, evaluations, training_seconds, prediction_seconds


def performance_table(evaluations):
    rows = []
    for model, result in evaluations.items():
        rows.append({
            "Model": model, "FeatureSource": MODEL_NAMES[model],
            "NumberOfFeatures": sum(len(FEATURE_GROUPS[g]) for g in MODEL_GROUPS[model]),
            **{key: result[key] for key in [
                "OA", "BalancedAccuracy", "Kappa", "MCC", "MacroF1", "mIoU",
                "AUROC", "AUPRC", "UrbanPrecision", "UrbanRecall", "UrbanF1",
                "UrbanIoU", "NonUrbanPrecision", "NonUrbanRecall", "NonUrbanF1", "NonUrbanIoU",
            ]},
        })
    return pd.DataFrame(rows)


def comparison_table(performance):
    indexed = performance.set_index("Model")
    rows = []
    for first, second in COMPARISONS:
        for metric in DIFFERENCE_METRICS:
            rows.append({
                "Comparison": f"{first} - {second}", "Metric": metric,
                "FirstValue": indexed.loc[first, metric], "SecondValue": indexed.loc[second, metric],
                "ObservedDifference": indexed.loc[first, metric] - indexed.loc[second, metric],
            })
    return pd.DataFrame(rows)


def bootstrap_primary(truth, predictions, replicates):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = [np.flatnonzero(truth == code) for code in (0, 1)]
    distributions = {model: {m: np.empty(replicates) for m in BOOTSTRAP_METRICS} for model in MODEL_GROUPS}
    progress = Progress(replicates, "Paired stratified bootstrap")
    for replicate in range(replicates):
        sampled = np.concatenate([rng.choice(part, len(part), replace=True) for part in indices])
        y = truth[sampled]
        for model, prediction in predictions.items():
            cm = np.bincount(2*y + prediction[sampled], minlength=4).reshape(2, 2)
            values = metrics_from_confusion(cm)
            for metric in BOOTSTRAP_METRICS:
                distributions[model][metric][replicate] = values[metric]
        progress.update()
    rows = []
    observed = {model: metrics_from_confusion(
        np.bincount(2*truth + prediction, minlength=4).reshape(2, 2)
    ) for model, prediction in predictions.items()}
    for first, second in COMPARISONS:
        for metric in BOOTSTRAP_METRICS:
            difference = distributions[first][metric] - distributions[second][metric]
            lower, upper = np.percentile(difference, [2.5, 97.5])
            rows.append({
                "Comparison": f"{first} - {second}", "Metric": metric,
                "ObservedDifference": observed[first][metric] - observed[second][metric],
                "BootstrapMeanDifference": float(difference.mean()),
                "CI_Lower_2.5": float(lower), "CI_Upper_97.5": float(upper),
                "CI_Includes_Zero": bool(lower <= 0 <= upper), "Replicates": replicates,
            })
    return pd.DataFrame(rows)


def prediction_table(frame, predictions, probabilities):
    output = frame[["Test_ID", "Object_ID", "X", "Y", "X_32639", "Y_32639", "TrueClass", "SourceClass3"]].copy()
    for model in MODEL_GROUPS:
        output[f"{model}_PredictedClass"] = [BINARY_NAMES[int(value)] for value in predictions[model]]
        output[f"{model}_P_Urban"] = probabilities[model]
    return output


def confusion_table(evaluations, test_name):
    rows = []
    for model, result in evaluations.items():
        cm = result["confusion"]
        order = [1, 0]
        reordered = cm[np.ix_(order, order)]
        normalized = reordered / reordered.sum(axis=1, keepdims=True)
        names = ["Urban", "NonUrban"]
        for i, true_name in enumerate(names):
            for j, predicted_name in enumerate(names):
                rows.append({
                    "TestSet": test_name, "Model": model, "TrueClass": true_name,
                    "PredictedClass": predicted_name, "RawCount": int(reordered[i, j]),
                    "RowNormalized": float(normalized[i, j]),
                })
    return pd.DataFrame(rows)


def plot_outputs(bundle, output_dir, sample, performance, evaluations):
    plt = bundle["plt"]
    counts = sample["SourceClass3"].value_counts().reindex(["Urban", "Vegetation", "BareScrub"])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#d95f02", "#1b9e77", "#7570b3"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Selected training pixels")
    ax.set_title("Balanced sampling before binary class merging")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "Pixel_Training_Class_Distribution.png", dpi=220)
    plt.close(fig)

    metrics = ["BalancedAccuracy", "MacroF1", "UrbanF1", "UrbanIoU"]
    x = np.arange(3)
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for offset, metric in enumerate(metrics):
        values = performance.set_index("Model").loc[list(MODEL_GROUPS), metric]
        ax.bar(x + (offset-1.5)*width, values, width, label=metric)
    ax.set_xticks(x, [MODEL_NAMES[m] for m in MODEL_GROUPS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Primary manual-test score")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "Pixel_Spectral_Sensor_Performance_Comparison.png", dpi=220)
    plt.close(fig)

    for model, result in evaluations.items():
        cm = result["confusion"][np.ix_([1, 0], [1, 0])]
        normalized = cm / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.6))
        for ax, matrix, title, fmt in zip(axes, [cm, normalized], ["Counts", "Row-normalized"], ["d", ".3f"]):
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=(matrix.max() if title == "Counts" else 1))
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center")
            ax.set_xticks([0, 1], ["Urban", "NonUrban"])
            ax.set_yticks([0, 1], ["Urban", "NonUrban"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
        fig.suptitle(f"{model}: {MODEL_NAMES[model]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"Pixel_{model}_Confusion_Matrix.png", dpi=220)
        plt.close(fig)


def map_profile(stack, dtype, nodata):
    profile = stack.master.profile.copy()
    profile.update(
        driver="GTiff", dtype=dtype, count=1, nodata=nodata,
        compress="DEFLATE", predictor=2 if dtype == "float32" else 1,
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES",
    )
    return profile


def map_full_scene(bundle, output_dir, stack, masks, windows, models):
    rasterio = bundle["rasterio"]
    class_paths = {
        "P1": output_dir / "Pixel_P1_NTL_Class.tif",
        "P2": output_dir / "Pixel_P2_Sentinel2_Class.tif",
        "P3": output_dir / "Pixel_P3_NTL_Sentinel2_Class.tif",
    }
    probability_paths = {
        "P1": output_dir / "Pixel_P1_NTL_PUrban.tif",
        "P2": output_dir / "Pixel_P2_Sentinel2_PUrban.tif",
        "P3": output_dir / "Pixel_P3_NTL_Sentinel2_PUrban.tif",
    }
    class_datasets = {model: rasterio.open(path, "w", **map_profile(stack, "uint8", 0)) for model, path in class_paths.items()}
    probability_datasets = {model: rasterio.open(path, "w", **map_profile(stack, "float32", -9999.0)) for model, path in probability_paths.items()}
    counts = {model: {1: 0, 2: 0} for model in MODEL_GROUPS}
    feature_names = list(PREDICTOR_PATHS)
    progress = Progress(len(windows), "Full-scene P1-P3 mapping")
    try:
        for window in windows:
            predictors, invalid, _ = read_predictor_arrays(stack, window)
            valid = masks.study(window) & ~invalid
            for model, groups in MODEL_GROUPS.items():
                classes = np.zeros(valid.shape, dtype="uint8")
                probability = np.full(valid.shape, -9999.0, dtype="float32")
                if valid.any():
                    names = [name for group in groups for name in FEATURE_GROUPS[group]]
                    indices = [feature_names.index(name) for name in names]
                    X = predictors[indices][:, valid].T
                    p = models[model].predict_proba(X)[:, 1].astype("float32")
                    pred = np.where(p >= THRESHOLD, 1, 2).astype("uint8")
                    probability[valid] = p
                    classes[valid] = pred
                    counts[model][1] += int((pred == 1).sum())
                    counts[model][2] += int((pred == 2).sum())
                class_datasets[model].write(classes, 1, window=window)
                probability_datasets[model].write(probability, 1, window=window)
            progress.update()
    finally:
        for dataset in list(class_datasets.values()) + list(probability_datasets.values()):
            dataset.close()
    rows = []
    for model in MODEL_GROUPS:
        with rasterio.open(class_paths[model]) as source:
            metadata_ok = same_grid(source, stack.master) and source.nodata == 0
            unique = set()
            post_counts = {1: 0, 2: 0}
            for _, window in source.block_windows(1):
                data = source.read(1, window=window)
                unique.update(np.unique(data).tolist())
                post_counts[1] += int((data == 1).sum())
                post_counts[2] += int((data == 2).sum())
        if not metadata_ok or not unique.issubset({0, 1, 2}) or post_counts != counts[model]:
            raise SystemExit(f"STOP: post-write map validation failed for {model}.")
        # The master is a nominal 10 m grid; use 0.0001 km2 per pixel as a
        # transparent diagnostic rather than an accuracy statement.
        rows.append({
            "Model": model, "ClassRaster": str(class_paths[model].resolve()),
            "ProbabilityRaster": str(probability_paths[model].resolve()),
            "UrbanPixels": counts[model][1], "NonUrbanPixels": counts[model][2],
            "ValidPixels": counts[model][1] + counts[model][2],
            "NominalClassifiedArea_km2": (counts[model][1] + counts[model][2]) * 0.0001,
            "CRSTransformShapeValidated": True, "NoData": 0,
            "ObservedClassCodes": ",".join(map(str, sorted(unique))),
        })
    return pd.DataFrame(rows)


def object_pixel_comparison(pixel_primary, pixel_full, output_dir):
    primary_path = OBJECT_SENSOR_DIR / "Spectral_Sensor_Primary_Performance.csv"
    full_path = OBJECT_SENSOR_DIR / "Spectral_Sensor_FullTest_Performance.csv"
    if not primary_path.exists() or not full_path.exists():
        return None
    mapping = {"M1": "P1", "M2": "P2", "M3": "P3"}
    rows = []
    for test_name, object_path, pixel in (
        ("Primary", primary_path, pixel_primary), ("Full", full_path, pixel_full)
    ):
        obj = pd.read_csv(object_path).copy()
        obj["Model"] = obj["Model"].map(mapping)
        obj = obj.set_index("Model")
        pix = pixel.set_index("Model")
        for model in MODEL_GROUPS:
            for metric in ("UrbanF1", "UrbanIoU", "MacroF1", "BalancedAccuracy"):
                rows.append({
                    "TestSet": test_name, "Model": model, "FeatureSource": MODEL_NAMES[model],
                    "Metric": metric, "ObjectBasedValue": obj.loc[model, metric],
                    "PixelBasedValue": pix.loc[model, metric],
                    "PixelMinusObject": pix.loc[model, metric] - obj.loc[model, metric],
                    "SupportCaution": "Descriptive only: object-wide versus point-pixel prediction support",
                })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "Object_vs_Pixel_Sensor_Comparison.csv", index=False)
    return result


def markdown_table(frame, columns=None, digits=4):
    if frame is None or frame.empty:
        return "_Not available._"
    data = frame.copy()
    if columns is not None:
        data = data[columns]
    for column in data.select_dtypes(include=["float", "float32", "float64"]):
        data[column] = data[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(c) for c in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def report_conclusion(performance, object_comparison):
    p = performance.set_index("Model")
    ntl_vs_s2 = p.loc["P1", "UrbanF1"] - p.loc["P2", "UrbanF1"]
    fusion = p.loc["P3", "UrbanF1"] - p.loc["P2", "UrbanF1"]
    if abs(ntl_vs_s2) < 0.01:
        q1 = "NTL and Sentinel-2 were practically close in Urban F1"
    elif ntl_vs_s2 > 0:
        q1 = "NTL exceeded Sentinel-2 in Urban F1"
    else:
        q1 = "Sentinel-2 exceeded NTL in Urban F1"
    q2 = "Fusion improved Urban F1 beyond Sentinel-2" if fusion > 0 else "Fusion did not improve Urban F1 beyond Sentinel-2"
    object_text = "No object-based comparison table was available."
    if object_comparison is not None:
        op = object_comparison
        primary = op[op["TestSet"] == "Primary"].set_index(["Model", "Metric"])
        obj_fusion = primary.loc[("P3", "UrbanF1"), "ObjectBasedValue"] - primary.loc[("P2", "UrbanF1"), "ObjectBasedValue"]
        relation = "stronger" if fusion > obj_fusion + 0.01 else ("weaker" if fusion < obj_fusion - 0.01 else "similar")
        object_text = (
            f"The pixel fusion Urban-F1 increment ({fusion:+.4f}) was {relation} than the descriptive "
            f"object increment ({obj_fusion:+.4f}). Different prediction support prevents a strict paired claim."
        )
    return q1, q2, object_text


def write_report(path, pilot, args, stack, manifest, dq, summary, candidate_counts,
                 class_counts, sample, primary, full, test_diagnostics, masks,
                 class_weights, folds, selected_parameters, cv_macro, cv_balanced,
                 primary_performance, full_performance, differences, bootstrap,
                 maps, object_comparison, runtimes, validation):
    mode = "PILOT" if pilot else "FULL DATASET"
    p = primary_performance.set_index("Model")
    q1, q2, object_text = report_conclusion(primary_performance, object_comparison)
    class_summary = summary[summary["Level"] == "SourceClass"]
    feature_lines = "\n".join(
        f"- **{model} ({MODEL_NAMES[model]}):** " + ", ".join(
            feature for group in groups for feature in FEATURE_GROUPS[group]
        ) for model, groups in MODEL_GROUPS.items()
    )
    alignment = pd.DataFrame(stack.audit_rows)[[
        "Layer", "NativeCRS", "NativeWidth", "NativeHeight", "NativeResolutionX",
        "NativeResolutionY", "ProcessingStatus", "PixelCenterAligned",
    ]]
    test_counts = pd.DataFrame([
        {"Set": name, "Count": len(frame), "Urban": int((frame["BinaryLabel"] == 1).sum()),
         "NonUrban": int((frame["BinaryLabel"] == 0).sum())}
        for name, frame in (("Primary", primary), ("Full", full))
    ])
    validation_frame = pd.DataFrame([{"Check": key, "Result": value} for key, value in validation.items()])
    map_text = markdown_table(maps) if maps is not None and not maps.empty else (
        "Full-scene maps were deliberately not generated in the pilot. The complete command enables chunked mapping."
        if pilot else "Full-scene maps were skipped by explicit command-line override."
    )
    deviations = (
        f"Pilot-only deviations: {args.pilot_windows} geographically dispersed raster windows; "
        f"target cap {args.pilot_target_per_class:,} per source class; {args.pilot_configurations} tuning "
        f"configurations with n_estimators capped at 120; 3 spatial folds; {args.pilot_bootstrap} bootstrap "
        "replicates; no full-scene maps. These limits validate the workflow but do not constitute the final experiment."
        if pilot else (
            "No methodological deviation." if not args.skip_maps else
            "Full-scene mapping was skipped via --skip-maps; this is a user-selected diagnostic deviation."
        )
    )
    report = f"""# Pixel-Based Spectral Sensor Comparison (XGBoost 01) — {mode}

Output directory: `{path.parent.resolve()}`

## A. Input rasters

- Master Sentinel-2 grid: `{MASTER_RASTER.resolve()}`
- Dynamic World reference: `{REFERENCE_RASTER.resolve()}`
- Manual independent test points: `{MANUAL_POINTS.resolve()}`
- Tehran Province boundary: `{PROVINCE_PATH.resolve()}`
- Segmentation (membership/exclusion only): `{SEGMENTS_PATH.resolve()}`
- Strict Rook graph: `{ADJACENCY_PATH.resolve()}`

{markdown_table(manifest, ["FeatureName", "Sensor", "SourceRaster", "ProcessingStatus"], 4)}

The four direct corrected/coregistered SDGSAT-1 layers used by the validated object spectral experiment were retained: PAN/HDR plus pansharpened R/G/B. PL and PH were not added separately because they were not separate predictors in that experiment and PAN/HDR is derived from them. No optical index was included: the controlled P2 set is direct B2/B3/B4/B8 only.

## B. Raster alignment

Master shape is `{stack.master.height} × {stack.master.width}`, CRS `{stack.master.crs}`, resolution `{stack.master.res}`, transform `{tuple(stack.master.transform)}`. All final readers passed exact CRS, shape, transform, extent, and pixel-center checks.

{markdown_table(alignment, digits=8)}

## C. Pixel training design

Dynamic World 1/2/3 pixels with all predictors valid, province-center inclusion, and outside the test/test-neighbour objects formed the candidate pool. Equal source-class sampling was applied before binary merging and distributed as evenly as capacity allowed over occupied EPSG:32639 5-km cells.

{markdown_table(class_summary, ["SourceClass3", "CandidateCount", "SelectedCount", "CandidateOccupiedCells", "SelectedOccupiedCells", "CandidateRetainedPercent"])}

- Selected binary counts: Urban `{int((sample.BinaryLabel == 1).sum()):,}`; NonUrban `{int((sample.BinaryLabel == 0).sum()):,}`.
- Selected spatial groups: `{sample.SpatialCell.nunique():,}`.
- Inverse-frequency weights: NonUrban `{class_weights[0]:.6f}`, Urban `{class_weights[1]:.6f}`; identical for P1–P3.

## D. Test data

{markdown_table(test_counts)}

Membership was inherited from the object-based primary/full CSVs. For multiple consistent points in one object, the smallest `Point_ID` (then source row) was chosen. Binary conflicts were excluded from evaluation but every labeled manual-test object was retained in the exclusion set. Training excluded `{test_diagnostics['all_test_objects_used_for_exclusion']:,}` manual-test objects and `{len(masks.neighbor_ids):,}` unique first-order Rook neighbours.

## E. Feature configurations

{feature_lines}

All values are direct containing-pixel raster values. There is no geometry, texture, neighbourhood statistic, spatial context, or object-level summary.

## F. Data quality audit

{markdown_table(dq, ["FeatureName", "AuditScope", "ValidPixelCount", "NaN_NoDataCount", "Minimum", "Maximum", "Mean", "StandardDeviation", "TrainingSampleUniqueValues", "ConstantRaster"])}

No constant or exact-duplicate training predictor was found; optical ranges passed the explicit sanity guard. All final project rasters are already pixel-aligned, so no unresolved resampling/coregistration operation remained.

## G. Spatial CV

GroupKFold used `{len(folds)}` folds and `{sample.SpatialCell.nunique()}` training-only 5-km groups. No test label entered tuning.

{markdown_table(folds)}

## H. Hyperparameter tuning

P3 alone was tuned by training-only spatial CV, with Macro-F1 primary and Balanced Accuracy secondary. Selected P3 CV Macro-F1 = `{cv_macro:.4f}` and Balanced Accuracy = `{cv_balanced:.4f}`. The selected parameters were frozen for P1–P3:

```json
{json.dumps(selected_parameters, indent=2)}
```

## I. Primary test results

{markdown_table(primary_performance)}

## J. Full test results

{markdown_table(full_performance)}

## K. Sensor contribution

{markdown_table(differences)}

Central P3−P2 primary differences: Urban F1 `{p.loc['P3','UrbanF1']-p.loc['P2','UrbanF1']:+.4f}` and Urban IoU `{p.loc['P3','UrbanIoU']-p.loc['P2','UrbanIoU']:+.4f}`.

## L. Bootstrap results

Paired stratified primary-test bootstrap; “CI includes zero” is reported descriptively and no automatic significance wording is used.

{markdown_table(bootstrap)}

## M. Pixel classification maps

{map_text}

Mapped area, when available, is a nominal 10-m pixel-count diagnostic only and is not reference accuracy.

## N. Comparison with object-based results

{object_text}

Object predictions apply to entire segments; pixel predictions use the containing pixel at each manual point. The table is descriptive, not a strict paired representation test. Consequently, whether object representation “improves the value extracted from NTL” can only be discussed descriptively from the relative sensor gaps, not asserted causally.

## O. Scientific conclusion

1. {q1} (difference `{p.loc['P1','UrbanF1']-p.loc['P2','UrbanF1']:+.4f}`).
2. Sentinel-2 versus NTL must be judged from the complete metric table; no narrative was forced.
3. {q2} (P3−P2 `{p.loc['P3','UrbanF1']-p.loc['P2','UrbanF1']:+.4f}`).
4. {object_text}
5. Representation implications remain descriptive because evaluation support differs.

## P. Runtime

{markdown_table(pd.DataFrame([runtimes]))}

## Q. Deviations

{deviations}

## Mandatory validation checks

{markdown_table(validation_frame)}

## Main scientific findings

- Primary Urban F1: P1 `{p.loc['P1','UrbanF1']:.4f}`, P2 `{p.loc['P2','UrbanF1']:.4f}`, P3 `{p.loc['P3','UrbanF1']:.4f}`.
- Primary Urban IoU: P1 `{p.loc['P1','UrbanIoU']:.4f}`, P2 `{p.loc['P2','UrbanIoU']:.4f}`, P3 `{p.loc['P3','UrbanIoU']:.4f}`.
- Primary P3−P2 Macro-F1: `{p.loc['P3','MacroF1']-p.loc['P2','MacroF1']:+.4f}`.
- {('These are **pilot findings**, not final estimates. Run the documented full command before manuscript inference.' if pilot else 'These are the complete-run estimates for the configured inputs and design.')}

## Implications for the manuscript

The results support only the predictor configurations demonstrated by the tables. Pixel-based NTL, direct pixel fusion, and object-based NTL should be described separately; object-vs-pixel differences remain descriptive because their prediction supports differ.{(' The full run is required before selecting a manuscript narrative.' if pilot else '')}
"""
    path.write_text(report, encoding="utf-8")


def main():
    total_started = time.perf_counter()
    args = parse_args()
    pilot = not args.full_run
    if args.block_size < 256:
        raise SystemExit("STOP: --block-size must be at least 256.")
    require_inputs()
    bundle = dependencies()
    if args.output_dir is None:
        suffix = "_pilot" if pilot else ""
        args.output_dir = SEGMENTATION_DIR / f"Pixel_Based_Spectral_Sensor_Comparison_XGBoost_01{suffix}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtimes = {}
    stack = None
    try:
        started = time.perf_counter()
        stack = RasterStack(bundle, output_dir)
        manifest = write_feature_manifest(stack, output_dir / "Pixel_Spectral_Sensor_Feature_Manifest.csv")
        segments, primary, full, test_ids, test_diagnostics = prepare_test_points(bundle, stack)
        masks = SpatialMasks(bundle, stack, segments, test_ids)
        primary = attach_point_predictors(stack, primary)
        full = attach_point_predictors(stack, full)
        primary[["Test_ID", "Object_ID", "Row", "Col", "X", "Y", "X_32639", "Y_32639", "SourceClass3", "TrueClass"]].to_csv(
            output_dir / "Pixel_Primary_Test_Points.csv", index=False
        )
        full[["Test_ID", "Object_ID", "Row", "Col", "X", "Y", "X_32639", "Y_32639", "SourceClass3", "TrueClass"]].to_csv(
            output_dir / "Pixel_Full_Test_Points.csv", index=False
        )
        windows_all = relevant_windows(bundle, stack, masks, args.block_size)
        windows = choose_pilot_windows(bundle, stack, masks, windows_all, args.pilot_windows) if pilot else windows_all
        runtimes["RasterAndVectorLoadingSeconds"] = time.perf_counter() - started

        started = time.perf_counter()
        candidate_counts, class_counts, class_cells, stats, study_pixels, transformer = first_pass(
            bundle, stack, masks, windows
        )
        cap = args.pilot_target_per_class if pilot else 50000
        target, quotas = quotas_from_counts(candidate_counts, class_counts, cap)
        sample = second_pass_sample(bundle, stack, masks, windows, candidate_counts, quotas, transformer)
        leaked = verify_no_exclusion_leakage(bundle, sample, masks, stack)
        expected = {1: target, 2: target, 3: target}
        observed = sample["SourceCode3"].value_counts().to_dict()
        if observed != expected:
            raise SystemExit(f"STOP: source classes are not exactly balanced: {observed}.")
        if not sample["SourceCode3"].isin([1, 2, 3]).all():
            raise SystemExit("STOP: uncertain DW code entered training.")
        sample.to_csv(output_dir / "Pixel_Training_Sample.csv", index=False)
        summary = sampling_summary(sample, candidate_counts, class_counts, class_cells)
        summary.to_csv(output_dir / "Pixel_Training_Sampling_Summary.csv", index=False)
        scope = f"Pilot representative windows ({len(windows)}/{len(windows_all)})" if pilot else "Full Tehran Province"
        dq, duplicate_pairs = data_quality_table(sample, stats, study_pixels, scope)
        dq.to_csv(output_dir / "Pixel_Predictor_Data_Quality_Audit.csv", index=False)
        runtimes["SamplingAndQualityAuditSeconds"] = time.perf_counter() - started

        y = sample["BinaryLabel"].to_numpy(dtype="int64")
        class_weights, sample_weights = binary_class_weights(y)
        configurations = sampled_configurations(args.pilot_configurations if pilot else 24, pilot)
        folds_number = 3 if pilot else 5
        started = time.perf_counter()
        parameters, cv_macro, cv_balanced, cv_results, fold_table = tune_p3(
            bundle, sample, sample_weights, configurations, folds_number, args.n_jobs
        )
        cv_results.to_csv(output_dir / "Pixel_Spectral_Sensor_Spatial_CV_Results.csv", index=False)
        fold_table.to_csv(output_dir / "Pixel_Spatial_CV_Fold_Audit.csv", index=False)
        runtimes["CVTuningSeconds"] = time.perf_counter() - started

        parameter_payload = {
            "mode": "pilot" if pilot else "full_run", "random_state": RANDOM_SEED,
            "threshold": THRESHOLD, "selected_parameters": parameters,
            "P3_mean_spatial_CV_MacroF1": cv_macro,
            "P3_mean_spatial_CV_BalancedAccuracy": cv_balanced,
            "class_weights": {BINARY_NAMES[key]: value for key, value in class_weights.items()},
            "feature_groups": FEATURE_GROUPS, "identical_for_P1_P2_P3": True,
        }
        (output_dir / "Pixel_Selected_Hyperparameters.json").write_text(
            json.dumps(parameter_payload, indent=2), encoding="utf-8"
        )

        started = time.perf_counter()
        models, predictions, probabilities, evaluations, train_seconds, predict_seconds = fit_and_evaluate(
            bundle, sample, primary, full, parameters, class_weights, args.n_jobs
        )
        runtimes["FinalModelTrainingSeconds"] = train_seconds
        runtimes["ManualTestPredictionSeconds"] = predict_seconds
        primary_performance = performance_table(evaluations["primary"])
        full_performance = performance_table(evaluations["full"])
        primary_performance.to_csv(output_dir / "Pixel_Spectral_Sensor_Primary_Performance.csv", index=False)
        full_performance.to_csv(output_dir / "Pixel_Spectral_Sensor_FullTest_Performance.csv", index=False)
        primary_predictions = prediction_table(primary, predictions["primary"], probabilities["primary"])
        full_predictions = prediction_table(full, predictions["full"], probabilities["full"])
        primary_predictions.to_csv(output_dir / "Pixel_Primary_Test_Predictions.csv", index=False)
        full_predictions.to_csv(output_dir / "Pixel_Full_Test_Predictions.csv", index=False)
        differences = comparison_table(primary_performance)
        differences.to_csv(output_dir / "Pixel_Spectral_Sensor_Performance_Differences.csv", index=False)
        confusion = pd.concat([
            confusion_table(evaluations["primary"], "Primary"),
            confusion_table(evaluations["full"], "Full"),
        ], ignore_index=True)
        confusion.to_csv(output_dir / "Pixel_Spectral_Sensor_Confusion_Matrices.csv", index=False)
        runtimes["ResultAssemblySeconds"] = time.perf_counter() - started

        started = time.perf_counter()
        replicates = args.pilot_bootstrap if pilot else 2000
        bootstrap = bootstrap_primary(
            primary["BinaryLabel"].to_numpy(dtype="int64"), predictions["primary"], replicates
        )
        bootstrap.to_csv(output_dir / "Pixel_Spectral_Sensor_Paired_Bootstrap.csv", index=False)
        runtimes["BootstrapSeconds"] = time.perf_counter() - started

        started = time.perf_counter()
        plot_outputs(bundle, output_dir, sample, primary_performance, evaluations["primary"])
        runtimes["FigureSeconds"] = time.perf_counter() - started

        maps = pd.DataFrame()
        started = time.perf_counter()
        if not pilot and not args.skip_maps:
            maps = map_full_scene(bundle, output_dir, stack, masks, windows_all, models)
            maps.to_csv(output_dir / "Pixel_Full_Map_Validation.csv", index=False)
        runtimes["FullRasterMappingSeconds"] = time.perf_counter() - started

        object_comparison = object_pixel_comparison(primary_performance, full_performance, output_dir)
        # Identical rows, labels, weights, parameters, and threshold are shared
        # programmatically rather than copied into independent workflows.
        validation = OrderedDict([
            ("All predictor readers exactly match the master grid", True),
            ("P1/P2/P3 use identical training pixels", True),
            ("P1/P2/P3 use identical primary and full test rows", True),
            ("P1/P2/P3 use identical class weights", True),
            ("P1/P2/P3 use identical frozen hyperparameters", True),
            ("P1/P2/P3 use threshold 0.50", True),
            ("Training pixels in test/test-neighbour objects", leaked),
            ("Uncertain DW pixels in training", int((~sample["SourceCode3"].isin([1, 2, 3])).sum())),
            ("Object geometry predictors", 0), ("Object texture predictors", 0),
            ("Object spatial-context predictors", 0),
            ("Manual test labels used in tuning", 0),
            ("Full maps generated", bool(not maps.empty)),
        ])
        runtimes["TotalSeconds"] = time.perf_counter() - total_started
        write_report(
            output_dir / "Pixel_Based_Spectral_Sensor_Comparison_XGBoost_01_Report.md",
            pilot, args, stack, manifest, dq, summary, candidate_counts, class_counts,
            sample, primary, full, test_diagnostics, masks, class_weights, fold_table,
            parameters, cv_macro, cv_balanced, primary_performance, full_performance,
            differences, bootstrap, maps, object_comparison, runtimes, validation,
        )
        print(f"Completed {('pilot' if pilot else 'full run')}: {output_dir}")
        print(f"Balanced source-class target: {target:,}; total training pixels: {len(sample):,}")
        if pilot:
            print("Full-scene maps were not run by design.")
    finally:
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
