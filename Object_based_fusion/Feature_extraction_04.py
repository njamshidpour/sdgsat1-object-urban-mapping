"""Raw object-level SDGSAT, optical, geometry, and GLCM feature extraction.

This is the developed successor to ``Feature_extraction_03.py``.  It keeps
exactextract for fast, coverage-weighted zonal statistics and adds:

* preservation of the original eCognition ``Seg_id``;
* metric geometry in an explicitly configured projected CRS;
* masked, windowed GLCM texture from the SDGSAT-1 panchromatic radiance;
* optional batched Sentinel-2 band/index statistics with exact fractional coverage;
* optical grid and per-raster object diagnostics;
* explicit CRS/NoData/invalid-geometry checks; and
* CSV plus optional GeoPackage/Shapefile output.

No raster is resampled or reprojected.  Geometry is transformed in memory
only where needed: to the raster CRS for extraction (disabled by default) or
to ``METRIC_CRS`` for metric shape measurements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import exactextract
from exactextract.raster import RasterioRasterSource
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely import make_valid
from shapely.geometry import mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


LOGGER = logging.getLogger("object_feature_extraction")


class TerminalProgress:
    """Compact single-line terminal progress with elapsed time and ETA."""

    def __init__(self, label: str, *, min_interval: float = 0.5, enabled: bool = True):
        self.label = label
        self.min_interval = min_interval
        self.enabled = enabled
        self.started = perf_counter()
        self.last_update = -math.inf
        self.last_fraction = -1.0

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or not np.isfinite(seconds):
            return "--:--"
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update(self, fraction: float, message: str = "", *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        fraction = min(1.0, max(0.0, float(fraction)))
        if not force and fraction < 1.0 and now - self.last_update < self.min_interval:
            return
        elapsed = now - self.started
        eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        width = 20
        filled = min(width, int(round(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"\r\033[2K[{bar}] {fraction * 100:5.1f}% {self.label}"
            f" | {self._duration(elapsed)} | ETA {self._duration(eta)}"
        )
        print(line, end="\n" if fraction >= 1.0 else "", file=sys.stderr, flush=True)
        self.last_update = now
        self.last_fraction = fraction

    def exactextract_callback(self, fraction: float, message: str) -> None:
        # exactextract currently sends "." as its status; omitting it keeps the
        # progress line short enough to avoid wrapping in ordinary terminals.
        self.update(fraction)

    def complete(self, message: str = "done") -> None:
        if self.last_fraction < 1.0:
            self.update(1.0, force=True)


# ---------------------------------------------------------------------------
# Project configuration: add/remove a raster only in RASTER_INPUTS.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"
OPTICAL_DATA_DIR = PROJECT_ROOT / "optic_data_codex"
NTL_DIR = PROJECT_ROOT / "output_corrected_bands"
NTL_INDEX_DIR = PROJECT_ROOT / "NTL_indices"

VECTOR_PATH = SEGMENT_DIR / "eCog_segments.shp"
OUTPUT_CSV = SEGMENT_DIR / "objects_features_v04.csv"

# Python dictionaries preserve insertion order.  The output spectral columns
# follow this order, with mean, std, and max emitted for each raster.
RASTER_INPUTS: dict[str, Path] = {
    "SDGSAT_Pan": NTL_DIR / "Pan_HDR_Radiance.tif",
    "SDGSAT_R": NTL_DIR / "Red_Radiance_salt_pepper_10m.tif",
    "SDGSAT_G": NTL_DIR / "Green_Radiance_salt_pepper_10m.tif",
    "SDGSAT_B": NTL_DIR / "Blue_Radiance_salt_pepper_10m.tif",
    "BR": NTL_INDEX_DIR / "BR_10m.tif",
    "ENUI": NTL_INDEX_DIR / "ENUI.tif",
    "NBDI": NTL_INDEX_DIR / "NBDI.tif",
    "NDIBG": NTL_INDEX_DIR / "NDIBG.tif",
    "NDIGR": NTL_INDEX_DIR / "NDIGR.tif",
    "NLCSI": NTL_INDEX_DIR / "NLCSI.tif",
    "SONDI": NTL_INDEX_DIR / "SONDI.tif",
    "TVANUI": NTL_INDEX_DIR / "TVANUI.tif",
    "VANUI": NTL_INDEX_DIR / "VANUI.tif",
}

# Object-level optical inputs. Keep these dictionaries empty until the exact
# user-selected Sentinel-2 bands/indices are supplied. Any number of single-band
# rasters can be added; relative paths are resolved under OPTICAL_DATA_DIR.
OPTICAL_BANDS: dict[str, Path | str] = {
    "S2_B2": OPTICAL_DATA_DIR / "Tehran_Raw_B2_10m.tif",
    "S2_B3": OPTICAL_DATA_DIR / "Tehran_Raw_B3_10m.tif",
    "S2_B4": OPTICAL_DATA_DIR / "Tehran_Raw_B4_10m.tif",
    "S2_B5": OPTICAL_DATA_DIR / "B05_10m_pansharpened_GS.tif",
    "S2_B6": OPTICAL_DATA_DIR / "B06_10m_pansharpened_GS.tif",
    "S2_B7": OPTICAL_DATA_DIR / "B07_10m_pansharpened_GS.tif",
    "S2_B8": OPTICAL_DATA_DIR / "Tehran_Raw_B8_10m.tif",
    "S2_B8A": OPTICAL_DATA_DIR / "B8A_10m_pansharpened_GS.tif",
    "S2_B11": OPTICAL_DATA_DIR / "B11_10m_pansharpened_GS.tif",
    "S2_B12": OPTICAL_DATA_DIR / "B12_10m_pansharpened_GS.tif",
}

OPTICAL_INDICES: dict[str, Path | str] = {
    "NDVI": OPTICAL_DATA_DIR / "Tehran_NDVI_2023.tif",
    "NDBI": OPTICAL_DATA_DIR / "Tehran_NDBI_2023.tif",
    "NDWI": OPTICAL_DATA_DIR / "Tehran_NDWI_2023.tif",
}

# Optical grid diagnostics are compared with this existing project grid.
# Grid mismatch is reported but does not invalidate independent zonal statistics;
# CRS mismatch with the segmentation vector is an error by default.
OPTICAL_ALIGNMENT_REFERENCE = RASTER_INPUTS["SDGSAT_Pan"]

# The source has a valid, unique Seg_id and a non-informative all-zero ID.
OBJECT_ID_FIELD = "Seg_id"
OUTPUT_ID_FIELD = "Object_ID"

# EPSG:32639 is WGS 84 / UTM zone 39N, the explicit metric CRS used for the
# Tehran study extent.  Only vector geometry is transformed for measurements.
METRIC_CRS = "EPSG:32639"

# The panchromatic HDR radiance directly represents total NTL intensity and is
# therefore preferable to colour ratios or NTL-derived indices for texture.
TEXTURE_RASTER_NAME = "SDGSAT_Pan"


@dataclass(frozen=True)
class GLCMConfig:
    """Reproducible settings for object-masked GLCM extraction."""

    levels: int = 32
    distances: tuple[int, ...] = (1,)
    angles_degrees: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    min_valid_pixels: int = 9
    all_touched: bool = False
    # One raster-wide robust range is estimated once, making object values
    # comparable. Values outside this range are clipped before quantization.
    quantile_range: tuple[float, float] = (2.0, 98.0)
    quantization_sample_size: int = 1_000_000


GLCM = GLCMConfig()

# feature-sequential performs local reads with predictable memory.  All
# configured rasters are still evaluated in one exactextract processor pass.
EXACTEXTRACT_STRATEGY = "feature-sequential"
EXACTEXTRACT_MAX_CELLS_IN_MEMORY = 3_000_000

SDGSAT_STATISTICS = (
    ("mean", "mean"),
    ("stdev", "std"),
    ("max", "max"),
)

# Optical bands and indices use the same compact summary as the SDGSAT inputs.
OPTICAL_STATISTICS = (
    ("mean", "mean", "mean"),
    ("stdev", "stdev", "std"),
    ("max", "max", "max"),
)


def _polygonal_part(geometry):
    """Return polygonal components of a repaired geometry, or None."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        polygons = [
            part
            for part in geometry.geoms
            if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty
        ]
        return unary_union(polygons) if polygons else None
    return None


def _repair_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    repaired = geometry if geometry.is_valid else make_valid(geometry)
    return _polygonal_part(repaired)


def verify_and_prepare_objects(
    vector_path: os.PathLike[str] | str,
    object_id_field: str = OBJECT_ID_FIELD,
    limit: int | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    """Read objects once, preserve IDs, and repair invalid polygons.

    ``limit`` is pushed into the vector reader, so validation does not load the
    complete 336k-object dataset and then slice it in memory.
    """
    vector_path = Path(vector_path)
    if not vector_path.exists():
        raise FileNotFoundError(f"Object vector not found: {vector_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer or None")

    rows = slice(0, limit) if limit is not None else None
    LOGGER.info("Loading objects: %s%s", vector_path, f" (first {limit})" if limit else "")
    gdf = gpd.read_file(vector_path, rows=rows, engine="pyogrio")
    if gdf.empty:
        raise ValueError("The object vector contains no rows.")
    if gdf.crs is None:
        raise ValueError("The object vector has no CRS; extraction cannot proceed safely.")

    # Resolve case only; do not silently replace the source IDs.
    case_matches = [c for c in gdf.columns if c.casefold() == object_id_field.casefold()]
    if len(case_matches) != 1:
        raise KeyError(
            f"Expected one object-ID field matching {object_id_field!r}; "
            f"available fields are {list(gdf.columns)!r}."
        )
    source_id_field = case_matches[0]
    ids = gdf[source_id_field]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError(f"{source_id_field!r} must be non-null and unique; IDs were not regenerated.")

    was_invalid = ~gdf.geometry.is_valid.fillna(False)
    was_empty = gdf.geometry.isna() | gdf.geometry.is_empty
    needs_repair = was_invalid & ~was_empty
    if needs_repair.any():
        LOGGER.warning("Repairing %d invalid polygon(s) with shapely.make_valid().", needs_repair.sum())
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.map(_repair_geometry)
    unusable = gdf.geometry.isna() | gdf.geometry.is_empty
    if unusable.any():
        LOGGER.warning(
            "%d empty/non-polygon geometry row(s) will be retained with NaN features.",
            unusable.sum(),
        )

    gdf[OUTPUT_ID_FIELD] = ids.to_numpy(copy=True)
    return gdf, source_id_field


def _validated_metric_crs(metric_crs: str | CRS) -> CRS:
    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected:
        raise ValueError(f"Geometry CRS must be projected, not {crs.to_string()}.")
    unit_names = {axis.unit_name.casefold() for axis in crs.axis_info if axis.unit_name}
    if not any(unit in {"metre", "meter"} for unit in unit_names):
        raise ValueError(f"Geometry CRS units must be metres; got {sorted(unit_names)!r}.")
    return crs


def _minimum_rectangle_dimensions(geometry) -> tuple[float, float]:
    """Return long and short side lengths of the oriented minimum rectangle."""
    if geometry is None or geometry.is_empty:
        return math.nan, math.nan
    rectangle = geometry.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return math.nan, math.nan
    coordinates = np.asarray(rectangle.exterior.coords, dtype="float64")
    side_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    positive = side_lengths[side_lengths > np.finfo(float).eps]
    if positive.size < 2:
        return math.nan, math.nan
    return float(positive.max()), float(positive.min())


def _ring_raw_moments(coordinates: np.ndarray) -> np.ndarray:
    """Return signed [area, Ax, Ay, integral(x^2), integral(y^2), integral(xy)]."""
    x0, y0 = coordinates[:-1, 0], coordinates[:-1, 1]
    x1, y1 = coordinates[1:, 0], coordinates[1:, 1]
    cross = x0 * y1 - x1 * y0
    return np.asarray(
        [
            0.5 * np.sum(cross),
            np.sum((x0 + x1) * cross) / 6.0,
            np.sum((y0 + y1) * cross) / 6.0,
            np.sum((x0 * x0 + x0 * x1 + x1 * x1) * cross) / 12.0,
            np.sum((y0 * y0 + y0 * y1 + y1 * y1) * cross) / 12.0,
            np.sum((2*x0*y0 + x0*y1 + x1*y0 + 2*x1*y1) * cross) / 24.0,
        ],
        dtype="float64",
    )


def _polygon_raw_moments(geometry) -> np.ndarray:
    """Integrate first/second moments over polygon area, subtracting holes."""
    total = np.zeros(6, dtype="float64")
    # Center coordinates before integration to prevent catastrophic cancellation
    # when UTM northings (~4e6 m) are squared and the centroid term is removed.
    min_x, min_y, max_x, max_y = geometry.bounds
    origin = np.asarray([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
    for polygon in polygons:
        # CCW exterior and CW holes make signed ring moments additive.
        polygon = orient(polygon, sign=1.0)
        total += _ring_raw_moments(
            np.asarray(polygon.exterior.coords, dtype="float64") - origin
        )
        for interior in polygon.interiors:
            total += _ring_raw_moments(np.asarray(interior.coords, dtype="float64") - origin)
    return total


def _moment_elongation(geometry) -> float:
    """Area-moment eccentricity sqrt(1-lambda_min/lambda_max), in [0, 1]."""
    if geometry is None or geometry.is_empty:
        return math.nan
    area, first_x, first_y, second_x, second_y, product_xy = _polygon_raw_moments(geometry)
    if area <= np.finfo(float).eps:
        return math.nan
    centroid_x, centroid_y = first_x / area, first_y / area
    covariance = np.asarray(
        [
            [second_x / area - centroid_x**2, product_xy / area - centroid_x * centroid_y],
            [product_xy / area - centroid_x * centroid_y, second_y / area - centroid_y**2],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda_min = max(0.0, float(eigenvalues[0]))
    lambda_max = max(0.0, float(eigenvalues[1]))
    if lambda_max <= np.finfo(float).eps:
        return math.nan
    return math.sqrt(max(0.0, min(1.0, 1.0 - lambda_min / lambda_max)))


def extract_geometry_features(
    gdf: gpd.GeoDataFrame,
    metric_crs: str | CRS = METRIC_CRS,
) -> pd.DataFrame:
    """Calculate six metric shape features for each object.

    Mathematical definitions (A=area, P=perimeter, L/W=long/short sides of
    the minimum rotated rectangle, lambda1>=lambda2 are eigenvalues of the
    polygon's area-weighted second central-moment/covariance matrix):

    * area_sqm = A
    * perimeter_m = P
    * compactness = 4*pi*A/P^2                 (circle=1)
    * shape_index = P/(2*sqrt(pi*A))           (circle=1)
    * elongation = sqrt(1-lambda2/lambda1)     (0=isotropic, ->1 linear)
    * length_width_ratio = L/W                 (>=1)

    The moment-based elongation is orientation invariant and deliberately not
    a deterministic rewrite of the minimum-rectangle L/W ratio.
    """
    metric_crs = _validated_metric_crs(metric_crs)
    metric_gdf = gdf.to_crs(metric_crs)
    geometry = metric_gdf.geometry
    valid = geometry.notna() & ~geometry.is_empty

    area = np.full(len(gdf), np.nan, dtype="float64")
    perimeter = np.full(len(gdf), np.nan, dtype="float64")
    area[valid.to_numpy()] = geometry[valid].area.to_numpy()
    perimeter[valid.to_numpy()] = geometry[valid].length.to_numpy()

    compactness = np.divide(
        4.0 * np.pi * area,
        perimeter**2,
        out=np.full_like(area, np.nan),
        where=(area > 0) & (perimeter > 0),
    )
    shape_index = np.divide(
        perimeter,
        2.0 * np.sqrt(np.pi * area),
        out=np.full_like(area, np.nan),
        where=(area > 0) & (perimeter > 0),
    )

    dimensions = [_minimum_rectangle_dimensions(geom) for geom in geometry]
    long_side = np.asarray([item[0] for item in dimensions])
    short_side = np.asarray([item[1] for item in dimensions])
    length_width_ratio = np.divide(
        long_side,
        short_side,
        out=np.full_like(long_side, np.nan),
        where=short_side > 0,
    )
    elongation = np.asarray([_moment_elongation(geom) for geom in geometry])

    return pd.DataFrame(
        {
            OUTPUT_ID_FIELD: gdf[OUTPUT_ID_FIELD].to_numpy(),
            "area_sqm": area,
            "perimeter_m": perimeter,
            "compactness": compactness,
            "shape_index": shape_index,
            "elongation": elongation,
            "length_width_ratio": length_width_ratio,
        }
    )


def _validate_open_raster(
    dataset: rasterio.io.DatasetReader,
    vector_crs,
    feature_name: str,
    allow_geometry_reprojection: bool,
) -> None:
    if dataset.crs is None:
        raise ValueError(f"Raster {dataset.name} has no CRS.")
    if dataset.count != 1:
        raise ValueError(
            f"Raster {dataset.name} has {dataset.count} bands; configure individual single-band inputs."
        )
    if CRS.from_user_input(vector_crs) != CRS.from_user_input(dataset.crs):
        message = (
            f"CRS mismatch for {feature_name}: vector={vector_crs}, raster={dataset.crs}. "
            "No raster will be reprojected or resampled."
        )
        if not allow_geometry_reprojection:
            raise ValueError(message + " Set allow_geometry_reprojection=True to transform vectors in memory.")
        LOGGER.warning("%s Transforming vector geometry in memory.", message)


def _raster_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, object]:
    """Capture alignment-relevant metadata without reading raster pixels."""
    return {
        "path": str(Path(dataset.name).resolve()),
        "crs": CRS.from_user_input(dataset.crs).to_string() if dataset.crs else None,
        "dimensions": (dataset.height, dataset.width),
        "resolution": tuple(float(value) for value in dataset.res),
        "extent": tuple(float(value) for value in dataset.bounds),
        "transform": tuple(float(value) for value in dataset.transform),
        "nodata": dataset.nodata,
        "dtype": dataset.dtypes[0],
    }


def _alignment_diagnostics(
    metadata: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, bool]:
    """Compare an optical raster grid with the configured project reference."""
    same_crs = metadata["crs"] == reference["crs"]
    same_dimensions = metadata["dimensions"] == reference["dimensions"]
    same_resolution = bool(
        np.allclose(metadata["resolution"], reference["resolution"], rtol=0.0, atol=1e-12)
    )
    same_extent = bool(
        np.allclose(metadata["extent"], reference["extent"], rtol=0.0, atol=1e-9)
    )
    same_transform = bool(
        np.allclose(metadata["transform"], reference["transform"], rtol=0.0, atol=1e-12)
    )
    return {
        "same_crs_as_reference": same_crs,
        "same_dimensions_as_reference": same_dimensions,
        "same_resolution_as_reference": same_resolution,
        "same_extent_as_reference": same_extent,
        "same_transform_as_reference": same_transform,
        "same_grid_as_reference": all(
            (same_crs, same_dimensions, same_resolution, same_extent, same_transform)
        ),
    }


def _validate_feature_names(names: Sequence[str], family: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate raster names in {family}: {names!r}")
    invalid = [name for name in names if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None]
    if invalid:
        raise ValueError(
            f"{family} names must start with an ASCII letter and contain only letters, "
            f"digits, and underscores; invalid names: {invalid!r}"
        )


def _resolve_optical_inputs(
    optical_bands: Mapping[str, os.PathLike[str] | str],
    optical_indices: Mapping[str, os.PathLike[str] | str],
) -> tuple[dict[str, Path], dict[str, str]]:
    """Combine bands then indices while retaining category and deterministic order."""
    band_names = list(optical_bands)
    index_names = list(optical_indices)
    _validate_feature_names(band_names, "OPTICAL_BANDS")
    _validate_feature_names(index_names, "OPTICAL_INDICES")
    overlap = set(band_names).intersection(index_names)
    if overlap:
        raise ValueError(f"Optical band/index names must be globally unique: {sorted(overlap)!r}")

    combined: dict[str, Path] = {}
    categories: dict[str, str] = {}
    for category, inputs in (("band", optical_bands), ("index", optical_indices)):
        for name, raw_path in inputs.items():
            path = Path(raw_path)
            combined[name] = path if path.is_absolute() else OPTICAL_DATA_DIR / path
            categories[name] = category
    return combined, categories


def extract_optical_features(
    gdf: gpd.GeoDataFrame,
    optical_bands: Mapping[str, os.PathLike[str] | str] = OPTICAL_BANDS,
    optical_indices: Mapping[str, os.PathLike[str] | str] = OPTICAL_INDICES,
    *,
    alignment_reference_raster: os.PathLike[str] | str = OPTICAL_ALIGNMENT_REFERENCE,
    allow_geometry_reprojection: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract exact mean, population stdev, and maximum per optical input.

    exactextract uses fractional cell coverage as the weight for boundary cells
    and excludes declared NoData/NaN cells. No full-raster mask and no per-object
    raster read is performed.
    """
    inputs, categories = _resolve_optical_inputs(optical_bands, optical_indices)
    output = pd.DataFrame({OUTPUT_ID_FIELD: gdf[OUTPUT_ID_FIELD].to_numpy()})
    diagnostics: dict[str, object] = {
        "band_count": len(optical_bands),
        "index_count": len(optical_indices),
        "inputs": {},
        "object_statistics": {},
        "skipped": not inputs,
    }
    if not inputs:
        LOGGER.info("Optical extraction skipped: OPTICAL_BANDS and OPTICAL_INDICES are empty.")
        return output, diagnostics

    conflicts = set(inputs).intersection(RASTER_INPUTS)
    if conflicts:
        raise ValueError(f"Optical names conflict with existing SDGSAT names: {sorted(conflicts)!r}")
    usable = gdf.geometry.notna() & ~gdf.geometry.is_empty
    row_positions = np.flatnonzero(usable.to_numpy())
    for feature_name in inputs:
        for _, _, output_suffix in OPTICAL_STATISTICS:
            output[f"{feature_name}_{output_suffix}"] = np.nan

    reference_path = Path(alignment_reference_raster)
    if not reference_path.exists():
        raise FileNotFoundError(f"Optical alignment reference not found: {reference_path}")
    with rasterio.open(reference_path) as reference_dataset:
        reference_metadata = _raster_metadata(reference_dataset)
    diagnostics["alignment_reference"] = reference_metadata

    with ExitStack() as stack:
        # Quantiles are grouped by the exact native grid. Mixing different
        # extents/transforms in one processor can introduce tiny interpolation
        # differences even when grids are mutually compatible.
        groups: dict[
            tuple[str, tuple[float, ...], tuple[int, int]],
            list[tuple[str, RasterioRasterSource]],
        ] = defaultdict(list)
        for feature_name, raster_path in inputs.items():
            if not raster_path.exists():
                raise FileNotFoundError(f"Optical raster for {feature_name!r} not found: {raster_path}")
            dataset = stack.enter_context(rasterio.open(raster_path))
            _validate_open_raster(
                dataset,
                gdf.crs,
                feature_name,
                allow_geometry_reprojection=allow_geometry_reprojection,
            )
            metadata = _raster_metadata(dataset)
            alignment = _alignment_diagnostics(metadata, reference_metadata)
            metadata.update(alignment)
            metadata["category"] = categories[feature_name]
            metadata["crs_matches_objects"] = (
                CRS.from_user_input(dataset.crs) == CRS.from_user_input(gdf.crs)
            )
            diagnostics["inputs"][feature_name] = metadata
            if not alignment["same_grid_as_reference"]:
                LOGGER.warning(
                    "Optical raster %s is not on the reference grid. Independent object "
                    "statistics remain valid because CRS matches; no alignment was performed.",
                    feature_name,
                )
            grid_key = (
                CRS.from_user_input(dataset.crs).to_wkt(),
                tuple(float(value) for value in dataset.transform),
                (dataset.height, dataset.width),
            )
            groups[grid_key].append(
                (feature_name, RasterioRasterSource(dataset, band_idx=1, name=feature_name))
            )

        operations = [operation for operation, _, _ in OPTICAL_STATISTICS]
        for grid_key, group in groups.items():
            extraction_gdf = gdf.loc[usable, [OUTPUT_ID_FIELD, "geometry"]]
            raster_crs = CRS.from_wkt(grid_key[0])
            if CRS.from_user_input(gdf.crs) != raster_crs:
                extraction_gdf = extraction_gdf.to_crs(raster_crs)
            names = [item[0] for item in group]
            sources = [item[1] for item in group]
            LOGGER.info("Batched optical statistics (%d raster(s)): %s", len(group), ", ".join(names))
            category_set = {categories[name] for name in names}
            family = next(iter(category_set)) if len(category_set) == 1 else "inputs"
            progress = TerminalProgress(
                f"Optical {family} ({len(group)} rasters)",
                enabled=show_progress,
            )
            stats = exactextract.exact_extract(
                sources,
                extraction_gdf,
                operations,
                strategy=EXACTEXTRACT_STRATEGY,
                max_cells_in_memory=EXACTEXTRACT_MAX_CELLS_IN_MEMORY,
                output="pandas",
                progress=progress.exactextract_callback if show_progress else False,
            )
            progress.complete()
            for feature_name in names:
                for _, source_suffix, output_suffix in OPTICAL_STATISTICS:
                    source_column = (
                        source_suffix if len(group) == 1 else f"{feature_name}_{source_suffix}"
                    )
                    if source_column not in stats:
                        raise RuntimeError(
                            f"Unexpected exactextract columns for {feature_name}: {list(stats.columns)!r}"
                        )
                    values = stats[source_column].to_numpy(dtype="float64", copy=True)
                    if np.isinf(values).any():
                        raise ValueError(
                            f"Infinite values contaminated {feature_name}_{output_suffix}; "
                            "encode invalid raster cells as NoData before extraction."
                        )
                    target_column = f"{feature_name}_{output_suffix}"
                    output.iloc[row_positions, output.columns.get_loc(target_column)] = values

    for feature_name in inputs:
        columns = [f"{feature_name}_{item[2]}" for item in OPTICAL_STATISTICS]
        values = output[columns]
        no_valid = values.isna().all(axis=1)
        diagnostics["object_statistics"][feature_name] = {
            "valid_objects": int((~no_valid).sum()),
            "objects_with_no_valid_pixels": int(no_valid.sum()),
            "nan_values": int(values.isna().sum().sum()),
        }
    return output, diagnostics


def extract_spectral_features(
    gdf: gpd.GeoDataFrame,
    raster_inputs: Mapping[str, os.PathLike[str] | str],
    allow_geometry_reprojection: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract coverage-weighted mean, population stdev, and max per object.

    exactextract processes all objects and all same-CRS rasters in one compiled
    batch, avoids full-image masks, respects raster NoData, and each configured
    raster dataset is opened exactly once here.
    """
    output = pd.DataFrame({OUTPUT_ID_FIELD: gdf[OUTPUT_ID_FIELD].to_numpy()})
    usable = gdf.geometry.notna() & ~gdf.geometry.is_empty

    for feature_name in raster_inputs:
        for suffix in ("mean", "std", "max"):
            output[f"{feature_name}_{suffix}"] = np.nan

    row_positions = np.flatnonzero(usable.to_numpy())
    with ExitStack() as stack:
        groups: dict[str, list[tuple[str, RasterioRasterSource]]] = defaultdict(list)
        for feature_name, raster_path in raster_inputs.items():
            raster_path = Path(raster_path)
            if not raster_path.exists():
                raise FileNotFoundError(f"Raster for {feature_name!r} not found: {raster_path}")
            dataset = stack.enter_context(rasterio.open(raster_path))
            _validate_open_raster(
                dataset,
                gdf.crs,
                feature_name,
                allow_geometry_reprojection=allow_geometry_reprojection,
            )
            crs_key = CRS.from_user_input(dataset.crs).to_wkt()
            groups[crs_key].append(
                (feature_name, RasterioRasterSource(dataset, band_idx=1, name=feature_name))
            )

        for crs_wkt, group in groups.items():
            extraction_gdf = gdf.loc[usable, [OUTPUT_ID_FIELD, "geometry"]]
            raster_crs = CRS.from_wkt(crs_wkt)
            if CRS.from_user_input(gdf.crs) != raster_crs:
                extraction_gdf = extraction_gdf.to_crs(raster_crs)
            names = [item[0] for item in group]
            sources = [item[1] for item in group]
            LOGGER.info("Batched spectral statistics (%d raster(s)): %s", len(group), ", ".join(names))
            progress = TerminalProgress(
                f"Nightlight/SDGSAT ({len(group)} rasters)",
                enabled=show_progress,
            )
            stats = exactextract.exact_extract(
                sources,
                extraction_gdf,
                ["mean", "stdev", "max"],
                strategy=EXACTEXTRACT_STRATEGY,
                max_cells_in_memory=EXACTEXTRACT_MAX_CELLS_IN_MEMORY,
                output="pandas",
                progress=progress.exactextract_callback if show_progress else False,
            )
            progress.complete()
            for feature_name in names:
                for source_suffix, output_suffix in (("mean", "mean"), ("stdev", "std"), ("max", "max")):
                    source_column = (
                        source_suffix if len(group) == 1 else f"{feature_name}_{source_suffix}"
                    )
                    if source_column not in stats:
                        raise RuntimeError(
                            f"Unexpected exactextract columns for {feature_name}: {list(stats.columns)!r}"
                        )
                    target_column = f"{feature_name}_{output_suffix}"
                    output.iloc[row_positions, output.columns.get_loc(target_column)] = stats[
                        source_column
                    ].to_numpy()

    return output


def _estimate_quantization_range(
    dataset: rasterio.io.DatasetReader,
    config: GLCMConfig,
) -> tuple[float, float]:
    """Estimate a deterministic raster-wide robust range using <=N pixels."""
    aspect = dataset.width / dataset.height
    sample_height = max(1, min(dataset.height, int(math.sqrt(config.quantization_sample_size / aspect))))
    sample_width = max(1, min(dataset.width, int(sample_height * aspect)))
    sampled = dataset.read(1, out_shape=(sample_height, sample_width), masked=True)
    values = sampled.compressed().astype("float64", copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Texture raster contains no finite valid samples: {dataset.name}")
    low, high = np.percentile(values, config.quantile_range)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("Texture raster has no usable radiometric variation for quantization.")
    return float(low), float(high)


def _bounded_window(geometry, dataset: rasterio.io.DatasetReader) -> Window | None:
    raster_bounds = dataset.bounds
    left, bottom, right, top = geometry.bounds
    if right <= raster_bounds.left or left >= raster_bounds.right:
        return None
    if top <= raster_bounds.bottom or bottom >= raster_bounds.top:
        return None
    left = max(left, raster_bounds.left)
    bottom = max(bottom, raster_bounds.bottom)
    right = min(right, raster_bounds.right)
    top = min(top, raster_bounds.top)
    window = from_bounds(left, bottom, right, top, transform=dataset.transform)
    col_start = max(0, math.floor(window.col_off))
    row_start = max(0, math.floor(window.row_off))
    col_stop = min(dataset.width, math.ceil(window.col_off + window.width))
    row_stop = min(dataset.height, math.ceil(window.row_off + window.height))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def _quantize(values: np.ndarray, valid: np.ndarray, levels: int, low: float, high: float) -> np.ndarray:
    quantized = np.zeros(values.shape, dtype=np.uint16)
    scaled = (np.clip(values[valid], low, high) - low) / (high - low)
    quantized[valid] = np.minimum((scaled * levels).astype(np.int64), levels - 1)
    return quantized


def _paired_slices(size: int, delta: int) -> tuple[slice, slice]:
    if delta >= 0:
        return slice(0, size - delta), slice(delta, size)
    return slice(-delta, size), slice(0, size + delta)


def _glcm_for_offset(
    quantized: np.ndarray,
    valid: np.ndarray,
    levels: int,
    row_delta: int,
    col_delta: int,
) -> tuple[np.ndarray | None, int]:
    if abs(row_delta) >= quantized.shape[0] or abs(col_delta) >= quantized.shape[1]:
        return None, 0
    row_source, row_target = _paired_slices(quantized.shape[0], row_delta)
    col_source, col_target = _paired_slices(quantized.shape[1], col_delta)
    source_valid = valid[row_source, col_source]
    target_valid = valid[row_target, col_target]
    pair_mask = source_valid & target_valid
    pair_count = int(pair_mask.sum())
    if pair_count == 0:
        return None, 0

    source = quantized[row_source, col_source][pair_mask].astype(np.int64)
    target = quantized[row_target, col_target][pair_mask].astype(np.int64)
    flat_index = source * levels + target
    counts = np.bincount(flat_index, minlength=levels * levels).reshape(levels, levels)
    # Symmetric GLCM: an unordered neighbouring pair contributes both i,j and j,i.
    matrix = counts.astype("float64") + counts.T
    matrix /= matrix.sum()
    return matrix, pair_count


def _glcm_properties(matrix: np.ndarray) -> dict[str, float]:
    """Return five documented Haralick-style properties of normalized P(i,j)."""
    levels = matrix.shape[0]
    i, j = np.indices((levels, levels), dtype="float64")
    difference_squared = (i - j) ** 2
    contrast = float(np.sum(matrix * difference_squared))
    homogeneity = float(np.sum(matrix / (1.0 + difference_squared)))
    # Energy follows skimage.graycoprops: sqrt(ASM) = sqrt(sum(P^2)).
    energy = float(np.sqrt(np.sum(matrix**2)))

    mean_i = float(np.sum(i * matrix))
    mean_j = float(np.sum(j * matrix))
    sigma_i = math.sqrt(float(np.sum(((i - mean_i) ** 2) * matrix)))
    sigma_j = math.sqrt(float(np.sum(((j - mean_j) ** 2) * matrix)))
    denominator = sigma_i * sigma_j
    correlation = (
        float(np.sum((i - mean_i) * (j - mean_j) * matrix) / denominator)
        if denominator > np.finfo(float).eps
        else 1.0
    )
    positive = matrix[matrix > 0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "entropy": entropy,
    }


def _object_glcm(
    values: np.ndarray,
    valid: np.ndarray,
    config: GLCMConfig,
    low: float,
    high: float,
) -> tuple[dict[str, float] | None, int]:
    if int(valid.sum()) < config.min_valid_pixels:
        return None, 0
    quantized = _quantize(values, valid, config.levels, low, high)
    properties: list[dict[str, float]] = []
    total_pairs = 0
    for distance in config.distances:
        if distance < 1:
            raise ValueError("All GLCM distances must be positive integers.")
        for angle in config.angles_degrees:
            radians = math.radians(angle)
            row_delta = int(round(-distance * math.sin(radians)))
            col_delta = int(round(distance * math.cos(radians)))
            matrix, pair_count = _glcm_for_offset(
                quantized, valid, config.levels, row_delta, col_delta
            )
            if matrix is not None:
                properties.append(_glcm_properties(matrix))
                total_pairs += pair_count
    if not properties:
        return None, 0
    # Each direction/distance receives equal weight, reducing orientation bias.
    return {
        key: float(np.mean([item[key] for item in properties]))
        for key in properties[0]
    }, total_pairs


def extract_texture_features(
    gdf: gpd.GeoDataFrame,
    raster_path: os.PathLike[str] | str,
    raster_name: str = TEXTURE_RASTER_NAME,
    config: GLCMConfig = GLCM,
    allow_geometry_reprojection: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Extract masked object GLCMs with one open raster and local windows only.

    Only neighbour pairs for which both pixels are inside the same polygon and
    both pixels are finite/non-NoData contribute to P(i,j).  No full-raster
    array or full-raster polygon mask is created.
    """
    if config.levels < 2 or config.levels > np.iinfo(np.uint16).max:
        raise ValueError("GLCM levels must be between 2 and 65535.")
    feature_names = ["contrast", "homogeneity", "energy", "correlation", "entropy"]
    result = pd.DataFrame({OUTPUT_ID_FIELD: gdf[OUTPUT_ID_FIELD].to_numpy()})
    for name in feature_names:
        result[f"{raster_name}_glcm_{name}"] = np.nan

    small_objects = outside_objects = no_pair_objects = 0
    with rasterio.open(raster_path) as dataset:
        _validate_open_raster(dataset, gdf.crs, raster_name, allow_geometry_reprojection)
        texture_gdf = gdf
        if CRS.from_user_input(gdf.crs) != CRS.from_user_input(dataset.crs):
            texture_gdf = gdf.to_crs(dataset.crs)
        low, high = _estimate_quantization_range(dataset, config)
        LOGGER.info(
            "GLCM %s: levels=%d, distances=%s, angles=%s, quantization=[%.8g, %.8g]",
            raster_name,
            config.levels,
            config.distances,
            config.angles_degrees,
            low,
            high,
        )
        progress = TerminalProgress(
            f"GLCM ({len(texture_gdf):,} objects)",
            enabled=show_progress,
        )
        progress.update(0.0, force=True)

        for position, geometry in enumerate(texture_gdf.geometry):
            progress.update(
                position / len(texture_gdf),
            )
            if geometry is None or geometry.is_empty:
                outside_objects += 1
                continue
            window = _bounded_window(geometry, dataset)
            if window is None:
                outside_objects += 1
                continue
            values = dataset.read(1, window=window, masked=True)
            inside = geometry_mask(
                [mapping(geometry)],
                out_shape=values.shape,
                transform=dataset.window_transform(window),
                invert=True,
                all_touched=config.all_touched,
            )
            data = np.asarray(values.data)
            valid = inside & ~np.ma.getmaskarray(values) & np.isfinite(data)
            if int(valid.sum()) < config.min_valid_pixels:
                small_objects += 1
                continue
            properties, _ = _object_glcm(data, valid, config, low, high)
            if properties is None:
                no_pair_objects += 1
                continue
            for name, value in properties.items():
                result.iat[position, result.columns.get_loc(f"{raster_name}_glcm_{name}")] = value

        progress.complete()

    diagnostics: dict[str, float | int] = {
        "quantization_low": low,
        "quantization_high": high,
        "small_objects": small_objects,
        "outside_or_empty_objects": outside_objects,
        "objects_without_valid_pairs": no_pair_objects,
    }
    if small_objects or outside_objects or no_pair_objects:
        LOGGER.warning("Texture NaN diagnostics: %s", diagnostics)
    return result, diagnostics


def _same_object_order(*frames: pd.DataFrame) -> bool:
    if not frames:
        return True
    reference = frames[0][OUTPUT_ID_FIELD].to_numpy()
    return all(np.array_equal(reference, frame[OUTPUT_ID_FIELD].to_numpy()) for frame in frames[1:])


def _write_vector_output(
    gdf: gpd.GeoDataFrame,
    features: pd.DataFrame,
    output_path: os.PathLike[str] | str,
) -> None:
    output_path = Path(output_path)
    if output_path.suffix.casefold() not in {".gpkg", ".shp"}:
        raise ValueError("Vector output must use .gpkg or .shp.")
    if output_path.suffix.casefold() == ".shp":
        LOGGER.warning("Shapefile truncates field names; GeoPackage is strongly recommended.")
    enriched = gdf.copy()
    for column in features.columns:
        if column != OUTPUT_ID_FIELD:
            enriched[column] = features[column].to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_file(output_path, driver="GPKG" if output_path.suffix.casefold() == ".gpkg" else None)


def run_feature_extraction_pipeline(
    vector_path: os.PathLike[str] | str,
    raster_inputs: Mapping[str, os.PathLike[str] | str],
    output_csv_path: os.PathLike[str] | str,
    *,
    object_id_field: str = OBJECT_ID_FIELD,
    metric_crs: str | CRS = METRIC_CRS,
    texture_raster_name: str = TEXTURE_RASTER_NAME,
    glcm_config: GLCMConfig = GLCM,
    optical_bands: Mapping[str, os.PathLike[str] | str] = OPTICAL_BANDS,
    optical_indices: Mapping[str, os.PathLike[str] | str] = OPTICAL_INDICES,
    optical_alignment_reference: os.PathLike[str] | str = OPTICAL_ALIGNMENT_REFERENCE,
    limit: int | None = None,
    vector_output_path: os.PathLike[str] | str | None = None,
    allow_geometry_reprojection: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the complete one-row-per-object extraction pipeline."""
    started = perf_counter()
    stage_timings: dict[str, float] = {}
    if texture_raster_name not in raster_inputs:
        raise KeyError(f"Texture raster {texture_raster_name!r} is not in RASTER_INPUTS.")

    LOGGER.info("Stage 1/6: preparing input vector")

    stage_started = perf_counter()
    gdf, source_id_field = verify_and_prepare_objects(vector_path, object_id_field, limit)
    stage_timings["input_vector_preparation"] = perf_counter() - stage_started
    LOGGER.info("Stage 2/6: extracting nightlight/SDGSAT features")

    stage_started = perf_counter()
    spectral_features = extract_spectral_features(
        gdf,
        raster_inputs,
        allow_geometry_reprojection=allow_geometry_reprojection,
        show_progress=show_progress,
    )
    stage_timings["sdgsat_spectral_extraction"] = perf_counter() - stage_started
    LOGGER.info("Stage 3/6: extracting optical features")

    stage_started = perf_counter()
    optical_features, optical_diagnostics = extract_optical_features(
        gdf,
        optical_bands,
        optical_indices,
        alignment_reference_raster=optical_alignment_reference,
        allow_geometry_reprojection=allow_geometry_reprojection,
        show_progress=show_progress,
    )
    stage_timings["optical_spectral_extraction"] = perf_counter() - stage_started
    LOGGER.info("Stage 4/6: extracting geometry features")

    stage_started = perf_counter()
    geometry_features = extract_geometry_features(gdf, metric_crs)
    stage_timings["geometry_extraction"] = perf_counter() - stage_started
    LOGGER.info("Stage 5/6: extracting GLCM texture features")

    stage_started = perf_counter()
    texture_features, texture_diagnostics = extract_texture_features(
        gdf,
        raster_inputs[texture_raster_name],
        raster_name=texture_raster_name,
        config=glcm_config,
        allow_geometry_reprojection=allow_geometry_reprojection,
        show_progress=show_progress,
    )
    stage_timings["texture_extraction"] = perf_counter() - stage_started
    if not _same_object_order(
        spectral_features, optical_features, geometry_features, texture_features
    ):
        raise RuntimeError("Object order changed during extraction; refusing an unsafe join.")

    final = pd.concat(
        [
            spectral_features,
            optical_features.drop(columns=OUTPUT_ID_FIELD),
            geometry_features.drop(columns=OUTPUT_ID_FIELD),
            texture_features.drop(columns=OUTPUT_ID_FIELD),
        ],
        axis=1,
    )
    if len(final) != len(gdf) or not final[OUTPUT_ID_FIELD].is_unique:
        raise RuntimeError("Output failed the one-row-per-unique-object invariant.")

    stage_started = perf_counter()
    LOGGER.info("Stage 6/6: writing output")
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv_path, index=False)
    if vector_output_path is not None:
        _write_vector_output(gdf, final, vector_output_path)
    stage_timings["output_writing"] = perf_counter() - stage_started

    sdgsat_feature_count = len(raster_inputs) * len(SDGSAT_STATISTICS)
    optical_band_feature_count = len(optical_bands) * len(OPTICAL_STATISTICS)
    optical_index_feature_count = len(optical_indices) * len(OPTICAL_STATISTICS)
    geometry_feature_count = 6
    texture_feature_count = 5

    metadata: dict[str, object] = {
        "source_id_field": source_id_field,
        "object_count": len(final),
        "feature_count": len(final.columns) - 1,
        "metric_crs": CRS.from_user_input(metric_crs).to_string(),
        "feature_counts": {
            "sdgsat_spectral": sdgsat_feature_count,
            "optical_bands": optical_band_feature_count,
            "optical_indices": optical_index_feature_count,
            "geometry": geometry_feature_count,
            "texture": texture_feature_count,
            "total_features": len(final.columns) - 1,
            "total_columns_including_object_id": len(final.columns),
        },
        "optical": optical_diagnostics,
        "texture_raster": str(raster_inputs[texture_raster_name]),
        "glcm": {
            "levels": glcm_config.levels,
            "distances": glcm_config.distances,
            "angles_degrees": glcm_config.angles_degrees,
            "min_valid_pixels": glcm_config.min_valid_pixels,
            **texture_diagnostics,
        },
        "stage_timings_seconds": stage_timings,
        "elapsed_seconds": perf_counter() - started,
        "columns": final.columns.tolist(),
    }
    LOGGER.info(
        "Feature counts: SDGSAT=%d, optical bands=%d, optical indices=%d, "
        "geometry=%d, texture=%d, total=%d (%d columns with Object_ID)",
        sdgsat_feature_count,
        optical_band_feature_count,
        optical_index_feature_count,
        geometry_feature_count,
        texture_feature_count,
        len(final.columns) - 1,
        len(final.columns),
    )
    LOGGER.info("Stage timings (seconds): %s", stage_timings)
    if optical_diagnostics["skipped"]:
        LOGGER.info("Optical diagnostics: 0 bands, 0 indices; extraction skipped cleanly.")
    else:
        for name, raster_diagnostics in optical_diagnostics["inputs"].items():
            LOGGER.info("Optical input %s: %s", name, raster_diagnostics)
            LOGGER.info(
                "Optical object statistics %s: %s",
                name,
                optical_diagnostics["object_statistics"][name],
            )
    LOGGER.info(
        "Completed %d objects, %d features in %.2f s -> %s",
        len(final),
        len(final.columns) - 1,
        metadata["elapsed_seconds"],
        output_csv_path,
    )
    return final, metadata


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Read only the first N objects (safe validation default: 100).",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Explicitly process every object; overrides --limit.",
    )
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--vector-output", type=Path, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    limit = None if args.full_dataset else args.limit
    _, metadata = run_feature_extraction_pipeline(
        VECTOR_PATH,
        RASTER_INPUTS,
        args.output_csv,
        limit=limit,
        vector_output_path=args.vector_output,
        show_progress=not args.no_progress,
    )
    LOGGER.info("Run metadata: %s", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
