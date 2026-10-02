"""NTL-driven object-level spatial-context extraction using first-order Rook adjacency.

This stage reads the completed object feature table and appends exactly six
contextual features. Existing spectral, optical, geometry, and texture features
are preserved and are not recomputed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.stats import norm
from shapely import make_valid
from shapely.ops import unary_union


LOGGER = logging.getLogger("spatial_context_extraction")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"
SEGMENTATION_PATH = SEGMENT_DIR / "eCog_segments.shp"
FEATURE_TABLE_PATH = SEGMENT_DIR / "objects_features_v04_full.csv"
OUTPUT_TABLE_PATH = SEGMENT_DIR / "objects_features_v04_spatial_context.csv"
DIAGNOSTICS_PATH = SEGMENT_DIR / "Spatial_Context_Extraction_01_Diagnostics.json"

POLYGON_ID_FIELD = "Seg_id"
TABLE_ID_FIELD = "Object_ID"
PAN_MEAN_FIELD = "SDGSAT_Pan_mean"
CONTEXT_COLUMNS = [
    "NTL_NeighborMean",
    "NTL_NeighborStd",
    "NTL_LocalContrast",
    "NTL_NormalizedContrast",
    "NTL_MeanAbsNeighborContrast",
    "NeighborCount",
]

# Epsilon is scaled to the observed radiance magnitude. At 1e-12 of the largest
# absolute PAN mean it is negligible for nonzero denominators and only prevents 0/0.
EPSILON_SCALE = 1e-12
GRAPH_CHUNK_SIZE = 10_000


class Progress:
    """Short single-line progress display that does not wrap ordinary terminals."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(1, total)
        self.started = perf_counter()
        self.last = 0.0

    @staticmethod
    def _time(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update(self, completed: int, *, force: bool = False) -> None:
        now = perf_counter()
        if not force and completed < self.total and now - self.last < 0.5:
            return
        fraction = min(1.0, completed / self.total)
        elapsed = now - self.started
        eta = elapsed * (1 - fraction) / fraction if fraction else math.nan
        eta_text = self._time(eta) if np.isfinite(eta) else "--:--"
        width = 18
        filled = int(round(width * fraction))
        line = (
            f"\r\033[2K[{'#' * filled}{'-' * (width - filled)}] "
            f"{100 * fraction:5.1f}% {self.label} | {self._time(elapsed)} | ETA {eta_text}"
        )
        print(line, end="\n" if completed >= self.total else "", file=sys.stderr, flush=True)
        self.last = now


def _polygonal_part(geometry):
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


def load_and_validate_inputs(
    segmentation_path: Path, feature_table_path: Path
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, object]]:
    for path in (segmentation_path, feature_table_path):
        if not path.exists():
            raise FileNotFoundError(path)

    header = pd.read_csv(feature_table_path, nrows=0)
    required_table = {TABLE_ID_FIELD, PAN_MEAN_FIELD}
    missing = required_table.difference(header.columns)
    if missing:
        raise KeyError(f"Missing required feature-table columns: {sorted(missing)}")
    existing_context = set(CONTEXT_COLUMNS).intersection(header.columns)
    if existing_context:
        raise ValueError(
            f"Input already contains spatial-context columns: {sorted(existing_context)}"
        )

    LOGGER.info("Reading segmentation geometry and %s", POLYGON_ID_FIELD)
    polygons = gpd.read_file(segmentation_path, columns=[POLYGON_ID_FIELD])
    if POLYGON_ID_FIELD not in polygons:
        raise KeyError(f"Polygon ID field not found: {POLYGON_ID_FIELD}")
    if polygons.crs is None:
        raise ValueError("Segmentation polygons have no CRS metadata.")
    if polygons[POLYGON_ID_FIELD].isna().any() or not polygons[POLYGON_ID_FIELD].is_unique:
        raise ValueError(f"{POLYGON_ID_FIELD} must be non-null and unique.")

    invalid = ~polygons.geometry.is_valid
    invalid_count = int(invalid.sum())
    if invalid_count:
        LOGGER.warning("Repairing %d invalid polygon(s) with shapely.make_valid().", invalid_count)
        repaired = [
            _polygonal_part(make_valid(geometry))
            for geometry in polygons.loc[invalid, "geometry"]
        ]
        polygons.loc[invalid, "geometry"] = repaired
    unusable = polygons.geometry.isna() | polygons.geometry.is_empty | ~polygons.geometry.is_valid
    if unusable.any():
        raise ValueError(f"{int(unusable.sum())} polygons remain empty/invalid after repair.")

    LOGGER.info("Reading completed feature table (existing features are not recomputed)")
    features = pd.read_csv(feature_table_path)
    if features[TABLE_ID_FIELD].isna().any() or not features[TABLE_ID_FIELD].is_unique:
        raise ValueError(f"{TABLE_ID_FIELD} must be non-null and unique.")
    if len(features) != len(polygons):
        raise ValueError(
            f"Row mismatch: {len(polygons):,} polygons vs {len(features):,} feature rows."
        )

    polygon_ids = pd.Index(polygons[POLYGON_ID_FIELD])
    table_ids = pd.Index(features[TABLE_ID_FIELD])
    if not polygon_ids.isin(table_ids).all() or not table_ids.isin(polygon_ids).all():
        raise ValueError("Seg_id and Object_ID do not contain exactly the same identifiers.")

    return polygons, features, {
        "segmentation_path": str(segmentation_path.resolve()),
        "feature_table_path": str(feature_table_path.resolve()),
        "polygon_id_field": POLYGON_ID_FIELD,
        "table_id_field": TABLE_ID_FIELD,
        "pan_mean_field": PAN_MEAN_FIELD,
        "object_count": len(features),
        "original_column_count": len(features.columns),
        "original_feature_count": len(features.columns) - 1,
        "invalid_geometries_repaired": invalid_count,
        "crs": str(polygons.crs),
    }


def build_rook_graph(
    geometries: np.ndarray, chunk_size: int = GRAPH_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build unique undirected strict-Rook edges using STRtree and DE-9IM.

    The accepted relation is F***1****: polygon interiors are disjoint and the
    boundary-boundary intersection has dimension 1. Boundary dimension 0 is a
    point-only contact and is explicitly excluded.
    """
    tree = shapely.STRtree(geometries)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    point_contacts = 0
    overlap_pairs = 0
    candidate_pairs = 0
    progress = Progress("Rook graph", len(geometries))

    for start in range(0, len(geometries), chunk_size):
        stop = min(start + chunk_size, len(geometries))
        pairs = tree.query(geometries[start:stop], predicate="intersects")
        left = pairs[0].astype(np.int64, copy=False) + start
        right = pairs[1].astype(np.int64, copy=False)
        unique_direction = left < right
        left = left[unique_direction]
        right = right[unique_direction]
        candidate_pairs += len(left)

        if len(left):
            relations = shapely.relate(geometries[left], geometries[right])
            rook = np.fromiter(
                (relation[0] == "F" and relation[4] == "1" for relation in relations),
                dtype=bool,
                count=len(relations),
            )
            point_contacts += sum(
                relation[0] == "F" and relation[4] == "0" for relation in relations
            )
            overlap_pairs += sum(relation[0] != "F" for relation in relations)
            if rook.any():
                left_parts.append(left[rook])
                right_parts.append(right[rook])
        progress.update(stop)

    edge_left = np.concatenate(left_parts) if left_parts else np.empty(0, dtype=np.int64)
    edge_right = np.concatenate(right_parts) if right_parts else np.empty(0, dtype=np.int64)
    if np.any(edge_left >= edge_right):
        raise RuntimeError("Graph edge canonicalization failed.")
    if len(edge_left):
        packed = edge_left * np.int64(len(geometries)) + edge_right
        if len(np.unique(packed)) != len(packed):
            raise RuntimeError("Duplicate undirected graph edges were generated.")

    return edge_left, edge_right, {
        "candidate_intersecting_pairs": candidate_pairs,
        "point_only_contacts_excluded": int(point_contacts),
        "overlapping_polygon_pairs_excluded": int(overlap_pairs),
    }


def graph_degrees(n: int, edge_left: np.ndarray, edge_right: np.ndarray) -> np.ndarray:
    return np.bincount(
        np.concatenate((edge_left, edge_right)), minlength=n
    ).astype(np.int64, copy=False)


def calculate_context(
    radiance: np.ndarray, edge_left: np.ndarray, edge_right: np.ndarray
) -> tuple[pd.DataFrame, float, dict[str, int]]:
    n = len(radiance)
    finite = np.isfinite(radiance)
    degree = graph_degrees(n, edge_left, edge_right)
    valid_edge = finite[edge_left] & finite[edge_right]
    left = edge_left[valid_edge]
    right = edge_right[valid_edge]

    valid_neighbor_count = np.bincount(
        np.concatenate((left, right)), minlength=n
    ).astype(np.int64, copy=False)
    neighbor_sum = np.bincount(left, weights=radiance[right], minlength=n)
    neighbor_sum += np.bincount(right, weights=radiance[left], minlength=n)
    neighbor_sum_sq = np.bincount(left, weights=radiance[right] ** 2, minlength=n)
    neighbor_sum_sq += np.bincount(right, weights=radiance[left] ** 2, minlength=n)
    abs_difference = np.abs(radiance[left] - radiance[right])
    neighbor_abs_difference = np.bincount(left, weights=abs_difference, minlength=n)
    neighbor_abs_difference += np.bincount(right, weights=abs_difference, minlength=n)

    has_radiometric_neighbors = finite & (valid_neighbor_count > 0)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    mean_abs = np.full(n, np.nan)
    mean[has_radiometric_neighbors] = (
        neighbor_sum[has_radiometric_neighbors] / valid_neighbor_count[has_radiometric_neighbors]
    )
    variance = np.full(n, np.nan)
    variance[has_radiometric_neighbors] = (
        neighbor_sum_sq[has_radiometric_neighbors]
        / valid_neighbor_count[has_radiometric_neighbors]
        - mean[has_radiometric_neighbors] ** 2
    )
    # Roundoff can make a theoretically zero variance slightly negative.
    std[has_radiometric_neighbors] = np.sqrt(
        np.maximum(variance[has_radiometric_neighbors], 0.0)
    )
    mean_abs[has_radiometric_neighbors] = (
        neighbor_abs_difference[has_radiometric_neighbors]
        / valid_neighbor_count[has_radiometric_neighbors]
    )
    contrast = radiance - mean
    max_abs_radiance = float(np.nanmax(np.abs(radiance))) if finite.any() else 0.0
    epsilon = max(np.finfo(np.float64).eps, EPSILON_SCALE * max_abs_radiance)
    normalized = contrast / (radiance + mean + epsilon)

    context = pd.DataFrame(
        {
            "NTL_NeighborMean": mean,
            "NTL_NeighborStd": std,
            "NTL_LocalContrast": contrast,
            "NTL_NormalizedContrast": normalized,
            "NTL_MeanAbsNeighborContrast": mean_abs,
            "NeighborCount": degree,
        }
    )
    affected = (~finite) | ((degree > 0) & (valid_neighbor_count < degree))
    return context, epsilon, {
        "missing_pan_objects": int((~finite).sum()),
        "objects_affected_by_missing_pan": int(affected.sum()),
        "objects_without_valid_radiometric_neighbors": int(
            (finite & (degree > 0) & (valid_neighbor_count == 0)).sum()
        ),
    }


def global_morans_i(
    radiance: np.ndarray, edge_left: np.ndarray, edge_right: np.ndarray
) -> dict[str, float | int | None]:
    """Global Moran's I and randomization-based analytical inference."""
    finite = np.isfinite(radiance)
    valid_edges = finite[edge_left] & finite[edge_right]
    left = edge_left[valid_edges]
    right = edge_right[valid_edges]
    degree = graph_degrees(len(radiance), left, right)
    n = int(finite.sum())
    if n < 4 or not len(left):
        return {"n": n, "I": None, "expected_I": None, "z_score": None, "p_value": None}

    centered = np.zeros(len(radiance), dtype=np.float64)
    centered[finite] = radiance[finite] - float(np.mean(radiance[finite]))
    row_sum = np.zeros(len(radiance), dtype=np.float64)
    nonisolated = degree > 0
    row_sum[nonisolated] = 1.0
    lag = np.zeros(len(radiance), dtype=np.float64)
    np.add.at(lag, left, centered[right] / degree[left])
    np.add.at(lag, right, centered[left] / degree[right])
    s0 = float(row_sum.sum())
    denominator = float(np.dot(centered[finite], centered[finite]))
    observed = (n / s0) * float(np.dot(centered, lag)) / denominator
    expected = -1.0 / (n - 1)

    inv_left_degree = 1.0 / degree[left]
    inv_right_degree = 1.0 / degree[right]
    s1 = float(np.sum((inv_left_degree + inv_right_degree) ** 2))
    column_sum = np.zeros(len(radiance), dtype=np.float64)
    np.add.at(column_sum, right, inv_left_degree)
    np.add.at(column_sum, left, inv_right_degree)
    s2 = float(np.sum((row_sum + column_sum) ** 2))
    m2 = float(np.mean(centered[finite] ** 2))
    b2 = float(np.mean(centered[finite] ** 4) / (m2**2))
    numerator = n * (
        (n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0
    ) - b2 * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0)
    variance = numerator / ((n - 1) * (n - 2) * (n - 3) * s0 * s0) - expected**2
    z_score = (observed - expected) / math.sqrt(variance) if variance > 0 else math.nan
    p_value = 2.0 * norm.sf(abs(z_score)) if np.isfinite(z_score) else math.nan
    log10_p_value = (
        (math.log(2.0) + norm.logsf(abs(z_score))) / math.log(10.0)
        if np.isfinite(z_score)
        else math.nan
    )
    return {
        "n": n,
        "nonisolated_valid_objects": int(nonisolated.sum()),
        "S0": s0,
        "I": observed,
        "expected_I": expected,
        "variance_randomization": variance,
        "z_score": z_score,
        "p_value_two_sided": p_value,
        "log10_p_value_two_sided": log10_p_value,
    }


def descriptive_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for column in CONTEXT_COLUMNS:
        values = frame[column].astype(float)
        finite = np.isfinite(values)
        valid = values[finite]
        result[column] = {
            "valid_count": int(finite.sum()),
            "nan_count": int(values.isna().sum()),
            "infinity_count": int(np.isinf(values).sum()),
            "min": float(valid.min()) if len(valid) else None,
            "q25": float(valid.quantile(0.25)) if len(valid) else None,
            "median": float(valid.median()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "q75": float(valid.quantile(0.75)) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
        }
    return result


def validate_results(
    geometries: np.ndarray,
    radiance: np.ndarray,
    context: pd.DataFrame,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
) -> dict[str, object]:
    n = len(geometries)
    degree = graph_degrees(n, edge_left, edge_right)
    if not np.array_equal(degree, context["NeighborCount"].to_numpy()):
        raise RuntimeError("NeighborCount does not agree with the edge list.")

    adjacency: dict[int, list[int]] = {}
    positive_nodes = np.flatnonzero(degree > 0)
    targets = [
        int(positive_nodes[0]),
        int(positive_nodes[np.argmin(np.abs(degree[positive_nodes] - np.median(degree[positive_nodes])))]),
        int(np.argmax(degree)),
    ]
    targets = list(dict.fromkeys(targets))
    for node in targets:
        neighbors = np.concatenate(
            (edge_right[edge_left == node], edge_left[edge_right == node])
        )
        adjacency[node] = neighbors.tolist()
        valid_neighbors = neighbors[np.isfinite(radiance[neighbors])]
        if np.isfinite(radiance[node]) and len(valid_neighbors):
            neighbor_values = radiance[valid_neighbors]
            expected = np.array(
                [
                    neighbor_values.mean(),
                    neighbor_values.std(ddof=0),
                    radiance[node] - neighbor_values.mean(),
                    (radiance[node] - neighbor_values.mean()),
                    np.mean(np.abs(radiance[node] - neighbor_values)),
                ]
            )
            actual = context.loc[
                node,
                [
                    "NTL_NeighborMean",
                    "NTL_NeighborStd",
                    "NTL_LocalContrast",
                    "NTL_NormalizedContrast",
                    "NTL_MeanAbsNeighborContrast",
                ],
            ].to_numpy(dtype=float)
            expected[3] = expected[3] / (
                radiance[node] + neighbor_values.mean()
                + max(np.finfo(float).eps, EPSILON_SCALE * np.nanmax(np.abs(radiance)))
            )
            if not np.allclose(actual, expected, rtol=1e-7, atol=1e-10, equal_nan=True):
                raise RuntimeError(f"Independent numerical validation failed for node {node}.")
            relations = shapely.relate(
                geometries[np.full(len(neighbors), node)], geometries[neighbors]
            )
            if not all(relation[0] == "F" and relation[4] == "1" for relation in relations):
                raise RuntimeError(f"Non-Rook neighbor found for validation node {node}.")

    # Synthetic topology test: corner-only contact must be excluded while an
    # edge-sharing polygon must pass the same DE-9IM criterion used above.
    synthetic = shapely.box(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
    )
    point_relation = shapely.relate(synthetic[0], synthetic[1])
    edge_relation = shapely.relate(synthetic[0], shapely.box(1.0, 0.0, 2.0, 1.0))
    if point_relation[4] != "0" or edge_relation[4] != "1":
        raise RuntimeError("Synthetic point/edge topology validation failed.")

    return {
        "representative_node_positions": targets,
        "representative_degrees": {str(node): int(degree[node]) for node in targets},
        "representative_neighbor_positions": {str(k): v for k, v in adjacency.items()},
        "numerical_validation": "passed",
        "rook_relation_validation": "passed",
        "synthetic_point_only_exclusion": "passed",
        "symmetry_validation": "passed (stored once as canonical undirected edges)",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    total_started = perf_counter()
    timings: dict[str, float] = {}

    stage_started = perf_counter()
    polygons, features, input_info = load_and_validate_inputs(
        args.segmentation, args.feature_table
    )
    timings["input_loading"] = perf_counter() - stage_started

    stage_started = perf_counter()
    geometries = polygons.geometry.to_numpy()
    edge_left, edge_right, topology = build_rook_graph(geometries, args.chunk_size)
    degree = graph_degrees(len(polygons), edge_left, edge_right)
    timings["rook_graph_construction"] = perf_counter() - stage_started

    stage_started = perf_counter()
    positions = pd.Index(features[TABLE_ID_FIELD]).get_indexer(polygons[POLYGON_ID_FIELD])
    if np.any(positions < 0):
        raise RuntimeError("Failed to map feature-table Object_ID to polygon Seg_id.")
    radiance = features[PAN_MEAN_FIELD].to_numpy(dtype=np.float64)[positions]
    timings["pan_mapping_to_graph_nodes"] = perf_counter() - stage_started

    stage_started = perf_counter()
    context_graph, epsilon, missing_info = calculate_context(radiance, edge_left, edge_right)
    timings["context_feature_calculation"] = perf_counter() - stage_started

    stage_started = perf_counter()
    moran = global_morans_i(radiance, edge_left, edge_right)
    timings["global_morans_i"] = perf_counter() - stage_started

    validation = validate_results(geometries, radiance, context_graph, edge_left, edge_right)
    validation["representative_object_ids"] = [
        int(polygons.iloc[position][POLYGON_ID_FIELD])
        for position in validation["representative_node_positions"]
    ]

    graph_order_by_table = pd.Index(polygons[POLYGON_ID_FIELD]).get_indexer(features[TABLE_ID_FIELD])
    context_table_order = context_graph.iloc[graph_order_by_table].reset_index(drop=True)
    final = pd.concat([features.reset_index(drop=True), context_table_order], axis=1)
    if len(final) != len(features) or not final[TABLE_ID_FIELD].equals(features[TABLE_ID_FIELD]):
        raise RuntimeError("Object ID/order changed while appending context features.")
    if not final.iloc[:, : len(features.columns)].equals(features):
        raise RuntimeError("An existing feature column changed unexpectedly.")

    stage_started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    timings["output_writing"] = perf_counter() - stage_started
    timings["total"] = perf_counter() - total_started

    isolated = int((degree == 0).sum())
    degree_series = pd.Series(degree, dtype="int64")
    diagnostics: dict[str, object] = {
        "inputs": input_info,
        "method": {
            "neighborhood": "first-order strict Rook adjacency",
            "rook_de9im_pattern": "F***1****",
            "weights": "binary row-standardized",
            "graph_representation": "unique undirected int64 edge list",
            "epsilon": epsilon,
            "epsilon_scale": EPSILON_SCALE,
            "optical_context_used": False,
        },
        "graph": {
            "objects": len(polygons),
            "unique_undirected_edges": len(edge_left),
            "isolated_objects": isolated,
            "isolated_percentage": 100.0 * isolated / len(polygons),
            "neighbor_count_min": int(degree.min()),
            "neighbor_count_q25": float(degree_series.quantile(0.25)),
            "neighbor_count_median": float(degree_series.median()),
            "neighbor_count_mean": float(degree_series.mean()),
            "neighbor_count_q75": float(degree_series.quantile(0.75)),
            "neighbor_count_q95": float(degree_series.quantile(0.95)),
            "neighbor_count_max": int(degree.max()),
            **topology,
        },
        "missing_pan": missing_info,
        "morans_i": moran,
        "feature_statistics": descriptive_statistics(final[CONTEXT_COLUMNS]),
        "validation": validation,
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(final),
            "original_columns": len(features.columns),
            "new_context_columns": len(CONTEXT_COLUMNS),
            "final_columns": len(final.columns),
        },
        "timings_seconds": timings,
    }
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Completed: %s", args.output)
    LOGGER.info("Diagnostics: %s", args.diagnostics)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, default=SEGMENTATION_PATH)
    parser.add_argument("--feature-table", type=Path, default=FEATURE_TABLE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_TABLE_PATH)
    parser.add_argument("--diagnostics", type=Path, default=DIAGNOSTICS_PATH)
    parser.add_argument("--chunk-size", type=int, default=GRAPH_CHUNK_SIZE)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
