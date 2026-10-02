"""Persist and verify the strict-Rook graph from Spatial_Context_Extraction_01."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Spatial_Context_Extraction_01 import (
    DIAGNOSTICS_PATH,
    FEATURE_TABLE_PATH,
    OUTPUT_TABLE_PATH,
    POLYGON_ID_FIELD,
    SEGMENTATION_PATH,
    TABLE_ID_FIELD,
    build_rook_graph,
    graph_degrees,
    load_and_validate_inputs,
)


SEGMENT_DIR = SEGMENTATION_PATH.parent
GRAPH_PATH = SEGMENT_DIR / "rook_adjacency_graph_v01.npz"
METADATA_PATH = SEGMENT_DIR / "rook_adjacency_graph_v01_metadata.json"


def degree_summary(degree: np.ndarray) -> dict[str, float | int]:
    values = pd.Series(degree, dtype="int64")
    return {
        "minimum": int(degree.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "maximum": int(degree.max()),
    }


def main() -> int:
    # This intentionally reuses the exact input checks, make_valid repair, and
    # DE-9IM graph builder from the completed spatial-context implementation.
    polygons, existing_features, input_info = load_and_validate_inputs(
        SEGMENTATION_PATH, FEATURE_TABLE_PATH
    )
    del existing_features

    edge_left, edge_right, topology = build_rook_graph(polygons.geometry.to_numpy())
    object_id = polygons[POLYGON_ID_FIELD].to_numpy(dtype=np.int64, copy=True)
    degree = graph_degrees(len(polygons), edge_left, edge_right)

    left_id = object_id[edge_left]
    right_id = object_id[edge_right]
    src_object_id = np.minimum(left_id, right_id)
    dst_object_id = np.maximum(left_id, right_id)
    order = np.lexsort((dst_object_id, src_object_id))
    src_object_id = src_object_id[order]
    dst_object_id = dst_object_id[order]
    if np.any(src_object_id >= dst_object_id):
        raise RuntimeError("Canonical Object_ID edge ordering failed.")
    edge_pairs = np.column_stack((src_object_id, dst_object_id))
    if len(np.unique(edge_pairs, axis=0)) != len(edge_pairs):
        raise RuntimeError("Duplicate undirected Object_ID edges detected.")

    # object_id defines the alignment of neighbor_count explicitly, so the NPZ
    # remains reusable even if a future table has a different row order.
    np.savez_compressed(
        GRAPH_PATH,
        src_object_id=src_object_id,
        dst_object_id=dst_object_id,
        object_id=object_id,
        neighbor_count=degree,
    )

    completed = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    expected_graph = completed["graph"]

    # Mandatory reload validation against both the rebuilt arrays and the
    # completed spatial-context output's per-object NeighborCount.
    with np.load(GRAPH_PATH, allow_pickle=False) as saved:
        saved_src = saved["src_object_id"]
        saved_dst = saved["dst_object_id"]
        saved_ids = saved["object_id"]
        saved_degree = saved["neighbor_count"]

    if len(saved_ids) != len(object_id) or not np.array_equal(saved_ids, object_id):
        raise RuntimeError("Reloaded object_id array does not match polygon Object_ID mapping.")
    if len(saved_src) != len(src_object_id) or not np.array_equal(saved_src, src_object_id):
        raise RuntimeError("Reloaded source edges differ from saved canonical edges.")
    if not np.array_equal(saved_dst, dst_object_id):
        raise RuntimeError("Reloaded destination edges differ from saved canonical edges.")
    if not np.array_equal(saved_degree, degree):
        raise RuntimeError("Reloaded NeighborCount differs from rebuilt graph degrees.")

    id_to_position = pd.Index(saved_ids)
    src_position = id_to_position.get_indexer(saved_src)
    dst_position = id_to_position.get_indexer(saved_dst)
    if np.any(src_position < 0) or np.any(dst_position < 0):
        raise RuntimeError("An edge references an Object_ID absent from object_id.")
    reloaded_degree = graph_degrees(len(saved_ids), src_position, dst_position)
    if not np.array_equal(reloaded_degree, saved_degree):
        raise RuntimeError("NeighborCount cannot be reproduced from reloaded edges.")

    reference = pd.read_csv(
        OUTPUT_TABLE_PATH, usecols=[TABLE_ID_FIELD, "NeighborCount"]
    )
    reference_positions = pd.Index(reference[TABLE_ID_FIELD]).get_indexer(saved_ids)
    if np.any(reference_positions < 0):
        raise RuntimeError("Saved graph IDs do not match completed spatial-context IDs.")
    reference_degree = reference["NeighborCount"].to_numpy()[reference_positions]
    if not np.array_equal(saved_degree, reference_degree):
        raise RuntimeError("Saved NeighborCount differs from completed spatial-context output.")

    summary = degree_summary(saved_degree)
    isolated = int((saved_degree == 0).sum())
    if len(saved_ids) != expected_graph["objects"]:
        raise RuntimeError("Object count differs from completed run diagnostics.")
    if len(saved_src) != expected_graph["unique_undirected_edges"]:
        raise RuntimeError("Edge count differs from completed run diagnostics.")
    if isolated != expected_graph["isolated_objects"]:
        raise RuntimeError("Isolated-object count differs from completed run diagnostics.")

    metadata = {
        "graph_version": "v01",
        "created_at": datetime.now().astimezone().isoformat(),
        "graph_path": str(GRAPH_PATH.resolve()),
        "file_format": "NumPy compressed NPZ sparse canonical edge list",
        "number_of_objects": len(saved_ids),
        "number_of_unique_undirected_edges": len(saved_src),
        "adjacency_definition": (
            "Strict first-order Rook: disjoint polygon interiors and a shared "
            "boundary segment of non-zero length (DE-9IM F***1****); point-only "
            "contacts excluded; binary adjacency."
        ),
        "id_field": TABLE_ID_FIELD,
        "source_polygon_id_field": POLYGON_ID_FIELD,
        "isolated_object_count": isolated,
        "neighbor_count_summary": summary,
        "source_segmentation_file": str(SEGMENTATION_PATH.resolve()),
        "source_feature_table": str(FEATURE_TABLE_PATH.resolve()),
        "geometry_repair": {
            "method": "shapely.make_valid with polygonal-part retention",
            "repaired_count": input_info["invalid_geometries_repaired"],
        },
        "topology_diagnostics": topology,
        "npz_contents": {
            "src_object_id": "int64; canonical smaller Object_ID for each edge",
            "dst_object_id": "int64; canonical larger Object_ID for each edge",
            "object_id": "int64; explicit alignment for neighbor_count",
            "neighbor_count": "int64; topological Rook degree aligned to object_id",
        },
        "reload_check": {
            "passed": True,
            "npz_reloaded_with_allow_pickle_false": True,
            "object_count_matches_completed_run": True,
            "edge_count_matches_completed_run": True,
            "neighbor_count_reconstructed_from_edges": True,
            "neighbor_count_matches_completed_spatial_context_table": True,
            "completed_run_diagnostics": str(DIAGNOSTICS_PATH.resolve()),
            "completed_spatial_context_table": str(OUTPUT_TABLE_PATH.resolve()),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
