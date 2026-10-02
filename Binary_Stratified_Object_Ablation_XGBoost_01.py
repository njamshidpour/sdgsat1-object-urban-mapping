#!/usr/bin/env python
"""Binary Urban/NonUrban object ablation with source-class stratification.

The script reconstructs binary training and independent test sets from the
master object-reference table and manual test points.  Urban, Vegetation and
BareScrub are balanced before Vegetation and BareScrub are merged into the
heterogeneous NonUrban class.  Feature families are inherited exactly from
Object_Based_Ablation_XGBoost_01.py and its saved manifest.

Pilot mode validates the full sampling and evaluation workflow with lighter
model fitting.  Full mode performs five-fold spatial CV, 24 configurations and
2,000 paired bootstrap replicates.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import Object_Based_Ablation_XGBoost_01 as previous


RANDOM_SEED = 20261001
PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"

DEFAULT_SEGMENTATION = SEGMENTATION_DIR / "eCog_segments.shp"
DEFAULT_MASTER_REFERENCE = (
    SEGMENTATION_DIR
    / "Object_Labeling_From_DW_01"
    / "Object_Reference_Labels_DW_01.csv"
)
DEFAULT_TEST_POINTS = PROJECT_ROOT / "GTs" / "Test_Points_1000.shp"
DEFAULT_ADJACENCY = SEGMENTATION_DIR / "rook_adjacency_graph_v01.npz"
DEFAULT_CONTEXT_TABLE = previous.DEFAULT_CONTEXT_TABLE
DEFAULT_TEXTURE_TABLE = previous.DEFAULT_TEXTURE_TABLE
DEFAULT_PREVIOUS_MANIFEST = (
    SEGMENTATION_DIR
    / "Object_Based_Ablation_XGBoost_01"
    / "Ablation_Feature_Group_Manifest.csv"
)
DEFAULT_PREVIOUS_PREDICTIONS = (
    SEGMENTATION_DIR
    / "Object_Based_Ablation_XGBoost_01"
    / "Ablation_Test_Predictions.csv"
)

SOURCE_CODES = {1: "Urban", 2: "Vegetation", 3: "BareScrub"}
BINARY_LABELS = {0: "NonUrban", 1: "Urban"}
THRESHOLD = 0.50

FEATURE_GROUPS = {key: list(value) for key, value in previous.FEATURE_GROUPS.items()}
MODEL_GROUPS = OrderedDict([
    ("B0", ("N",)),
    ("B1", ("N", "G")),
    ("B2", ("N", "T")),
    ("B3", ("N", "C")),
    ("B4", ("N", "G", "T")),
    ("B5", ("N", "G", "C")),
    ("B6", ("N", "T", "C")),
    ("B7", ("N", "G", "T", "C")),
])

BOOTSTRAP_METRICS = [
    "BalancedAccuracy", "MacroF1", "Urban_F1", "Urban_IoU", "mIoU"
]
REQUIRED_COMPARISONS = [
    ("B1", "B0"), ("B2", "B0"), ("B3", "B0"),
    ("B4", "B1"), ("B4", "B2"),
    ("B5", "B1"), ("B5", "B3"),
    ("B6", "B2"), ("B6", "B3"),
    ("B7", "B4"), ("B7", "B5"), ("B7", "B6"),
    ("B7", "B0"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary stratified object-level XGBoost feature ablation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--master-reference", type=Path, default=DEFAULT_MASTER_REFERENCE)
    parser.add_argument("--test-points", type=Path, default=DEFAULT_TEST_POINTS)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--context-table", type=Path, default=DEFAULT_CONTEXT_TABLE)
    parser.add_argument("--texture-table", type=Path, default=DEFAULT_TEXTURE_TABLE)
    parser.add_argument("--previous-manifest", type=Path, default=DEFAULT_PREVIOUS_MANIFEST)
    parser.add_argument(
        "--previous-three-class-predictions", type=Path,
        default=DEFAULT_PREVIOUS_PREDICTIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--search-configs", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--pilot-configs", type=int, default=3)
    parser.add_argument("--pilot-bootstrap", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def dependency_bundle():
    bundle = previous.dependency_bundle()
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        roc_auc_score,
    )
    bundle.update({
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "matthews_corrcoef": matthews_corrcoef,
        "roc_auc_score": roc_auc_score,
    })
    return bundle


def validate_previous_manifest(path: Path):
    if not path.is_file():
        raise SystemExit(
            "STOP: previous final feature manifest is unavailable: " + str(path)
        )
    manifest = pd.read_csv(path)
    required = {"FeatureGroup", "OrderWithinGroup", "Feature"}
    if not required.issubset(manifest.columns):
        raise SystemExit("STOP: previous feature manifest has an invalid structure.")
    resolved = {}
    for group in ("N", "G", "T", "C"):
        rows = manifest[manifest["FeatureGroup"] == group].sort_values("OrderWithinGroup")
        resolved[group] = rows["Feature"].tolist()
        if resolved[group] != FEATURE_GROUPS[group]:
            raise SystemExit(
                f"STOP: current {group} definition differs from the previous final manifest."
            )
    if resolved["T"] != [
        "SDGSAT_Pan_bgglcm_contrast",
        "SDGSAT_Pan_bgglcm_homogeneity",
        "SDGSAT_Pan_bgglcm_entropy",
    ]:
        raise SystemExit("STOP: previous manifest does not contain the required reduced BG-Log T.")
    return manifest


def save_binary_manifest(previous_manifest, source_path: Path, output_path: Path):
    output = previous_manifest.copy()
    output.insert(0, "BinaryManifestSource", str(source_path.resolve()))
    output.to_csv(output_path, index=False)
    return output


def load_master_reference(path: Path):
    required = [
        "Object_ID", "N_labeled_pixels", "DominantClass", "DominantClassCode",
        "Purity", "Coverage",
    ]
    data = pd.read_csv(path)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit("STOP: master-reference columns missing: " + ", ".join(missing))
    if data["Object_ID"].isna().any() or data["Object_ID"].duplicated().any():
        raise SystemExit("STOP: master Object_ID is missing or duplicated.")
    data["Object_ID"] = data["Object_ID"].astype("int64")
    return data


def map_manual_test_pool(gpd, points_path: Path, segments):
    points = gpd.read_file(points_path)
    if points.crs is None:
        raise SystemExit("STOP: manual test points have no CRS.")
    candidates = [
        column for column in ("Class_ID", "Class", "class_id", "class", "Label", "label")
        if column in points.columns
    ]
    if not candidates:
        raise SystemExit("STOP: no recognizable class field in manual test points.")
    label_field = candidates[0]
    points = points.to_crs(segments.crs).reset_index(names="SourcePointIndex")
    joined = gpd.sjoin(
        points[["SourcePointIndex", label_field, "geometry"]],
        segments[["Seg_id", "geometry"]],
        how="left", predicate="within",
    )
    joined["SourceCode3"] = joined[label_field].map(previous.manual_label_to_three_class)
    valid = joined.dropna(subset=["Seg_id", "SourceCode3"]).copy()
    valid["Object_ID"] = valid["Seg_id"].astype("int64")
    valid["SourceCode3"] = valid["SourceCode3"].astype("int64")
    valid["BinaryLabel"] = (valid["SourceCode3"] == 1).astype("int64")

    grouped = valid.groupby("Object_ID", sort=True)
    rows = []
    conflict_ids = []
    mixed_nonurban_ids = []
    for object_id, group in grouped:
        binary_values = sorted(group["BinaryLabel"].unique().tolist())
        source_values = sorted(group["SourceCode3"].unique().tolist())
        if len(binary_values) != 1:
            conflict_ids.append(int(object_id))
            continue
        if len(source_values) == 1:
            source_class = SOURCE_CODES[source_values[0]]
        else:
            source_class = "MixedNonUrban"
            mixed_nonurban_ids.append(int(object_id))
        rows.append({
            "Object_ID": int(object_id),
            "SourceClass3": source_class,
            "BinaryLabel": int(binary_values[0]),
            "BinaryClass": BINARY_LABELS[int(binary_values[0])],
            "ManualPointCount": int(len(group)),
        })
    pool = pd.DataFrame(rows).sort_values("Object_ID").reset_index(drop=True)
    diagnostics = {
        "source_points": int(len(points)),
        "label_field": label_field,
        "unlabeled_or_unknown_points": int(
            points[label_field].map(previous.manual_label_to_three_class).isna().sum()
        ),
        "strictly_unmatched_labeled_points": int(
            joined.loc[joined["SourceCode3"].notna(), "Seg_id"].isna().sum()
        ),
        "valid_point_assignments": int(len(valid)),
        "unique_nonconflicting_objects": int(len(pool)),
        "binary_conflicting_objects": int(len(conflict_ids)),
        "binary_conflicting_object_ids": conflict_ids,
        "mixed_nonurban_source_objects": int(len(mixed_nonurban_ids)),
        "mixed_nonurban_object_ids": mixed_nonurban_ids,
        "duplicate_point_assignments": int(len(valid) - valid["Object_ID"].nunique()),
    }
    return pool, diagnostics


def rook_neighbors(adjacency_path: Path, object_ids):
    graph = np.load(adjacency_path, allow_pickle=False)
    required = {"src_object_id", "dst_object_id"}
    if not required.issubset(graph.files):
        raise SystemExit("STOP: saved Rook graph lacks canonical edge arrays.")
    src = graph["src_object_id"].astype("int64", copy=False)
    dst = graph["dst_object_id"].astype("int64", copy=False)
    object_ids = np.asarray(object_ids, dtype="int64")
    src_mask = np.isin(src, object_ids)
    dst_mask = np.isin(dst, object_ids)
    neighbors = np.unique(np.concatenate([dst[src_mask], src[dst_mask]]))
    return neighbors[~np.isin(neighbors, object_ids)]


def add_centroid_cells(frame, segment_index):
    ids = frame["Object_ID"].to_numpy(dtype="int64")
    missing = pd.Index(ids).difference(segment_index.index)
    if len(missing):
        raise SystemExit(f"STOP: {len(missing)} required IDs are missing from segmentation.")
    geometry = segment_index.loc[ids].to_crs("EPSG:32639")
    centroids = geometry.geometry.centroid
    gx = np.floor(centroids.x.to_numpy() / 5000.0).astype("int64")
    gy = np.floor(centroids.y.to_numpy() / 5000.0).astype("int64")
    result = frame.copy()
    result["Centroid_X_32639"] = centroids.x.to_numpy()
    result["Centroid_Y_32639"] = centroids.y.to_numpy()
    result["GridX"] = gx
    result["GridY"] = gy
    result["SpatialCell"] = [f"{x}_{y}" for x, y in zip(gx, gy)]
    return result


def tertile(values, labels):
    values = pd.Series(values)
    if values.isna().any():
        raise SystemExit("STOP: a required stratification variable contains NaN.")
    ranks = values.rank(method="first")
    return pd.qcut(ranks, q=3, labels=labels).astype(str)


def allocate_proportional(capacity: pd.Series, total: int, ensure_coverage: bool):
    capacity = capacity.astype("int64").sort_index()
    if total < 0 or total > int(capacity.sum()):
        raise ValueError("Allocation target exceeds available capacity.")
    quota = pd.Series(0, index=capacity.index, dtype="int64")
    positive = capacity[capacity > 0]
    if ensure_coverage and total >= len(positive):
        quota.loc[positive.index] = 1
    remaining = total - int(quota.sum())
    while remaining > 0:
        available = capacity - quota
        available = available[available > 0]
        if available.empty:
            raise RuntimeError("Unable to fill proportional allocation.")
        raw = remaining * available / available.sum()
        addition = np.floor(raw).astype("int64")
        addition = np.minimum(addition, available)
        if int(addition.sum()) > 0:
            quota.loc[addition.index] += addition
            remaining -= int(addition.sum())
            continue
        order = sorted(
            available.index,
            key=lambda key: (-(raw.loc[key] - np.floor(raw.loc[key])), str(key)),
        )
        for key in order:
            if remaining == 0:
                break
            if quota.loc[key] < capacity.loc[key]:
                quota.loc[key] += 1
                remaining -= 1
    return quota


def allocate_even(capacity: pd.Series, total: int):
    capacity = capacity.astype("int64").sort_index()
    if total > int(capacity.sum()):
        raise ValueError("Even allocation target exceeds capacity.")
    quota = pd.Series(0, index=capacity.index, dtype="int64")
    remaining = total
    while remaining > 0:
        active = [key for key in capacity.index if quota.loc[key] < capacity.loc[key]]
        if not active:
            raise RuntimeError("Unable to fill even allocation.")
        active.sort(key=lambda key: (quota.loc[key], str(key)))
        for key in active:
            if remaining == 0:
                break
            quota.loc[key] += 1
            remaining -= 1
    return quota


def hierarchical_stratified_sample(frame, target, within_columns, rng, even_within):
    """Allocate across cells first, then within-cell strata, with exact target."""
    if target >= len(frame):
        return frame.copy()
    cell_capacity = frame.groupby("SpatialCell", observed=True).size()
    cell_quota = allocate_proportional(cell_capacity, target, ensure_coverage=True)
    selections = []
    for cell in sorted(cell_capacity.index):
        quota = int(cell_quota.loc[cell])
        if quota == 0:
            continue
        cell_frame = frame[frame["SpatialCell"] == cell]
        if within_columns:
            group_key = within_columns[0] if len(within_columns) == 1 else within_columns
            capacities = cell_frame.groupby(group_key, observed=True).size()
            allocation = (
                allocate_even(capacities, quota)
                if even_within else allocate_proportional(capacities, quota, ensure_coverage=False)
            )
            for stratum in capacities.index:
                amount = int(allocation.loc[stratum])
                if amount == 0:
                    continue
                key_tuple = stratum if isinstance(stratum, tuple) else (stratum,)
                mask = np.ones(len(cell_frame), dtype=bool)
                for column, value in zip(within_columns, key_tuple):
                    mask &= cell_frame[column].to_numpy() == value
                indices = cell_frame.index.to_numpy()[mask]
                chosen = rng.choice(indices, size=amount, replace=False)
                selections.extend(chosen.tolist())
        else:
            indices = cell_frame.index.to_numpy()
            selections.extend(rng.choice(indices, size=quota, replace=False).tolist())
    if len(selections) != target or len(set(selections)) != target:
        raise SystemExit(
            f"STOP: hierarchical sampler selected {len(selections)} rows; expected {target}."
        )
    return frame.loc[selections].sort_values("Object_ID").copy()


def build_primary_test(pool, rng):
    eligible_source = pool[pool["SourceClass3"].isin(SOURCE_CODES.values())].copy()
    urban = eligible_source[eligible_source["SourceClass3"] == "Urban"]
    vegetation = eligible_source[eligible_source["SourceClass3"] == "Vegetation"]
    bare = eligible_source[eligible_source["SourceClass3"] == "BareScrub"].copy()
    target_bare = min(len(vegetation), len(bare))
    bare["SizeStratum"] = tertile(
        bare["area_sqm"], ["small", "medium", "large"]
    )
    selected_bare = hierarchical_stratified_sample(
        bare, target_bare, ["SizeStratum"], rng, even_within=False
    )
    selected = pd.concat([urban, vegetation, selected_bare], ignore_index=True)
    selected = selected.sort_values("Object_ID").reset_index(drop=True)
    return selected, {
        "available_urban": int(len(urban)),
        "available_vegetation": int(len(vegetation)),
        "available_barescrub": int(len(bare)),
        "selected_barescrub": int(len(selected_bare)),
        "primary_urban": int((selected["BinaryLabel"] == 1).sum()),
        "primary_nonurban": int((selected["BinaryLabel"] == 0).sum()),
        "primary_total": int(len(selected)),
        "primary_occupied_cells": int(selected["SpatialCell"].nunique()),
        "bare_selected_cells": int(selected_bare["SpatialCell"].nunique()),
        "bare_available_cells": int(bare["SpatialCell"].nunique()),
    }


def construct_training_candidates(master, feature_index, full_pool_ids, neighbor_ids):
    confidence = (
        (master["Purity"] >= 0.90)
        & (master["Coverage"] >= 0.50)
        & (master["N_labeled_pixels"] >= 5)
        & master["DominantClass"].isin(SOURCE_CODES.values())
    )
    high_confidence = master.loc[
        confidence,
        ["Object_ID", "DominantClass", "Purity", "Coverage", "N_labeled_pixels"],
    ].copy()
    excluded = np.union1d(
        np.asarray(full_pool_ids, dtype="int64"),
        np.asarray(neighbor_ids, dtype="int64"),
    )
    candidates = high_confidence[~high_confidence["Object_ID"].isin(excluded)].copy()
    candidates = candidates.rename(columns={"DominantClass": "SourceClass3"})
    candidates["BinaryLabel"] = (candidates["SourceClass3"] == "Urban").astype("int64")
    candidates["BinaryClass"] = candidates["BinaryLabel"].map(BINARY_LABELS)
    missing_features = pd.Index(candidates["Object_ID"]).difference(feature_index.index)
    if len(missing_features):
        raise SystemExit(f"STOP: {len(missing_features)} candidates lack features.")
    candidates = candidates.join(
        feature_index[["area_sqm", "SDGSAT_Pan_mean"]], on="Object_ID", how="left",
        validate="one_to_one",
    )
    return candidates, {
        "high_confidence_before_test_exclusion": int(len(high_confidence)),
        "excluded_high_confidence_test_objects": int(
            high_confidence["Object_ID"].isin(full_pool_ids).sum()
        ),
        "excluded_high_confidence_test_neighbors": int(
            high_confidence["Object_ID"].isin(neighbor_ids).sum()
        ),
        "eligible_after_exclusion": int(len(candidates)),
    }


def stratified_training_selection(candidates, rng):
    class_counts = candidates["SourceClass3"].value_counts()
    if not set(SOURCE_CODES.values()).issubset(class_counts.index):
        raise SystemExit("STOP: one or more source classes have no training candidates.")
    target = int(class_counts[list(SOURCE_CODES.values())].min())
    selected_parts = []
    diagnostics = []
    enriched_parts = []
    for source_class in SOURCE_CODES.values():
        group = candidates[candidates["SourceClass3"] == source_class].copy()
        group["SizeStratum"] = tertile(
            group["area_sqm"], ["small", "medium", "large"]
        )
        group["RadianceStratum"] = tertile(
            group["SDGSAT_Pan_mean"], ["low", "medium", "high"]
        )
        enriched_parts.append(group)
        selected = hierarchical_stratified_sample(
            group, target, ["SizeStratum", "RadianceStratum"],
            rng, even_within=True,
        )
        selected_parts.append(selected)
        diagnostics.append({
            "SourceClass3": source_class,
            "CandidateCount": int(len(group)),
            "SelectedCount": int(len(selected)),
            "PercentageRetained": 100.0 * len(selected) / len(group),
            "CandidateSpatialCells": int(group["SpatialCell"].nunique()),
            "SelectedSpatialCells": int(selected["SpatialCell"].nunique()),
        })
    enriched = pd.concat(enriched_parts).sort_values("Object_ID")
    selected = pd.concat(selected_parts).sort_values("Object_ID").reset_index(drop=True)
    if selected["Object_ID"].duplicated().any():
        raise SystemExit("STOP: stratified training selection produced duplicate Object_ID values.")
    counts = selected["SourceClass3"].value_counts()
    if any(int(counts[name]) != target for name in SOURCE_CODES.values()):
        raise SystemExit("STOP: source classes are not exactly balanced before binary merge.")
    return enriched, selected, target, pd.DataFrame(diagnostics)


def binary_class_weights(y):
    y = np.asarray(y, dtype="int64")
    total = len(y)
    weights = {}
    for code in (0, 1):
        count = int((y == code).sum())
        if count == 0:
            raise SystemExit("STOP: selected binary training class has zero objects.")
        weights[code] = total / (2 * count)
    sample_weight = np.asarray([weights[int(value)] for value in y], dtype="float64")
    return weights, sample_weight


def make_estimator(xgb, parameters, n_jobs):
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        missing=np.nan,
        verbosity=0,
        **parameters,
    )


def tune_b7(bundle, X, y, groups, sample_weight, configurations, n_splits, n_jobs):
    splits = list(bundle["GroupKFold"](n_splits=n_splits).split(X, y, groups=groups))
    progress = previous.SimpleProgress(
        len(configurations) * len(splits), "Binary spatial CV tuning"
    )
    rows = []
    for config_id, parameters in enumerate(configurations, start=1):
        macro_scores = []
        balanced_scores = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            if len(np.unique(y[train_index])) != 2 or len(np.unique(y[validation_index])) != 2:
                raise SystemExit(f"STOP: binary class absent from spatial CV fold {fold}.")
            model = make_estimator(bundle["xgb"], parameters, n_jobs)
            model.fit(
                X[train_index], y[train_index],
                sample_weight=sample_weight[train_index],
            )
            probability = model.predict_proba(X[validation_index])[:, 1]
            prediction = (probability >= THRESHOLD).astype("int64")
            macro_scores.append(bundle["f1_score"](
                y[validation_index], prediction, average="macro", zero_division=0
            ))
            balanced_scores.append(bundle["balanced_accuracy_score"](
                y[validation_index], prediction
            ))
            progress.update()
        row = {"Configuration": config_id, **parameters}
        for index, value in enumerate(macro_scores, start=1):
            row[f"Fold_{index}_MacroF1"] = float(value)
        for index, value in enumerate(balanced_scores, start=1):
            row[f"Fold_{index}_BalancedAccuracy"] = float(value)
        row["Mean_MacroF1"] = float(np.mean(macro_scores))
        row["Std_MacroF1"] = float(np.std(macro_scores, ddof=1))
        row["Mean_BalancedAccuracy"] = float(np.mean(balanced_scores))
        rows.append(row)
    results = pd.DataFrame(rows)
    maximum = results["Mean_MacroF1"].max()
    tied = results[results["Mean_MacroF1"] >= maximum - 1e-4].sort_values(
        ["max_depth", "n_estimators", "Mean_MacroF1"],
        ascending=[True, True, False],
    )
    selected = tied.iloc[0]
    parameters = {
        key: type(previous.PARAMETER_SPACE[key][0])(selected[key])
        for key in previous.PARAMETER_SPACE
    }
    return (
        parameters,
        float(selected["Mean_MacroF1"]),
        float(selected["Mean_BalancedAccuracy"]),
        results,
        splits,
    )


def binary_metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype="float64")
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, predicted, out=np.zeros(2), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(2), where=support > 0)
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros(2), where=(precision + recall) > 0,
        )
        iou = np.divide(
            tp, support + predicted - tp,
            out=np.zeros(2), where=(support + predicted - tp) > 0,
        )
    return {
        "OA": float(tp.sum() / total),
        "BalancedAccuracy": float(np.mean(recall)),
        "MacroF1": float(np.mean(f1)),
        "mIoU": float(np.mean(iou)),
        "NonUrban_Precision": float(precision[0]),
        "NonUrban_Recall": float(recall[0]),
        "NonUrban_F1": float(f1[0]),
        "NonUrban_IoU": float(iou[0]),
        "Urban_Precision": float(precision[1]),
        "Urban_Recall": float(recall[1]),
        "Urban_F1": float(f1[1]),
        "Urban_IoU": float(iou[1]),
        "support": support.astype("int64"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def evaluate_binary(bundle, y_true, prediction, p_urban):
    cm = bundle["confusion_matrix"](y_true, prediction, labels=[0, 1])
    metrics = binary_metrics_from_confusion(cm)
    metrics["Kappa"] = float(bundle["cohen_kappa_score"](y_true, prediction))
    metrics["MCC"] = float(bundle["matthews_corrcoef"](y_true, prediction))
    metrics["AUROC"] = float(bundle["roc_auc_score"](y_true, p_urban))
    metrics["AUPRC"] = float(bundle["average_precision_score"](y_true, p_urban))
    metrics["confusion"] = cm
    return metrics


def fit_models(bundle, features, training, primary, secondary, parameters, n_jobs):
    indexed = features.set_index("Object_ID")
    train_ids = training["Object_ID"].to_numpy(dtype="int64")
    primary_ids = primary["Object_ID"].to_numpy(dtype="int64")
    secondary_ids = secondary["Object_ID"].to_numpy(dtype="int64")
    y_train = training["BinaryLabel"].to_numpy(dtype="int64")
    y_primary = primary["BinaryLabel"].to_numpy(dtype="int64")
    y_secondary = secondary["BinaryLabel"].to_numpy(dtype="int64")
    weights, sample_weight = binary_class_weights(y_train)
    predictions = {"primary": {}, "secondary": {}}
    probabilities = {"primary": {}, "secondary": {}}
    evaluations = {"primary": {}, "secondary": {}}
    training_seconds = 0.0
    prediction_seconds = 0.0
    progress = previous.SimpleProgress(len(MODEL_GROUPS), "Binary ablation models")
    for model_name, group_names in MODEL_GROUPS.items():
        columns = [column for group in group_names for column in FEATURE_GROUPS[group]]
        X_train = indexed.loc[train_ids, columns].to_numpy(dtype="float32", copy=True)
        X_primary = indexed.loc[primary_ids, columns].to_numpy(dtype="float32", copy=True)
        X_secondary = indexed.loc[secondary_ids, columns].to_numpy(dtype="float32", copy=True)
        model = make_estimator(bundle["xgb"], parameters, n_jobs)
        started = time.perf_counter()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        p_primary = model.predict_proba(X_primary)[:, 1]
        p_secondary = model.predict_proba(X_secondary)[:, 1]
        prediction_seconds += time.perf_counter() - started
        if not np.all(np.isfinite(p_primary)) or not np.all(np.isfinite(p_secondary)):
            raise SystemExit(f"STOP: {model_name} produced non-finite probabilities.")
        pred_primary = (p_primary >= THRESHOLD).astype("int64")
        pred_secondary = (p_secondary >= THRESHOLD).astype("int64")
        probabilities["primary"][model_name] = p_primary
        probabilities["secondary"][model_name] = p_secondary
        predictions["primary"][model_name] = pred_primary
        predictions["secondary"][model_name] = pred_secondary
        evaluations["primary"][model_name] = evaluate_binary(
            bundle, y_primary, pred_primary, p_primary
        )
        evaluations["secondary"][model_name] = evaluate_binary(
            bundle, y_secondary, pred_secondary, p_secondary
        )
        progress.update()
    return (
        predictions, probabilities, evaluations, weights,
        training_seconds, prediction_seconds,
    )


def evaluation_tables(evaluations, test_name):
    model_rows = []
    class_rows = []
    confusion_rows = []
    for model, result in evaluations.items():
        groups = MODEL_GROUPS[model]
        model_rows.append({
            "Model": model,
            "FeatureGroups": "+".join(groups),
            "NumberOfFeatures": sum(len(FEATURE_GROUPS[g]) for g in groups),
            "OA": result["OA"],
            "BalancedAccuracy": result["BalancedAccuracy"],
            "Kappa": result["Kappa"],
            "MCC": result["MCC"],
            "MacroF1": result["MacroF1"],
            "mIoU": result["mIoU"],
            "AUROC": result["AUROC"],
            "AUPRC": result["AUPRC"],
            "Urban_Precision": result["Urban_Precision"],
            "Urban_Recall": result["Urban_Recall"],
            "Urban_F1": result["Urban_F1"],
            "Urban_IoU": result["Urban_IoU"],
            "NonUrban_Precision": result["NonUrban_Precision"],
            "NonUrban_Recall": result["NonUrban_Recall"],
            "NonUrban_F1": result["NonUrban_F1"],
            "NonUrban_IoU": result["NonUrban_IoU"],
        })
        for index, code in enumerate((0, 1)):
            class_rows.append({
                "TestSet": test_name,
                "Model": model,
                "Class": BINARY_LABELS[code],
                "Precision": result["precision"][index],
                "Recall": result["recall"][index],
                "F1": result["f1"][index],
                "IoU": result["iou"][index],
                "Support": result["support"][index],
            })
        for true_index, true_code in enumerate((0, 1)):
            for predicted_index, predicted_code in enumerate((0, 1)):
                confusion_rows.append({
                    "TestSet": test_name,
                    "Model": model,
                    "TrueClass": BINARY_LABELS[true_code],
                    "PredictedClass": BINARY_LABELS[predicted_code],
                    "Count": int(result["confusion"][true_index, predicted_index]),
                })
    return pd.DataFrame(model_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def save_predictions(path, test_frame, predictions, probabilities):
    output = test_frame[["Object_ID", "BinaryClass", "SourceClass3"]].rename(
        columns={"BinaryClass": "TrueBinaryClass"}
    ).copy()
    for model in MODEL_GROUPS:
        output[f"{model}_PredictedClass"] = [
            BINARY_LABELS[int(value)] for value in predictions[model]
        ]
        output[f"{model}_P_Urban"] = probabilities[model]
        output[f"{model}_P_NonUrban"] = 1.0 - probabilities[model]
    output.to_csv(path, index=False)
    return output


def paired_bootstrap(y_true, predictions, replicates):
    rng = np.random.default_rng(RANDOM_SEED)
    indices_by_class = [np.flatnonzero(y_true == code) for code in (0, 1)]
    distributions = {
        model: {metric: np.empty(replicates) for metric in BOOTSTRAP_METRICS}
        for model in MODEL_GROUPS
    }
    progress = previous.SimpleProgress(replicates, "Binary paired bootstrap")
    for replicate in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in indices_by_class
        ])
        true_sample = y_true[sampled]
        for model, predicted in predictions.items():
            cm = np.bincount(
                2 * true_sample + predicted[sampled], minlength=4
            ).reshape(2, 2)
            metrics = binary_metrics_from_confusion(cm)
            for metric in BOOTSTRAP_METRICS:
                distributions[model][metric][replicate] = metrics[metric]
        progress.update()
    ci_rows = []
    for model in MODEL_GROUPS:
        original_cm = np.bincount(
            2 * y_true + predictions[model], minlength=4
        ).reshape(2, 2)
        observed = binary_metrics_from_confusion(original_cm)
        for metric in BOOTSTRAP_METRICS:
            values = distributions[model][metric]
            ci_rows.append({
                "Model": model,
                "Metric": metric,
                "ObservedValue": observed[metric],
                "BootstrapMean": float(np.mean(values)),
                "CI_Lower_2.5": float(np.percentile(values, 2.5)),
                "CI_Upper_97.5": float(np.percentile(values, 97.5)),
                "Replicates": replicates,
            })
    required = set(REQUIRED_COMPARISONS)
    pairs = [(right, left) for left, right in itertools.combinations(MODEL_GROUPS, 2)]
    difference_rows = []
    for first, second in pairs:
        for metric in BOOTSTRAP_METRICS:
            difference = distributions[first][metric] - distributions[second][metric]
            lower, upper = np.percentile(difference, [2.5, 97.5])
            difference_rows.append({
                "Comparison": f"{first} - {second}",
                "Model_1": first,
                "Model_2": second,
                "Metric": metric,
                "MeanDifference": float(np.mean(difference)),
                "CI_Lower_2.5": float(lower),
                "CI_Upper_97.5": float(upper),
                "CI_Includes_Zero": bool(lower <= 0 <= upper),
                "RequiredComparison": bool((first, second) in required),
                "Replicates": replicates,
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(difference_rows)


def marginal_contributions(performance):
    values = performance.set_index("Model")
    formulas = {
        "Geometry": [("B1", "B0"), ("B4", "B2"), ("B5", "B3"), ("B7", "B6")],
        "Texture": [("B2", "B0"), ("B4", "B1"), ("B6", "B3"), ("B7", "B5")],
        "SpatialContext": [("B3", "B0"), ("B5", "B1"), ("B6", "B2"), ("B7", "B4")],
    }
    rows = []
    for family, pairs in formulas.items():
        for metric in ("MacroF1", "mIoU", "Urban_F1", "Urban_IoU", "BalancedAccuracy"):
            components = [float(values.loc[a, metric] - values.loc[b, metric]) for a, b in pairs]
            row = {
                "FeatureFamily": family,
                "Metric": metric,
                "AverageMarginalContribution": float(np.mean(components)),
            }
            for position, ((a, b), value) in enumerate(zip(pairs, components), start=1):
                row[f"Component_{position}"] = f"{a}-{b}"
                row[f"Component_{position}_Difference"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_includes_zero(paired, first, second, metric):
    rows = paired[
        (
            ((paired["Model_1"] == first) & (paired["Model_2"] == second))
            | ((paired["Model_1"] == second) & (paired["Model_2"] == first))
        )
        & (paired["Metric"] == metric)
    ]
    return bool(not rows.empty and rows.iloc[0]["CI_Includes_Zero"])


def select_parsimonious_model(performance, paired, pilot):
    if pilot:
        best = performance.sort_values("MacroF1", ascending=False).iloc[0]["Model"]
        return best, "Pilot only: this is a provisional plotting model, not a final selection."
    values = performance.set_index("Model")
    best = values["MacroF1"].idxmax()
    criteria = ["Urban_F1", "Urban_IoU", "MacroF1", "BalancedAccuracy"]
    eligible = performance.copy()
    for metric in criteria:
        eligible = eligible[eligible[metric] >= values.loc[best, metric] - 0.01]
    eligible = eligible.sort_values(["NumberOfFeatures", "MacroF1"], ascending=[True, False])
    supported = []
    for candidate in eligible["Model"]:
        if candidate == best or all(
            comparison_includes_zero(paired, candidate, best, metric)
            for metric in criteria
        ):
            supported.append(candidate)
    selected = supported[0] if supported else best
    explanation = (
        "The selected model is the smallest configuration within 0.01 of the best-Macro-F1 "
        "model on Urban F1, Urban IoU, Macro-F1 and Balanced Accuracy, with paired intervals "
        "including zero relative to the best model."
    )
    return selected, explanation


def posthoc_three_class_comparison(path, full_test, new_predictions, bundle):
    if not path.is_file():
        return None, "Previous full three-class predictions were unavailable; comparison skipped."
    old = pd.read_csv(path)
    if "Object_ID" not in old.columns:
        return None, "Previous prediction file lacked Object_ID; comparison skipped."
    old["Object_ID"] = old["Object_ID"].astype("int64")
    full_lookup = full_test.set_index("Object_ID")
    common = pd.Index(old["Object_ID"]).intersection(full_lookup.index)
    if common.empty:
        return None, "No common manual test objects existed; comparison skipped."
    old_indexed = old.set_index("Object_ID").loc[common]
    test_positions = pd.Series(
        np.arange(len(full_test)), index=full_test["Object_ID"].to_numpy(dtype="int64")
    )
    positions = test_positions.loc[common].to_numpy(dtype="int64")
    y_true = full_lookup.loc[common, "BinaryLabel"].to_numpy(dtype="int64")
    rows = []
    for number in range(8):
        old_column = f"A{number}_PredictedClass"
        new_model = f"B{number}"
        if old_column not in old_indexed.columns:
            continue
        old_binary = (old_indexed[old_column].astype(str).str.lower() == "urban").astype("int64").to_numpy()
        new_binary = new_predictions[new_model][positions]
        for classifier, prediction in (
            ("PostHocThreeClass", old_binary),
            ("RetrainedBinary", new_binary),
        ):
            cm = bundle["confusion_matrix"](y_true, prediction, labels=[0, 1])
            metrics = binary_metrics_from_confusion(cm)
            rows.append({
                "FeatureConfiguration": f"{number}",
                "OldModel": f"A{number}",
                "NewModel": new_model,
                "Classifier": classifier,
                "CommonTestObjects": len(common),
                "Urban_Precision": metrics["Urban_Precision"],
                "Urban_Recall": metrics["Urban_Recall"],
                "Urban_F1": metrics["Urban_F1"],
                "Urban_IoU": metrics["Urban_IoU"],
                "BalancedAccuracy": metrics["BalancedAccuracy"],
                "MacroF1": metrics["MacroF1"],
            })
    result = pd.DataFrame(rows)
    return result, f"Compared on {len(common)} common manual objects without tuning."


def training_sampling_summary(enriched, selected, basic_summary):
    rows = []
    for source_class in SOURCE_CODES.values():
        before = enriched[enriched["SourceClass3"] == source_class]
        after = selected[selected["SourceClass3"] == source_class]
        base = basic_summary[basic_summary["SourceClass3"] == source_class].iloc[0].to_dict()
        for stage, frame in (("Candidate", before), ("Selected", after)):
            row = {
                **base,
                "Stage": stage,
                "Objects": int(len(frame)),
                "OccupiedSpatialCells": int(frame["SpatialCell"].nunique()),
                "Purity_Min": frame["Purity"].min(),
                "Purity_Median": frame["Purity"].median(),
                "Purity_Mean": frame["Purity"].mean(),
                "Coverage_Min": frame["Coverage"].min(),
                "Coverage_Median": frame["Coverage"].median(),
                "Coverage_Mean": frame["Coverage"].mean(),
            }
            for value in ("small", "medium", "large"):
                row[f"Size_{value}"] = int((frame["SizeStratum"] == value).sum())
            for value in ("low", "medium", "high"):
                row[f"Radiance_{value}"] = int((frame["RadianceStratum"] == value).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_training_class_distribution(plt, summary, path):
    pivot = summary.pivot(index="SourceClass3", columns="Stage", values="Objects")
    order = ["Urban", "Vegetation", "BareScrub"]
    pivot = pivot.loc[order]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(x - width / 2, pivot["Candidate"], width, label="Eligible candidates")
    ax.bar(x + width / 2, pivot["Selected"], width, label="Selected")
    ax.set_xticks(x, ["Urban", "Vegetation-derived\nNonUrban", "BareScrub-derived\nNonUrban"])
    ax.set_ylabel("Number of training objects")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_spatial_distribution(plt, selected, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    for ax, source_class in zip(axes, SOURCE_CODES.values()):
        subset = selected[selected["SourceClass3"] == source_class]
        cells = subset.groupby(["GridX", "GridY"], observed=True).size().reset_index(name="Count")
        sizes = 12 + 55 * np.sqrt(cells["Count"] / cells["Count"].max())
        scatter = ax.scatter(cells["GridX"], cells["GridY"], s=sizes, c=cells["Count"])
        ax.set_title(source_class)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("5-km grid X")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Selected objects")
    axes[0].set_ylabel("5-km grid Y")
    fig.suptitle("Spatial distribution of stratified training objects")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance(plt, performance, path):
    metrics = ["MacroF1", "mIoU", "Urban_IoU", "BalancedAccuracy"]
    x = np.arange(len(performance))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.7))
    for position, metric in enumerate(metrics):
        ax.bar(x + (position - 1.5) * width, performance[metric], width, label=metric)
    ax.set_xticks(x, performance["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Primary independent-test score")
    ax.set_xlabel("Binary ablation model")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_marginal(plt, marginal, path):
    metrics = ["MacroF1", "mIoU", "Urban_IoU", "BalancedAccuracy"]
    families = ["Geometry", "Texture", "SpatialContext"]
    x = np.arange(len(families))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for position, metric in enumerate(metrics):
        values = [
            marginal.loc[
                (marginal["FeatureFamily"] == family) & (marginal["Metric"] == metric),
                "AverageMarginalContribution",
            ].iloc[0]
            for family in families
        ]
        ax.bar(x + (position - 1.5) * width, values, width, label=metric)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["Geometry", "Texture", "Spatial context"])
    ax.set_ylabel("Average marginal score difference")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binary_confusion(plt, cm, model, path):
    normalized = np.divide(
        cm, cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype="float64"),
        where=cm.sum(axis=1, keepdims=True) > 0,
    )
    labels = ["NonUrban", "Urban"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, matrix, title, percent in (
        (axes[0], cm, "Raw counts", False),
        (axes[1], normalized * 100, "Row-normalized (%)", True),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        threshold = matrix.max() / 2 if matrix.max() else 0
        for row in range(2):
            for column in range(2):
                label = f"{matrix[row, column]:.1f}%" if percent else str(int(matrix[row, column]))
                ax.text(
                    column, row, label, ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        ax.set_xticks(range(2), labels)
        ax.set_yticks(range(2), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model} primary independent-test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, digits=4):
    return previous.markdown_table(frame, digits=digits)


def report_text(
    mode, args, input_info, test_diagnostics, primary_diagnostics,
    exclusion_diagnostics, target, sampling_summary, weights, fold_summary,
    cv_results, selected_parameters, cv_macro, cv_balanced,
    primary_performance, full_performance, marginal, bootstrap_ci, paired,
    selected_model, selection_explanation, old_comparison, old_comparison_note,
    nan_diagnostics, runtimes, versions, output_dir,
):
    groups = []
    for group in ("N", "G", "T", "C"):
        groups.append(
            f"### {group} ({len(FEATURE_GROUPS[group])})\n\n"
            + ", ".join(FEATURE_GROUPS[group])
        )
    required_paired = paired[paired["RequiredComparison"]][[
        "Comparison", "Metric", "MeanDifference", "CI_Lower_2.5",
        "CI_Upper_97.5", "CI_Includes_Zero",
    ]]
    comparison_section = (
        markdown_table(old_comparison)
        if old_comparison is not None and not old_comparison.empty
        else old_comparison_note
    )
    comparison_interpretation = "No matched interpretation was possible."
    if old_comparison is not None and not old_comparison.empty:
        number = selected_model.removeprefix("B")
        matched = old_comparison[old_comparison["FeatureConfiguration"].astype(str) == number]
        old_row = matched[matched["Classifier"] == "PostHocThreeClass"]
        new_row = matched[matched["Classifier"] == "RetrainedBinary"]
        if not old_row.empty and not new_row.empty:
            changes = {
                metric: float(new_row.iloc[0][metric] - old_row.iloc[0][metric])
                for metric in ("Urban_Precision", "Urban_Recall", "Urban_F1", "Urban_IoU")
            }
            comparison_interpretation = (
                f"For the matched {selected_model}/A{number} feature configuration, binary retraining "
                f"changed Urban Precision by {changes['Urban_Precision']:+.4f}, Recall by "
                f"{changes['Urban_Recall']:+.4f}, F1 by {changes['Urban_F1']:+.4f}, and IoU by "
                f"{changes['Urban_IoU']:+.4f} relative to post-hoc merging. "
                "Thus an aggregate binary gain is not called an Urban-delineation improvement unless "
                "Urban F1 and IoU also increase."
            )
    if mode == "PILOT":
        scientific = (
            "This pilot validates the redesigned data construction and modelling pipeline. It does "
            "not establish whether imbalance caused previous errors or whether a feature family "
            "should be retained. Those conclusions require the full five-fold/2,000-bootstrap run."
        )
    else:
        best = primary_performance.set_index("Model").loc[selected_model]
        scientific = (
            f"The parsimonious model {selected_model} achieved Urban F1={best['Urban_F1']:.4f}, "
            f"Urban IoU={best['Urban_IoU']:.4f}, Macro-F1={best['MacroF1']:.4f}, and "
            f"Balanced Accuracy={best['BalancedAccuracy']:.4f}. The source-class balancing "
            "removes BareScrub dominance by design, but causal attribution of any improvement "
            "requires the matched post-hoc comparison and its uncertainty. Remaining discrepancies "
            "must be interpreted jointly as feature-representation, pseudo-reference mismatch, "
            "and finite manual-test support rather than assigned to one cause without evidence."
        )
    lines = [
        f"# Binary Stratified Object Ablation XGBoost 01 — {mode}",
        "",
        "## A. Binary class definition",
        "",
        "Urban remains Urban. Vegetation and BareScrub are separately retained as SourceClass3 "
        "during sampling and then merged to NonUrban. Models are newly trained with binary:logistic; "
        "three-class predictions are never merged to create the new classifier.",
        "",
        "## B. Independent test construction",
        "",
        f"Manual points: {test_diagnostics['source_points']:,}; full non-conflicting pool: "
        f"{test_diagnostics['unique_nonconflicting_objects']:,}; binary conflicts: "
        f"{test_diagnostics['binary_conflicting_objects']}; unmatched labeled points: "
        f"{test_diagnostics['strictly_unmatched_labeled_points']}.",
        f"Primary set: Urban={primary_diagnostics['primary_urban']:,}, "
        f"NonUrban={primary_diagnostics['primary_nonurban']:,}, total={primary_diagnostics['primary_total']:,}, "
        f"occupied 5-km cells={primary_diagnostics['primary_occupied_cells']:,}.",
        f"Vegetation available={primary_diagnostics['available_vegetation']:,}; BareScrub sampled="
        f"{primary_diagnostics['selected_barescrub']:,} from {primary_diagnostics['available_barescrub']:,} "
        "using spatial-cell and size-tertile strata.",
        f"Full-pool source composition: Urban={primary_diagnostics['available_urban']:,}, "
        f"Vegetation={primary_diagnostics['available_vegetation']:,}, "
        f"BareScrub={primary_diagnostics['available_barescrub']:,}; binary composition: Urban="
        f"{primary_diagnostics['available_urban']:,}, NonUrban="
        f"{primary_diagnostics['available_vegetation'] + primary_diagnostics['available_barescrub']:,}.",
        "The complete pool is the secondary robustness test. Neither test set enters sampling, "
        "weighting, tuning, or threshold selection.",
        "",
        "## C. Training candidates",
        "",
        f"High-confidence objects before manual-test exclusion: "
        f"{exclusion_diagnostics['high_confidence_before_test_exclusion']:,}. Eligible after "
        f"complete-pool and Rook-neighbor exclusion: {exclusion_diagnostics['eligible_after_exclusion']:,}.",
        "",
        markdown_table(sampling_summary[sampling_summary["Stage"] == "Candidate"]),
        "",
        "## D. Stratified training design",
        "",
        f"N_target = {target:,}. Exactly {target:,} Urban, {target:,} Vegetation and {target:,} "
        "BareScrub objects were selected before binary merging. The final binary counts are "
        f"Urban={target:,} and NonUrban={2*target:,}.",
        "Spatial quotas first retained broad coverage across occupied 5-km cells. Within each cell, "
        "selection was distributed across size × PAN-radiance tertiles. Sparse strata automatically "
        "returned unused capacity to available strata inside the same source class.",
        "",
        "## E. Training distribution",
        "",
        markdown_table(sampling_summary),
        "",
        f"Binary class weights: NonUrban={weights[0]:.6f}; Urban={weights[1]:.6f}.",
        "",
        "## F. Train/test independence",
        "",
        f"Complete manual objects excluded: {input_info['full_test_objects']:,}; distinct non-test "
        f"first-order neighbors excluded: {input_info['rook_neighbor_objects']:,}. Training/full-test "
        f"overlap={input_info['training_test_overlap']}; training/neighbor overlap="
        f"{input_info['training_neighbor_overlap']}.",
        "",
        "## G. Feature groups",
        "",
        *groups,
        "",
        "The exact previous full manifest was validated. T contains only reduced BG-Log features; "
        "C excludes NTL_NormalizedContrast.",
        "",
        "## H. Spatial CV and hyperparameter tuning",
        "",
        f"Occupied training cells={input_info['training_spatial_cells']:,}; folds={len(fold_summary)}; "
        f"configurations={len(cv_results)}. Selected CV Macro-F1={cv_macro:.6f}; "
        f"Balanced Accuracy={cv_balanced:.6f}.",
        "",
        markdown_table(fold_summary),
        "",
        "Selected parameters:",
        "",
        "    " + json.dumps(selected_parameters, sort_keys=True),
        "",
        "## I. Primary binary ablation results",
        "",
        markdown_table(primary_performance),
        "",
        "## J. Full-test robustness results",
        "",
        markdown_table(full_performance),
        "",
        "## K. Urban performance",
        "",
        markdown_table(primary_performance[[
            "Model", "Urban_Precision", "Urban_Recall", "Urban_F1", "Urban_IoU",
        ]]),
        "",
        "## L. Factorial contributions",
        "",
        markdown_table(marginal[["FeatureFamily", "Metric", "AverageMarginalContribution"]]),
        "",
        "These values are descriptive factorial contributions, not causal effects.",
        "",
        "## M. Bootstrap uncertainty",
        "",
        f"Primary-test stratified paired bootstrap replicates: {int(bootstrap_ci['Replicates'].iloc[0])}.",
        "",
        markdown_table(required_paired),
        "",
        "## N. Three-class versus binary retraining",
        "",
        old_comparison_note,
        "",
        comparison_section,
        "",
        comparison_interpretation,
        "",
        "## O. Final parsimonious model",
        "",
        f"Selected/provisional model: {selected_model}. {selection_explanation}",
        "",
        "## P. Scientific interpretation",
        "",
        scientific,
        "",
        "## Q. Runtime",
        "",
        markdown_table(pd.DataFrame([{"Stage": key, "Seconds": value} for key, value in runtimes.items()])),
        "",
        "## R. Deviations",
        "",
        "- The final feature representation remains an Object_ID-validated join of Spatial Context 02 "
        "for N/G/C and Feature Extraction 05 texture-only for final BG-Log T.",
        "- The previous final feature manifest was used without changing any N/G/T/C definition.",
        (
            "- Pilot modelling used three spatial folds, three capped 120-tree configurations and "
            "200 bootstrap replicates. Sampling itself used the complete eligible candidate pool."
            if mode == "PILOT" else
            "- Full modelling used five folds, 24 unmodified configurations and 2,000 bootstrap replicates."
        ),
        f"- Non-texture NaN cells in selected training/primary/full test: "
        f"{nan_diagnostics['training_non_texture']:,}/"
        f"{nan_diagnostics['primary_non_texture']:,}/"
        f"{nan_diagnostics['full_non_texture']:,}. BG-Log NaNs were preserved.",
        f"- Software: Python {versions['python']}; XGBoost {versions['xgboost']}; "
        f"scikit-learn {versions['sklearn']}.",
        "- The fixed threshold was 0.50 for every model and was never optimized.",
        "- All B0–B7 models used identical training IDs, primary/full test IDs, binary weights, "
        "frozen hyperparameters and threshold. Test performance was not accessed during CV tuning.",
        "- Training sampling used source class, training-object location, size and PAN radiance only; "
        "the independently sampled test subset did not advance the training RNG stream.",
        "",
        "## Output directory",
        "",
        str(output_dir.resolve()),
    ]
    return "\n".join(lines) + "\n"


def main():
    total_started = time.perf_counter()
    args = parse_args()
    bundle = dependency_bundle()
    suffix = "_pilot" if args.pilot else ""
    if args.preflight_only:
        suffix = "_preflight"
    if args.output_dir is None:
        args.output_dir = SEGMENTATION_DIR / f"Binary_Stratified_Object_Ablation_XGBoost_01{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous.require_files([
        args.segmentation, args.master_reference, args.test_points, args.adjacency,
        args.context_table, args.texture_table, args.previous_manifest,
    ])

    previous_manifest = validate_previous_manifest(args.previous_manifest)
    save_binary_manifest(
        previous_manifest, args.previous_manifest,
        args.output_dir / "Binary_Ablation_Feature_Group_Manifest.csv",
    )

    stage_started = time.perf_counter()
    print("Loading master reference and final object features...")
    master = load_master_reference(args.master_reference)
    features = previous.load_feature_view(args.context_table, args.texture_table)
    load_seconds = time.perf_counter() - stage_started
    if set(master["Object_ID"]) != set(features["Object_ID"]):
        raise SystemExit("STOP: master reference and feature tables do not have identical Object_ID domains.")
    feature_index = features.set_index("Object_ID")

    stage_started = time.perf_counter()
    print("Reconstructing the independent manual test pool...")
    gpd = bundle["gpd"]
    segments = gpd.read_file(args.segmentation, columns=["Seg_id", "geometry"])
    if segments.crs is None:
        raise SystemExit("STOP: segmentation has no CRS.")
    segments["Seg_id"] = segments["Seg_id"].astype("int64")
    if segments["Seg_id"].duplicated().any():
        raise SystemExit("STOP: segmentation Seg_id is not unique.")
    if set(segments["Seg_id"]) != set(features["Object_ID"]):
        raise SystemExit("STOP: segmentation and features have different Object_ID domains.")
    segment_index = segments.set_index("Seg_id", drop=False)
    full_pool, test_diagnostics = map_manual_test_pool(gpd, args.test_points, segments)
    full_pool = full_pool.join(
        feature_index[["area_sqm"]], on="Object_ID", how="left", validate="one_to_one"
    )
    full_pool = add_centroid_cells(full_pool, segment_index)
    full_pool["SizeStratum"] = "not_used"
    full_pool.to_csv(args.output_dir / "Binary_Full_Manual_Test_Pool.csv", index=False)
    full_pool.to_csv(args.output_dir / "Binary_Secondary_Full_Test_Set.csv", index=False)

    # Independent deterministic streams prevent test-set sampling from advancing
    # or otherwise influencing the training-sampling random state.
    test_rng = np.random.default_rng(RANDOM_SEED)
    primary_test, primary_diagnostics = build_primary_test(full_pool, test_rng)
    primary_test.to_csv(
        args.output_dir / "Binary_Primary_Stratified_Test_Set.csv", index=False
    )

    neighbors = rook_neighbors(args.adjacency, full_pool["Object_ID"].to_numpy())
    candidates, exclusion_diagnostics = construct_training_candidates(
        master, feature_index, full_pool["Object_ID"].to_numpy(), neighbors
    )
    candidates = add_centroid_cells(candidates, segment_index)
    training_rng = np.random.default_rng(RANDOM_SEED)
    enriched_candidates, selected_training, target, basic_summary = (
        stratified_training_selection(candidates, training_rng)
    )
    sampling_summary = training_sampling_summary(
        enriched_candidates, selected_training, basic_summary
    )
    sampling_summary.to_csv(
        args.output_dir / "Binary_Training_Sampling_Summary.csv", index=False
    )
    selected_training[[
        "Object_ID", "SourceClass3", "BinaryClass", "BinaryLabel",
        "Purity", "Coverage", "N_labeled_pixels", "SpatialCell",
        "SizeStratum", "RadianceStratum", "area_sqm", "SDGSAT_Pan_mean",
    ]].to_csv(args.output_dir / "Binary_Stratified_Training_Set.csv", index=False)
    sampling_seconds = time.perf_counter() - stage_started

    full_ids = full_pool["Object_ID"].to_numpy(dtype="int64")
    training_ids = selected_training["Object_ID"].to_numpy(dtype="int64")
    primary_ids = primary_test["Object_ID"].to_numpy(dtype="int64")
    train_test_overlap = np.intersect1d(training_ids, full_ids)
    train_neighbor_overlap = np.intersect1d(training_ids, neighbors)
    if len(train_test_overlap) or len(train_neighbor_overlap):
        raise SystemExit(
            f"STOP: leakage detected: test overlap={len(train_test_overlap)}, "
            f"neighbor overlap={len(train_neighbor_overlap)}."
        )
    if selected_training["Object_ID"].duplicated().any():
        raise SystemExit("STOP: duplicate training Object_ID.")
    if primary_test["Object_ID"].duplicated().any() or full_pool["Object_ID"].duplicated().any():
        raise SystemExit("STOP: duplicate independent-test Object_ID.")

    y_train = selected_training["BinaryLabel"].to_numpy(dtype="int64")
    weights, sample_weight = binary_class_weights(y_train)
    groups = selected_training["SpatialCell"].to_numpy()
    folds = 3 if args.pilot else 5
    if len(np.unique(groups)) < folds:
        raise SystemExit("STOP: too few occupied spatial cells for GroupKFold.")

    all_features = [column for group in FEATURE_GROUPS.values() for column in group]
    non_texture = [column for column in all_features if column not in FEATURE_GROUPS["T"]]
    views = {
        "training": feature_index.loc[training_ids],
        "primary": feature_index.loc[primary_ids],
        "full": feature_index.loc[full_ids],
    }
    nan_diagnostics = {
        f"{name}_non_texture": int(view[non_texture].isna().sum().sum())
        for name, view in views.items()
    }
    nan_diagnostics.update({
        f"{name}_texture": int(view[FEATURE_GROUPS["T"]].isna().sum().sum())
        for name, view in views.items()
    })

    input_info = {
        "objects": int(len(features)),
        "full_test_objects": int(len(full_pool)),
        "primary_test_objects": int(len(primary_test)),
        "rook_neighbor_objects": int(len(neighbors)),
        "eligible_candidates": int(len(candidates)),
        "selected_training": int(len(selected_training)),
        "training_test_overlap": int(len(train_test_overlap)),
        "training_neighbor_overlap": int(len(train_neighbor_overlap)),
        "training_spatial_cells": int(selected_training["SpatialCell"].nunique()),
    }
    validation = {
        "inputs": {
            key: str(getattr(args, key).resolve()) for key in (
                "segmentation", "master_reference", "test_points", "adjacency",
                "context_table", "texture_table", "previous_manifest",
            )
        },
        "input_info": input_info,
        "test_diagnostics": test_diagnostics,
        "primary_test_diagnostics": primary_diagnostics,
        "training_exclusion": exclusion_diagnostics,
        "N_target": target,
        "binary_weights": {BINARY_LABELS[k]: v for k, v in weights.items()},
        "nan_diagnostics": nan_diagnostics,
        "feature_groups": FEATURE_GROUPS,
        "probability_threshold": THRESHOLD,
        "random_seed": RANDOM_SEED,
    }
    (args.output_dir / "Binary_Ablation_Input_Validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8"
    )

    plot_training_class_distribution(
        bundle["plt"], sampling_summary,
        args.output_dir / "Binary_Training_Class_Distribution.png",
    )
    plot_training_spatial_distribution(
        bundle["plt"], selected_training,
        args.output_dir / "Binary_Training_Spatial_Cell_Distribution.png",
    )

    if args.preflight_only:
        print("Binary preflight passed; no classifier was fitted.")
        print(f"Output: {args.output_dir.resolve()}")
        return 0

    configuration_count = args.pilot_configs if args.pilot else args.search_configs
    bootstrap_replicates = args.pilot_bootstrap if args.pilot else args.bootstrap_replicates
    configurations = previous.sampled_parameter_configurations(
        configuration_count, pilot=args.pilot
    )
    b7_columns = [column for group in MODEL_GROUPS["B7"] for column in FEATURE_GROUPS[group]]
    X_b7 = feature_index.loc[training_ids, b7_columns].to_numpy(dtype="float32", copy=True)

    stage_started = time.perf_counter()
    (
        selected_parameters, cv_macro, cv_balanced, cv_results, splits,
    ) = tune_b7(
        bundle, X_b7, y_train, groups, sample_weight,
        configurations, folds, args.n_jobs,
    )
    cv_seconds = time.perf_counter() - stage_started
    cv_results.to_csv(args.output_dir / "Binary_Ablation_Spatial_CV_Results.csv", index=False)
    fold_rows = []
    fold_assignment = np.full(len(selected_training), -1, dtype="int64")
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        fold_assignment[validation_index] = fold
        fold_rows.append({
            "Fold": fold,
            "TrainingObjects": len(train_index),
            "ValidationObjects": len(validation_index),
            "ValidationSpatialCells": len(np.unique(groups[validation_index])),
            "ValidationUrban": int((y_train[validation_index] == 1).sum()),
            "ValidationNonUrban": int((y_train[validation_index] == 0).sum()),
        })
    fold_summary = pd.DataFrame(fold_rows)
    spatial_folds = selected_training[["Object_ID", "SpatialCell"]].copy()
    spatial_folds["ValidationFold"] = fold_assignment
    spatial_folds.to_csv(
        args.output_dir / "Binary_Ablation_Training_Spatial_Groups.csv", index=False
    )

    stage_started = time.perf_counter()
    (
        predictions, probabilities, evaluations, final_weights,
        final_training_seconds, prediction_seconds,
    ) = fit_models(
        bundle, features, selected_training, primary_test, full_pool,
        selected_parameters, args.n_jobs,
    )
    fit_total_seconds = time.perf_counter() - stage_started
    primary_performance, primary_classwise, primary_confusion = evaluation_tables(
        evaluations["primary"], "PrimaryStratified"
    )
    full_performance, full_classwise, full_confusion = evaluation_tables(
        evaluations["secondary"], "SecondaryFull"
    )
    primary_performance.to_csv(
        args.output_dir / "Binary_Ablation_Primary_Performance.csv", index=False
    )
    full_performance.to_csv(
        args.output_dir / "Binary_Ablation_FullTest_Performance.csv", index=False
    )
    pd.concat([primary_classwise, full_classwise]).to_csv(
        args.output_dir / "Binary_Ablation_Classwise_Performance.csv", index=False
    )
    pd.concat([primary_confusion, full_confusion]).to_csv(
        args.output_dir / "Binary_Ablation_Confusion_Matrices.csv", index=False
    )
    primary_predictions_table = save_predictions(
        args.output_dir / "Binary_Ablation_Primary_Test_Predictions.csv",
        primary_test, predictions["primary"], probabilities["primary"],
    )
    full_predictions_table = save_predictions(
        args.output_dir / "Binary_Ablation_Full_Test_Predictions.csv",
        full_pool, predictions["secondary"], probabilities["secondary"],
    )

    stage_started = time.perf_counter()
    bootstrap_ci, paired = paired_bootstrap(
        primary_test["BinaryLabel"].to_numpy(dtype="int64"),
        predictions["primary"], bootstrap_replicates,
    )
    bootstrap_seconds = time.perf_counter() - stage_started
    bootstrap_ci.to_csv(
        args.output_dir / "Binary_Ablation_Bootstrap_CI.csv", index=False
    )
    paired.to_csv(
        args.output_dir / "Binary_Ablation_Paired_Bootstrap_Differences.csv", index=False
    )
    marginal = marginal_contributions(primary_performance)
    marginal.to_csv(
        args.output_dir / "Binary_Ablation_Marginal_Contributions.csv", index=False
    )

    selected_model, selection_explanation = select_parsimonious_model(
        primary_performance, paired, args.pilot
    )
    old_comparison, old_comparison_note = posthoc_three_class_comparison(
        args.previous_three_class_predictions,
        full_pool,
        predictions["secondary"],
        bundle,
    )
    if old_comparison is not None:
        old_comparison.to_csv(
            args.output_dir / "Binary_vs_PostHoc_ThreeClass_Comparison.csv", index=False
        )

    hyperparameters = {
        "random_state": RANDOM_SEED,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "fixed_probability_threshold": THRESHOLD,
        "selection_metric": "mean spatial-CV Macro-F1",
        "secondary_cv_metric": "Balanced Accuracy",
        "spatial_cv_folds": folds,
        "tested_configuration_count": len(configurations),
        "selected_validation_macro_f1": cv_macro,
        "selected_validation_balanced_accuracy": cv_balanced,
        "selected_hyperparameters": selected_parameters,
        "frozen_for_models": list(MODEL_GROUPS),
        "provisional_or_final_parsimonious_model": selected_model,
    }
    (args.output_dir / "Binary_Ablation_Selected_Hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2), encoding="utf-8"
    )

    plot_performance(
        bundle["plt"], primary_performance,
        args.output_dir / "Binary_Ablation_Performance_Comparison.png",
    )
    plot_marginal(
        bundle["plt"], marginal,
        args.output_dir / "Binary_Ablation_Marginal_Contributions.png",
    )
    plot_binary_confusion(
        bundle["plt"], evaluations["primary"]["B0"]["confusion"], "B0",
        args.output_dir / "B0_Primary_Confusion_Matrix.png",
    )
    plot_binary_confusion(
        bundle["plt"], evaluations["primary"][selected_model]["confusion"], selected_model,
        args.output_dir / f"{selected_model}_Primary_Confusion_Matrix.png",
    )

    versions = {
        "python": sys.version.split()[0],
        "xgboost": bundle["xgb"].__version__,
        "sklearn": importlib.import_module("sklearn").__version__,
    }
    total_seconds = time.perf_counter() - total_started
    runtimes = {
        "Feature and master loading": load_seconds,
        "Test construction and stratified sampling": sampling_seconds,
        "Spatial CV tuning": cv_seconds,
        "Final model training": final_training_seconds,
        "Primary and full-test prediction": prediction_seconds,
        "Model orchestration overhead": max(
            0.0, fit_total_seconds - final_training_seconds - prediction_seconds
        ),
        "Paired bootstrap": bootstrap_seconds,
        "Total": total_seconds,
    }
    report_name = (
        "Binary_Stratified_Object_Ablation_XGBoost_01_Report_PILOT.md"
        if args.pilot else "Binary_Stratified_Object_Ablation_XGBoost_01_Report.md"
    )
    report = report_text(
        "PILOT" if args.pilot else "FULL RUN",
        args, input_info, test_diagnostics, primary_diagnostics,
        exclusion_diagnostics, target, sampling_summary, weights, fold_summary,
        cv_results, selected_parameters, cv_macro, cv_balanced,
        primary_performance, full_performance, marginal, bootstrap_ci, paired,
        selected_model, selection_explanation, old_comparison, old_comparison_note,
        nan_diagnostics, runtimes, versions, args.output_dir,
    )
    (args.output_dir / report_name).write_text(report, encoding="utf-8")
    print(f"Completed {'pilot' if args.pilot else 'full binary run'}.")
    print(f"Report: {(args.output_dir / report_name).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
