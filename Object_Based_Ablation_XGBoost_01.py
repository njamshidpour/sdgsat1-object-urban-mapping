#!/usr/bin/env python
"""Spatially independent XGBoost ablation for object-based NTL classification.

The full run performs the requested 2^3 factorial ablation of geometry (G),
BG-log texture (T), and spatial context (C) on top of the common NTL baseline
(N).  A deliberately lighter pilot mode validates the complete workflow
without producing final research conclusions.

No imagery is read and no feature or training label is recomputed.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20261001
PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"

DEFAULT_CONTEXT_TABLE = (
    SEGMENTATION_DIR
    / "Spatial_Context_Extraction_02"
    / "objects_features_with_spatial_context_02.csv"
)
DEFAULT_TEXTURE_TABLE = SEGMENTATION_DIR / "objects_features_v05_texture_only_full.csv"
DEFAULT_TRAINING_LABELS = (
    SEGMENTATION_DIR
    / "Object_Labeling_From_DW_01"
    / "Object_Training_Labels_DW_HighConfidence_01.csv"
)
DEFAULT_TEST_POINTS = PROJECT_ROOT / "GTs" / "Test_Points_1000.shp"
DEFAULT_SEGMENTATION = SEGMENTATION_DIR / "eCog_segments.shp"
DEFAULT_ADJACENCY = SEGMENTATION_DIR / "rook_adjacency_graph_v01.npz"

CLASS_CODES = [1, 2, 3]
CLASS_NAMES = {1: "Urban", 2: "Vegetation", 3: "BareScrub"}
METRIC_NAMES = ["OA", "MacroF1", "mIoU", "Urban_F1", "Urban_IoU"]

# Explicit, reviewable feature definitions.  No substring matching is used.
NTL_FEATURES = [
    "SDGSAT_Pan_mean", "SDGSAT_Pan_std", "SDGSAT_Pan_max",
    "SDGSAT_R_mean", "SDGSAT_R_std", "SDGSAT_R_max",
    "SDGSAT_G_mean", "SDGSAT_G_std", "SDGSAT_G_max",
    "SDGSAT_B_mean", "SDGSAT_B_std", "SDGSAT_B_max",
    "BR_mean", "BR_std", "BR_max",
    "ENUI_mean", "ENUI_std", "ENUI_max",
    "NBDI_mean", "NBDI_std", "NBDI_max",
    "NDIBG_mean", "NDIBG_std", "NDIBG_max",
    "NDIGR_mean", "NDIGR_std", "NDIGR_max",
    "NLCSI_mean", "NLCSI_std", "NLCSI_max",
    "SONDI_mean", "SONDI_std", "SONDI_max",
    "TVANUI_mean", "TVANUI_std", "TVANUI_max",
    "VANUI_mean", "VANUI_std", "VANUI_max",
]

# ShapeIndex is excluded because shape_index = 1/sqrt(compactness).  The prior
# report documented this identity but did not actually nominate one descriptor.
# Compactness is retained because it is the base descriptor used in the earlier
# extraction scripts; the choice and this minor workflow deviation are reported.
GEOMETRY_FEATURES = [
    "area_sqm", "perimeter_m", "compactness", "elongation",
    "length_width_ratio",
]

TEXTURE_FEATURES = [
    "SDGSAT_Pan_bgglcm_contrast",
    "SDGSAT_Pan_bgglcm_homogeneity",
    "SDGSAT_Pan_bgglcm_entropy",
]

CONTEXT_FEATURES = [
    "NTL_NeighborMean",
    "NTL_NeighborStd",
    "NTL_LocalContrast",
    "NTL_MeanAbsNeighborContrast",
    "NeighborCount",
]

FEATURE_GROUPS = {
    "N": NTL_FEATURES,
    "G": GEOMETRY_FEATURES,
    "T": TEXTURE_FEATURES,
    "C": CONTEXT_FEATURES,
}

MODEL_GROUPS = OrderedDict([
    ("A0", ("N",)),
    ("A1", ("N", "G")),
    ("A2", ("N", "T")),
    ("A3", ("N", "C")),
    ("A4", ("N", "G", "T")),
    ("A5", ("N", "G", "C")),
    ("A6", ("N", "T", "C")),
    ("A7", ("N", "G", "T", "C")),
])

REQUIRED_PAIRED_COMPARISONS = [
    ("A1", "A0"), ("A2", "A0"), ("A3", "A0"),
    ("A4", "A1"), ("A4", "A2"),
    ("A5", "A1"), ("A5", "A3"),
    ("A6", "A2"), ("A6", "A3"),
    ("A7", "A4"), ("A7", "A5"), ("A7", "A6"),
    ("A7", "A0"),
]

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


def dependency_bundle():
    """Import mandatory third-party packages with an explicit failure."""
    missing = []
    for package in ("xgboost", "sklearn", "geopandas", "matplotlib"):
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    if missing:
        raise SystemExit(
            "STOP: mandatory package(s) unavailable: " + ", ".join(missing)
            + ". XGBoost will not be replaced by another classifier."
        )

    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import GroupKFold
    return {
        "gpd": gpd, "plt": plt, "xgb": xgb,
        "accuracy_score": accuracy_score,
        "cohen_kappa_score": cohen_kappa_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "GroupKFold": GroupKFold,
    }


class SimpleProgress:
    """A clean, single-line progress indicator without terminal log flooding."""

    def __init__(self, total: int, label: str):
        self.total = max(int(total), 1)
        self.label = label
        self.current = 0
        self.started = time.perf_counter()
        self.last_drawn = self.started
        self._draw()

    def _draw(self):
        fraction = min(self.current / self.total, 1.0)
        width = 28
        done = int(round(width * fraction))
        bar = "#" * done + "-" * (width - done)
        elapsed = time.perf_counter() - self.started
        print(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} "
            f"({100*fraction:5.1f}%) {elapsed:,.1f}s",
            end="", flush=True,
        )

    def update(self, amount: int = 1):
        self.current += amount
        now = time.perf_counter()
        # Fast loops (especially bootstrap) must not create hundreds of terminal
        # lines in consoles that do not render carriage returns in place.
        if self.current >= self.total or (now - self.last_drawn) >= 0.5:
            self._draw()
            self.last_drawn = now
        if self.current >= self.total:
            print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eight-model object-based XGBoost feature-family ablation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Run a lightweight validation pilot.")
    mode.add_argument("--full-run", action="store_true", help="Run the complete research analysis.")
    mode.add_argument("--preflight-only", action="store_true", help="Validate all inputs; fit no model.")
    parser.add_argument("--context-table", type=Path, default=DEFAULT_CONTEXT_TABLE)
    parser.add_argument("--texture-table", type=Path, default=DEFAULT_TEXTURE_TABLE)
    parser.add_argument("--training-labels", type=Path, default=DEFAULT_TRAINING_LABELS)
    parser.add_argument("--test-points", type=Path, default=DEFAULT_TEST_POINTS)
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--search-configs", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--pilot-train-size", type=int, default=9000)
    parser.add_argument("--pilot-configs", type=int, default=3)
    parser.add_argument("--pilot-bootstrap", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit("STOP: required input file(s) missing:\n- " + "\n- ".join(missing))


def read_header(path: Path):
    return list(pd.read_csv(path, nrows=0).columns)


def validate_feature_headers(context_path: Path, texture_path: Path):
    context_columns = read_header(context_path)
    texture_columns = read_header(texture_path)
    primary_required = ["Object_ID", *NTL_FEATURES, *GEOMETRY_FEATURES, *CONTEXT_FEATURES]
    missing_primary = [c for c in primary_required if c not in context_columns]
    missing_texture = [c for c in ["Object_ID", *TEXTURE_FEATURES] if c not in texture_columns]
    if missing_primary:
        raise SystemExit(
            "STOP: canonical spatial-context table is missing required columns:\n- "
            + "\n- ".join(missing_primary)
        )
    if missing_texture:
        raise SystemExit(
            "STOP: final full-population BG-Log texture is unavailable. Missing columns:\n- "
            + "\n- ".join(missing_texture)
            + "\nConventional SDGSAT_Pan_glcm_* columns will NOT be substituted."
        )
    forbidden = {
        "SDGSAT_Pan_glcm_contrast",
        "SDGSAT_Pan_glcm_homogeneity",
        "SDGSAT_Pan_glcm_entropy",
    }
    if any(c in TEXTURE_FEATURES for c in forbidden):
        raise SystemExit("STOP: internal texture manifest incorrectly references conventional GLCM.")
    return context_columns, texture_columns


def load_feature_view(context_path: Path, texture_path: Path):
    validate_feature_headers(context_path, texture_path)
    primary_columns = ["Object_ID", *NTL_FEATURES, *GEOMETRY_FEATURES, *CONTEXT_FEATURES]
    texture_columns = ["Object_ID", *TEXTURE_FEATURES]
    primary = pd.read_csv(context_path, usecols=primary_columns)
    texture = pd.read_csv(texture_path, usecols=texture_columns)
    for name, frame in (("context", primary), ("texture", texture)):
        if frame["Object_ID"].isna().any():
            raise SystemExit(f"STOP: {name} table contains missing Object_ID values.")
        frame["Object_ID"] = frame["Object_ID"].astype("int64")
        if frame["Object_ID"].duplicated().any():
            raise SystemExit(f"STOP: {name} table contains duplicate Object_ID values.")
    primary_ids = pd.Index(primary["Object_ID"])
    texture_ids = pd.Index(texture["Object_ID"])
    if len(primary) != len(texture) or set(primary_ids) != set(texture_ids):
        raise SystemExit(
            "STOP: context and BG-Log tables do not have the same full Object_ID domain."
        )
    features = primary.merge(texture, on="Object_ID", how="left", validate="one_to_one")
    expected = 1 + sum(len(v) for v in FEATURE_GROUPS.values())
    if features.shape[1] != expected:
        raise SystemExit(
            f"STOP: resolved feature view has {features.shape[1]} columns; expected {expected}."
        )
    return features


def load_training_labels(path: Path):
    required = [
        "Object_ID", "FinalTrainingLabel", "FinalTrainingClass",
        "Purity", "Coverage", "N_labeled_pixels",
    ]
    data = pd.read_csv(path)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise SystemExit("STOP: training-label columns missing: " + ", ".join(missing))
    data = data[required].copy()
    data["Object_ID"] = data["Object_ID"].astype("int64")
    data["FinalTrainingLabel"] = data["FinalTrainingLabel"].astype("int64")
    if data["Object_ID"].duplicated().any():
        raise SystemExit("STOP: training-label file contains duplicate Object_ID values.")
    invalid_threshold = (
        (data["Purity"] < 0.90)
        | (data["Coverage"] < 0.50)
        | (data["N_labeled_pixels"] < 5)
    )
    if invalid_threshold.any():
        raise SystemExit(
            f"STOP: {int(invalid_threshold.sum())} rows violate fixed training thresholds."
        )
    if set(data["FinalTrainingLabel"].unique()) != set(CLASS_CODES):
        raise SystemExit("STOP: training labels are not exactly the adopted classes 1, 2, 3.")
    return data


def manual_label_to_three_class(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "").replace("/", "")
        if "urban" in normalized or "built" in normalized:
            return 1
        if "veget" in normalized:
            return 2
        if "bare" in normalized or "scrub" in normalized:
            return 3
        try:
            value = float(value)
        except ValueError:
            return np.nan
    numeric = int(value)
    return {1: 1, 2: 2, 3: 3, 4: 3}.get(numeric, np.nan)


def build_test_objects(gpd, test_points_path: Path, segments):
    points = gpd.read_file(test_points_path)
    if points.crs is None:
        raise SystemExit("STOP: test points have no CRS.")
    label_candidates = [
        c for c in ("Class_ID", "Class", "class_id", "class", "Label", "label")
        if c in points.columns
    ]
    if not label_candidates:
        raise SystemExit("STOP: no recognizable manual class field exists in test points.")
    label_field = label_candidates[0]
    points = points.to_crs(segments.crs)
    points = points.reset_index(names="SourcePointIndex")
    joined = gpd.sjoin(
        points[["SourcePointIndex", label_field, "geometry"]],
        segments[["Seg_id", "geometry"]],
        how="left",
        predicate="within",
    )
    joined["MappedClass"] = joined[label_field].map(manual_label_to_three_class)
    valid = joined.dropna(subset=["Seg_id", "MappedClass"]).copy()
    valid["Object_ID"] = valid["Seg_id"].astype("int64")
    valid["MappedClass"] = valid["MappedClass"].astype("int64")
    class_sets = valid.groupby("Object_ID")["MappedClass"].agg(lambda x: tuple(sorted(set(x))))
    conflicting_ids = class_sets[class_sets.map(len) > 1].index.astype("int64").tolist()
    consistent = class_sets[class_sets.map(len) == 1]
    test = pd.DataFrame({
        "Object_ID": consistent.index.astype("int64"),
        "TrueLabel": [values[0] for values in consistent],
    }).sort_values("Object_ID").reset_index(drop=True)
    diagnostics = {
        "source_points": int(len(points)),
        "label_field": label_field,
        "unlabeled_or_unknown_points": int(points[label_field].map(manual_label_to_three_class).isna().sum()),
        "strictly_unmatched_labeled_points": int(
            joined.loc[joined["MappedClass"].notna(), "Seg_id"].isna().sum()
        ),
        "valid_point_object_assignments": int(len(valid)),
        "duplicate_point_assignments_removed": int(len(valid) - valid["Object_ID"].nunique()),
        "conflicting_test_objects": int(len(conflicting_ids)),
        "conflicting_object_ids": conflicting_ids,
        "unique_test_objects": int(len(test)),
        "class_counts": {
            CLASS_NAMES[k]: int((test["TrueLabel"] == k).sum()) for k in CLASS_CODES
        },
    }
    return test, diagnostics


def validate_graph_exclusion(adjacency_path: Path, training_ids, test_ids):
    graph = np.load(adjacency_path, allow_pickle=False)
    required = {"src_object_id", "dst_object_id"}
    if not required.issubset(graph.files):
        raise SystemExit("STOP: adjacency graph lacks src_object_id/dst_object_id arrays.")
    src = graph["src_object_id"].astype("int64", copy=False)
    dst = graph["dst_object_id"].astype("int64", copy=False)
    test_ids = np.asarray(test_ids, dtype="int64")
    src_is_test = np.isin(src, test_ids)
    dst_is_test = np.isin(dst, test_ids)
    neighbors = np.unique(np.concatenate([dst[src_is_test], src[dst_is_test]]))
    neighbors = neighbors[~np.isin(neighbors, test_ids)]
    overlap = np.intersect1d(np.asarray(training_ids, dtype="int64"), neighbors)
    return {
        "test_neighbor_count": int(len(neighbors)),
        "training_test_neighbor_overlap_count": int(len(overlap)),
        "overlap_example_ids": overlap[:20].astype(int).tolist(),
    }


def stratified_pilot_sample(labels: pd.DataFrame, target_size: int):
    if target_size >= len(labels):
        return labels.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    pieces = []
    remaining = target_size
    counts = labels["FinalTrainingLabel"].value_counts().sort_index()
    for position, code in enumerate(CLASS_CODES):
        group = labels[labels["FinalTrainingLabel"] == code]
        if position == len(CLASS_CODES) - 1:
            amount = min(remaining, len(group))
        else:
            amount = max(1, int(round(target_size * counts[code] / len(labels))))
            amount = min(amount, len(group), remaining - (len(CLASS_CODES) - position - 1))
        chosen = rng.choice(group.index.to_numpy(), size=amount, replace=False)
        pieces.append(labels.loc[chosen])
        remaining -= amount
    sample = pd.concat(pieces, ignore_index=True)
    return sample.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def spatial_groups_for_training(segments, training_ids):
    segment_index = segments.set_index("Seg_id", drop=False)
    missing = pd.Index(training_ids).difference(segment_index.index)
    if len(missing):
        raise SystemExit(f"STOP: {len(missing)} training objects are absent from segmentation.")
    selected = segment_index.loc[np.asarray(training_ids, dtype="int64")].copy()
    metric = selected.to_crs("EPSG:32639")
    centroids = metric.geometry.centroid
    gx = np.floor(centroids.x.to_numpy() / 5000.0).astype("int64")
    gy = np.floor(centroids.y.to_numpy() / 5000.0).astype("int64")
    groups = np.asarray([f"{x}_{y}" for x, y in zip(gx, gy)], dtype=object)
    return groups, centroids.x.to_numpy(), centroids.y.to_numpy()


def class_weights(y_codes):
    y_codes = np.asarray(y_codes, dtype="int64")
    total = len(y_codes)
    weights = {}
    for code in CLASS_CODES:
        count = int((y_codes == code).sum())
        if count == 0:
            raise SystemExit(f"STOP: training class {code} has zero objects.")
        weights[code] = total / (len(CLASS_CODES) * count)
    sample_weight = np.asarray([weights[int(v)] for v in y_codes], dtype="float64")
    return weights, sample_weight


def sampled_parameter_configurations(number: int, pilot: bool):
    keys = list(PARAMETER_SPACE)
    combinations = list(itertools.product(*(PARAMETER_SPACE[k] for k in keys)))
    if number > len(combinations):
        raise SystemExit("STOP: requested more configurations than the parameter space contains.")
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(len(combinations), size=number, replace=False)
    configurations = [dict(zip(keys, combinations[i])) for i in chosen]
    if pilot:
        # Runtime-only pilot deviation; full-run values remain untouched.
        for config in configurations:
            config["n_estimators"] = min(int(config["n_estimators"]), 120)
    return configurations


def make_estimator(xgb, parameters, n_jobs):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        missing=np.nan,
        verbosity=0,
        **parameters,
    )


def tune_a7(bundle, X, y_zero, groups, sample_weight, configurations, n_splits, n_jobs):
    splitter = bundle["GroupKFold"](n_splits=n_splits)
    splits = list(splitter.split(X, y_zero, groups=groups))
    progress = SimpleProgress(len(configurations) * len(splits), "Spatial CV tuning")
    rows = []
    for config_id, parameters in enumerate(configurations, start=1):
        fold_scores = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            model = make_estimator(bundle["xgb"], parameters, n_jobs)
            model.fit(
                X[train_index], y_zero[train_index],
                sample_weight=sample_weight[train_index],
            )
            prediction = model.predict(X[validation_index]).astype("int64")
            score = bundle["f1_score"](
                y_zero[validation_index], prediction, average="macro", zero_division=0
            )
            fold_scores.append(float(score))
            progress.update()
        row = {
            "Configuration": config_id,
            **parameters,
            **{f"Fold_{i+1}_MacroF1": v for i, v in enumerate(fold_scores)},
            "Mean_MacroF1": float(np.mean(fold_scores)),
            "Std_MacroF1": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        }
        rows.append(row)
    results = pd.DataFrame(rows)
    best_score = float(results["Mean_MacroF1"].max())
    # Scores within 1e-4 are treated as practically tied; prefer shallower and
    # then smaller models without consulting the independent test set.
    tied = results[results["Mean_MacroF1"] >= best_score - 1e-4].sort_values(
        ["max_depth", "n_estimators", "Mean_MacroF1"],
        ascending=[True, True, False],
    )
    selected_row = tied.iloc[0]
    selected_parameters = {
        key: type(PARAMETER_SPACE[key][0])(selected_row[key]) for key in PARAMETER_SPACE
    }
    selected_score = float(selected_row["Mean_MacroF1"])
    return selected_parameters, selected_score, results, splits


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype="float64")
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(3),
            where=(precision + recall) > 0,
        )
        iou = np.divide(
            tp,
            support + predicted - tp,
            out=np.zeros(3),
            where=(support + predicted - tp) > 0,
        )
    return {
        "OA": float(tp.sum() / total) if total else np.nan,
        "MacroF1": float(np.mean(f1)),
        "mIoU": float(np.mean(iou)),
        "Urban_F1": float(f1[0]),
        "Urban_IoU": float(iou[0]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": support.astype("int64"),
    }


def evaluate_predictions(bundle, y_true_zero, y_pred_zero):
    cm = bundle["confusion_matrix"](y_true_zero, y_pred_zero, labels=[0, 1, 2])
    compact = metrics_from_confusion(cm)
    compact["Kappa"] = float(bundle["cohen_kappa_score"](y_true_zero, y_pred_zero))
    compact["confusion"] = cm
    return compact


def fit_all_models(bundle, feature_frame, train_ids, y_train_codes, test_ids,
                   y_test_codes, selected_parameters, n_jobs):
    indexed = feature_frame.set_index("Object_ID", drop=True)
    X_train_all = indexed.loc[np.asarray(train_ids, dtype="int64")]
    X_test_all = indexed.loc[np.asarray(test_ids, dtype="int64")]
    y_train_zero = np.asarray(y_train_codes, dtype="int64") - 1
    y_test_zero = np.asarray(y_test_codes, dtype="int64") - 1
    weights, sample_weight = class_weights(y_train_codes)
    predictions = {}
    probabilities = {}
    evaluations = {}
    training_seconds = 0.0
    prediction_seconds = 0.0
    progress = SimpleProgress(len(MODEL_GROUPS), "Final ablation models")
    for model_name, group_names in MODEL_GROUPS.items():
        columns = [column for group in group_names for column in FEATURE_GROUPS[group]]
        X_train = X_train_all[columns].to_numpy(dtype="float32", copy=True)
        X_test = X_test_all[columns].to_numpy(dtype="float32", copy=True)
        model = make_estimator(bundle["xgb"], selected_parameters, n_jobs)
        started = time.perf_counter()
        model.fit(X_train, y_train_zero, sample_weight=sample_weight)
        training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1).astype("int64")
        prediction_seconds += time.perf_counter() - started
        if not np.all(np.isfinite(proba)) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise SystemExit(f"STOP: invalid class probabilities produced by {model_name}.")
        predictions[model_name] = pred
        probabilities[model_name] = proba
        evaluations[model_name] = evaluate_predictions(bundle, y_test_zero, pred)
        progress.update()
    return (
        predictions, probabilities, evaluations, weights,
        training_seconds, prediction_seconds,
    )


def performance_tables(evaluations):
    model_rows = []
    class_rows = []
    confusion_rows = []
    for model, result in evaluations.items():
        groups = MODEL_GROUPS[model]
        number_features = sum(len(FEATURE_GROUPS[g]) for g in groups)
        model_rows.append({
            "Model": model,
            "FeatureGroups": "+".join(groups),
            "NumberOfFeatures": number_features,
            "OA": result["OA"],
            "Kappa": result["Kappa"],
            "MacroF1": result["MacroF1"],
            "mIoU": result["mIoU"],
            "Urban_Precision": result["precision"][0],
            "Urban_Recall": result["recall"][0],
            "Urban_F1": result["Urban_F1"],
            "Urban_IoU": result["Urban_IoU"],
        })
        for index, code in enumerate(CLASS_CODES):
            class_rows.append({
                "Model": model,
                "Class": CLASS_NAMES[code],
                "Precision": result["precision"][index],
                "Recall": result["recall"][index],
                "F1": result["f1"][index],
                "IoU": result["iou"][index],
                "Support": result["support"][index],
            })
        for true_index, true_code in enumerate(CLASS_CODES):
            for pred_index, pred_code in enumerate(CLASS_CODES):
                confusion_rows.append({
                    "Model": model,
                    "TrueClass": CLASS_NAMES[true_code],
                    "PredictedClass": CLASS_NAMES[pred_code],
                    "Count": int(result["confusion"][true_index, pred_index]),
                })
    return pd.DataFrame(model_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def bootstrap_analysis(y_true_zero, predictions, replicates):
    rng = np.random.default_rng(RANDOM_SEED)
    class_indices = [np.flatnonzero(y_true_zero == value) for value in range(3)]
    distributions = {
        model: {metric: np.empty(replicates, dtype="float64") for metric in METRIC_NAMES}
        for model in MODEL_GROUPS
    }
    progress = SimpleProgress(replicates, "Paired stratified bootstrap")
    for replicate in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in class_indices
        ])
        true_sample = y_true_zero[sampled]
        for model, predicted in predictions.items():
            cm = np.bincount(
                3 * true_sample + predicted[sampled], minlength=9
            ).reshape(3, 3)
            metrics = metrics_from_confusion(cm)
            for metric in METRIC_NAMES:
                distributions[model][metric][replicate] = metrics[metric]
        progress.update()

    ci_rows = []
    for model in MODEL_GROUPS:
        observed_cm = np.bincount(
            3 * y_true_zero + predictions[model], minlength=9
        ).reshape(3, 3)
        observed = metrics_from_confusion(observed_cm)
        for metric in METRIC_NAMES:
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

    required_set = set(REQUIRED_PAIRED_COMPARISONS)
    all_pairs = list(itertools.combinations(MODEL_GROUPS.keys(), 2))
    comparison_pairs = [(right, left) for left, right in all_pairs]
    difference_rows = []
    for first, second in comparison_pairs:
        for metric in METRIC_NAMES:
            differences = distributions[first][metric] - distributions[second][metric]
            lower, upper = np.percentile(differences, [2.5, 97.5])
            difference_rows.append({
                "Comparison": f"{first} - {second}",
                "Model_1": first,
                "Model_2": second,
                "Metric": metric,
                "MeanDifference": float(np.mean(differences)),
                "CI_Lower_2.5": float(lower),
                "CI_Upper_97.5": float(upper),
                "CI_Includes_Zero": bool(lower <= 0 <= upper),
                "RequiredComparison": bool((first, second) in required_set),
                "Replicates": replicates,
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(difference_rows), distributions


def marginal_contributions(performance):
    values = performance.set_index("Model")
    formulas = {
        "Geometry": [("A1", "A0"), ("A4", "A2"), ("A5", "A3"), ("A7", "A6")],
        "Texture": [("A2", "A0"), ("A4", "A1"), ("A6", "A3"), ("A7", "A5")],
        "SpatialContext": [("A3", "A0"), ("A5", "A1"), ("A6", "A2"), ("A7", "A4")],
    }
    rows = []
    for family, pairs in formulas.items():
        for metric in ("MacroF1", "mIoU", "Urban_F1", "Urban_IoU"):
            contributions = [float(values.loc[a, metric] - values.loc[b, metric]) for a, b in pairs]
            row = {
                "FeatureFamily": family,
                "Metric": metric,
                "AverageMarginalContribution": float(np.mean(contributions)),
            }
            for index, ((a, b), value) in enumerate(zip(pairs, contributions), start=1):
                row[f"Component_{index}"] = f"{a}-{b}"
                row[f"Component_{index}_Difference"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(output_dir: Path):
    rows = []
    provenance = {
        "N": "Spatial_Context_Extraction_02 canonical table",
        "G": "Spatial_Context_Extraction_02 canonical table",
        "T": "Feature Extraction 05 texture-only BG-log full table",
        "C": "Spatial_Context_Extraction_02 canonical table",
    }
    for group, columns in FEATURE_GROUPS.items():
        for order, column in enumerate(columns, start=1):
            rows.append({
                "FeatureGroup": group,
                "OrderWithinGroup": order,
                "Feature": column,
                "Provenance": provenance[group],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Ablation_Feature_Group_Manifest.csv", index=False)
    return frame


def save_predictions(path, test_objects, predictions, probabilities):
    output = pd.DataFrame({
        "Object_ID": test_objects["Object_ID"].to_numpy(dtype="int64"),
        "TrueClass": test_objects["TrueLabel"].map(CLASS_NAMES).to_numpy(),
    })
    for model in MODEL_GROUPS:
        output[f"{model}_PredictedClass"] = [CLASS_NAMES[int(v) + 1] for v in predictions[model]]
        output[f"{model}_P_Urban"] = probabilities[model][:, 0]
        output[f"{model}_P_Vegetation"] = probabilities[model][:, 1]
        output[f"{model}_P_BareScrub"] = probabilities[model][:, 2]
    output.to_csv(path, index=False)


def plot_performance(plt, performance, path):
    metrics = ["MacroF1", "mIoU", "Urban_IoU"]
    x = np.arange(len(performance))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, metric in enumerate(metrics):
        ax.bar(x + (offset - 1) * width, performance[metric], width, label=metric)
    ax.set_xticks(x, performance["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Independent-test score")
    ax.set_xlabel("Ablation model")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_marginal(plt, marginal, path):
    selected = marginal[marginal["Metric"].isin(["MacroF1", "mIoU"])].copy()
    families = ["Geometry", "Texture", "SpatialContext"]
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for position, metric in enumerate(["MacroF1", "mIoU"]):
        values = [
            selected.loc[
                (selected["FeatureFamily"] == family) & (selected["Metric"] == metric),
                "AverageMarginalContribution",
            ].iloc[0]
            for family in families
        ]
        ax.bar(x + (position - 0.5) * width, values, width, label=metric)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["Geometry", "Texture", "Spatial context"])
    ax.set_ylabel("Average marginal score difference")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(plt, cm, model, path):
    normalized = np.divide(
        cm,
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype="float64"),
        where=cm.sum(axis=1, keepdims=True) > 0,
    )
    labels = [CLASS_NAMES[c] for c in CLASS_CODES]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, matrix, title, percent in (
        (axes[0], cm, "Raw counts", False),
        (axes[1], normalized * 100, "Row-normalized (%)", True),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(3):
            for column in range(3):
                text = f"{matrix[row, column]:.1f}%" if percent else f"{int(matrix[row, column])}"
                threshold = matrix.max() / 2 if matrix.max() else 0
                ax.text(column, row, text, ha="center", va="center",
                        color="white" if matrix[row, column] > threshold else "black")
        ax.set_xticks(range(3), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model} independent-test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, digits=4):
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "float32", "float64"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def paired_row(paired, first, second, metric):
    rows = paired[
        (paired["Model_1"] == first)
        & (paired["Model_2"] == second)
        & (paired["Metric"] == metric)
    ]
    return None if rows.empty else rows.iloc[0]


def scientific_decisions(performance, marginal, paired, pilot):
    if pilot:
        return {
            "texture": "Pilot only: no scientific retention/removal decision is made.",
            "context": "Pilot only: no scientific retention/removal decision is made.",
            "parsimonious": "Pilot only: a final parsimonious model is not selected.",
        }
    texture_checks = [paired_row(paired, "A7", "A5", m) for m in ("MacroF1", "mIoU")]
    context_checks = [paired_row(paired, "A7", "A4", m) for m in ("MacroF1", "mIoU")]
    texture_mean = marginal.loc[
        (marginal["FeatureFamily"] == "Texture")
        & (marginal["Metric"].isin(["MacroF1", "mIoU"])),
        "AverageMarginalContribution",
    ].mean()
    context_mean = marginal.loc[
        (marginal["FeatureFamily"] == "SpatialContext")
        & (marginal["Metric"].isin(["MacroF1", "mIoU"])),
        "AverageMarginalContribution",
    ].mean()
    texture_supported = any(row is not None and row["CI_Lower_2.5"] > 0 for row in texture_checks)
    context_supported = any(row is not None and row["CI_Lower_2.5"] > 0 for row in context_checks)
    texture_text = (
        "Reduced BG-Log texture shows supported incremental value beyond N+G+C."
        if texture_supported
        else "Reduced BG-Log texture does not show a clearly supported increment beyond N+G+C; "
             "removal is defensible if its marginal effects are also practically negligible."
    )
    context_text = (
        "Reduced spatial context shows supported incremental value beyond N+G+T."
        if context_supported
        else "Reduced spatial context does not show a clearly supported increment beyond N+G+T."
    )
    values = performance.set_index("Model")
    best_model = values["MacroF1"].idxmax()
    best_macro = values.loc[best_model, "MacroF1"]
    best_iou = values.loc[best_model, "mIoU"]
    eligible_by_effect = performance[
        (performance["MacroF1"] >= best_macro - 0.01)
        & (performance["mIoU"] >= best_iou - 0.01)
    ].sort_values(["NumberOfFeatures", "MacroF1"], ascending=[True, False])

    def uncertainty_includes_zero(candidate, reference, metric):
        if candidate == reference:
            return True
        rows = paired[
            (
                ((paired["Model_1"] == candidate) & (paired["Model_2"] == reference))
                | ((paired["Model_1"] == reference) & (paired["Model_2"] == candidate))
            )
            & (paired["Metric"] == metric)
        ]
        return bool(not rows.empty and rows.iloc[0]["CI_Includes_Zero"])

    uncertainty_supported = []
    for candidate in eligible_by_effect["Model"]:
        if all(
            uncertainty_includes_zero(candidate, best_model, metric)
            for metric in ("MacroF1", "mIoU")
        ):
            uncertainty_supported.append(candidate)
    chosen = uncertainty_supported[0] if uncertainty_supported else best_model
    parsimonious = (
        f"Using a predeclared 0.01 practical-equivalence band for both Macro-F1 and mIoU, "
        f"plus paired intervals that include zero relative to the highest-Macro-F1 model, "
        f"the smallest supported candidate is {chosen} ({'+'.join(MODEL_GROUPS[chosen])}). "
        f"This recommendation is descriptive, not additional model tuning."
    )
    return {
        "texture": texture_text + f" Mean Macro-F1/mIoU marginal contribution = {texture_mean:.4f}.",
        "context": context_text + f" Mean Macro-F1/mIoU marginal contribution = {context_mean:.4f}.",
        "parsimonious": parsimonious,
    }


def write_report(path, mode_name, args, versions, input_info, manifest, class_table,
                 test_diagnostics, graph_diagnostics, fold_summary, cv_results,
                 selected_parameters, selected_cv_score, performance, classwise,
                 marginal, bootstrap_ci, paired, runtimes, nan_diagnostics,
                 decisions, output_dir):
    feature_sections = []
    for group in ("N", "G", "T", "C"):
        feature_sections.append(
            f"### {group} ({len(FEATURE_GROUPS[group])} features)\n\n"
            + ", ".join(FEATURE_GROUPS[group])
        )
    required_bootstrap = paired[paired["RequiredComparison"]].copy()
    lines = [
        f"# Object-Based XGBoost Ablation 01 — {mode_name}",
        "",
        "## Scope",
        "",
        (
            "This is a workflow-validation pilot. Values are real pilot results but must not be "
            "reported as the final 57,388-object experiment."
            if mode_name == "PILOT"
            else "This is the complete predefined eight-model feature-family ablation."
        ),
        "",
        "## A. Inputs",
        "",
        f"- Primary N/G/C table: {args.context_table.resolve()}",
        f"- Supplemental final BG-Log table: {args.texture_table.resolve()}",
        f"- Training labels: {args.training_labels.resolve()}",
        f"- Test points: {args.test_points.resolve()}",
        f"- Segmentation: {args.segmentation.resolve()}",
        f"- Rook graph: {args.adjacency.resolve()}",
        f"- Primary feature rows / unique IDs: {input_info['feature_rows']:,} / {input_info['feature_unique_ids']:,}",
        f"- Eligible training-label rows in source: {input_info['training_label_rows']:,}",
        f"- Training objects used in this run: {input_info['training_used']:,}",
        f"- Independent test objects: {test_diagnostics['unique_test_objects']:,}",
        "",
        "The resolved feature view is an Object_ID-validated join: the newer spatial-context output "
        "supplies N/G/C, while the validated Feature Extraction 05 texture-only full output supplies T. "
        "Older SDGSAT_Pan_glcm_* fields are ignored.",
        "",
        "## B. Feature groups",
        "",
        *feature_sections,
        "",
        "T contains only BG-Log Contrast, Homogeneity, and Entropy. C excludes NTL_NormalizedContrast.",
        "",
        "## C. Train/test design",
        "",
        markdown_table(class_table),
        "",
        f"Strict mapping used the test field {test_diagnostics['label_field']}. "
        f"There were {test_diagnostics['strictly_unmatched_labeled_points']} unmatched labeled points, "
        f"{test_diagnostics['duplicate_point_assignments_removed']} duplicate point assignments removed, "
        f"and {test_diagnostics['conflicting_test_objects']} conflicting objects.",
        f"Train/test Object_ID overlap: {input_info['train_test_overlap']}.",
        f"First-order test-neighbor overlap in training: {graph_diagnostics['training_test_neighbor_overlap_count']} "
        f"of {graph_diagnostics['test_neighbor_count']:,} distinct non-test neighbors.",
        "Training weights were computed only from this run's training labels. Test labels were used only "
        "after tuning and fitting, for independent evaluation.",
        "",
        "## D. Spatial cross-validation",
        "",
        "Training centroids were projected to EPSG:32639 and assigned to globally anchored 5 km × 5 km "
        "groups using floor(Easting/5000), floor(Northing/5000). GroupKFold was used without random "
        "object mixing.",
        f"Spatial groups: {input_info['spatial_group_count']:,}; folds: {len(fold_summary)}; "
        f"tested configurations: {len(cv_results)}.",
        "",
        markdown_table(fold_summary),
        "",
        f"Selected validation Macro-F1: {selected_cv_score:.6f}.",
        "Configurations within 0.0001 Macro-F1 were treated as practically tied; shallower depth and "
        "then fewer trees were preferred.",
        f"The best independent-test Macro-F1 in this run was {performance['MacroF1'].max():.6f}. "
        "The CV/test difference should be interpreted in light of the different label sources "
        "(Dynamic World-derived training labels versus manual independent test labels), not hidden.",
        "",
        "Selected configuration:",
        "",
        "    " + json.dumps(selected_parameters, sort_keys=True),
        "",
        "Top CV configurations:",
        "",
        markdown_table(cv_results.sort_values("Mean_MacroF1", ascending=False).head(10)),
        "",
        "## E. Ablation results",
        "",
        markdown_table(performance),
        "",
        "## F. Class-specific results",
        "",
        markdown_table(classwise),
        "",
        "Urban precision, recall, F1, and IoU are also presented in the model-level table above.",
        "",
        "## G. Average marginal contributions",
        "",
        markdown_table(marginal[["FeatureFamily", "Metric", "AverageMarginalContribution"]]),
        "",
        "These are descriptive factorial contributions, not causal effects.",
        "",
        "## H. Paired bootstrap",
        "",
        f"Stratified paired bootstrap replicates: {int(bootstrap_ci['Replicates'].iloc[0])}.",
        "",
        markdown_table(required_bootstrap[[
            "Comparison", "Metric", "MeanDifference", "CI_Lower_2.5",
            "CI_Upper_97.5", "CI_Includes_Zero",
        ]]),
        "",
        "A confidence interval including zero is reported as uncertainty, not automatically as a "
        "significance claim.",
        "",
        "## I. Texture decision",
        "",
        decisions["texture"],
        "",
        "## J. Context decision",
        "",
        decisions["context"],
        "",
        "## K. Final parsimonious model",
        "",
        decisions["parsimonious"],
        "",
        "## L. Runtime",
        "",
        markdown_table(pd.DataFrame([{"Stage": k, "Seconds": v} for k, v in runtimes.items()])),
        "",
        "## M. Deviations and validation notes",
        "",
        "- No single existing table contained both the final BG-Log texture and Spatial Context 02. "
        "The script therefore builds a one-to-one Object_ID-resolved view from the two authoritative outputs.",
        "- The final redundancy report documented Compactness–ShapeIndex as deterministic but did not "
        "record a final choice. Compactness was retained because it is the earlier base definition; "
        "ShapeIndex was excluded. This judgment is explicit rather than hidden.",
        "- Optical controls O1/O2 were intentionally omitted: they are outside the core factorial question "
        "and including them would add a secondary analysis not needed to answer it.",
        (
            "- Pilot deviations: a stratified training subset, three spatial folds, capped 120-tree "
            "configurations, fewer sampled configurations, and fewer bootstrap replicates were used."
            if mode_name == "PILOT" else
            "- Full run used five spatial folds, the unmodified search space, and 2,000 bootstrap replicates."
        ),
        f"- Unexpected non-texture NaN cells (training/test): "
        f"{nan_diagnostics['non_texture_train_nan_cells']:,} / {nan_diagnostics['non_texture_test_nan_cells']:,}.",
        f"- BG-Log NaN cells were preserved (training/test): "
        f"{nan_diagnostics['texture_train_nan_cells']:,} / {nan_diagnostics['texture_test_nan_cells']:,}.",
        f"- Software: Python {versions['python']}; XGBoost {versions['xgboost']}; "
        f"scikit-learn {versions['sklearn']}.",
        "- All A0–A7 models used identical Object_ID sets, class weights, frozen hyperparameters, and seed.",
        "- Dynamic World-derived training labels were never used as independent test labels.",
        "",
        "## Output directory",
        "",
        str(output_dir.resolve()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started_total = time.perf_counter()
    args = parse_args()
    bundle = dependency_bundle()
    if args.output_dir is None:
        suffix = "_pilot" if args.pilot else ""
        if args.preflight_only:
            suffix = "_preflight"
        args.output_dir = SEGMENTATION_DIR / f"Object_Based_Ablation_XGBoost_01{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    require_files([
        args.context_table, args.texture_table, args.training_labels,
        args.test_points, args.segmentation, args.adjacency,
    ])
    validate_feature_headers(args.context_table, args.texture_table)
    manifest = write_manifest(args.output_dir)

    stage_started = time.perf_counter()
    print("Loading validated object tables...")
    features = load_feature_view(args.context_table, args.texture_table)
    labels_full = load_training_labels(args.training_labels)
    load_seconds = time.perf_counter() - stage_started

    print("Loading segmentation and mapping independent test points (strict within)...")
    gpd = bundle["gpd"]
    segments = gpd.read_file(args.segmentation, columns=["Seg_id", "geometry"])
    if segments.crs is None:
        raise SystemExit("STOP: segmentation has no CRS.")
    segments["Seg_id"] = segments["Seg_id"].astype("int64")
    if segments["Seg_id"].duplicated().any():
        raise SystemExit("STOP: segmentation Seg_id is not unique.")
    test_objects, test_diagnostics = build_test_objects(gpd, args.test_points, segments)

    feature_ids = pd.Index(features["Object_ID"])
    missing_training = pd.Index(labels_full["Object_ID"]).difference(feature_ids)
    missing_test = pd.Index(test_objects["Object_ID"]).difference(feature_ids)
    if len(missing_training) or len(missing_test):
        raise SystemExit(
            f"STOP: feature join failures: training={len(missing_training)}, test={len(missing_test)}."
        )
    train_test_overlap = np.intersect1d(
        labels_full["Object_ID"].to_numpy(), test_objects["Object_ID"].to_numpy()
    )
    if len(train_test_overlap):
        raise SystemExit(
            f"STOP: train/test leakage detected for {len(train_test_overlap)} Object_ID values."
        )
    graph_diagnostics = validate_graph_exclusion(
        args.adjacency,
        labels_full["Object_ID"].to_numpy(),
        test_objects["Object_ID"].to_numpy(),
    )
    if graph_diagnostics["training_test_neighbor_overlap_count"]:
        raise SystemExit(
            "STOP: training labels include first-order Rook neighbors of independent test objects. "
            f"Examples: {graph_diagnostics['overlap_example_ids']}"
        )

    if len(labels_full) != 57388:
        print(f"WARNING: training label count is {len(labels_full):,}, not expected 57,388.")
    if len(test_objects) != 996:
        print(f"WARNING: unique test-object count is {len(test_objects):,}, not expected 996.")

    labels_used = (
        stratified_pilot_sample(labels_full, args.pilot_train_size)
        if args.pilot else labels_full.copy()
    )
    groups, centroid_x, centroid_y = spatial_groups_for_training(
        segments, labels_used["Object_ID"].to_numpy()
    )
    number_splits = 3 if args.pilot else 5
    if len(np.unique(groups)) < number_splits:
        raise SystemExit("STOP: insufficient distinct spatial groups for GroupKFold.")

    indexed = features.set_index("Object_ID")
    train_ids = labels_used["Object_ID"].to_numpy(dtype="int64")
    test_ids = test_objects["Object_ID"].to_numpy(dtype="int64")
    y_train_codes = labels_used["FinalTrainingLabel"].to_numpy(dtype="int64")
    y_test_codes = test_objects["TrueLabel"].to_numpy(dtype="int64")
    all_columns = [c for group in FEATURE_GROUPS.values() for c in group]
    non_texture = [c for c in all_columns if c not in TEXTURE_FEATURES]
    train_view = indexed.loc[train_ids]
    test_view = indexed.loc[test_ids]
    nan_diagnostics = {
        "non_texture_train_nan_cells": int(train_view[non_texture].isna().sum().sum()),
        "non_texture_test_nan_cells": int(test_view[non_texture].isna().sum().sum()),
        "texture_train_nan_cells": int(train_view[TEXTURE_FEATURES].isna().sum().sum()),
        "texture_test_nan_cells": int(test_view[TEXTURE_FEATURES].isna().sum().sum()),
        "per_feature_train_nan": train_view[all_columns].isna().sum().astype(int).to_dict(),
        "per_feature_test_nan": test_view[all_columns].isna().sum().astype(int).to_dict(),
    }

    weights, sample_weight = class_weights(y_train_codes)
    class_rows = []
    for code in CLASS_CODES:
        class_rows.append({
            "Class": CLASS_NAMES[code],
            "TrainingSourceCount": int((labels_full["FinalTrainingLabel"] == code).sum()),
            "TrainingUsedCount": int((y_train_codes == code).sum()),
            "TrainingWeight": weights[code],
            "IndependentTestCount": int((y_test_codes == code).sum()),
        })
    class_table = pd.DataFrame(class_rows)
    class_table.to_csv(args.output_dir / "Ablation_Class_Weights_and_Counts.csv", index=False)

    input_info = {
        "feature_rows": int(len(features)),
        "feature_unique_ids": int(features["Object_ID"].nunique()),
        "training_label_rows": int(len(labels_full)),
        "training_used": int(len(labels_used)),
        "test_objects": int(len(test_objects)),
        "train_test_overlap": int(len(train_test_overlap)),
        "spatial_group_count": int(len(np.unique(groups))),
    }
    validation_payload = {
        "inputs": {k: str(getattr(args, k).resolve()) for k in (
            "context_table", "texture_table", "training_labels",
            "test_points", "segmentation", "adjacency",
        )},
        "input_info": input_info,
        "test_mapping": test_diagnostics,
        "graph_exclusion": graph_diagnostics,
        "nan_diagnostics": nan_diagnostics,
        "feature_groups": FEATURE_GROUPS,
        "random_seed": RANDOM_SEED,
    }
    (args.output_dir / "Ablation_Input_Validation.json").write_text(
        json.dumps(validation_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    spatial_assignment = pd.DataFrame({
        "Object_ID": train_ids,
        "SpatialGroup": groups,
        "Centroid_EPSG32639_X": centroid_x,
        "Centroid_EPSG32639_Y": centroid_y,
    })

    if args.preflight_only:
        spatial_assignment.to_csv(
            args.output_dir / "Ablation_Training_Spatial_Groups.csv", index=False
        )
        print("Preflight passed. No model was fitted.")
        print(f"Output: {args.output_dir.resolve()}")
        return 0

    config_count = args.pilot_configs if args.pilot else args.search_configs
    bootstrap_replicates = args.pilot_bootstrap if args.pilot else args.bootstrap_replicates
    configurations = sampled_parameter_configurations(config_count, pilot=args.pilot)
    A7_columns = [c for group in MODEL_GROUPS["A7"] for c in FEATURE_GROUPS[group]]
    X_a7 = train_view[A7_columns].to_numpy(dtype="float32", copy=True)
    y_train_zero = y_train_codes - 1

    stage_started = time.perf_counter()
    selected_parameters, selected_cv_score, cv_results, splits = tune_a7(
        bundle, X_a7, y_train_zero, groups, sample_weight,
        configurations, number_splits, args.n_jobs,
    )
    cv_seconds = time.perf_counter() - stage_started
    cv_results.to_csv(args.output_dir / "Ablation_Spatial_CV_Results.csv", index=False)

    fold_id = np.full(len(labels_used), -1, dtype="int64")
    fold_rows = []
    for fold, (training_index, validation_index) in enumerate(splits, start=1):
        fold_id[validation_index] = fold
        row = {
            "Fold": fold,
            "TrainingObjects": len(training_index),
            "ValidationObjects": len(validation_index),
            "ValidationSpatialGroups": len(np.unique(groups[validation_index])),
        }
        for code in CLASS_CODES:
            row[f"Validation_{CLASS_NAMES[code]}"] = int((y_train_codes[validation_index] == code).sum())
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)
    spatial_assignment["ValidationFold"] = fold_id
    spatial_assignment.to_csv(
        args.output_dir / "Ablation_Training_Spatial_Groups.csv", index=False
    )

    stage_started = time.perf_counter()
    (
        predictions, probabilities, evaluations, final_weights,
        final_training_seconds, prediction_seconds,
    ) = fit_all_models(
        bundle, features, train_ids, y_train_codes, test_ids, y_test_codes,
        selected_parameters, args.n_jobs,
    )
    fit_prediction_seconds = time.perf_counter() - stage_started
    performance, classwise, confusion_long = performance_tables(evaluations)
    performance.to_csv(args.output_dir / "Ablation_Model_Performance.csv", index=False)
    classwise.to_csv(args.output_dir / "Ablation_Classwise_Performance.csv", index=False)
    confusion_long.to_csv(args.output_dir / "Ablation_Confusion_Matrices.csv", index=False)
    save_predictions(
        args.output_dir / "Ablation_Test_Predictions.csv",
        test_objects, predictions, probabilities,
    )

    stage_started = time.perf_counter()
    bootstrap_ci, paired, distributions = bootstrap_analysis(
        y_test_codes - 1, predictions, bootstrap_replicates
    )
    bootstrap_seconds = time.perf_counter() - stage_started
    bootstrap_ci.to_csv(args.output_dir / "Ablation_Bootstrap_CI.csv", index=False)
    paired.to_csv(args.output_dir / "Ablation_Paired_Bootstrap_Differences.csv", index=False)
    marginal = marginal_contributions(performance)
    marginal.to_csv(
        args.output_dir / "Ablation_Average_Marginal_Contributions.csv", index=False
    )

    hyperparameter_payload = {
        "random_state": RANDOM_SEED,
        "objective": "multi:softprob",
        "num_class": 3,
        "selection_metric": "mean spatial-CV Macro-F1",
        "spatial_cv_folds": number_splits,
        "tested_configuration_count": len(configurations),
        "selected_validation_macro_f1": selected_cv_score,
        "selected_hyperparameters": selected_parameters,
        "frozen_for_models": list(MODEL_GROUPS),
    }
    (args.output_dir / "Ablation_Selected_Hyperparameters.json").write_text(
        json.dumps(hyperparameter_payload, indent=2), encoding="utf-8"
    )

    plot_performance(
        bundle["plt"], performance,
        args.output_dir / "Ablation_Performance_Comparison.png",
    )
    plot_marginal(
        bundle["plt"], marginal,
        args.output_dir / "Ablation_Marginal_Contribution.png",
    )
    plot_confusion(
        bundle["plt"], evaluations["A0"]["confusion"], "A0",
        args.output_dir / "A0_Confusion_Matrix.png",
    )
    plot_confusion(
        bundle["plt"], evaluations["A7"]["confusion"], "A7",
        args.output_dir / "A7_Confusion_Matrix.png",
    )

    versions = {
        "python": sys.version.split()[0],
        "xgboost": bundle["xgb"].__version__,
        "sklearn": importlib.import_module("sklearn").__version__,
    }
    total_seconds = time.perf_counter() - started_total
    runtimes = {
        "Feature and label loading": load_seconds,
        "Spatial CV tuning": cv_seconds,
        "Final training": final_training_seconds,
        "Prediction": prediction_seconds,
        "Final model orchestration overhead": max(
            0.0, fit_prediction_seconds - final_training_seconds - prediction_seconds
        ),
        "Paired bootstrap": bootstrap_seconds,
        "Total": total_seconds,
    }
    decisions = scientific_decisions(performance, marginal, paired, args.pilot)
    report_name = (
        "Object_Based_Ablation_XGBoost_01_Report_PILOT.md"
        if args.pilot else "Object_Based_Ablation_XGBoost_01_Report.md"
    )
    write_report(
        args.output_dir / report_name,
        "PILOT" if args.pilot else "FULL RUN",
        args, versions, input_info, manifest, class_table,
        test_diagnostics, graph_diagnostics, fold_summary, cv_results,
        selected_parameters, selected_cv_score, performance, classwise,
        marginal, bootstrap_ci, paired, runtimes, nan_diagnostics,
        decisions, args.output_dir,
    )
    print(f"Completed {'pilot' if args.pilot else 'full run'}.")
    print(f"Report: {(args.output_dir / report_name).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
