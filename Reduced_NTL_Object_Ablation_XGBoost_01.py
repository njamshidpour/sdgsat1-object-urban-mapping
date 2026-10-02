#!/usr/bin/env python
"""Reduced NTL object-family ablation for binary Urban/NonUrban mapping.

The default mode is a bounded pilot.  ``--full-run`` performs the complete
five-model, independently tuned experiment with 2,000 paired bootstraps.
Feature reduction is unsupervised, family-wise, training-only, and repeated
inside every spatial cross-validation training fold.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import Object_Based_Ablation_XGBoost_01 as object_core
import Binary_Stratified_Object_Ablation_XGBoost_01 as binary_core
import Spectral_Sensor_Comparison_XGBoost_01 as spectral_core


RANDOM_SEED = 20261001
THRESHOLD = 0.50
PRIMARY_CORRELATION_THRESHOLD = 0.90
SENSITIVITY_THRESHOLDS = (0.85, 0.90, 0.95)
ROOT = Path(__file__).resolve().parent
SEG_DIR = ROOT / "Object_based_fusion" / "Segmentation_output"
BINARY_DIR = SEG_DIR / "Binary_Stratified_Object_Ablation_XGBoost_01"
ABLATION_DIR = SEG_DIR / "Object_Based_Ablation_XGBoost_01"

CONTEXT_TABLE = SEG_DIR / "Spatial_Context_Extraction_02" / "objects_features_with_spatial_context_02.csv"
TEXTURE_TABLE = SEG_DIR / "objects_features_v05_texture_only_full.csv"
TRAINING_PATH = BINARY_DIR / "Binary_Stratified_Training_Set.csv"
PRIMARY_PATH = BINARY_DIR / "Binary_Primary_Stratified_Test_Set.csv"
FULL_PATH = BINARY_DIR / "Binary_Secondary_Full_Test_Set.csv"
ADJACENCY_PATH = SEG_DIR / "rook_adjacency_graph_v01.npz"
SOURCE_MANIFEST = ABLATION_DIR / "Ablation_Feature_Group_Manifest.csv"

EXPECTED_GROUPS = {
    "R": list(object_core.NTL_FEATURES),
    "G": list(object_core.GEOMETRY_FEATURES),
    "T": list(object_core.TEXTURE_FEATURES),
    "C": list(object_core.CONTEXT_FEATURES),
}
MODEL_GROUPS = OrderedDict([
    ("R0", ("R",)),
    ("R1", ("R", "G")),
    ("R2", ("R", "C")),
    ("R3", ("R", "T")),
    ("R4", ("R", "G", "C", "T")),
])
MODEL_LABELS = {
    "R0": "NTL Radiometric",
    "R1": "NTL + Geometry",
    "R2": "NTL + Context",
    "R3": "NTL + Texture",
    "R4": "NTL + Geometry + Context + Texture",
}
COMPARISONS = [
    ("R1", "R0", "Geometry effect"),
    ("R2", "R0", "Context effect"),
    ("R3", "R0", "Texture effect"),
    ("R4", "R0", "All object features"),
]
BOOTSTRAP_METRICS = ["BalancedAccuracy", "MacroF1", "mIoU", "Urban_F1", "Urban_IoU"]
CONTRIBUTION_METRICS = ["MacroF1", "mIoU", "UrbanF1", "UrbanIoU"]
PARAMETER_SPACE = OrderedDict([
    ("n_estimators", [300, 500, 800]),
    ("max_depth", [3, 5, 7]),
    ("learning_rate", [0.03, 0.05, 0.10]),
    ("min_child_weight", [1, 3, 5]),
    ("gamma", [0, 0.1, 0.5]),
    ("subsample", [0.8, 1.0]),
    ("colsample_bytree", [0.7, 0.9, 1.0]),
    ("reg_lambda", [1, 5, 10]),
    ("reg_alpha", [0, 0.1, 0.5]),
])
FAMILY_NAMES = {
    "R": "NTL Radiometric", "G": "Geometry", "T": "Texture", "C": "Spatial Context"
}
REDUCTION_FILES = {
    "R": "NTL_Radiometric_Reduction.csv",
    "G": "Geometry_Reduction.csv",
    "T": "Texture_Reduction.csv",
    "C": "Spatial_Context_Reduction.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Bounded validation pilot (default).")
    mode.add_argument("--full-run", action="store_true", help="Complete research experiment.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--search-configs", type=int, default=30)
    parser.add_argument("--pilot-configs", type=int, default=3)
    parser.add_argument("--pilot-per-source", type=int, default=1000)
    parser.add_argument("--pilot-bootstrap", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def dependencies():
    bundle = binary_core.dependency_bundle()
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise SystemExit("STOP: scipy is required for hierarchical feature clustering.") from exc
    bundle.update({"fcluster": fcluster, "linkage": linkage, "squareform": squareform})
    return bundle


def require_inputs():
    required = [CONTEXT_TABLE, TEXTURE_TABLE, TRAINING_PATH, PRIMARY_PATH, FULL_PATH,
                ADJACENCY_PATH, SOURCE_MANIFEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("STOP: required input(s) missing:\n" + "\n".join(missing))


def load_groups():
    manifest = pd.read_csv(SOURCE_MANIFEST)
    required = {"FeatureGroup", "OrderWithinGroup", "Feature"}
    if not required.issubset(manifest.columns):
        raise SystemExit("STOP: previous ablation manifest has an invalid structure.")
    mapping = {"R": "N", "G": "G", "T": "T", "C": "C"}
    resolved = {}
    for family, old_group in mapping.items():
        values = manifest[manifest.FeatureGroup == old_group].sort_values("OrderWithinGroup").Feature.tolist()
        if values != EXPECTED_GROUPS[family]:
            raise SystemExit(f"STOP: {family} differs from the validated ablation manifest: {values}")
        resolved[family] = values
    forbidden_texture = {"SDGSAT_Pan_glcm_energy", "SDGSAT_Pan_glcm_correlation"}
    if forbidden_texture & set(resolved["T"]):
        raise SystemExit("STOP: conventional or discarded GLCM features entered the texture group.")
    return resolved


def load_memberships():
    training = spectral_core.load_reused_dataset(TRAINING_PATH, "training")
    primary = spectral_core.load_reused_dataset(PRIMARY_PATH, "primary")
    full = spectral_core.load_reused_dataset(FULL_PATH, "full")
    for name, frame in (("training", training), ("primary", primary), ("full", full)):
        if frame.Object_ID.isna().any() or frame.Object_ID.duplicated().any():
            raise SystemExit(f"STOP: invalid Object_ID values in {name} membership.")
    overlap = np.intersect1d(training.Object_ID, full.Object_ID)
    if len(overlap):
        raise SystemExit(f"STOP: {len(overlap)} training/full-test objects overlap.")
    if not set(primary.Object_ID).issubset(set(full.Object_ID)):
        raise SystemExit("STOP: primary test is not a subset of full manual test.")
    neighbors = binary_core.rook_neighbors(ADJACENCY_PATH, full.Object_ID.to_numpy(dtype="int64"))
    neighbor_overlap = np.intersect1d(training.Object_ID.to_numpy(dtype="int64"), neighbors)
    if len(neighbor_overlap):
        raise SystemExit(f"STOP: {len(neighbor_overlap)} test-neighbour objects occur in training.")
    return training, primary, full, neighbors


def load_features(groups):
    context_columns = ["Object_ID", *groups["R"], *groups["G"], *groups["C"]]
    texture_columns = ["Object_ID", *groups["T"]]
    context_header = pd.read_csv(CONTEXT_TABLE, nrows=0).columns.tolist()
    texture_header = pd.read_csv(TEXTURE_TABLE, nrows=0).columns.tolist()
    missing_context = [column for column in context_columns if column not in context_header]
    missing_texture = [column for column in texture_columns if column not in texture_header]
    if missing_context or missing_texture:
        raise SystemExit("STOP: exact candidate features are missing:\n" + "\n".join(missing_context + missing_texture))
    context = pd.read_csv(CONTEXT_TABLE, usecols=context_columns)
    texture = pd.read_csv(TEXTURE_TABLE, usecols=texture_columns)
    for name, frame in (("context", context), ("texture", texture)):
        if frame.Object_ID.isna().any() or frame.Object_ID.duplicated().any():
            raise SystemExit(f"STOP: {name} feature table has invalid Object_ID values.")
        frame["Object_ID"] = frame.Object_ID.astype("int64")
    if len(context) != len(texture) or set(context.Object_ID) != set(texture.Object_ID):
        raise SystemExit("STOP: context and texture feature tables have different Object_ID domains.")
    features = context.merge(texture, on="Object_ID", validate="one_to_one")
    if features.shape[1] != 1 + sum(map(len, groups.values())):
        raise SystemExit("STOP: merged candidate feature view has an unexpected column count.")
    forbidden = [c for c in features.columns if c.startswith("S2_") or c.startswith(("NDVI", "NDBI", "NDWI"))]
    if forbidden:
        raise SystemExit("STOP: optical classification predictors entered the experiment: " + ", ".join(forbidden))
    return features


def verify_joins(features, named_memberships):
    available = pd.Index(features.Object_ID)
    for name, frame in named_memberships:
        missing = pd.Index(frame.Object_ID).difference(available)
        if len(missing):
            raise SystemExit(f"STOP: {len(missing)} {name} objects lack features.")


def pilot_training_subset(training, per_source):
    rng = np.random.default_rng(RANDOM_SEED)
    pieces = []
    for source in ("Urban", "Vegetation", "BareScrub"):
        frame = training[training.SourceClass3 == source]
        amount = min(per_source, len(frame))
        pieces.append(binary_core.hierarchical_stratified_sample(frame, amount, [], rng, False))
    result = pd.concat(pieces).sort_values("Object_ID").reset_index(drop=True)
    counts = result.SourceClass3.value_counts()
    if counts.nunique() != 1:
        raise SystemExit("STOP: pilot source-class strata are not balanced.")
    return result


def missingness_audit(features, groups, memberships):
    indexed = features.set_index("Object_ID")
    scopes = {"FullObjectPopulation": features}
    scopes.update({name: indexed.loc[frame.Object_ID].reset_index() for name, frame in memberships})
    rows = []
    for scope, frame in scopes.items():
        for family, columns in groups.items():
            for feature in columns:
                values = pd.to_numeric(frame[feature], errors="coerce")
                finite = np.isfinite(values.dropna().to_numpy(dtype="float64"))
                rows.append({
                    "Scope": scope, "Family": family, "Feature": feature, "Rows": len(frame),
                    "MissingCount": int(values.isna().sum()),
                    "MissingPercent": float(100 * values.isna().mean()),
                    "NonFiniteNonMissingCount": int((~finite).sum()),
                })
    audit = pd.DataFrame(rows)
    if audit.NonFiniteNonMissingCount.sum():
        raise SystemExit("STOP: infinity detected in candidate features.")
    return audit


def is_near_constant(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(values) < 2 or np.unique(values).size <= 1:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(np.std(values)) <= 1e-12 * scale


def reduce_family(bundle, training_values, columns, threshold):
    active = [column for column in columns if not is_near_constant(training_values[column])]
    removed_constant = [column for column in columns if column not in active]
    rows = []
    if active:
        correlation = training_values[active].corr(method="spearman", min_periods=2)
        correlation = correlation.astype("float64")
        np.fill_diagonal(correlation.values, 1.0)
        absolute = correlation.abs().fillna(0.0)
        np.fill_diagonal(absolute.values, 1.0)
        if len(active) == 1:
            labels = np.ones(1, dtype="int64")
        else:
            distance = (1.0 - absolute).clip(0.0, 1.0)
            np.fill_diagonal(distance.values, 0.0)
            condensed = bundle["squareform"](distance.to_numpy(), checks=False)
            hierarchy = bundle["linkage"](condensed, method="average")
            labels = bundle["fcluster"](hierarchy, t=1.0 - threshold + 1e-12, criterion="distance")
        for cluster_id in sorted(np.unique(labels)):
            members = [active[i] for i in np.flatnonzero(labels == cluster_id)]
            if len(members) == 1:
                representative = members[0]
                maximum = 0.0
            else:
                within = absolute.loc[members, members].copy()
                averages = (within.sum(axis=1) - 1.0) / (len(members) - 1)
                best = float(averages.max())
                tied = [column for column in members if math.isclose(float(averages[column]), best, abs_tol=1e-12)]
                representative = min(tied, key=lambda name: columns.index(name))
                masked = within.to_numpy().copy()
                np.fill_diagonal(masked, np.nan)
                maximum = float(np.nanmax(masked))
            for member in members:
                rows.append({
                    "OriginalFeature": member, "ClusterID": int(cluster_id),
                    "RepresentativeFeature": representative,
                    "Retained": member == representative,
                    "MaximumWithinClusterAbsCorrelation": maximum,
                    "RemovalReason": "retained medoid" if member == representative else "correlation redundancy",
                })
    for offset, feature in enumerate(removed_constant, start=1):
        rows.append({
            "OriginalFeature": feature, "ClusterID": 0,
            "RepresentativeFeature": "", "Retained": False,
            "MaximumWithinClusterAbsCorrelation": np.nan,
            "RemovalReason": "constant or near-zero variance",
        })
    table = pd.DataFrame(rows)
    table["_order"] = table.OriginalFeature.map({name: i for i, name in enumerate(columns)})
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    retained = table.loc[table.Retained, "OriginalFeature"].tolist()
    if not retained:
        raise SystemExit("STOP: reduction removed every feature from a family.")
    return table, retained


def full_training_reductions(bundle, train_view, groups, output_dir):
    reductions, retained, sensitivity_rows = {}, {}, []
    for family, columns in groups.items():
        for threshold in SENSITIVITY_THRESHOLDS:
            table, chosen = reduce_family(bundle, train_view, columns, threshold)
            sensitivity_rows.append({
                "Family": family, "FamilyName": FAMILY_NAMES[family], "AbsSpearmanThreshold": threshold,
                "OriginalFeatures": len(columns), "RetainedFeatures": len(chosen),
                "ReductionPercent": 100.0 * (1.0 - len(chosen) / len(columns)),
            })
            if math.isclose(threshold, PRIMARY_CORRELATION_THRESHOLD):
                reductions[family], retained[family] = table, chosen
        reductions[family].to_csv(output_dir / REDUCTION_FILES[family], index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "Reduction_Sensitivity.csv", index=False)
    return reductions, retained, sensitivity


def fold_specific_reductions(bundle, train_view, groups, splits):
    fold_retained, rows = {}, []
    progress = object_core.SimpleProgress(len(splits) * len(groups), "Strict fold reductions")
    for fold, (train_index, _) in enumerate(splits, start=1):
        fold_retained[fold] = {}
        for family, columns in groups.items():
            table, retained = reduce_family(bundle, train_view.iloc[train_index], columns, PRIMARY_CORRELATION_THRESHOLD)
            fold_retained[fold][family] = retained
            for row in table.to_dict("records"):
                rows.append({"Fold": fold, "Family": family, **row})
            progress.update()
    return fold_retained, pd.DataFrame(rows)


def parameter_configurations(number, pilot):
    keys = list(PARAMETER_SPACE)
    all_combinations = list(itertools.product(*(PARAMETER_SPACE[key] for key in keys)))
    if not 1 <= number <= len(all_combinations):
        raise SystemExit("STOP: invalid requested number of parameter configurations.")
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.choice(len(all_combinations), size=number, replace=False)
    configurations = [dict(zip(keys, all_combinations[position])) for position in positions]
    if pilot:
        for configuration in configurations:
            configuration["n_estimators"] = min(120, int(configuration["n_estimators"]))
    return configurations


def make_estimator(bundle, parameters, n_jobs):
    return bundle["xgb"].XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=n_jobs, missing=np.nan, verbosity=0,
        **parameters,
    )


def select_tuning_row(results):
    candidates = results[results.Mean_MacroF1 >= results.Mean_MacroF1.max() - 1e-4]
    candidates = candidates[candidates.Mean_UrbanF1 >= candidates.Mean_UrbanF1.max() - 1e-4]
    candidates = candidates[candidates.Mean_BalancedAccuracy >= candidates.Mean_BalancedAccuracy.max() - 1e-4]
    return candidates.sort_values(
        ["max_depth", "n_estimators", "reg_lambda", "reg_alpha", "Configuration"],
        ascending=[True, True, False, False, True],
    ).iloc[0]


def tune_models(bundle, train_view, training, groups, fold_features, splits, configurations, n_jobs):
    y = training.BinaryLabel.to_numpy(dtype="int64")
    _, weights = binary_core.binary_class_weights(y)
    rows, selected, selected_cv = [], {}, []
    progress = object_core.SimpleProgress(
        len(MODEL_GROUPS) * len(configurations) * len(splits), "Independent spatial CV tuning"
    )
    for model_name, family_names in MODEL_GROUPS.items():
        model_rows = []
        for config_id, parameters in enumerate(configurations, start=1):
            fold_metrics = []
            for fold, (train_index, validation_index) in enumerate(splits, start=1):
                columns = [column for family in family_names for column in fold_features[fold][family]]
                model = make_estimator(bundle, parameters, n_jobs)
                model.fit(train_view.iloc[train_index][columns], y[train_index], sample_weight=weights[train_index])
                probability = model.predict_proba(train_view.iloc[validation_index][columns])[:, 1]
                prediction = (probability >= THRESHOLD).astype("int64")
                evaluated = binary_core.evaluate_binary(bundle, y[validation_index], prediction, probability)
                fold_metrics.append(evaluated)
                progress.update()
            row = {"Model": model_name, "Configuration": config_id, **parameters}
            for metric in ("MacroF1", "Urban_F1", "BalancedAccuracy"):
                values = [entry[metric] for entry in fold_metrics]
                label = "UrbanF1" if metric == "Urban_F1" else metric
                for fold, value in enumerate(values, start=1):
                    row[f"Fold_{fold}_{label}"] = value
                row[f"Mean_{label}"] = float(np.mean(values))
                row[f"Std_{label}"] = float(np.std(values, ddof=1))
            rows.append(row)
            model_rows.append(row)
        result = pd.DataFrame(model_rows)
        chosen = select_tuning_row(result)
        selected[model_name] = {
            key: type(PARAMETER_SPACE[key][0])(chosen[key]) for key in PARAMETER_SPACE
        }
        selected_cv.append({
            "Model": model_name, "SelectedConfiguration": int(chosen.Configuration),
            "Mean_MacroF1": float(chosen.Mean_MacroF1),
            "Mean_UrbanF1": float(chosen.Mean_UrbanF1),
            "Mean_BalancedAccuracy": float(chosen.Mean_BalancedAccuracy),
        })
    return selected, pd.DataFrame(rows), pd.DataFrame(selected_cv)


def final_model_features(retained):
    return {
        model: [column for family in families for column in retained[family]]
        for model, families in MODEL_GROUPS.items()
    }


def fit_final_models(bundle, indexed, training, primary, full, model_features, parameters, n_jobs):
    y_train = training.BinaryLabel.to_numpy(dtype="int64")
    _, sample_weight = binary_core.binary_class_weights(y_train)
    predictions = {"primary": {}, "full": {}}
    probabilities = {"primary": {}, "full": {}}
    evaluations = {"primary": {}, "full": {}}
    models, timing = {}, {"training": 0.0, "prediction": 0.0}
    progress = object_core.SimpleProgress(len(MODEL_GROUPS), "Final reduced models")
    for model_name in MODEL_GROUPS:
        columns = model_features[model_name]
        model = make_estimator(bundle, parameters[model_name], n_jobs)
        started = time.perf_counter()
        model.fit(indexed.loc[training.Object_ID, columns], y_train, sample_weight=sample_weight)
        timing["training"] += time.perf_counter() - started
        models[model_name] = model
        started = time.perf_counter()
        for test_name, frame in (("primary", primary), ("full", full)):
            truth = frame.BinaryLabel.to_numpy(dtype="int64")
            probability = model.predict_proba(indexed.loc[frame.Object_ID, columns])[:, 1]
            if not np.isfinite(probability).all():
                raise SystemExit(f"STOP: {model_name} produced non-finite probabilities.")
            prediction = (probability >= THRESHOLD).astype("int64")
            probabilities[test_name][model_name] = probability
            predictions[test_name][model_name] = prediction
            evaluations[test_name][model_name] = binary_core.evaluate_binary(
                bundle, truth, prediction, probability
            )
        timing["prediction"] += time.perf_counter() - started
        progress.update()
    return models, predictions, probabilities, evaluations, timing


def performance_table(evaluations, model_features):
    mapping = OrderedDict([
        ("OA", "OA"), ("BalancedAccuracy", "BalancedAccuracy"), ("Kappa", "Kappa"),
        ("MCC", "MCC"), ("MacroF1", "MacroF1"), ("mIoU", "mIoU"),
        ("AUROC", "AUROC"), ("AUPRC", "AUPRC"),
        ("UrbanPrecision", "Urban_Precision"), ("UrbanRecall", "Urban_Recall"),
        ("UrbanF1", "Urban_F1"), ("UrbanIoU", "Urban_IoU"),
        ("NonUrbanPrecision", "NonUrban_Precision"), ("NonUrbanRecall", "NonUrban_Recall"),
        ("NonUrbanF1", "NonUrban_F1"), ("NonUrbanIoU", "NonUrban_IoU"),
    ])
    rows = []
    for model_name, result in evaluations.items():
        rows.append({
            "Model": model_name, "Configuration": MODEL_LABELS[model_name],
            "FeatureGroups": "+".join(MODEL_GROUPS[model_name]),
            "NumberOfFeatures": len(model_features[model_name]),
            **{output: float(result[internal]) for output, internal in mapping.items()},
        })
    return pd.DataFrame(rows)


def contribution_table(primary_performance):
    indexed = primary_performance.set_index("Model")
    rows = []
    for first, second, label in COMPARISONS:
        for metric in CONTRIBUTION_METRICS:
            rows.append({
                "Comparison": f"{first} - {second}", "Contribution": label, "Metric": metric,
                "FirstValue": indexed.loc[first, metric], "BaselineValue": indexed.loc[second, metric],
                "ObservedDifference": indexed.loc[first, metric] - indexed.loc[second, metric],
            })
    return pd.DataFrame(rows)


def paired_bootstrap(truth, predictions, replicates):
    rng = np.random.default_rng(RANDOM_SEED)
    class_indices = [np.flatnonzero(truth == code) for code in (0, 1)]
    distributions = {
        (first, second): {metric: np.empty(replicates) for metric in BOOTSTRAP_METRICS}
        for first, second, _ in COMPARISONS
    }
    progress = object_core.SimpleProgress(replicates, "Paired stratified bootstrap")
    for replicate in range(replicates):
        sample = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in class_indices])
        y = truth[sample]
        cached = {}
        for model_name in MODEL_GROUPS:
            cm = np.zeros((2, 2), dtype="int64")
            np.add.at(cm, (y, predictions[model_name][sample]), 1)
            cached[model_name] = binary_core.binary_metrics_from_confusion(cm)
        for first, second, _ in COMPARISONS:
            for metric in BOOTSTRAP_METRICS:
                distributions[(first, second)][metric][replicate] = cached[first][metric] - cached[second][metric]
        progress.update()
    truth_metrics = {}
    for model_name in MODEL_GROUPS:
        cm = np.zeros((2, 2), dtype="int64")
        np.add.at(cm, (truth, predictions[model_name]), 1)
        truth_metrics[model_name] = binary_core.binary_metrics_from_confusion(cm)
    rows = []
    for first, second, label in COMPARISONS:
        for metric in BOOTSTRAP_METRICS:
            values = distributions[(first, second)][metric]
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({
                "Comparison": f"{first} - {second}", "Contribution": label,
                "Metric": metric.replace("_", ""),
                "ObservedDifference": truth_metrics[first][metric] - truth_metrics[second][metric],
                "BootstrapMeanDifference": float(values.mean()),
                "CI_Lower_2.5": float(lower), "CI_Upper_97.5": float(upper),
                "CI_IncludesZero": bool(lower <= 0 <= upper), "Replicates": replicates,
            })
    return pd.DataFrame(rows)


def model_complexity(selected_parameters, model_features, groups):
    original_counts = {
        model: sum(len(groups[family]) for family in families)
        for model, families in MODEL_GROUPS.items()
    }
    columns = {
        "n_estimators": "SelectedNEstimators", "max_depth": "SelectedMaxDepth",
        "learning_rate": "SelectedLearningRate", "min_child_weight": "SelectedMinChildWeight",
        "gamma": "SelectedGamma", "subsample": "SelectedSubsample",
        "colsample_bytree": "SelectedColsampleBytree", "reg_lambda": "SelectedRegLambda",
        "reg_alpha": "SelectedRegAlpha",
    }
    return pd.DataFrame([{
        "Model": model, "Configuration": MODEL_LABELS[model],
        "OriginalPredictorCount": original_counts[model],
        "ReducedPredictorCount": len(model_features[model]),
        **{output: selected_parameters[model][parameter] for parameter, output in columns.items()},
    } for model in MODEL_GROUPS])


def save_predictions(path, membership, predictions, probabilities):
    result = membership[["Object_ID", "BinaryClass", "SourceClass3"]].rename(
        columns={"BinaryClass": "TrueClass"}
    ).copy()
    for model in MODEL_GROUPS:
        result[f"{model}_PredictedClass"] = [binary_core.BINARY_LABELS[int(value)] for value in predictions[model]]
        result[f"{model}_P_Urban"] = probabilities[model]
    result.to_csv(path, index=False)


def save_confusions(path, evaluations):
    rows = []
    for test_name, by_model in evaluations.items():
        for model, result in by_model.items():
            cm = result["confusion"]
            for true in range(2):
                for predicted in range(2):
                    rows.append({
                        "TestSet": test_name, "Model": model,
                        "TrueClass": binary_core.BINARY_LABELS[true],
                        "PredictedClass": binary_core.BINARY_LABELS[predicted],
                        "Count": int(cm[true, predicted]),
                        "RowPercent": float(100 * cm[true, predicted] / cm[true].sum()),
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def save_heatmap(plt, correlation, path, title):
    size = max(6.0, min(15.0, 4.5 + 0.23 * len(correlation)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(correlation.to_numpy(), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(correlation)), labels=correlation.columns, rotation=90,
                  fontsize=5 if len(correlation) > 15 else 8)
    ax.set_yticks(range(len(correlation)), labels=correlation.index,
                  fontsize=5 if len(correlation) > 15 else 8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def correlation_outputs(bundle, train_view, groups, retained, output_dir):
    stems = {"R": "NTL_Radiometric", "G": "Geometry", "T": "Texture", "C": "Spatial_Context"}
    diagnostics = []
    for family, columns in groups.items():
        before = train_view[columns].corr(method="spearman", min_periods=2)
        after = train_view[retained[family]].corr(method="spearman", min_periods=2)
        before.to_csv(output_dir / f"{stems[family]}_Spearman_Before.csv")
        after.to_csv(output_dir / f"{stems[family]}_Spearman_After.csv")
        save_heatmap(bundle["plt"], before, output_dir / f"{stems[family]}_Correlation_Before.png",
                     f"{FAMILY_NAMES[family]}: before reduction")
        save_heatmap(bundle["plt"], after, output_dir / f"{stems[family]}_Correlation_After.png",
                     f"{FAMILY_NAMES[family]}: after reduction")
        for stage, matrix in (("Before", before), ("After", after)):
            values = matrix.abs().to_numpy(dtype="float64")
            values[np.eye(len(values), dtype=bool)] = np.nan
            diagnostics.append({
                "Family": family, "Stage": stage, "FeatureCount": len(matrix),
                "MaximumAbsoluteOffDiagonalSpearman": float(np.nanmax(values)) if len(matrix) > 1 else 0.0,
                "PairsAbsRhoAtLeast090": int(np.nansum(np.triu(values >= 0.90, 1))),
            })
    return pd.DataFrame(diagnostics)


def performance_figure(plt, performance, path):
    metrics = ["MacroF1", "mIoU", "UrbanF1", "UrbanIoU"]
    x = np.arange(len(performance))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 1.5) * width, performance[metric], width, label=metric)
    ax.set_xticks(x, performance.Model)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Reduced feature configuration")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def importance_outputs(models, model_features, output_dir):
    score = models["R4"].get_booster().get_score(importance_type="gain")
    rows = []
    total = sum(score.values())
    feature_family = {
        feature: family for family, columns in EXPECTED_GROUPS.items() for feature in columns
    }
    for feature in model_features["R4"]:
        gain = float(score.get(feature, 0.0))
        rows.append({
            "Feature": feature, "Family": feature_family[feature], "Gain": gain,
            "NormalizedGain": gain / total if total else 0.0,
        })
    detail = pd.DataFrame(rows).sort_values("NormalizedGain", ascending=False)
    summary = detail.groupby("Family", as_index=False).agg(
        RetainedFeatures=("Feature", "count"), NormalizedGain=("NormalizedGain", "sum")
    )
    detail.to_csv(output_dir / "R4_Gain_Feature_Importance.csv", index=False)
    summary.to_csv(output_dir / "R4_Gain_Family_Importance.csv", index=False)
    return detail, summary


def choose_parsimonious(performance, bootstrap, pilot):
    supported = []
    for model in ("R1", "R2", "R3", "R4"):
        subset = bootstrap[(bootstrap.Comparison == f"{model} - R0") &
                           (bootstrap.Metric.isin(["MacroF1", "mIoU", "UrbanF1", "UrbanIoU"]))]
        if len(subset) == 4 and bool((subset["CI_Lower_2.5"] > 0).all()):
            supported.append(model)
    if not supported:
        selected = "R0"
        reason = "No augmented configuration had consistently positive four-metric paired intervals over R0."
    else:
        candidate = performance[performance.Model.isin(supported)].copy()
        best = candidate.MacroF1.max()
        candidate = candidate[candidate.MacroF1 >= best - 0.001]
        selected = candidate.sort_values(["NumberOfFeatures", "UrbanF1"], ascending=[True, False]).iloc[0].Model
        reason = "Selected among bootstrap-supported models by Macro-F1, then predictor-count parsimony."
    return {
        "selected_model": selected, "selected_configuration": MODEL_LABELS[selected],
        "reason": reason, "bootstrap_supported_models": supported, "pilot": pilot,
    }


def markdown_table(frame, digits=4):
    return spectral_core.markdown_table(frame, digits=digits)


def report(path, pilot, inputs, groups, reductions, retained, sensitivity, correlation_diag,
           training_full, training_run, primary, full, neighbor_count, fold_audit,
           selected_parameters, selected_cv, primary_perf, full_perf, contribution,
           bootstrap, complexity, importance_summary, selection, runtimes, missing_path,
           output_dir):
    reduction_summary = pd.DataFrame([{
        "Family": FAMILY_NAMES[family], "Original": len(groups[family]),
        "Retained": len(retained[family]),
        "ReductionPercent": 100 * (1 - len(retained[family]) / len(groups[family])),
        "RemovedFeatures": ", ".join(reductions[family].loc[~reductions[family].Retained, "OriginalFeature"]),
    } for family in ("R", "G", "T", "C")])
    counts = pd.DataFrame([
        {"Dataset": "Full validated training", "Objects": len(training_full),
         "Urban": int((training_full.BinaryLabel == 1).sum()), "NonUrban": int((training_full.BinaryLabel == 0).sum()),
         "SpatialCells": training_full.SpatialCell.nunique()},
        {"Dataset": "Training used in this run", "Objects": len(training_run),
         "Urban": int((training_run.BinaryLabel == 1).sum()), "NonUrban": int((training_run.BinaryLabel == 0).sum()),
         "SpatialCells": training_run.SpatialCell.nunique()},
        {"Dataset": "Primary test", "Objects": len(primary), "Urban": int((primary.BinaryLabel == 1).sum()),
         "NonUrban": int((primary.BinaryLabel == 0).sum()), "SpatialCells": primary.SpatialCell.nunique()},
        {"Dataset": "Full manual test", "Objects": len(full), "Urban": int((full.BinaryLabel == 1).sum()),
         "NonUrban": int((full.BinaryLabel == 0).sum()), "SpatialCells": full.SpatialCell.nunique()},
    ])
    selected_params_table = pd.DataFrame([
        {"Model": model, **parameters} for model, parameters in selected_parameters.items()
    ])
    performance_columns = ["Model", "Configuration", "NumberOfFeatures", "OA", "BalancedAccuracy",
                           "MacroF1", "mIoU", "UrbanPrecision", "UrbanRecall", "UrbanF1", "UrbanIoU"]
    primary_index = primary_perf.set_index("Model")
    contributions = {
        label: primary_index.loc[first, "UrbanF1"] - primary_index.loc[second, "UrbanF1"]
        for first, second, label in COMPARISONS
    }
    clusters = []
    for family in ("R", "G", "T", "C"):
        table = reductions[family]
        for cluster_id, cluster in table[table.ClusterID > 0].groupby("ClusterID"):
            clusters.append({
                "Family": family, "Cluster": int(cluster_id),
                "Representative": cluster.RepresentativeFeature.iloc[0],
                "Members": ", ".join(cluster.OriginalFeature),
            })
    lines = [
        f"# Reduced NTL Object Ablation XGBoost 01 — {'PILOT' if pilot else 'FULL RUN'}", "",
        f"Output directory: `{output_dir.resolve()}`", "",
        "## A. Objective", "",
        "This binary Urban/NonUrban experiment tests Geometry, BG-Log Texture, and NTL Spatial Context beyond a reduced NTL radiometric baseline. Redundancy is removed first to limit correlated dimensionality while preserving the physical identity of each family. Sentinel-2 affects only the pre-existing segmentation and supplies no classification predictor.", "",
        "## B. Original feature space", "",
    ]
    for family in ("R", "G", "T", "C"):
        lines += [f"- **{FAMILY_NAMES[family]} ({len(groups[family])}):** " + ", ".join(groups[family])]
    lines += ["", "## C. Unsupervised reduction method", "",
              "Within each family, constant/near-zero-variance variables are removed; pairwise-valid training-only Spearman correlations define distance `1−|rho|`; average-linkage hierarchical clustering is cut at `|rho|=0.90`; and each cluster is represented by its medoid (highest mean within-cluster absolute correlation, with original manifest order as the deterministic interpretability tie-break). Labels, importance, and test performance are never used. During spatial CV this procedure is relearned inside each fold-training partition.", "",
              "## D. Reduction results", "", markdown_table(reduction_summary), "", markdown_table(pd.DataFrame(clusters)), "",
              "Sensitivity counts at 0.85, 0.90, and 0.95:", "", markdown_table(sensitivity), "",
              "## E. Redundancy diagnostics", "", markdown_table(correlation_diag), "",
              "Before/after matrices and blue–white–red heatmaps were saved for every family. A value of zero after reduction is reported for a single retained feature because no off-diagonal pair remains.", "",
              "## F. Training and test design", "", markdown_table(counts), "",
              f"Training/full-test overlap = `0`; training/test-neighbour overlap = `0`; unique first-order full-test neighbours = `{neighbor_count}`. Memberships and labels were read unchanged from the validated binary datasets.", "", markdown_table(fold_audit), "",
              "## G. Hyperparameter tuning", "", markdown_table(selected_params_table), "",
              "All models used the same reproducibly sampled search configurations, independent optimization, five identical GroupKFold spatial splits, inverse-frequency binary weights, seed 20261001, and a fixed 0.50 decision threshold.", "",
              "## H. Spatial CV performance", "", markdown_table(selected_cv), "",
              "The reported CV scores use strict fold-specific family reduction learned without each validation fold.", "",
              "## I. Primary independent results", "", markdown_table(primary_perf[performance_columns]), "",
              "## J. Full test results", "", markdown_table(full_perf[performance_columns]), "",
              "## K. Geometry contribution", "", markdown_table(contribution[contribution.Contribution == "Geometry effect"]), "",
              "## L. Context contribution", "", markdown_table(contribution[contribution.Contribution == "Context effect"]), "",
              "## M. Texture contribution", "", markdown_table(contribution[contribution.Contribution == "Texture effect"]), "",
              "## N. Total object-specific contribution", "", markdown_table(contribution[contribution.Contribution == "All object features"]), "",
              "## O. Bootstrap uncertainty", "", markdown_table(bootstrap), "",
              "Intervals are paired, class-stratified 95% intervals; they are not automatically labelled statistically significant.", "",
              "## P. Model complexity", "", markdown_table(complexity), "",
              "R4 gain importance is secondary interpretation only and did not influence reduction or selection:", "", markdown_table(importance_summary), "",
              "## Q. Scientific interpretation", "",
              f"1. Geometry Urban-F1 gain over R0: `{contributions['Geometry effect']:+.4f}`.",
              f"2. Spatial Context Urban-F1 gain over R0: `{contributions['Context effect']:+.4f}`.",
              f"3. Texture Urban-F1 gain over R0: `{contributions['Texture effect']:+.4f}`.",
              f"4. All object families Urban-F1 gain over R0: `{contributions['All object features']:+.4f}`.",
              f"5. Largest independent Urban-F1 gain: **{max(contributions, key=contributions.get)}** (`{max(contributions.values()):+.4f}`).",
              "6. Primary and full-test tables above show whether each gain is maintained on the complete manual test pool.", "",
              "## R. Manuscript implication", "",
              "The experiment isolates whether object geometry, local NTL structure, and BG-Log texture remain informative after redundancy within the NTL radiometric baseline and each object family is removed without label supervision. Conclusions must be based on the independent-test differences and their paired uncertainty, not on model size alone.", "",
              "## S. Runtime", "", markdown_table(pd.DataFrame([runtimes])), "",
              "## T. Deviations", ""]
    if pilot:
        lines += ["Pilot-only deviations: 1,000 objects per original source class (or configured cap), three sampled hyperparameter configurations per model, estimator count capped at 120, and 200 bootstrap replicates. All reduction, strict-fold CV, tuning, fitting, and evaluation logic remains active. Full-dataset conclusions were not produced.", ""]
    else:
        lines += ["No methodological deviations. The requested 30 sampled configurations and 2,000 paired bootstrap replicates were used.", ""]
    lines += [
        "## Validation checks", "",
        "- No Sentinel-2 optical classification predictor: **passed**.",
        "- Reduction is label-free, training-only, family-wise, and fold-specific during CV: **passed**.",
        "- Identical memberships, spatial folds, class-weight rule, and search protocol across R0–R4: **passed**.",
        "- Independent test data used in reduction or tuning: **0 objects**.",
        "- Arbitrary imputation: **none**; XGBoost receives structural NaNs natively.",
        "- Texture discarded solely for NaNs: **none**.",
        "- Training/full-test overlap and test-neighbour leakage: **zero**.",
        f"- Complete missingness audit: `{missing_path.resolve()}`.", "",
        "## Input and output paths", "",
    ]
    for label, value in inputs.items():
        lines.append(f"- {label}: `{Path(value).resolve()}`")
    lines += ["", "## Main scientific findings", "",
              f"- Geometry gain (Urban F1): `{contributions['Geometry effect']:+.4f}`.",
              f"- Context gain (Urban F1): `{contributions['Context effect']:+.4f}`.",
              f"- Texture gain (Urban F1): `{contributions['Texture effect']:+.4f}`.",
              f"- All-feature gain (Urban F1): `{contributions['All object features']:+.4f}`.", "",
              "## Recommended manuscript configuration", "",
              f"**{selection['selected_model']} — {selection['selected_configuration']}**. {selection['reason']}"
              + (" This recommendation is preliminary and must be replaced by the full-run result." if pilot else ""), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pilot = not args.full_run
    output_dir = args.output_dir or (
        SEG_DIR / ("Reduced_NTL_Object_Ablation_XGBoost_01_pilot" if pilot else "Reduced_NTL_Object_Ablation_XGBoost_01")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    runtimes = {}

    started = time.perf_counter()
    require_inputs()
    bundle = dependencies()
    groups = load_groups()
    training_full, primary, full, neighbors = load_memberships()
    features = load_features(groups)
    verify_joins(features, [("training", training_full), ("primary", primary), ("full", full)])
    training = pilot_training_subset(training_full, args.pilot_per_source) if pilot else training_full.copy()
    missing = missingness_audit(features, groups, [
        ("Training", training), ("PrimaryTest", primary), ("FullTest", full)
    ])
    missing_path = output_dir / "Candidate_Feature_Missingness.csv"
    missing.to_csv(missing_path, index=False)
    indexed = features.set_index("Object_ID")
    train_view = indexed.loc[training.Object_ID, [c for values in groups.values() for c in values]].reset_index(drop=True)
    runtimes["InputAndValidationSeconds"] = time.perf_counter() - started

    started = time.perf_counter()
    reductions, retained, sensitivity = full_training_reductions(bundle, train_view, groups, output_dir)
    model_features = final_model_features(retained)
    manifest_rows = []
    for family in ("R", "G", "T", "C"):
        for order, feature in enumerate(retained[family], start=1):
            manifest_rows.append({
                "FeatureFamily": family, "FamilyName": FAMILY_NAMES[family],
                "OrderWithinFamily": order, "Feature": feature,
                "IncludedInModels": ",".join(model for model, columns in model_features.items() if feature in columns),
            })
    pd.DataFrame(manifest_rows).to_csv(output_dir / "Reduced_Feature_Manifest.csv", index=False)
    correlation_diag = correlation_outputs(bundle, train_view, groups, retained, output_dir)
    correlation_diag.to_csv(output_dir / "Reduction_Correlation_Diagnostics.csv", index=False)
    runtimes["FullTrainingReductionSeconds"] = time.perf_counter() - started

    y_train = training.BinaryLabel.to_numpy(dtype="int64")
    spatial_groups = training.SpatialCell.to_numpy()
    splits = list(bundle["GroupKFold"](n_splits=5).split(np.zeros((len(training), 1)), y_train, spatial_groups))
    fold_rows = []
    for fold, (train_i, valid_i) in enumerate(splits, start=1):
        if len(np.unique(y_train[train_i])) != 2 or len(np.unique(y_train[valid_i])) != 2:
            raise SystemExit(f"STOP: a binary class is absent from spatial fold {fold}.")
        fold_rows.append({
            "Fold": fold, "TrainingObjects": len(train_i), "ValidationObjects": len(valid_i),
            "TrainingSpatialCells": len(np.unique(spatial_groups[train_i])),
            "ValidationSpatialCells": len(np.unique(spatial_groups[valid_i])),
            "TrainingUrban": int((y_train[train_i] == 1).sum()),
            "TrainingNonUrban": int((y_train[train_i] == 0).sum()),
            "ValidationUrban": int((y_train[valid_i] == 1).sum()),
            "ValidationNonUrban": int((y_train[valid_i] == 0).sum()),
        })
    fold_audit = pd.DataFrame(fold_rows)
    fold_audit.to_csv(output_dir / "Spatial_CV_Fold_Audit.csv", index=False)

    started = time.perf_counter()
    fold_features, fold_reduction_table = fold_specific_reductions(bundle, train_view, groups, splits)
    fold_reduction_table.to_csv(output_dir / "Strict_Fold_Reduction_Manifest.csv", index=False)
    runtimes["StrictFoldReductionSeconds"] = time.perf_counter() - started

    number_configs = args.pilot_configs if pilot else args.search_configs
    configurations = parameter_configurations(number_configs, pilot)
    pd.DataFrame(configurations).to_csv(output_dir / "Shared_Hyperparameter_Search_Configurations.csv", index=False)
    started = time.perf_counter()
    selected_parameters, tuning_results, selected_cv = tune_models(
        bundle, train_view, training, groups, fold_features, splits, configurations, args.n_jobs
    )
    tuning_results.to_csv(output_dir / "Reduced_NTL_Object_Spatial_CV_Tuning.csv", index=False)
    selected_cv.to_csv(output_dir / "Reduced_NTL_Object_Selected_CV_Performance.csv", index=False)
    (output_dir / "Reduced_NTL_Object_Selected_Hyperparameters.json").write_text(
        json.dumps({"random_state": RANDOM_SEED, "threshold": THRESHOLD,
                    "selected_hyperparameters": selected_parameters}, indent=2), encoding="utf-8"
    )
    runtimes["IndependentTuningSeconds"] = time.perf_counter() - started

    started = time.perf_counter()
    models, predictions, probabilities, evaluations, fit_timing = fit_final_models(
        bundle, indexed, training, primary, full, model_features, selected_parameters, args.n_jobs
    )
    runtimes["FinalModelTrainingSeconds"] = fit_timing["training"]
    runtimes["IndependentPredictionSeconds"] = fit_timing["prediction"]
    runtimes["FinalFitStageSeconds"] = time.perf_counter() - started

    primary_perf = performance_table(evaluations["primary"], model_features)
    full_perf = performance_table(evaluations["full"], model_features)
    primary_perf.to_csv(output_dir / "Reduced_NTL_Object_Ablation_Performance.csv", index=False)
    full_perf.to_csv(output_dir / "Reduced_NTL_Object_Ablation_FullTest_Performance.csv", index=False)
    contribution = contribution_table(primary_perf)
    contribution.to_csv(output_dir / "Reduced_NTL_Object_Feature_Contribution.csv", index=False)
    complexity = model_complexity(selected_parameters, model_features, groups)
    complexity.to_csv(output_dir / "Reduced_NTL_Object_Model_Complexity.csv", index=False)
    save_predictions(output_dir / "Reduced_NTL_Object_Primary_Predictions.csv", primary,
                     predictions["primary"], probabilities["primary"])
    save_predictions(output_dir / "Reduced_NTL_Object_FullTest_Predictions.csv", full,
                     predictions["full"], probabilities["full"])
    save_confusions(output_dir / "Reduced_NTL_Object_Confusion_Matrices.csv", evaluations)

    started = time.perf_counter()
    replicates = args.pilot_bootstrap if pilot else 2000
    bootstrap = paired_bootstrap(primary.BinaryLabel.to_numpy(dtype="int64"), predictions["primary"], replicates)
    bootstrap.to_csv(output_dir / "Reduced_NTL_Object_Paired_Bootstrap.csv", index=False)
    selection = choose_parsimonious(primary_perf, bootstrap, pilot)
    (output_dir / "Reduced_NTL_Object_Parsimony_Selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    runtimes["BootstrapAndSelectionSeconds"] = time.perf_counter() - started

    started = time.perf_counter()
    performance_figure(bundle["plt"], primary_perf, output_dir / "Reduced_NTL_Object_Ablation_Performance.png")
    _, importance_summary = importance_outputs(models, model_features, output_dir)
    runtimes["FiguresAndImportanceSeconds"] = time.perf_counter() - started
    runtimes["TotalSeconds"] = time.perf_counter() - total_started

    report(
        output_dir / "Reduced_NTL_Object_Ablation_XGBoost_01_Report.md", pilot,
        {"Context feature table": CONTEXT_TABLE, "Texture feature table": TEXTURE_TABLE,
         "Training membership": TRAINING_PATH, "Primary test": PRIMARY_PATH,
         "Full manual test": FULL_PATH, "Rook adjacency": ADJACENCY_PATH,
         "Validated family manifest": SOURCE_MANIFEST},
        groups, reductions, retained, sensitivity, correlation_diag, training_full, training,
        primary, full, len(neighbors), fold_audit, selected_parameters, selected_cv,
        primary_perf, full_perf, contribution, bootstrap, complexity, importance_summary,
        selection, runtimes, missing_path, output_dir,
    )
    print(f"Completed {'pilot' if pilot else 'full run'}: {output_dir.resolve()}")
    print("Retained features:", {family: len(values) for family, values in retained.items()})
    print("Preliminary selection:" if pilot else "Final selection:",
          selection["selected_model"], selection["selected_configuration"])
    if pilot:
        print("Full dataset was not run. Use --full-run manually after reviewing the pilot report.")


if __name__ == "__main__":
    main()
