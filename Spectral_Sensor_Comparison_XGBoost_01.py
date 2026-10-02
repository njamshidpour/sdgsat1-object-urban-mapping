#!/usr/bin/env python
"""Controlled NTL versus Sentinel-2 object-spectral XGBoost comparison.

M1 uses only SDGSAT-1 nighttime-light spectral/radiometric attributes, M2
uses only corrected Sentinel-2 spectral attributes, and M3 combines both.
Training and independent test memberships are reused unchanged from the full
binary stratified experiment.  Geometry, texture and spatial context are not
predictors in this experiment.
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

import Object_Based_Ablation_XGBoost_01 as multiclass_core
import Binary_Stratified_Object_Ablation_XGBoost_01 as binary_core


RANDOM_SEED = 20261001
THRESHOLD = 0.50
PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = PROJECT_ROOT / "Object_based_fusion" / "Segmentation_output"
BINARY_OUTPUT_DIR = SEGMENTATION_DIR / "Binary_Stratified_Object_Ablation_XGBoost_01"

DEFAULT_TRAINING = BINARY_OUTPUT_DIR / "Binary_Stratified_Training_Set.csv"
DEFAULT_PRIMARY_TEST = BINARY_OUTPUT_DIR / "Binary_Primary_Stratified_Test_Set.csv"
DEFAULT_FULL_TEST = BINARY_OUTPUT_DIR / "Binary_Secondary_Full_Test_Set.csv"
DEFAULT_SEGMENTATION = SEGMENTATION_DIR / "eCog_segments.shp"
DEFAULT_FEATURE_TABLE = (
    SEGMENTATION_DIR
    / "Spatial_Context_Extraction_02"
    / "objects_features_with_spatial_context_02.csv"
)
DEFAULT_BINARY_MANIFEST = BINARY_OUTPUT_DIR / "Binary_Ablation_Feature_Group_Manifest.csv"

NTL_FEATURES = list(multiclass_core.FEATURE_GROUPS["N"])
S2_FEATURES = [
    "S2_B2_mean", "S2_B2_std", "S2_B2_max",
    "S2_B3_mean", "S2_B3_std", "S2_B3_max",
    "S2_B4_mean", "S2_B4_std", "S2_B4_max",
    "S2_B5_mean", "S2_B5_std", "S2_B5_max",
    "S2_B6_mean", "S2_B6_std", "S2_B6_max",
    "S2_B7_mean", "S2_B7_std", "S2_B7_max",
    "S2_B8_mean", "S2_B8_std", "S2_B8_max",
    "S2_B8A_mean", "S2_B8A_std", "S2_B8A_max",
    "S2_B11_mean", "S2_B11_std", "S2_B11_max",
    "S2_B12_mean", "S2_B12_std", "S2_B12_max",
    "NDVI_mean", "NDVI_std", "NDVI_max",
    "NDBI_mean", "NDBI_std", "NDBI_max",
    "NDWI_mean", "NDWI_std", "NDWI_max",
]

FEATURE_GROUPS = {
    "NTL": NTL_FEATURES,
    "S2": S2_FEATURES,
}
MODEL_GROUPS = OrderedDict([
    ("M1", ("NTL",)),
    ("M2", ("S2",)),
    ("M3", ("NTL", "S2")),
])
MODEL_SOURCES = {
    "M1": "NTL",
    "M2": "Sentinel-2",
    "M3": "NTL + Sentinel-2",
}
COMPARISONS = [("M2", "M1"), ("M3", "M1"), ("M3", "M2")]
DIFFERENCE_METRICS = [
    "BalancedAccuracy", "MacroF1", "mIoU", "UrbanPrecision",
    "UrbanRecall", "UrbanF1", "UrbanIoU",
]
BOOTSTRAP_METRICS = [
    "BalancedAccuracy", "MacroF1", "mIoU", "UrbanF1", "UrbanIoU"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled NTL/Sentinel-2 object-spectral XGBoost comparison."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--training-set", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--primary-test-set", type=Path, default=DEFAULT_PRIMARY_TEST)
    parser.add_argument("--full-test-set", type=Path, default=DEFAULT_FULL_TEST)
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--binary-manifest", type=Path, default=DEFAULT_BINARY_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--search-configs", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--pilot-configs", type=int, default=3)
    parser.add_argument("--pilot-bootstrap", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def dependency_bundle():
    return binary_core.dependency_bundle()


def load_reused_dataset(path: Path, name: str):
    data = pd.read_csv(path)
    required = {"Object_ID", "SourceClass3", "BinaryClass", "BinaryLabel"}
    if name == "training":
        required.add("SpatialCell")
    missing = sorted(required.difference(data.columns))
    if missing:
        raise SystemExit(f"STOP: {name} dataset lacks columns: {', '.join(missing)}")
    if data["Object_ID"].isna().any() or data["Object_ID"].duplicated().any():
        raise SystemExit(f"STOP: {name} Object_ID is missing or duplicated.")
    data["Object_ID"] = data["Object_ID"].astype("int64")
    data["BinaryLabel"] = data["BinaryLabel"].astype("int64")
    expected_class = data["BinaryLabel"].map({0: "NonUrban", 1: "Urban"})
    if not expected_class.equals(data["BinaryClass"]):
        raise SystemExit(f"STOP: {name} binary class names and codes disagree.")
    if set(data["BinaryLabel"].unique()) != {0, 1}:
        raise SystemExit(f"STOP: {name} does not contain both binary classes.")
    return data


def validate_ntl_manifest(path: Path):
    manifest = pd.read_csv(path)
    required = {"FeatureGroup", "OrderWithinGroup", "Feature"}
    if not required.issubset(manifest.columns):
        raise SystemExit("STOP: binary feature manifest has an invalid structure.")
    rows = manifest[manifest["FeatureGroup"] == "N"].sort_values("OrderWithinGroup")
    if rows["Feature"].tolist() != NTL_FEATURES:
        raise SystemExit("STOP: NTL group differs from the previous final binary experiment.")
    return manifest


def load_spectral_features(path: Path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["Object_ID", *NTL_FEATURES, *S2_FEATURES]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(
            "STOP: corrected canonical feature table lacks required spectral columns:\n- "
            + "\n- ".join(missing)
        )
    data = pd.read_csv(path, usecols=required)
    if data["Object_ID"].isna().any() or data["Object_ID"].duplicated().any():
        raise SystemExit("STOP: canonical feature table has missing or duplicate Object_ID.")
    data["Object_ID"] = data["Object_ID"].astype("int64")
    return data


def exact_duplicate_columns(frame: pd.DataFrame, columns):
    duplicates = []
    for position, first in enumerate(columns):
        for second in columns[position + 1:]:
            if frame[first].equals(frame[second]):
                duplicates.append((first, second))
    return duplicates


def audit_s2_features(features: pd.DataFrame):
    rows = []
    defects = []
    for column in S2_FEATURES:
        series = features[column]
        valid = series.dropna()
        nan_count = int(series.isna().sum())
        nan_percentage = 100.0 * nan_count / len(series)
        finite = np.isfinite(valid.to_numpy(dtype="float64")).all()
        unique = int(valid.nunique())
        constant = unique <= 1
        rows.append({
            "FeatureName": column,
            "NaNCount": nan_count,
            "NaNPercentage": nan_percentage,
            "Minimum": float(valid.min()) if len(valid) else np.nan,
            "Maximum": float(valid.max()) if len(valid) else np.nan,
            "Mean": float(valid.mean()) if len(valid) else np.nan,
            "StandardDeviation": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            "UniqueValues": unique,
            "IsConstant": bool(constant),
            "AllValidValuesFinite": bool(finite),
        })
        if nan_percentage > 5.0:
            defects.append(f"{column}: NaN percentage {nan_percentage:.3f}% exceeds 5%")
        if constant:
            defects.append(f"{column}: constant predictor")
        if not finite:
            defects.append(f"{column}: non-finite valid values")

    duplicates = exact_duplicate_columns(features, S2_FEATURES)
    for first, second in duplicates:
        defects.append(f"exact duplicate columns: {first} and {second}")

    for index_name in ("NDVI", "NDBI", "NDWI"):
        for statistic in ("mean", "max"):
            column = f"{index_name}_{statistic}"
            if features[column].min() < -1.000001 or features[column].max() > 1.000001:
                defects.append(f"{column}: outside theoretical [-1, 1] extent")
        std_column = f"{index_name}_std"
        if features[std_column].min() < -1e-12 or features[std_column].max() > 1.000001:
            defects.append(f"{std_column}: outside theoretical [0, 1] extent")

    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    for band in band_names:
        if features[f"S2_{band}_std"].min() < -1e-12:
            defects.append(f"S2_{band}_std: negative within-object standard deviation")
        violation = (
            features[f"S2_{band}_max"] + 1e-9 < features[f"S2_{band}_mean"]
        ).sum()
        if violation:
            defects.append(f"S2_{band}: {int(violation)} objects have max < mean")

    b11_b12 = []
    for statistic in ("mean", "std", "max"):
        first = features[f"S2_B11_{statistic}"]
        second = features[f"S2_B12_{statistic}"]
        b11_b12.append({
            "Statistic": statistic,
            "ExactEqualityFraction": float((first == second).mean()),
            "PearsonCorrelation": float(first.corr(second, method="pearson")),
            "SpearmanCorrelation": float(first.corr(second, method="spearman")),
        })
    if all(row["ExactEqualityFraction"] == 1.0 for row in b11_b12):
        defects.append("B11 and B12 remain exactly duplicated for mean/std/max")

    duplicate_row_mask = features.duplicated(subset=S2_FEATURES, keep=False)
    duplicate_vector_rows = int(duplicate_row_mask.sum())
    duplicate_vector_groups = int(
        features.loc[duplicate_row_mask, S2_FEATURES].drop_duplicates().shape[0]
    )
    diagnostics = {
        "feature_count": len(S2_FEATURES),
        "exact_duplicate_column_pairs": [list(pair) for pair in duplicates],
        "duplicate_feature_vector_rows": duplicate_vector_rows,
        "duplicate_feature_vector_groups": duplicate_vector_groups,
        "b11_b12": b11_b12,
        "defects": defects,
        "passed": len(defects) == 0,
    }
    return pd.DataFrame(rows), diagnostics


def audit_ntl_missing(features: pd.DataFrame):
    return pd.DataFrame([
        {
            "FeatureName": column,
            "NaNCount": int(features[column].isna().sum()),
            "NaNPercentage": 100.0 * features[column].isna().mean(),
            "NonFiniteValidCount": int((~np.isfinite(features[column].dropna())).sum()),
        }
        for column in NTL_FEATURES
    ])


def write_manifest(path: Path):
    rows = []
    for group, columns in FEATURE_GROUPS.items():
        for order, feature in enumerate(columns, start=1):
            rows.append({
                "FeatureName": feature,
                "SensorGroup": group,
                "OrderWithinGroup": order,
            })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(path, index=False)
    return manifest


def make_estimator(xgb, parameters, n_jobs):
    return binary_core.make_estimator(xgb, parameters, n_jobs)


def tune_m3(bundle, X, y, groups, sample_weight, configurations, folds, n_jobs):
    splits = list(bundle["GroupKFold"](n_splits=folds).split(X, y, groups=groups))
    progress = multiclass_core.SimpleProgress(
        len(configurations) * len(splits), "M3 spatial CV tuning"
    )
    rows = []
    for config_number, parameters in enumerate(configurations, start=1):
        macro_scores = []
        balanced_scores = []
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            if len(np.unique(y[train_index])) != 2 or len(np.unique(y[validation_index])) != 2:
                raise SystemExit(f"STOP: class absent from spatial fold {fold_number}.")
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
        row = {"Configuration": config_number, **parameters}
        for index, value in enumerate(macro_scores, start=1):
            row[f"Fold_{index}_MacroF1"] = float(value)
        for index, value in enumerate(balanced_scores, start=1):
            row[f"Fold_{index}_BalancedAccuracy"] = float(value)
        row["Mean_MacroF1"] = float(np.mean(macro_scores))
        row["Std_MacroF1"] = float(np.std(macro_scores, ddof=1))
        row["Mean_BalancedAccuracy"] = float(np.mean(balanced_scores))
        rows.append(row)
    results = pd.DataFrame(rows)
    best_score = results["Mean_MacroF1"].max()
    tied = results[results["Mean_MacroF1"] >= best_score - 1e-4].sort_values(
        ["max_depth", "n_estimators", "Mean_MacroF1"],
        ascending=[True, True, False],
    )
    selected = tied.iloc[0]
    parameters = {
        key: type(multiclass_core.PARAMETER_SPACE[key][0])(selected[key])
        for key in multiclass_core.PARAMETER_SPACE
    }
    return (
        parameters,
        float(selected["Mean_MacroF1"]),
        float(selected["Mean_BalancedAccuracy"]),
        results,
        splits,
    )


def fit_models(bundle, features, training, primary, full_test, parameters, n_jobs):
    indexed = features.set_index("Object_ID")
    train_ids = training["Object_ID"].to_numpy(dtype="int64")
    primary_ids = primary["Object_ID"].to_numpy(dtype="int64")
    full_ids = full_test["Object_ID"].to_numpy(dtype="int64")
    y_train = training["BinaryLabel"].to_numpy(dtype="int64")
    y_primary = primary["BinaryLabel"].to_numpy(dtype="int64")
    y_full = full_test["BinaryLabel"].to_numpy(dtype="int64")
    weights, sample_weight = binary_core.binary_class_weights(y_train)
    predictions = {"primary": {}, "full": {}}
    probabilities = {"primary": {}, "full": {}}
    evaluations = {"primary": {}, "full": {}}
    training_seconds = 0.0
    prediction_seconds = 0.0
    progress = multiclass_core.SimpleProgress(len(MODEL_GROUPS), "Sensor models")
    for model_name, group_names in MODEL_GROUPS.items():
        columns = [column for group in group_names for column in FEATURE_GROUPS[group]]
        X_train = indexed.loc[train_ids, columns].to_numpy(dtype="float32", copy=True)
        X_primary = indexed.loc[primary_ids, columns].to_numpy(dtype="float32", copy=True)
        X_full = indexed.loc[full_ids, columns].to_numpy(dtype="float32", copy=True)
        model = make_estimator(bundle["xgb"], parameters, n_jobs)
        started = time.perf_counter()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        p_primary = model.predict_proba(X_primary)[:, 1]
        p_full = model.predict_proba(X_full)[:, 1]
        prediction_seconds += time.perf_counter() - started
        if not np.isfinite(p_primary).all() or not np.isfinite(p_full).all():
            raise SystemExit(f"STOP: {model_name} produced non-finite probabilities.")
        pred_primary = (p_primary >= THRESHOLD).astype("int64")
        pred_full = (p_full >= THRESHOLD).astype("int64")
        probabilities["primary"][model_name] = p_primary
        probabilities["full"][model_name] = p_full
        predictions["primary"][model_name] = pred_primary
        predictions["full"][model_name] = pred_full
        evaluations["primary"][model_name] = binary_core.evaluate_binary(
            bundle, y_primary, pred_primary, p_primary
        )
        evaluations["full"][model_name] = binary_core.evaluate_binary(
            bundle, y_full, pred_full, p_full
        )
        progress.update()
    return (
        predictions, probabilities, evaluations, weights,
        training_seconds, prediction_seconds,
    )


def performance_table(evaluations):
    rows = []
    for model, result in evaluations.items():
        groups = MODEL_GROUPS[model]
        rows.append({
            "Model": model,
            "FeatureSource": MODEL_SOURCES[model],
            "NumberOfFeatures": sum(len(FEATURE_GROUPS[group]) for group in groups),
            "OA": result["OA"],
            "BalancedAccuracy": result["BalancedAccuracy"],
            "Kappa": result["Kappa"],
            "MCC": result["MCC"],
            "MacroF1": result["MacroF1"],
            "mIoU": result["mIoU"],
            "AUROC": result["AUROC"],
            "AUPRC": result["AUPRC"],
            "UrbanPrecision": result["Urban_Precision"],
            "UrbanRecall": result["Urban_Recall"],
            "UrbanF1": result["Urban_F1"],
            "UrbanIoU": result["Urban_IoU"],
            "NonUrbanPrecision": result["NonUrban_Precision"],
            "NonUrbanRecall": result["NonUrban_Recall"],
            "NonUrbanF1": result["NonUrban_F1"],
            "NonUrbanIoU": result["NonUrban_IoU"],
        })
    return pd.DataFrame(rows)


def point_differences(performance: pd.DataFrame):
    values = performance.set_index("Model")
    rows = []
    for first, second in COMPARISONS:
        for metric in DIFFERENCE_METRICS:
            rows.append({
                "Comparison": f"{first} - {second}",
                "Metric": metric,
                "Model_1_Value": values.loc[first, metric],
                "Model_2_Value": values.loc[second, metric],
                "ObservedDifference": values.loc[first, metric] - values.loc[second, metric],
            })
    return pd.DataFrame(rows)


def paired_bootstrap(y_true, predictions, replicates):
    rng = np.random.default_rng(RANDOM_SEED)
    class_indices = [np.flatnonzero(y_true == code) for code in (0, 1)]
    distributions = {
        model: {metric: np.empty(replicates) for metric in BOOTSTRAP_METRICS}
        for model in MODEL_GROUPS
    }
    progress = multiclass_core.SimpleProgress(replicates, "Sensor paired bootstrap")
    for replicate in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in class_indices
        ])
        true_sample = y_true[sampled]
        for model, prediction in predictions.items():
            cm = np.bincount(
                2 * true_sample + prediction[sampled], minlength=4
            ).reshape(2, 2)
            metrics = binary_core.binary_metrics_from_confusion(cm)
            conversion = {
                "BalancedAccuracy": metrics["BalancedAccuracy"],
                "MacroF1": metrics["MacroF1"],
                "mIoU": metrics["mIoU"],
                "UrbanF1": metrics["Urban_F1"],
                "UrbanIoU": metrics["Urban_IoU"],
            }
            for metric in BOOTSTRAP_METRICS:
                distributions[model][metric][replicate] = conversion[metric]
        progress.update()

    model_ci_rows = []
    for model in MODEL_GROUPS:
        for metric in BOOTSTRAP_METRICS:
            values = distributions[model][metric]
            model_ci_rows.append({
                "Model": model,
                "Metric": metric,
                "BootstrapMean": float(np.mean(values)),
                "CI_Lower_2.5": float(np.percentile(values, 2.5)),
                "CI_Upper_97.5": float(np.percentile(values, 97.5)),
                "Replicates": replicates,
            })
    difference_rows = []
    for first, second in COMPARISONS:
        for metric in BOOTSTRAP_METRICS:
            difference = distributions[first][metric] - distributions[second][metric]
            lower, upper = np.percentile(difference, [2.5, 97.5])
            difference_rows.append({
                "Comparison": f"{first} - {second}",
                "Metric": metric,
                "BootstrapMeanDifference": float(np.mean(difference)),
                "CI_Lower_2.5": float(lower),
                "CI_Upper_97.5": float(upper),
                "CI_Includes_Zero": bool(lower <= 0 <= upper),
                "Replicates": replicates,
            })
    return pd.DataFrame(model_ci_rows), pd.DataFrame(difference_rows)


def save_predictions(path, test, predictions, probabilities):
    output = pd.DataFrame({
        "Object_ID": test["Object_ID"].to_numpy(dtype="int64"),
        "TrueClass": test["BinaryClass"].to_numpy(),
    })
    for model in MODEL_GROUPS:
        output[f"{model}_PredictedClass"] = [
            "Urban" if value == 1 else "NonUrban" for value in predictions[model]
        ]
        output[f"{model}_P_Urban"] = probabilities[model]
    output.to_csv(path, index=False)
    return output


def confusion_table(evaluations, test_name):
    rows = []
    order = [(1, "Urban"), (0, "NonUrban")]
    for model, result in evaluations.items():
        cm = result["confusion"]
        reordered = cm[np.ix_([1, 0], [1, 0])]
        normalized = np.divide(
            reordered, reordered.sum(axis=1, keepdims=True),
            out=np.zeros_like(reordered, dtype="float64"),
            where=reordered.sum(axis=1, keepdims=True) > 0,
        )
        for row_index, (_, true_name) in enumerate(order):
            for column_index, (_, predicted_name) in enumerate(order):
                rows.append({
                    "TestSet": test_name,
                    "Model": model,
                    "TrueClass": true_name,
                    "PredictedClass": predicted_name,
                    "RawCount": int(reordered[row_index, column_index]),
                    "RowNormalizedPercentage": 100.0 * normalized[row_index, column_index],
                })
    return pd.DataFrame(rows)


def plot_performance(plt, performance, path):
    metrics = ["BalancedAccuracy", "MacroF1", "UrbanF1", "UrbanIoU"]
    x = np.arange(len(performance))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for position, metric in enumerate(metrics):
        ax.bar(x + (position - 1.5) * width, performance[metric], width, label=metric)
    ax.set_xticks(x, ["M1\nNTL", "M2\nSentinel-2", "M3\nNTL + Sentinel-2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Primary independent-test score")
    ax.set_xlabel("Predictor source")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(plt, cm, model, path):
    matrix = cm[np.ix_([1, 0], [1, 0])]
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype="float64"),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    labels = ["Urban", "NonUrban"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, values, title, percent in (
        (axes[0], matrix, "Raw counts", False),
        (axes[1], normalized * 100, "Row-normalized (%)", True),
    ):
        image = ax.imshow(values, cmap="Blues", vmin=0)
        threshold = values.max() / 2 if values.max() else 0
        for row in range(2):
            for column in range(2):
                text = f"{values[row, column]:.1f}%" if percent else str(int(values[row, column]))
                ax.text(
                    column, row, text, ha="center", va="center",
                    color="white" if values[row, column] > threshold else "black",
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


def plot_probability_distributions(plt, y_true, probabilities, path):
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, model in zip(axes, MODEL_GROUPS):
        ax.hist(
            probabilities[model][y_true == 1], bins=bins, alpha=0.55,
            density=True, label="True Urban",
        )
        ax.hist(
            probabilities[model][y_true == 0], bins=bins, alpha=0.55,
            density=True, label="True NonUrban",
        )
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{model}: {MODEL_SOURCES[model]}")
        ax.set_xlabel("P(Urban)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False)
    fig.suptitle("Primary-test Urban-probability distributions (diagnostic only)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_lookup(paired, comparison, metric):
    rows = paired[(paired["Comparison"] == comparison) & (paired["Metric"] == metric)]
    return None if rows.empty else rows.iloc[0]


def scientific_interpretation(performance, paired, pilot):
    values = performance.set_index("Model")
    m2_m1 = values.loc["M2", "MacroF1"] - values.loc["M1", "MacroF1"]
    m3_m2 = values.loc["M3", "MacroF1"] - values.loc["M2", "MacroF1"]
    m3_m1 = values.loc["M3", "MacroF1"] - values.loc["M1", "MacroF1"]

    def supported(comparison, metric, positive=True):
        row = paired_lookup(paired, comparison, metric)
        if row is None:
            return False
        return bool(row["CI_Lower_2.5"] > 0) if positive else bool(row["CI_Upper_97.5"] < 0)

    fusion_supported = (
        m3_m2 > 0
        and supported("M3 - M2", "MacroF1")
        and (
            supported("M3 - M2", "UrbanF1")
            or supported("M3 - M2", "UrbanIoU")
        )
    )
    s2_supported = m2_m1 > 0 and supported("M2 - M1", "MacroF1")
    ntl_supported = m2_m1 < 0 and supported("M2 - M1", "MacroF1", positive=False)
    ntl_competitive = abs(m2_m1) <= 0.01 or not (s2_supported or ntl_supported)
    m1_near_m3 = abs(m3_m1) <= 0.01 and bool(
        paired_lookup(paired, "M3 - M1", "MacroF1")["CI_Includes_Zero"]
    )

    if fusion_supported and not m1_near_m3:
        direction = "genuinely complementary NTL–Sentinel-2 fusion"
        case = (
            "M3 improves on M2 with paired support and is not practically equivalent to M1: "
            "both sensor families contribute complementary information."
        )
    elif s2_supported:
        direction = "Sentinel-2-dominated object-based classification"
        case = "Sentinel-2 is stronger than NTL and M3 lacks clear supported improvement over M2."
    else:
        direction = "NTL-driven object-based urban delineation"
        case = "NTL remains competitive and direct fusion lacks clear supported incremental benefit."

    if pilot:
        direction = "Pilot only — no final manuscript direction"
        case = "Pilot estimates validate the workflow but cannot set the manuscript framing."
        ntl_competitive_value = "Not concluded from pilot"
        s2_improves_value = "Not concluded from pilot"
        ntl_adds_value = "Not concluded from pilot"
        fusion_justified_value = "Not concluded from pilot"
    else:
        ntl_competitive_value = ntl_competitive
        s2_improves_value = s2_supported
        ntl_adds_value = fusion_supported
        fusion_justified_value = direction == "genuinely complementary NTL–Sentinel-2 fusion"
    return {
        "ntl_competitive": ntl_competitive_value,
        "s2_improves": s2_improves_value,
        "ntl_adds_to_s2": ntl_adds_value,
        "fusion_justified": fusion_justified_value,
        "m1_near_m3": m1_near_m3,
        "case": case,
        "direction": direction,
        "m2_m1_macro": m2_m1,
        "m3_m1_macro": m3_m1,
        "m3_m2_macro": m3_m2,
    }


def markdown_table(frame, digits=4):
    return multiclass_core.markdown_table(frame, digits=digits)


def write_report(
    path, mode, args, input_info, class_info, manifest, s2_audit,
    s2_diagnostics, ntl_audit, fold_summary, cv_results, parameters,
    cv_macro, cv_balanced, primary_performance, full_performance,
    point_difference, bootstrap_difference, interpretation, runtimes,
    versions, output_dir,
):
    s2_problem_text = (
        "No blocking optical preprocessing defect was detected."
        if s2_diagnostics["passed"] else
        "Blocking defects: " + "; ".join(s2_diagnostics["defects"])
    )
    lines = [
        f"# Spectral Sensor Comparison XGBoost 01 — {mode}",
        "",
        "## A. Inputs",
        "",
        f"- Reused training set: {args.training_set.resolve()}",
        f"- Reused primary test: {args.primary_test_set.resolve()}",
        f"- Reused full test: {args.full_test_set.resolve()}",
        f"- Segmentation: {args.segmentation.resolve()}",
        f"- Corrected spectral feature table: {args.feature_table.resolve()}",
        f"- Segmentation/features object domain: {input_info['feature_objects']:,} unique Object_ID values.",
        "",
        "## B. Training and test data",
        "",
        markdown_table(class_info),
        "",
        f"Training objects={input_info['training_objects']:,}; primary test="
        f"{input_info['primary_objects']:,}; full test={input_info['full_objects']:,}; "
        f"train/full overlap={input_info['train_full_overlap']}.",
        "The three existing binary stratified CSVs were reused unchanged; no sample was regenerated, "
        "added, removed, under-sampled, or over-sampled.",
        "",
        "## C. Feature definitions",
        "",
        f"### NTL ({len(NTL_FEATURES)})\n\n" + ", ".join(NTL_FEATURES),
        "",
        f"### Sentinel-2 ({len(S2_FEATURES)})\n\n" + ", ".join(S2_FEATURES),
        "",
        f"M1 predictors={len(NTL_FEATURES)}; M2 predictors={len(S2_FEATURES)}; "
        f"M3 predictors={len(NTL_FEATURES) + len(S2_FEATURES)}.",
        "Geometry, GLCM/texture, spatial context, neighborhood variables, labels and confidence "
        "variables are excluded by explicit manifests.",
        "",
        "## D. Optical feature quality audit",
        "",
        s2_problem_text,
        f"Exact duplicate column pairs={len(s2_diagnostics['exact_duplicate_column_pairs'])}; "
        f"duplicate full S2 feature-vector rows={s2_diagnostics['duplicate_feature_vector_rows']}; "
        f"groups={s2_diagnostics['duplicate_feature_vector_groups']}.",
        "B11/B12 diagnostics:",
        "",
        markdown_table(pd.DataFrame(s2_diagnostics["b11_b12"])),
        "",
        markdown_table(s2_audit),
        "",
        f"NTL missing cells across the full feature table: {int(ntl_audit['NaNCount'].sum()):,}.",
        "",
        "## E. Spatial CV",
        "",
        f"The stored 5-km SpatialCell field was reused. Groups={input_info['spatial_groups']:,}; "
        f"folds={len(fold_summary)}.",
        "",
        markdown_table(fold_summary),
        "",
        "## F. Hyperparameter tuning",
        "",
        f"M3 alone was tuned on training data using {len(cv_results)} deterministic sampled "
        f"configurations. Selected CV Macro-F1={cv_macro:.6f}; Balanced Accuracy={cv_balanced:.6f}.",
        "Configurations within 0.0001 Macro-F1 were treated as near-tied; shallower depth and fewer "
        "trees were preferred.",
        "",
        "    " + json.dumps(parameters, sort_keys=True),
        "",
        "## G. Primary test results",
        "",
        markdown_table(primary_performance),
        "",
        "## H. Full test results",
        "",
        markdown_table(full_performance),
        "",
        "## I. Urban delineation",
        "",
        markdown_table(primary_performance[[
            "Model", "FeatureSource", "UrbanPrecision", "UrbanRecall", "UrbanF1", "UrbanIoU",
        ]]),
        "",
        "M1 is NTL-only object attributes using Sentinel-2-derived object boundaries. It is not a "
        "fully Sentinel-2-independent NTL workflow.",
        "",
        "## J. Sensor contribution",
        "",
        markdown_table(point_difference),
        "",
        "M3−M2 is the direct incremental NTL-beyond-Sentinel-2 comparison; M3−M1 measures the "
        "increment from adding daytime optical attributes to NTL.",
        "",
        "## K. Bootstrap uncertainty",
        "",
        f"Paired class-stratified primary-test replicates: "
        f"{int(bootstrap_difference['Replicates'].iloc[0])}.",
        "",
        markdown_table(bootstrap_difference),
        "",
        "Intervals including zero are reported as uncertainty and not labelled significant.",
        "",
        "## L. Scientific conclusion",
        "",
        f"1. Is NTL alone competitive with Sentinel-2? {interpretation['ntl_competitive']}.",
        f"2. Does Sentinel-2 improve over NTL? {interpretation['s2_improves']}.",
        f"3. Does adding NTL to Sentinel-2 improve the classifier? {interpretation['ntl_adds_to_s2']}.",
        f"4. Does evidence justify direct fusion framing? {interpretation['fusion_justified']}.",
        f"5. Evidence-based case: {interpretation['case']}",
        "The common object boundaries originate from Sentinel-2 segmentation, so conclusions concern "
        "attribute information conditional on those shared boundaries.",
        "",
        "## M. Runtime",
        "",
        markdown_table(pd.DataFrame([{"Stage": key, "Seconds": value} for key, value in runtimes.items()])),
        "",
        "## N. Deviations",
        "",
        (
            "- Pilot modelling used three folds, three capped 120-tree configurations and 200 "
            "bootstrap replicates; all sample memberships remained the full reused memberships."
            if mode == "PILOT" else
            "- Full modelling used five folds, 24 unmodified configurations and 2,000 bootstrap replicates."
        ),
        "- No geometry, texture or spatial-context feature was admitted.",
        "- Probability-distribution plots are diagnostic only and did not alter the fixed threshold.",
        f"- Software: Python {versions['python']}; XGBoost {versions['xgboost']}; "
        f"scikit-learn {versions['sklearn']}.",
        "- M1/M2/M3 used identical training/test memberships, labels, weights, hyperparameters, "
        "random seed and threshold. No test information entered tuning or feature definitions.",
        "",
        "## Recommended manuscript direction",
        "",
        interpretation["direction"],
        "",
        "## Output directory",
        "",
        str(output_dir.resolve()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    total_started = time.perf_counter()
    args = parse_args()
    bundle = dependency_bundle()
    if args.output_dir is None:
        suffix = "_pilot" if args.pilot else ""
        if args.preflight_only:
            suffix = "_preflight"
        args.output_dir = SEGMENTATION_DIR / f"Spectral_Sensor_Comparison_XGBoost_01{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiclass_core.require_files([
        args.training_set, args.primary_test_set, args.full_test_set,
        args.segmentation, args.feature_table, args.binary_manifest,
    ])

    stage_started = time.perf_counter()
    print("Loading reused binary samples and corrected spectral features...")
    training = load_reused_dataset(args.training_set, "training")
    primary = load_reused_dataset(args.primary_test_set, "primary test")
    full_test = load_reused_dataset(args.full_test_set, "full test")
    validate_ntl_manifest(args.binary_manifest)
    features = load_spectral_features(args.feature_table)
    loading_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    print("Auditing Sentinel-2 feature quality...")
    s2_audit, s2_diagnostics = audit_s2_features(features)
    ntl_audit = audit_ntl_missing(features)
    s2_audit.to_csv(args.output_dir / "Spectral_Sensor_S2_Quality_Audit.csv", index=False)
    ntl_audit.to_csv(args.output_dir / "Spectral_Sensor_NTL_Missing_Audit.csv", index=False)
    (args.output_dir / "Spectral_Sensor_S2_Quality_Diagnostics.json").write_text(
        json.dumps(s2_diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not s2_diagnostics["passed"]:
        raise SystemExit(
            "STOP: corrected Sentinel-2 audit failed:\n- "
            + "\n- ".join(s2_diagnostics["defects"])
        )
    quality_seconds = time.perf_counter() - stage_started
    manifest = write_manifest(
        args.output_dir / "Spectral_Sensor_Feature_Manifest.csv"
    )

    feature_ids = pd.Index(features["Object_ID"])
    for name, frame in (("training", training), ("primary", primary), ("full", full_test)):
        missing = pd.Index(frame["Object_ID"]).difference(feature_ids)
        if len(missing):
            raise SystemExit(f"STOP: {len(missing)} {name} IDs fail the feature join.")
    if not set(primary["Object_ID"]).issubset(set(full_test["Object_ID"])):
        raise SystemExit("STOP: primary test is not a subset of the full independent test.")
    train_full_overlap = np.intersect1d(training["Object_ID"], full_test["Object_ID"])
    if len(train_full_overlap):
        raise SystemExit(f"STOP: train/test leakage for {len(train_full_overlap)} objects.")

    # Verify Seg_id, without ever using the unrelated source ID field.
    segmentation_ids = bundle["gpd"].read_file(
        args.segmentation, columns=["Seg_id"], ignore_geometry=True
    )
    if "Seg_id" not in segmentation_ids.columns:
        raise SystemExit("STOP: segmentation lacks canonical Seg_id.")
    segmentation_index = pd.Index(segmentation_ids["Seg_id"].astype("int64"))
    if segmentation_index.has_duplicates or set(segmentation_index) != set(feature_ids):
        raise SystemExit("STOP: Seg_id and feature Object_ID domains are not one-to-one identical.")

    y_train = training["BinaryLabel"].to_numpy(dtype="int64")
    weights, sample_weight = binary_core.binary_class_weights(y_train)
    groups = training["SpatialCell"].astype(str).to_numpy()
    folds = 3 if args.pilot else 5
    if len(np.unique(groups)) < folds:
        raise SystemExit("STOP: insufficient stored spatial groups for GroupKFold.")

    class_rows = []
    for dataset_name, frame in (
        ("Training", training), ("PrimaryTest", primary), ("FullTest", full_test)
    ):
        class_rows.append({
            "Dataset": dataset_name,
            "Objects": len(frame),
            "Urban": int((frame["BinaryLabel"] == 1).sum()),
            "NonUrban": int((frame["BinaryLabel"] == 0).sum()),
            "UrbanSource": int((frame["SourceClass3"] == "Urban").sum()),
            "VegetationSource": int((frame["SourceClass3"] == "Vegetation").sum()),
            "BareScrubSource": int((frame["SourceClass3"] == "BareScrub").sum()),
            "UrbanWeight": weights[1] if dataset_name == "Training" else np.nan,
            "NonUrbanWeight": weights[0] if dataset_name == "Training" else np.nan,
        })
    class_info = pd.DataFrame(class_rows)
    class_info.to_csv(args.output_dir / "Spectral_Sensor_Dataset_Summary.csv", index=False)

    input_info = {
        "feature_objects": int(len(features)),
        "training_objects": int(len(training)),
        "primary_objects": int(len(primary)),
        "full_objects": int(len(full_test)),
        "train_full_overlap": int(len(train_full_overlap)),
        "spatial_groups": int(len(np.unique(groups))),
        "ntl_features": len(NTL_FEATURES),
        "s2_features": len(S2_FEATURES),
        "m3_features": len(NTL_FEATURES) + len(S2_FEATURES),
        "threshold": THRESHOLD,
    }
    validation = {
        "inputs": {
            name: str(getattr(args, name).resolve()) for name in (
                "training_set", "primary_test_set", "full_test_set",
                "segmentation", "feature_table", "binary_manifest",
            )
        },
        "input_info": input_info,
        "feature_groups": FEATURE_GROUPS,
        "class_weights": {"Urban": weights[1], "NonUrban": weights[0]},
        "s2_quality": s2_diagnostics,
        "random_seed": RANDOM_SEED,
    }
    (args.output_dir / "Spectral_Sensor_Input_Validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8"
    )

    if args.preflight_only:
        print("Spectral sensor preflight passed; no classifier was fitted.")
        print(f"Output: {args.output_dir.resolve()}")
        return 0

    configuration_count = args.pilot_configs if args.pilot else args.search_configs
    bootstrap_replicates = args.pilot_bootstrap if args.pilot else args.bootstrap_replicates
    configurations = multiclass_core.sampled_parameter_configurations(
        configuration_count, pilot=args.pilot
    )
    indexed = features.set_index("Object_ID")
    m3_features = [*NTL_FEATURES, *S2_FEATURES]
    X_m3 = indexed.loc[
        training["Object_ID"].to_numpy(dtype="int64"), m3_features
    ].to_numpy(dtype="float32", copy=True)

    stage_started = time.perf_counter()
    parameters, cv_macro, cv_balanced, cv_results, splits = tune_m3(
        bundle, X_m3, y_train, groups, sample_weight,
        configurations, folds, args.n_jobs,
    )
    cv_seconds = time.perf_counter() - stage_started
    cv_results.to_csv(args.output_dir / "Spectral_Sensor_Spatial_CV_Results.csv", index=False)
    fold_rows = []
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        fold_rows.append({
            "Fold": fold,
            "TrainingObjects": len(train_index),
            "ValidationObjects": len(validation_index),
            "ValidationSpatialGroups": len(np.unique(groups[validation_index])),
            "ValidationUrban": int((y_train[validation_index] == 1).sum()),
            "ValidationNonUrban": int((y_train[validation_index] == 0).sum()),
        })
    fold_summary = pd.DataFrame(fold_rows)

    stage_started = time.perf_counter()
    (
        predictions, probabilities, evaluations, final_weights,
        fitting_seconds, prediction_seconds,
    ) = fit_models(
        bundle, features, training, primary, full_test, parameters, args.n_jobs
    )
    fit_orchestration_seconds = time.perf_counter() - stage_started
    primary_performance = performance_table(evaluations["primary"])
    full_performance = performance_table(evaluations["full"])
    primary_performance.to_csv(
        args.output_dir / "Spectral_Sensor_Primary_Performance.csv", index=False
    )
    full_performance.to_csv(
        args.output_dir / "Spectral_Sensor_FullTest_Performance.csv", index=False
    )
    point_difference = point_differences(primary_performance)
    point_difference.to_csv(
        args.output_dir / "Spectral_Sensor_Performance_Differences.csv", index=False
    )
    save_predictions(
        args.output_dir / "Spectral_Sensor_Primary_Test_Predictions.csv",
        primary, predictions["primary"], probabilities["primary"],
    )
    save_predictions(
        args.output_dir / "Spectral_Sensor_Full_Test_Predictions.csv",
        full_test, predictions["full"], probabilities["full"],
    )
    pd.concat([
        confusion_table(evaluations["primary"], "PrimaryStratified"),
        confusion_table(evaluations["full"], "SecondaryFull"),
    ]).to_csv(args.output_dir / "Spectral_Sensor_Confusion_Matrices.csv", index=False)

    stage_started = time.perf_counter()
    bootstrap_ci, bootstrap_difference = paired_bootstrap(
        primary["BinaryLabel"].to_numpy(dtype="int64"),
        predictions["primary"], bootstrap_replicates,
    )
    bootstrap_seconds = time.perf_counter() - stage_started
    bootstrap_ci.to_csv(
        args.output_dir / "Spectral_Sensor_Bootstrap_CI.csv", index=False
    )
    observed_lookup = point_difference.set_index(["Comparison", "Metric"])["ObservedDifference"]
    bootstrap_difference.insert(
        2,
        "ObservedDifference",
        [
            observed_lookup.loc[(row.Comparison, row.Metric)]
            for row in bootstrap_difference.itertuples()
        ],
    )
    bootstrap_difference.to_csv(
        args.output_dir / "Spectral_Sensor_Paired_Bootstrap_Differences.csv", index=False
    )

    interpretation = scientific_interpretation(
        primary_performance, bootstrap_difference, args.pilot
    )
    hyperparameters = {
        "random_state": RANDOM_SEED,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "fixed_probability_threshold": THRESHOLD,
        "tuned_model": "M3",
        "spatial_cv_folds": folds,
        "tested_configuration_count": len(configurations),
        "selected_validation_macro_f1": cv_macro,
        "selected_validation_balanced_accuracy": cv_balanced,
        "selected_hyperparameters": parameters,
        "frozen_for_models": list(MODEL_GROUPS),
    }
    (args.output_dir / "Spectral_Sensor_Selected_Hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2), encoding="utf-8"
    )

    plot_performance(
        bundle["plt"], primary_performance,
        args.output_dir / "Spectral_Sensor_Performance_Comparison.png",
    )
    for model in MODEL_GROUPS:
        plot_confusion(
            bundle["plt"], evaluations["primary"][model]["confusion"], model,
            args.output_dir / f"{model}_Confusion_Matrix.png",
        )
    plot_probability_distributions(
        bundle["plt"], primary["BinaryLabel"].to_numpy(dtype="int64"),
        probabilities["primary"],
        args.output_dir / "Spectral_Sensor_Probability_Distributions.png",
    )

    versions = {
        "python": sys.version.split()[0],
        "xgboost": bundle["xgb"].__version__,
        "sklearn": importlib.import_module("sklearn").__version__,
    }
    total_seconds = time.perf_counter() - total_started
    runtimes = {
        "Data loading": loading_seconds,
        "Quality checks": quality_seconds,
        "Spatial CV tuning": cv_seconds,
        "Model fitting": fitting_seconds,
        "Prediction": prediction_seconds,
        "Model orchestration overhead": max(
            0.0, fit_orchestration_seconds - fitting_seconds - prediction_seconds
        ),
        "Paired bootstrap": bootstrap_seconds,
        "Total": total_seconds,
    }
    report_name = (
        "Spectral_Sensor_Comparison_XGBoost_01_Report_PILOT.md"
        if args.pilot else "Spectral_Sensor_Comparison_XGBoost_01_Report.md"
    )
    write_report(
        args.output_dir / report_name,
        "PILOT" if args.pilot else "FULL RUN",
        args, input_info, class_info, manifest, s2_audit, s2_diagnostics,
        ntl_audit, fold_summary, cv_results, parameters, cv_macro, cv_balanced,
        primary_performance, full_performance, point_difference,
        bootstrap_difference, interpretation, runtimes, versions, args.output_dir,
    )
    print(f"Completed {'pilot' if args.pilot else 'full sensor comparison'}.")
    print(f"Report: {(args.output_dir / report_name).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
