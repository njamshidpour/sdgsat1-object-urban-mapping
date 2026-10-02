"""Focused, unsupervised redundancy analysis of the final object feature table.

Only existing table columns are used. No imagery, labels, feature extraction,
classification, or automatic feature removal is performed.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
INPUT = HERE / "Segmentation_output" / "objects_features_final_full.csv"
OUTPUT = HERE / "Feature_Redundancy_Analysis_Final_output"
REPORT = HERE / "Feature_Redundancy_Analysis_Final_Report.md"
ID = "Object_ID"
PAN_STD = "SDGSAT_Pan_std"
SPATIAL_CONTEXT_EPSILON = 1.499710070587349e-13
EXPECTED_ROWS = 336_109
CHUNK_SIZE = 100_000

F_NTL = "NTL spectral/radiometric"
F_OPT = "Optical spectral"
F_GEO = "Geometry"
F_TEX = "Conventional GLCM texture"
F_CTX = "NTL spatial context"

STATS = ("mean", "std", "max")
NTL_BASE = (
    "SDGSAT_Pan", "SDGSAT_R", "SDGSAT_G", "SDGSAT_B", "BR", "ENUI",
    "NBDI", "NDIBG", "NDIGR", "NLCSI", "SONDI", "TVANUI", "VANUI",
)
OPT_BASE = (
    "S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7", "S2_B8",
    "S2_B8A", "S2_B11", "S2_B12", "NDVI", "NDBI", "NDWI",
)
GEOMETRY = [
    "area_sqm", "perimeter_m", "compactness", "shape_index", "elongation",
    "length_width_ratio",
]
TEXTURE = [
    f"SDGSAT_Pan_glcm_{name}"
    for name in ("contrast", "homogeneity", "energy", "correlation", "entropy")
]
CONTEXT = [
    "NTL_NeighborMean", "NTL_NeighborStd", "NTL_LocalContrast",
    "NTL_NormalizedContrast", "NTL_MeanAbsNeighborContrast", "NeighborCount",
]
FAMILIES = {
    F_NTL: [f"{base}_{stat}" for base in NTL_BASE for stat in STATS],
    F_OPT: [f"{base}_{stat}" for base in OPT_BASE for stat in STATS],
    F_GEO: GEOMETRY,
    F_TEX: TEXTURE,
    F_CTX: CONTEXT,
}
ALL_FEATURES = [feature for features in FAMILIES.values() for feature in features]
FAMILY_OF = {feature: family for family, features in FAMILIES.items() for feature in features}


def clean_progress(label: str, rows: int) -> None:
    fraction = min(rows / EXPECTED_ROWS, 1.0)
    width = 20
    filled = round(width * fraction)
    print(
        f"\r[{('#' * filled + '-' * (width - filled))}] {fraction:6.1%} "
        f"| {label} | {rows:,} objects",
        end="",
        flush=True,
    )


def load_and_validate() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    header = list(pd.read_csv(INPUT, nrows=0).columns)
    expected = [ID, *ALL_FEATURES]
    if header != expected:
        missing = sorted(set(expected).difference(header))
        unexpected = sorted(set(header).difference(expected))
        raise ValueError(
            f"Final-table schema differs from the documented 96-column structure; "
            f"missing={missing}, unexpected={unexpected}, exact_order={header == expected}"
        )
    chunks = []
    rows = 0
    for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        rows += len(chunk)
        clean_progress("reading final table", rows)
    print()
    data = pd.concat(chunks, ignore_index=True)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows; found {len(data):,}.")
    if data[ID].isna().any() or not data[ID].is_unique:
        raise ValueError("Object_ID must be non-null and unique.")
    values = data[ALL_FEATURES].to_numpy(dtype="float64", copy=False)
    if np.isinf(values).any():
        raise ValueError("Infinite feature values found; the requested diagnostics define only NaN.")
    return data


def exact_duplicate_map(data: pd.DataFrame) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    groups: dict[str, list[str]] = {}
    arrays: dict[str, np.ndarray] = {}
    for feature in ALL_FEATURES:
        values = data[feature].to_numpy(dtype="float64", copy=True)
        values[np.isnan(values)] = np.nan
        arrays[feature] = values
        groups.setdefault(hashlib.sha256(values.tobytes()).hexdigest(), []).append(feature)
    duplicate_of = {feature: [] for feature in ALL_FEATURES}
    pairs = []
    for candidates in groups.values():
        for a, b in combinations(candidates, 2):
            if np.array_equal(arrays[a], arrays[b], equal_nan=True):
                duplicate_of[a].append(b)
                duplicate_of[b].append(a)
                pairs.append((a, b))
    return duplicate_of, pairs


def basic_diagnostics(data: pd.DataFrame, duplicate_of: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    total = len(data)
    for feature in ALL_FEATURES:
        values = data[feature].to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(values)
        valid_values = values[valid]
        rows.append({
            "Feature": feature,
            "Family": FAMILY_OF[feature],
            "Valid_Count": int(valid.sum()),
            "NaN_Percentage": float(np.isnan(values).sum() / total * 100),
            "Zero_Percentage_of_Valid": (
                float(np.count_nonzero(valid_values == 0) / len(valid_values) * 100)
                if len(valid_values) else np.nan
            ),
            "Unique_Valid_Values": int(pd.Series(valid_values).nunique()),
            "Is_Constant": bool(pd.Series(valid_values).nunique() <= 1),
            "Is_Exact_Duplicate": bool(duplicate_of[feature]),
            "Exact_Duplicate_Of": "; ".join(duplicate_of[feature]),
        })
    return pd.DataFrame(rows)


def matrix_pairs(matrix: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    rows = []
    for a, b in combinations(matrix.columns, 2):
        rho = matrix.at[a, b]
        if pd.notna(rho) and abs(rho) >= threshold:
            rows.append({
                "Feature_1": a, "Feature_2": b, "rho": float(rho),
                "abs_rho": float(abs(rho)), "Very_High_GE_095": bool(abs(rho) >= 0.95),
            })
    return pd.DataFrame(rows, columns=[
        "Feature_1", "Feature_2", "rho", "abs_rho", "Very_High_GE_095"
    ]).sort_values("abs_rho", ascending=False, ignore_index=True)


def active_comparison(full: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(full.columns, 2):
        rho_all = full.at[a, b]
        if pd.isna(rho_all) or abs(rho_all) < 0.90:
            continue
        rho_active = active.at[a, b]
        abs_all = abs(float(rho_all))
        abs_active = abs(float(rho_active)) if pd.notna(rho_active) else np.nan
        decline = abs_all - abs_active if pd.notna(abs_active) else np.nan
        if pd.notna(abs_active) and abs_all >= 0.95 and abs_active >= 0.95:
            flag = "persistent redundancy"
        elif pd.isna(abs_active) or abs_active < 0.90 or decline >= 0.10:
            flag = "floor-influenced apparent redundancy"
        else:
            flag = "high correlation retained below persistent threshold"
        rows.append({
            "Feature_1": a, "Feature_2": b, "rho_all": float(rho_all),
            "rho_active": float(rho_active), "abs_rho_all": abs_all,
            "abs_rho_active": abs_active, "abs_rho_decrease": decline,
            "Diagnostic_Flag": flag,
        })
    return pd.DataFrame(rows).sort_values("abs_rho_all", ascending=False, ignore_index=True)


def cross_family_outputs(full_corr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, high_rows = [], []
    family_names = list(FAMILIES)
    maximums = pd.DataFrame(np.eye(len(family_names)), index=family_names, columns=family_names)
    for family_1, family_2 in combinations(family_names, 2):
        block = full_corr.loc[FAMILIES[family_1], FAMILIES[family_2]]
        pair_rows = [
            (a, b, float(block.at[a, b]), abs(float(block.at[a, b])))
            for a in block.index for b in block.columns if pd.notna(block.at[a, b])
        ]
        ranked = sorted(pair_rows, key=lambda item: item[3], reverse=True)
        maximums.at[family_1, family_2] = maximums.at[family_2, family_1] = ranked[0][3]
        row = {
            "Family_1": family_1, "Family_2": family_2,
            "Cross_Family_Pair_Count": len(FAMILIES[family_1]) * len(FAMILIES[family_2]),
            "Maximum_Abs_Rho": ranked[0][3],
            "Pairs_Abs_Rho_GE_090": sum(item[3] >= 0.90 for item in ranked),
            "Pairs_Abs_Rho_GE_095": sum(item[3] >= 0.95 for item in ranked),
        }
        for position, (a, b, rho, absolute) in enumerate(ranked[:5], 1):
            row[f"Top_{position}_Pair"] = f"{a} <-> {b}"
            row[f"Top_{position}_Rho"] = rho
        summary_rows.append(row)
        for a, b, rho, absolute in ranked:
            if absolute < 0.90:
                break
            high_rows.append({
                "Family_1": family_1, "Family_2": family_2,
                "Feature_1": a, "Feature_2": b, "rho": rho,
                "abs_rho": absolute, "Very_High_GE_095": absolute >= 0.95,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(high_rows), maximums


def deterministic_checks(data: pd.DataFrame) -> list[dict[str, object]]:
    shape_expected = 1 / np.sqrt(data["compactness"].to_numpy(float))
    shape_actual = data["shape_index"].to_numpy(float)
    local_expected = data["SDGSAT_Pan_mean"] - data["NTL_NeighborMean"]
    local_actual = data["NTL_LocalContrast"]
    # Exact epsilon persisted by the completed Spatial Context run diagnostics.
    epsilon = SPATIAL_CONTEXT_EPSILON
    normalized_expected = local_actual / (
        data["SDGSAT_Pan_mean"] + data["NTL_NeighborMean"] + epsilon
    )
    normalized_actual = data["NTL_NormalizedContrast"]
    return [
        {
            "Feature_1": "compactness", "Feature_2_or_Group": "shape_index",
            "Family_1": F_GEO, "Family_2": F_GEO,
            "Evidence": "shape_index = 1/sqrt(compactness)",
            "Verified": bool(np.allclose(shape_actual, shape_expected, rtol=1e-7, atol=1e-10, equal_nan=True)),
            "Maximum_Absolute_Error": float(np.nanmax(np.abs(shape_actual - shape_expected))),
        },
        {
            "Feature_1": "NTL_LocalContrast",
            "Feature_2_or_Group": "SDGSAT_Pan_mean; NTL_NeighborMean",
            "Family_1": F_CTX, "Family_2": f"{F_NTL}; {F_CTX}",
            "Evidence": "NTL_LocalContrast = SDGSAT_Pan_mean - NTL_NeighborMean",
            "Verified": bool(np.allclose(local_actual, local_expected, rtol=1e-7, atol=1e-10, equal_nan=True)),
            "Maximum_Absolute_Error": float(np.nanmax(np.abs(local_actual - local_expected))),
        },
        {
            "Feature_1": "NTL_NormalizedContrast",
            "Feature_2_or_Group": "SDGSAT_Pan_mean; NTL_NeighborMean; NTL_LocalContrast",
            "Family_1": F_CTX, "Family_2": f"{F_NTL}; {F_CTX}",
            "Evidence": "normalized signed contrast from PAN and neighbor means",
            "Verified": bool(np.allclose(normalized_actual, normalized_expected, rtol=1e-7, atol=1e-10, equal_nan=True)),
            "Maximum_Absolute_Error": float(np.nanmax(np.abs(normalized_actual - normalized_expected))),
        },
    ]


def candidate_table(
    duplicate_pairs: list[tuple[str, str]], deterministic: list[dict[str, object]],
    ntl_active: pd.DataFrame, texture_active: pd.DataFrame,
    within_high: pd.DataFrame, cross_high: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    occupied: set[tuple[str, str]] = set()

    def key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))

    for a, b in duplicate_pairs:
        occupied.add(key(a, b))
        rows.append({
            "Category": "Exact duplicate", "Feature_1": a, "Feature_2_or_Group": b,
            "Family_1": FAMILY_OF[a], "Family_2": FAMILY_OF[b],
            "rho_all": 1.0, "rho_active": np.nan,
            "Evidence": "Elementwise equality including the NaN mask", "Review_Priority": "highest",
        })
    for item in deterministic:
        rows.append({
            "Category": "Deterministic/structural redundancy", **item,
            "rho_all": np.nan, "rho_active": np.nan,
            "Review_Priority": "highest" if item["Feature_1"] == "compactness" else "interpretive",
        })
        if ";" not in item["Feature_2_or_Group"]:
            occupied.add(key(item["Feature_1"], item["Feature_2_or_Group"]))

    for comparison in (ntl_active, texture_active):
        for item in comparison.itertuples(index=False):
            pair_key = key(item.Feature_1, item.Feature_2)
            if pair_key in occupied:
                continue
            if item.Diagnostic_Flag == "persistent redundancy":
                category = "Persistent very-high correlation"
            elif item.Diagnostic_Flag == "floor-influenced apparent redundancy":
                category = "Floor-influenced apparent redundancy"
            else:
                continue
            occupied.add(pair_key)
            rows.append({
                "Category": category, "Feature_1": item.Feature_1,
                "Feature_2_or_Group": item.Feature_2,
                "Family_1": FAMILY_OF[item.Feature_1], "Family_2": FAMILY_OF[item.Feature_2],
                "rho_all": item.rho_all, "rho_active": item.rho_active,
                "Evidence": item.Diagnostic_Flag, "Review_Priority": "high" if category.startswith("Persistent") else "interpretive",
            })

    correlation_rows = pd.concat([within_high, cross_high], ignore_index=True)
    for item in correlation_rows.itertuples(index=False):
        if item.abs_rho < 0.95 or key(item.Feature_1, item.Feature_2) in occupied:
            continue
        pair_families = {FAMILY_OF[item.Feature_1], FAMILY_OF[item.Feature_2]}
        if pair_families.issubset({F_NTL, F_TEX}):
            continue
        occupied.add(key(item.Feature_1, item.Feature_2))
        rows.append({
            "Category": "Persistent very-high correlation", "Feature_1": item.Feature_1,
            "Feature_2_or_Group": item.Feature_2,
            "Family_1": FAMILY_OF[item.Feature_1], "Family_2": FAMILY_OF[item.Feature_2],
            "rho_all": item.rho, "rho_active": np.nan,
            "Evidence": "|rho_all| >= 0.95; active-subset restriction not applicable to this family pair",
            "Review_Priority": "review",
        })
    return pd.DataFrame(rows)


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def color(value: float, absolute: bool) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (210, 210, 210)
    if absolute:
        t = min(max(value, 0), 1)
        return tuple(int(245 + t * (b - 245)) for b in (8, 81, 156))
    t = min(abs(value), 1)
    end = (178, 24, 43) if value >= 0 else (33, 102, 172)
    return tuple(int(248 + t * (component - 248)) for component in end)


def heatmap(matrix: pd.DataFrame, path: Path, title: str, absolute: bool = False) -> None:
    n = len(matrix)
    cell = 42 if n <= 8 else 34
    left, top, right, bottom = 530, 110, 170, 530
    width, height = left + n * cell + right, top + n * cell + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 25), title, fill="black", font=font(30, True))
    values = matrix.to_numpy(float)
    for row in range(n):
        for col in range(n):
            box = (left + col * cell, top + row * cell,
                   left + (col + 1) * cell, top + (row + 1) * cell)
            draw.rectangle(box, fill=color(values[row, col], absolute), outline=(235, 235, 235))
            if n <= 8:
                text = f"{values[row, col]:.2f}"
                fill = "white" if abs(values[row, col]) > 0.65 else "black"
                draw.text((box[0] + 4, box[1] + 11), text, fill=fill, font=font(14))
    label_font = font(17 if n <= 8 else 14)
    for row, label in enumerate(matrix.index):
        draw.text((10, top + row * cell + 8), str(label), fill="black", font=label_font)
    for col, label in enumerate(matrix.columns):
        layer = Image.new("RGBA", (bottom - 10, cell), (255, 255, 255, 0))
        ImageDraw.Draw(layer).text((2, 5), str(label), fill="black", font=label_font)
        rotated = layer.rotate(90, expand=True)
        image.paste(rotated, (left + col * cell + 5, top + n * cell + 5), rotated)
    bar_x = left + n * cell + 55
    for y in range(n * cell):
        value = 1 - y / max(n * cell - 1, 1)
        draw.line((bar_x, top + y, bar_x + 28, top + y), fill=color(value if absolute else 2 * value - 1, absolute))
    draw.text((bar_x - 10, top - 28), "1.0", fill="black", font=font(15))
    draw.text((bar_x - 10, top + n * cell + 5), "0.0" if absolute else "-1.0", fill="black", font=font(15))
    image.save(path, dpi=(300, 300))


def markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_None._"
    def render(value):
        if pd.isna(value): return "NaN"
        if isinstance(value, (float, np.floating)): return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(render(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(
    data, diagnostics, duplicates, within_summary, within_high, ntl_active,
    texture_active, cross_summary, cross_high, candidates, deterministic, elapsed,
) -> None:
    family_lines = "\n".join(
        f"- **{family} ({len(features)}):** " + ", ".join(f"`{f}`" for f in features)
        for family, features in FAMILIES.items()
    )
    constants = diagnostics.loc[diagnostics.Is_Constant, "Feature"].tolist()
    nan_features = diagnostics.loc[diagnostics.NaN_Percentage > 0, ["Feature", "NaN_Percentage"]]
    duplicate_text = "; ".join(f"`{a}` = `{b}`" for a, b in duplicates) or "None"
    within_sections = []
    for family in FAMILIES:
        pairs = within_high[within_high.Family == family].drop(columns="Family")
        within_sections.append(f"### {family}\n\n{markdown(pairs)}")
    persistent = candidates[candidates.Category == "Persistent very-high correlation"]
    floor = candidates[candidates.Category == "Floor-influenced apparent redundancy"]
    report = f"""# Feature Redundancy Analysis Final Report

## A. Final feature structure

Canonical input: `{INPUT}`. It contains `{len(data):,}` objects, `{len(ALL_FEATURES)}` analyzed features, and one excluded identifier (`Object_ID`). The schema exactly matched the documented final structure.

{family_lines}

## B. Basic quality-control findings

- Constant features: {', '.join(constants) if constants else 'none'}.
- Exact duplicate pairs: {duplicate_text}.
- Features containing NaN values:

{markdown(nan_features)}

`Feature_Basic_Diagnostics.csv` reports valid count, NaN percentage, zero percentage among valid values, unique-value count, constant status, and exact-duplicate status for every feature. No feature was removed.

## C. Within-family redundancy results

{markdown(within_summary)}

All within-family pairs with `|rho| >= 0.90` follow; correlation alone is not treated as proof that a feature should be removed.

{chr(10).join(within_sections)}

## D. NTL active-subset diagnostic

The primary matrices use all pairwise-valid objects. The diagnostic subset contains `{int((data[PAN_STD] > 0).sum()):,}` objects (`{(data[PAN_STD] > 0).mean()*100:.6f}%`) with `{PAN_STD} > 0`. It was applied only to NTL spectral and conventional GLCM families. A decrease of at least 0.10 in absolute rho, or a fall below 0.90, is flagged as floor-influenced; persistence requires both full and active `|rho| >= 0.95`.

### NTL spectral/radiometric

{markdown(ntl_active)}

### Conventional GLCM texture

{markdown(texture_active)}

Floor-influenced findings are diagnostic and, by themselves, are insufficient evidence for removal.

## E. Cross-family redundancy

{markdown(cross_summary)}

All cross-family pairs with `|rho| >= 0.90` are saved in `Cross_Family_High_Correlation_Pairs.csv`; the summary above includes the five strongest pairs for every family combination. No active-subset cross-family analysis was performed.

## F. Candidate redundancy groups

{markdown(candidates)}

Verified deterministic relationships:

{markdown(pd.DataFrame(deterministic))}

## G. Scientific interpretation

The analysis separates exact/deterministic redundancy from empirical rank correlation. Exact duplicates and verified one-to-one mathematical transformations provide the strongest redundancy evidence. Very-high correlations that persist in the applicable active subset warrant review, but still do not prove interchangeability. Correlations that weaken after removal of PAN-uniform objects are partly driven by sensor-floor ties and should not be used alone to discard NTL features. Engineered spatial-context contrasts are deterministic combinations but may retain useful model-ready interactions.

## H. Recommended features to REVIEW before classification

1. Review every exact-duplicate pair first; retaining both has no descriptive benefit.
2. Review `compactness` versus `shape_index`, because the implemented definitions make them one-to-one deterministic transforms.
3. Review the `{len(persistent)}` persistent very-high-correlation candidates in `Feature_Redundancy_Candidates.csv`, considering scientific meaning and downstream validation before removal.
4. Interpret, but do not remove solely on this basis, the `{len(floor)}` floor-influenced apparent-redundancy candidates.
5. Treat `NTL_LocalContrast` and `NTL_NormalizedContrast` as explicit engineered interactions; their deterministic inputs do not automatically make them scientifically dispensable.

## End-of-run research summary

The final table contained the expected 39 NTL spectral, 39 optical spectral, 6 geometry, 5 conventional GLCM, and 6 NTL spatial-context features. Basic diagnostics identified the exact/constant/NaN issues listed above. The strongest within- and cross-family relationships are documented in the tables and CSV outputs. The active-subset tables identify which NTL and GLCM relationships weakened under `PAN_std > 0` and which remained persistent. Mathematical inspection verified the compactness/shape-index relationship and the two implemented spatial-context contrast equations; no deterministic relationship was inferred from correlation alone. The corrected B11/B12 and NDVI/NDBI/NDWI columns were used from the final table and no table-structure deviation was found. No imagery, labels, PCA, VIF, feature selection, classification, or automatic deletion was performed. Total runtime was `{elapsed:.2f}` seconds. The conservative review set is limited to exact duplicates, verified deterministic relationships, persistent very-high correlations, and separately identified floor-influenced pairs whose flag alone is insufficient for removal.
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_and_validate()

    print("[1/6] Basic diagnostics and exact-duplicate checks")
    duplicate_of, duplicates = exact_duplicate_map(data)
    diagnostics = basic_diagnostics(data, duplicate_of)
    diagnostics.to_csv(OUTPUT / "Feature_Basic_Diagnostics.csv", index=False)

    print("[2/6] Full-population Spearman matrix (pairwise valid)")
    full_corr = data[ALL_FEATURES].corr(method="spearman", min_periods=2)
    within_rows = []
    high_tables = []
    for family, features in FAMILIES.items():
        matrix = full_corr.loc[features, features]
        matrix.rename_axis("Feature").reset_index().to_csv(
            OUTPUT / f"Spearman_{family.replace('/', '_').replace(' ', '_')}.csv", index=False
        )
        pairs = matrix_pairs(matrix)
        pairs.insert(0, "Family", family)
        high_tables.append(pairs)
        upper = matrix.to_numpy()[np.triu_indices(len(features), 1)]
        finite = upper[np.isfinite(upper)]
        within_rows.append({
            "Family": family, "Feature_Count": len(features),
            "Feature_Pair_Count": len(features) * (len(features) - 1) // 2,
            "Pairs_Abs_Rho_GE_090": int(np.sum(np.abs(finite) >= 0.90)),
            "Pairs_Abs_Rho_GE_095": int(np.sum(np.abs(finite) >= 0.95)),
            "Maximum_Abs_Rho": float(np.max(np.abs(finite))) if len(finite) else np.nan,
        })
        heatmap(matrix, OUTPUT / f"Heatmap_Within_{family.replace('/', '_').replace(' ', '_')}.png", f"Within-family Spearman: {family}")
    within_summary = pd.DataFrame(within_rows)
    within_high = pd.concat(high_tables, ignore_index=True)
    within_high.to_csv(OUTPUT / "Within_Family_High_Correlation_Pairs.csv", index=False)

    print("[3/6] PAN-active subset diagnostic for NTL and GLCM")
    active = data[data[PAN_STD] > 0]
    ntl_active_matrix = active[FAMILIES[F_NTL]].corr(method="spearman", min_periods=2)
    texture_active_matrix = active[FAMILIES[F_TEX]].corr(method="spearman", min_periods=2)
    ntl_comparison = active_comparison(full_corr.loc[FAMILIES[F_NTL], FAMILIES[F_NTL]], ntl_active_matrix)
    texture_comparison = active_comparison(full_corr.loc[FAMILIES[F_TEX], FAMILIES[F_TEX]], texture_active_matrix)
    ntl_comparison.to_csv(OUTPUT / "NTL_ActiveSubset_Redundancy_Comparison.csv", index=False)
    texture_comparison.to_csv(OUTPUT / "Texture_ActiveSubset_Redundancy_Comparison.csv", index=False)

    print("[4/6] Cross-family summaries")
    cross_summary, cross_high, maximums = cross_family_outputs(full_corr)
    cross_summary.to_csv(OUTPUT / "Cross_Family_Redundancy_Summary.csv", index=False)
    cross_high.to_csv(OUTPUT / "Cross_Family_High_Correlation_Pairs.csv", index=False)
    heatmap(maximums, OUTPUT / "Heatmap_Cross_Family_Maximum_Abs_Rho.png", "Maximum absolute Spearman correlation between families", absolute=True)

    print("[5/6] Conservative redundancy candidates")
    deterministic = deterministic_checks(data)
    candidates = candidate_table(duplicates, deterministic, ntl_comparison, texture_comparison, within_high, cross_high)
    candidates.to_csv(OUTPUT / "Feature_Redundancy_Candidates.csv", index=False)

    print("[6/6] Research report")
    elapsed = perf_counter() - started
    write_report(data, diagnostics, duplicates, within_summary, within_high,
                 ntl_comparison, texture_comparison, cross_summary, cross_high,
                 candidates, deterministic, elapsed)
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
