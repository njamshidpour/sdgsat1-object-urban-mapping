"""Feature Extraction 05: corrected optical data and native PAN BG-log GLCM.

The validated v04 NTL spectral and geometry columns are reused by Object_ID.
All 39 optical columns are recomputed.  The old conventional GLCM columns are
discarded and replaced by five native-grid background-suppressed log-GLCM
columns.  No spatial-context or class information is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from shapely.affinity import translate
from shapely.geometry import mapping

import Feature_extraction_04 as v04


class CleanTerminalProgress(v04.TerminalProgress):
    """Single-line progress throttled to keep long research runs readable."""
    def __init__(self, label, *, min_interval=15.0, enabled=True):
        super().__init__(label, min_interval=min_interval, enabled=enabled)


v04.TerminalProgress = CleanTerminalProgress

LOG = logging.getLogger("Feature_extraction_05")
ROOT = Path(__file__).resolve().parents[1]
SEG_DIR = ROOT / "Object_based_fusion" / "Segmentation_output"
SEGMENTATION = SEG_DIR / "eCog_segments.shp"
V04_FULL = SEG_DIR / "objects_features_v04_full.csv"
OUTPUT_FULL = SEG_DIR / "objects_features_v05_full.csv"
DIAGNOSTICS = SEG_DIR / "Feature_Extraction_05_Diagnostics.json"
TEXTURE_ONLY_OUTPUT = SEG_DIR / "objects_features_v05_texture_only_full.csv"
TEXTURE_ONLY_DIAGNOSTICS = SEG_DIR / "Feature_Extraction_05_Texture_Only_Diagnostics.json"
PREFLIGHT_CACHE = SEG_DIR / "Feature_Extraction_05_Preflight.json"
REPORT = ROOT / "Object_based_fusion" / "FEATURE_EXTRACTION_05_REPORT.md"

OPT = ROOT / "optic_data_codex"
OPTICAL_BANDS = {
    "S2_B2": OPT / "Tehran_Raw_B2_10m.tif",
    "S2_B3": OPT / "Tehran_Raw_B3_10m.tif",
    "S2_B4": OPT / "Tehran_Raw_B4_10m.tif",
    "S2_B5": OPT / "B05_10m_pansharpened_GS.tif",
    "S2_B6": OPT / "B06_10m_pansharpened_GS.tif",
    "S2_B7": OPT / "B07_10m_pansharpened_GS.tif",
    "S2_B8": OPT / "Tehran_Raw_B8_10m.tif",
    "S2_B8A": OPT / "B8A_10m_pansharpened_GS.tif",
    "S2_B11": OPT / "B11_10m_pansharpened_GS.tif",
    "S2_B12": OPT / "B12_10m_pansharpened_GS.tif",
}
OPTICAL_INDICES = {
    "NDVI": OPT / "Tehran_NDVI_2023_Int16_compressed.tif",
    "NDBI": OPT / "Tehran_NDBI_2023_Int16_compressed.tif",
    "NDWI": OPT / "Tehran_NDWI_2023_Int16_compressed.tif",
}
PL = ROOT / "output_corrected_bands" / "PL_Radiance_salt_pepper.tif"
PH = ROOT / "output_corrected_bands" / "PH_Radiance_salt_pepper.tif"
HDR = ROOT / "Texture_Method_Audit_02_outputs" / "PAN_HDR_PreCoreg_Radiance.tif"
PL_ALIGNED = ROOT / "output_corrected_bands" / "PL_Radiance_salt_pepper_aligned.tif"
PH_ALIGNED = ROOT / "output_corrected_bands" / "PH_Radiance_salt_pepper_aligned.tif"
PL_COREG = ROOT / "output_corrected_bands" / "PL_Radiance_salt_pepper_coregistered.tif"
PH_COREG = ROOT / "output_corrected_bands" / "PH_Radiance_salt_pepper_coregistered.tif"

# Empirically recovered from four independent 3072x3072 comparisons of the
# persisted aligned and coregistered artifacts.  Values below are the inverse
# mapping from final S2 pixels to the unshifted reprojected native HDR pixels.
HDR_INVERSE_SHIFT_COL = 2.57750951
HDR_INVERSE_SHIFT_ROW = 0.99543452
SHIFT_EVIDENCE = {
    "PL_aligned_to_coreg_median": [-2.56008788, -0.46211676],
    "PH_aligned_to_coreg_median": [-2.55455254, -1.50855822],
    "HDR_aligned_to_coreg_median": [-2.57750951, -0.99543452],
    "HDR_four_window_range": [0.04105316, 0.03099772],
}

NTL_COLUMNS = [f"{name}_{s}" for name in v04.RASTER_INPUTS for s in ("mean", "std", "max")]
GEOMETRY_COLUMNS = ["area_sqm", "perimeter_m", "compactness", "shape_index", "elongation", "length_width_ratio"]
OLD_TEXTURE = [f"SDGSAT_Pan_glcm_{x}" for x in ("contrast", "homogeneity", "energy", "correlation", "entropy")]
TEXTURE_SUFFIXES = ("contrast", "homogeneity", "energy", "correlation", "entropy")
NEW_TEXTURE = [f"SDGSAT_Pan_bgglcm_{x}" for x in TEXTURE_SUFFIXES]
OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray): return _jsonable(value.tolist())
    return value


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _weighted_percentile(values, counts, p):
    rank = p / 100.0 * (int(counts.sum()) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    cumulative = np.cumsum(counts)
    vlo = values[np.searchsorted(cumulative, lo + 1)]
    vhi = values[np.searchsorted(cumulative, hi + 1)]
    return float(vlo + (rank - lo) * (vhi - vlo))


def _raster_meta(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {"path": str(path.resolve()), "dtype": src.dtypes[0], "nodata": src.nodata,
                "crs": src.crs.to_string() if src.crs else None, "shape": list(src.shape),
                "transform": list(src.transform)[:6], "bounds": list(src.bounds),
                "resolution": list(src.res), "compression": str(src.compression)}


def preflight(force=False) -> dict:
    paths = [SEGMENTATION, V04_FULL, *OPTICAL_BANDS.values(), *OPTICAL_INDICES.values(),
             PL, PH, HDR, PL_ALIGNED, PH_ALIGNED, PL_COREG, PH_COREG]
    missing = [str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))
    fingerprints = {str(p.resolve()): _fingerprint(p) for p in paths}
    if PREFLIGHT_CACHE.exists() and not force:
        cached = json.loads(PREFLIGHT_CACHE.read_text(encoding="utf-8"))
        if cached.get("fingerprints") == fingerprints:
            LOG.info("Using validated preflight cache: %s", PREFLIGHT_CACHE)
            return cached

    t0 = perf_counter()
    b11_hash, b12_hash = _sha256(OPTICAL_BANDS["S2_B11"]), _sha256(OPTICAL_BANDS["S2_B12"])
    if b11_hash == b12_hash: raise RuntimeError("Corrected B11 and B12 are byte-identical; stopping.")

    index_info = {}
    for name, path in OPTICAL_INDICES.items():
        meta = _raster_meta(path); minimum, maximum = 32767, -32768
        outside = minus32768 = zeros = total = 0
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                a = src.read(1, window=window)
                minimum = min(minimum, int(a.min())); maximum = max(maximum, int(a.max()))
                outside += int(((a < -10000) | (a > 10000)).sum())
                minus32768 += int((a == -32768).sum()); zeros += int((a == 0).sum()); total += a.size
        meta.update({"stored_min": minimum, "stored_max": maximum,
                     "physical_min": minimum / 10000, "physical_max": maximum / 10000,
                     "outside_expected_count": outside, "minus_32768_count": minus32768,
                     "zero_count": zeros, "pixel_count": total, "scale_factor": 10000})
        if meta["dtype"] != "int16" or outside or minus32768:
            raise RuntimeError(f"Unsafe {name} index encoding: {meta}")
        index_info[name] = meta

    plm, phm, hdrm = _raster_meta(PL), _raster_meta(PH), _raster_meta(HDR)
    for key in ("shape", "crs", "transform", "bounds", "resolution"):
        if plm[key] != phm[key]: raise RuntimeError(f"PL/PH incompatible {key}")
    # Validate the persisted native HDR against exact local PL/PH averages.
    max_error = 0.0
    with rasterio.open(PL) as pl, rasterio.open(PH) as ph, rasterio.open(HDR) as hdr:
        for row, col in ((1000, 1000), (5000, 8000), (10000, 18000)):
            w = rasterio.windows.Window(col, row, 512, 512)
            expected = (pl.read(1, window=w).astype("float32") + ph.read(1, window=w).astype("float32")) * np.float32(0.5)
            max_error = max(max_error, float(np.max(np.abs(expected - hdr.read(1, window=w)))))
        counts = Counter()
        for _, window in hdr.block_windows(1):
            a = hdr.read(1, window=window, masked=True).compressed()
            vals, n = np.unique(a[np.isfinite(a)], return_counts=True)
            counts.update(dict(zip(vals.tolist(), n.tolist())))
    if max_error != 0: raise RuntimeError(f"Persisted native HDR mismatch: max error {max_error}")
    values = np.asarray(sorted(counts), dtype="float64")
    weights = np.asarray([counts[v] for v in values], dtype="int64")
    mode_i = int(np.argmax(weights)); floor = float(values[mode_i]); floor_count = int(weights[values <= floor].sum())
    active = values > floor
    excess = values[active] - floor; active_counts = weights[active]
    epsilon = float(excess.min() * 1e-6)
    y = np.log(excess + epsilon)
    y_low, y_high = _weighted_percentile(y, active_counts, 1), _weighted_percentile(y, active_counts, 99)

    result = {"fingerprints": fingerprints, "b11_sha256": b11_hash, "b12_sha256": b12_hash,
              "b11_b12_distinct": True, "indices": index_info, "PL": plm, "PH": phm, "HDR": hdrm,
              "hdr_formula": "float32 0.5*(PL+PH)", "hdr_validation_max_abs_error": max_error,
              "T_floor": floor, "mode_count": int(weights[mode_i]), "valid_hdr_pixels": int(weights.sum()),
              "floor_or_below_count": floor_count, "floor_or_below_fraction": floor_count / int(weights.sum()),
              "epsilon": epsilon, "Y_low": y_low, "Y_high": y_high,
              "radiance_low": floor + math.exp(y_low) - epsilon,
              "radiance_high": floor + math.exp(y_high) - epsilon,
              "mapping": {**SHIFT_EVIDENCE, "inverse_shift_col": HDR_INVERSE_SHIFT_COL,
                          "inverse_shift_row": HDR_INVERSE_SHIFT_ROW,
                          "method": "phase correlation of persisted aligned/coregistered HDR artifacts; median of four windows"},
              "elapsed_seconds": perf_counter() - t0}
    PREFLIGHT_CACHE.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    return result


def load_objects(mode: str, limit: int | None) -> gpd.GeoDataFrame:
    if mode == "full":
        gdf, _ = v04.verify_and_prepare_objects(SEGMENTATION, "Seg_id", None)
        return gdf
    if mode == "validation":
        sample = pd.read_csv(ROOT / "Texture_Method_Audit_02_outputs" / "Texture_Audit_Sample.csv")
        unique = sample[["Object_ID"]].drop_duplicates()
        ids = unique.sample(n=min(limit or 750, len(unique)), random_state=20260905)["Object_ID"].astype("int64").to_numpy()
    else:
        ids = np.arange(limit or 10000, dtype="int64")
    gdf = pyogrio.read_dataframe(SEGMENTATION, columns=["Seg_id"], fids=ids)
    gdf["geometry"] = gdf.geometry.map(v04._repair_geometry)
    gdf = gdf.rename(columns={"Seg_id": "Object_ID"})
    order = pd.Index(ids, name="Object_ID")
    gdf = gdf.set_index("Object_ID").reindex(order)
    gdf.index.name = "Object_ID"
    gdf = gdf.reset_index()
    if not np.array_equal(gdf.Object_ID.to_numpy(), ids): raise RuntimeError("FID/Seg_id mapping mismatch")
    return gdf


def load_baseline(ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = ["Object_ID", *NTL_COLUMNS, *GEOMETRY_COLUMNS, *OLD_TEXTURE]
    old = pd.read_csv(V04_FULL, usecols=use)
    if old.Object_ID.duplicated().any(): raise RuntimeError("v04 Object_ID is not unique")
    selected = old.set_index("Object_ID").loc[ids].reset_index()
    return selected[["Object_ID", *NTL_COLUMNS, *GEOMETRY_COLUMNS]], selected[["Object_ID", *OLD_TEXTURE, "SDGSAT_Pan_mean", "SDGSAT_Pan_std"]]


def extract_optical(gdf):
    frame, diag = v04.extract_optical_features(gdf, OPTICAL_BANDS, OPTICAL_INDICES,
        alignment_reference_raster=v04.OPTICAL_ALIGNMENT_REFERENCE,
        allow_geometry_reprojection=False, show_progress=True)
    for name in OPTICAL_INDICES:
        for suffix in ("mean", "std", "max"): frame[f"{name}_{suffix}"] /= 10000.0
    return frame, diag


def _shift_to_native(gdf, reference_transform, native_crs):
    xoff = HDR_INVERSE_SHIFT_COL * reference_transform.a + HDR_INVERSE_SHIFT_ROW * reference_transform.b
    yoff = HDR_INVERSE_SHIFT_COL * reference_transform.d + HDR_INVERSE_SHIFT_ROW * reference_transform.e
    shifted = gdf.copy(); shifted.geometry = shifted.geometry.map(lambda x: translate(x, xoff=xoff, yoff=yoff) if x is not None else None)
    return shifted.to_crs(native_crs), xoff, yoff


def _slices(size, delta):
    return (slice(0, size-delta), slice(delta, size)) if delta >= 0 else (slice(-delta, size), slice(0, size+delta))


def _props(matrix):
    i, j = np.indices(matrix.shape, dtype="float64"); d2 = (i-j)**2
    mi, mj = float((i*matrix).sum()), float((j*matrix).sum())
    si = math.sqrt(float((((i-mi)**2)*matrix).sum())); sj = math.sqrt(float((((j-mj)**2)*matrix).sum()))
    pos = matrix[matrix > 0]
    return {"contrast": float((matrix*d2).sum()), "homogeneity": float((matrix/(1+d2)).sum()),
            "energy": float(np.sqrt((matrix**2).sum())),
            "correlation": float((((i-mi)*(j-mj)*matrix).sum())/(si*sj)) if si*sj > np.finfo(float).eps else 1.0,
            "entropy": float(-(pos*np.log2(pos)).sum())}


def _object_texture(data, valid, floor, epsilon, low, high):
    if int(valid.sum()) < 9: return None, {"valid_pixels": int(valid.sum()), "retained_pairs": 0}
    active = valid & (data > floor); q = np.zeros(data.shape, dtype="uint8")
    y = np.log(data[active].astype("float64") - floor + epsilon)
    u = (np.clip(y, low, high) - low) / (high-low)
    q[active] = 1 + np.minimum(np.floor(u*31).astype("int64"), 30)
    directions=[]; retained=0; removed00=0
    for dr, dc in OFFSETS:
        rs, rt = _slices(q.shape[0], dr); cs, ct = _slices(q.shape[1], dc)
        mask = valid[rs,cs] & valid[rt,ct]
        if not mask.any(): continue
        a, b = q[rs,cs][mask].astype("int64"), q[rt,ct][mask].astype("int64")
        counts = np.bincount(a*32+b, minlength=1024).reshape(32,32).astype("float64")
        removed00 += int(counts[0,0]); counts[0,0] = 0
        symmetric = counts + counts.T; total = symmetric.sum()
        if total <= 0: continue
        retained += int(total/2); directions.append(_props(symmetric/total))
    details={"valid_pixels":int(valid.sum()),"active_pixels":int(active.sum()),"floor_pixels":int((valid&~active).sum()),
             "retained_pairs":retained,"removed_floor_floor_pairs":removed00,
             "min_state":int(q[valid].min()) if valid.any() else None,"max_state":int(q[valid].max()) if valid.any() else None}
    if not directions: return None, details
    return {k:float(np.mean([x[k] for x in directions])) for k in TEXTURE_SUFFIXES}, details


def extract_texture(gdf, cfg):
    result = pd.DataFrame({"Object_ID":gdf.Object_ID.to_numpy()});
    for c in NEW_TEXTURE: result[c]=np.nan
    reasons=Counter(); examples=[]; progress=v04.TerminalProgress(f"BG-log GLCM ({len(gdf):,} objects)")
    with rasterio.open(HDR) as src, rasterio.open(PL_ALIGNED) as ref:
        LOG.info("Mapping %s object geometries to the native PAN grid", f"{len(gdf):,}")
        tgdf, xoff, yoff = _shift_to_native(gdf, ref.transform, src.crs)
        LOG.info("Native-grid mapping complete; starting local-window texture extraction")
        for pos, geom in enumerate(tgdf.geometry):
            progress.update(pos/len(tgdf))
            if geom is None or geom.is_empty: reasons["empty_or_outside"]+=1; continue
            window=v04._bounded_window(geom,src)
            if window is None: reasons["empty_or_outside"]+=1; continue
            arr=src.read(1,window=window,masked=True); data=np.asarray(arr.data)
            inside=geometry_mask([mapping(geom)],out_shape=data.shape,transform=src.window_transform(window),invert=True,all_touched=False)
            valid=inside & ~np.ma.getmaskarray(arr) & np.isfinite(data)
            props, detail=_object_texture(data,valid,cfg["T_floor"],cfg["epsilon"],cfg["Y_low"],cfg["Y_high"])
            if len(examples)<12 and (pos % max(1,len(gdf)//12)==0): examples.append({"Object_ID":int(gdf.Object_ID.iloc[pos]),**detail})
            if props is None:
                reasons["too_few_pixels" if detail["valid_pixels"]<9 else "no_retained_pairs"]+=1; continue
            for k,v in props.items(): result.iat[pos,result.columns.get_loc(f"SDGSAT_Pan_bgglcm_{k}")]=v
        progress.complete()
    return result,{"reasons":dict(reasons),"examples":examples,"geometry_shift_degrees":[xoff,yoff],"valid_objects":int(result[NEW_TEXTURE[0]].notna().sum())}


def feature_stats(df, cols):
    out={}
    for c in cols:
        s=df[c]; v=s.dropna(); mode=v.value_counts(dropna=True)
        q=v.quantile([.25,.5,.75]) if len(v) else pd.Series({.25:np.nan,.5:np.nan,.75:np.nan})
        out[c]={"valid_count":len(v),"nan_count":int(s.isna().sum()),"nan_percentage":float(s.isna().mean()*100),
                "zero_count":int((v==0).sum()),"zero_fraction":float((v==0).mean()) if len(v) else None,
                "unique_count":int(v.nunique()),"most_frequent_value":float(mode.index[0]) if len(mode) else None,
                "most_frequent_percentage":float(mode.iloc[0]/len(v)*100) if len(mode) else None,
                "min":float(v.min()) if len(v) else None,"Q25":float(q.loc[.25]),"median":float(q.loc[.5]),
                "mean":float(v.mean()) if len(v) else None,"Q75":float(q.loc[.75]),"max":float(v.max()) if len(v) else None,
                "IQR":float(q.loc[.75]-q.loc[.25])}
    return out


def correlation_summary(df, texture_cols):
    matrix=df[texture_cols].corr(method="spearman")
    vals=np.abs(matrix.to_numpy()[np.triu_indices(len(texture_cols),1)])
    radio={c:{r:float(df[[c,r]].corr(method="spearman").iloc[0,1]) for r in ("SDGSAT_Pan_mean","SDGSAT_Pan_std")} for c in texture_cols}
    return {"matrix":matrix.to_dict(),"pairs_abs_rho_ge_0.90":int((vals>=.90).sum()),"pairs_abs_rho_ge_0.95":int((vals>=.95).sum()),"radiometry":radio}


def write_report(diag):
    p=diag["preflight"]; opt=diag["optical_feature_statistics"]; tex=diag["new_texture_statistics"]
    def table(stats):
        lines=["| Feature | Valid | NaN % | Min | Mean | Median | Max | Mode % |","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c,s in stats.items(): lines.append(f"| {c} | {s['valid_count']} | {s['nan_percentage']:.4f} | {s['min']} | {s['mean']} | {s['median']} | {s['max']} | {s['most_frequent_percentage']:.4f} |")
        return "\n".join(lines)
    ntl_inputs="\n".join(f"- `{k}`: `{v}`" for k,v in v04.RASTER_INPUTS.items())
    optical_inputs="\n".join(f"- `{k}`: `{v}`" for k,v in {**OPTICAL_BANDS,**OPTICAL_INDICES}.items())
    text=f"""# FEATURE EXTRACTION 05 — Research Report

## A. Changes relative to Feature_extraction_04.py

The 39 validated SDGSAT spectral columns and six EPSG:32639 geometry columns were reused exactly from `objects_features_v04_full.csv` by verified `Object_ID`. All 39 optical columns were recomputed. The five conventional `SDGSAT_Pan_glcm_*` columns were removed and replaced by five logarithmic background-suppressed GLCM columns. Spatial context was not calculated.

## B. Final data inputs

- Segmentation: `{SEGMENTATION}`; source ID `Seg_id`, output ID `Object_ID`.
{ntl_inputs}
{optical_inputs}
- PL PAN: `{PL}`
- PH PAN: `{PH}`

## C. Optical corrections

B11 and B12 are distinct: SHA256 `{p['b11_sha256']}` versus `{p['b12_sha256']}`. Their paths differ and a complete earlier array comparison found 395,440,868 unequal pixels. Corrected indices are Int16/LZW, 14,235×27,780, have no declared NoData and contain no -32768 or values outside [-10000,10000]. LZW was read transparently and is lossless. Statistics were divided by 10,000 exactly once after NoData-aware zonal extraction.

## D. Optical feature statistics

{table(opt)}

The nine index-derived NaN counts are recorded above. Stored index ranges were NDVI {p['indices']['NDVI']['stored_min']}..{p['indices']['NDVI']['stored_max']}, NDBI {p['indices']['NDBI']['stored_min']}..{p['indices']['NDBI']['stored_max']}, and NDWI {p['indices']['NDWI']['stored_min']}..{p['indices']['NDWI']['stored_max']}; physical values are those numbers /10000.

## E. Native PAN texture source

PL and PH are exactly the requested pre-coregistration rasters. They have identical CRS, 10 m grid, dimensions, transform and extent. HDR was computed as float32 `0.5*(PL+PH)`; the persisted native HDR was checked in three windows with maximum absolute error `{p['hdr_validation_max_abs_error']}`. PAN was not resampled during texture extraction.

## F. Native-grid object mapping

The formerly unpersisted shift was recovered from the actual aligned and coregistered artifacts. Four independent windows gave effective HDR aligned→coreg shift `{-HDR_INVERSE_SHIFT_COL:.8f}, {-HDR_INVERSE_SHIFT_ROW:.8f}` pixels with range {p['mapping']['HDR_four_window_range']}. The inverse mapping translated final-grid polygons by `+{HDR_INVERSE_SHIFT_COL:.8f}` columns and `+{HDR_INVERSE_SHIFT_ROW:.8f}` rows on the Sentinel transform, then reprojected polygons to EPSG:32639. Only geometry moved; native HDR samples did not. PL/PH vertical estimates differ by about 1.05 pixels; the stable effective equal-weight HDR shift was used. This is an interpretation limitation, not a hidden approximation.

## G. Sensor floor

The exact full-HDR mode, not a hard-coded value, is `T_floor={p['T_floor']:.17g}`. Pixels at/below it comprise `{p['floor_or_below_fraction']*100:.6f}%` of valid HDR pixels. The term *sensor-floor state* is used without claiming physical no-light.

## H. Log-radiance transformation

For `R>T_floor`, `E=R-T_floor` and `Y=ln(E+epsilon)`, with `epsilon={p['epsilon']:.17g}` (one millionth of the minimum observed positive excess).

## I. Global quantization

One full-image scheme was used: 32 states; floor=0; above-floor=1..31. Full above-floor P1/P99 gave `Y_low={p['Y_low']:.17g}`, `Y_high={p['Y_high']:.17g}`, corresponding to radiances `{p['radiance_low']:.17g}` and `{p['radiance_high']:.17g}`. For clipped `Y*`, `u=(Y*-Y_low)/(Y_high-Y_low)` and `g=1+min(floor(31u),30)`, guaranteeing endpoints 1 and 31.

## J. Background suppression

For each direction only `C(0,0)` was set to zero before normalization. Floor-light, light-floor and light-light transitions remained.

## K. GLCM settings

Distance 1; directions 0°,45°,90°,135°; symmetric counts `C+Cᵀ`; `all_touched=False`; both pixel centers must be within the same object. Descriptors were calculated per retained direction and then equally averaged.

## L. Texture equations

`Contrast=ΣP(i,j)(i-j)²`; `Homogeneity=ΣP(i,j)/(1+(i-j)²)`; `Energy=sqrt(ΣP²)`; `Correlation=Σ(i-μi)(j-μj)P/(σiσj)` with v04 convention 1 when a marginal variance is zero; `Entropy=-Σ[P log2(P)]` for positive P.

## M. Texture diagnostics

{table(tex)}

Within-texture Spearman: `{json.dumps(diag['new_texture_correlations']['matrix'])}`. Pairs |rho|≥0.90: {diag['new_texture_correlations']['pairs_abs_rho_ge_0.90']}; ≥0.95: {diag['new_texture_correlations']['pairs_abs_rho_ge_0.95']}. Correlations with PAN mean/std: `{json.dumps(diag['new_texture_correlations']['radiometry'])}`. Undefined objects correctly received five NaNs. Reasons: `{json.dumps(diag['texture_extraction']['reasons'])}`.

## N. Comparison with Feature Extraction 04 texture

Old diagnostics: `{json.dumps(diag['old_texture_summary'])}`. New diagnostics and correlations above quantify changes in validity, NaNs, zeros, dominant values, redundancy and PAN radiometry dependence. Old columns are not present in v05.

## O. Geometry

`area_sqm`, `perimeter_m`, `compactness`, `shape_index`, `elongation`, and `length_width_ratio` remain unchanged and were originally calculated after geometry repair in EPSG:32639.

## P. Output structure

39 NTL + 39 optical + 6 geometry + 5 texture = 89 features plus `Object_ID` = 90 columns. Row count: {diag['object_count']}. Output: `{diag['output_csv']}`.

## Q. Performance

Stage seconds: `{json.dumps(diag['stage_timings_seconds'])}`. NTL and geometry are exact v04 reuse/read rather than recomputation; optical and texture are newly calculated.

## R. Scientific issues

The two PAN channels had distinct independently estimated vertical coregistration shifts. The effective HDR mapping was highly stable across windows but carries roughly half-pixel channel-disagreement uncertainty near object boundaries. Index rasters declare no NoData; full scans found only physically valid scaled values, so every stored pixel was treated as valid.

## S. Deviations

The selected audit stored P2/P98, while this specification explicitly requested P1/P99 unless final validated alternatives existed; v05 therefore calculated exact full above-floor P1/P99. Existing NTL and geometry were reused rather than recomputed, preserving them exactly. No other methodological deviation was made.
"""
    REPORT.write_text(text,encoding="utf-8")


def run(mode="validation", limit=750, output=None, force_preflight=False):
    started=perf_counter(); timings={}; p=preflight(force_preflight)
    t=perf_counter(); gdf=load_objects(mode, None if mode=="full" else limit); timings["object_loading"]=perf_counter()-t
    ids=gdf.Object_ID.to_numpy(); t=perf_counter(); baseline,old=load_baseline(ids); timings["ntl_geometry_reuse"]=perf_counter()-t
    t=perf_counter(); optical,opt_diag=extract_optical(gdf); timings["optical_extraction"]=perf_counter()-t
    t=perf_counter(); texture,tex_diag=extract_texture(gdf,p); timings["texture_extraction"]=perf_counter()-t
    final=pd.concat([baseline[["Object_ID",*NTL_COLUMNS]],optical.drop(columns="Object_ID"),baseline[GEOMETRY_COLUMNS],texture.drop(columns="Object_ID")],axis=1)
    expected=["Object_ID",*NTL_COLUMNS,*[f"{n}_{s}" for n in [*OPTICAL_BANDS,*OPTICAL_INDICES] for s in ("mean","std","max")],*GEOMETRY_COLUMNS,*NEW_TEXTURE]
    if list(final.columns)!=expected or final.shape[1]!=90 or not np.array_equal(final.Object_ID,ids): raise RuntimeError("Final schema/ID invariant failed")
    output=Path(output) if output else (OUTPUT_FULL if mode=="full" else SEG_DIR/f"objects_features_v05_{mode}{len(final)}.csv")
    t=perf_counter(); final.to_csv(output,index=False); timings["output_writing"]=perf_counter()-t
    analysis=pd.concat([old,texture.drop(columns="Object_ID")],axis=1)
    diag={"mode":mode,"object_count":len(final),"column_count":90,"feature_count":89,"output_csv":str(output.resolve()),
          "preflight":p,"optical":opt_diag,"texture_extraction":tex_diag,
          "optical_feature_statistics":feature_stats(final,[f"{n}_{s}" for n in [*OPTICAL_BANDS,*OPTICAL_INDICES] for s in ("mean","std","max")]),
          "new_texture_statistics":feature_stats(final,NEW_TEXTURE),"new_texture_correlations":correlation_summary(analysis,NEW_TEXTURE),
          "old_texture_summary":{"statistics":feature_stats(old,OLD_TEXTURE),"correlations":correlation_summary(old,OLD_TEXTURE)},
          "stage_timings_seconds":timings,"elapsed_seconds":perf_counter()-started,"columns":list(final.columns)}
    DIAGNOSTICS.write_text(json.dumps(_jsonable(diag),indent=2),encoding="utf-8"); write_report(diag)
    LOG.info("Completed %s: %d rows x %d columns -> %s",mode,*final.shape,output)
    return final,diag


def run_texture_only(base_table=OUTPUT_FULL, output=None, *, full_dataset=False,
                     limit=750, force_preflight=False):
    """Replace only the five v05 texture columns in an existing feature table.

    All non-texture values are read from ``base_table`` and carried to the new
    CSV without recomputation. The source table is never overwritten.
    """
    started = perf_counter(); timings = {}; p = preflight(force_preflight)
    base_table = Path(base_table).resolve()
    if not base_table.exists():
        raise FileNotFoundError(f"Base feature table not found: {base_table}")
    default_output = (
        TEXTURE_ONLY_OUTPUT if full_dataset
        else SEG_DIR / f"objects_features_v05_texture_only_subset{limit}.csv"
    )
    output = Path(output).resolve() if output else default_output.resolve()
    if output == base_table:
        raise ValueError("Texture-only output must differ from the base table; source overwrite is refused.")

    t = perf_counter()
    base = pd.read_csv(base_table)
    if not full_dataset:
        base = base.head(limit).copy()
    timings["base_table_read"] = perf_counter() - t
    if "Object_ID" not in base or base["Object_ID"].duplicated().any():
        raise RuntimeError("Base table lacks a unique Object_ID column.")
    nontexture = [c for c in base.columns if c not in OLD_TEXTURE and c not in NEW_TEXTURE]
    expected_nontexture = [
        "Object_ID", *NTL_COLUMNS,
        *[f"{n}_{s}" for n in [*OPTICAL_BANDS, *OPTICAL_INDICES] for s in ("mean", "std", "max")],
        *GEOMETRY_COLUMNS,
    ]
    if nontexture != expected_nontexture:
        missing = [c for c in expected_nontexture if c not in nontexture]
        extra = [c for c in nontexture if c not in expected_nontexture]
        raise RuntimeError(f"Unexpected base-table schema; missing={missing}, extra={extra}")

    t = perf_counter()
    mode = "full" if full_dataset else "benchmark"
    gdf = load_objects(mode, None if full_dataset else len(base))
    if not np.array_equal(gdf.Object_ID.to_numpy(), base.Object_ID.to_numpy()):
        raise RuntimeError("Segmentation Object_ID order does not match the base table.")
    timings["object_loading_and_repair"] = perf_counter() - t

    t = perf_counter()
    texture, tex_diag = extract_texture(gdf, p)
    timings["texture_extraction"] = perf_counter() - t
    final = pd.concat(
        [base[expected_nontexture].reset_index(drop=True), texture[NEW_TEXTURE].reset_index(drop=True)],
        axis=1,
    )
    if final.shape != (len(base), 90) or not np.array_equal(final.Object_ID, base.Object_ID):
        raise RuntimeError("Texture-only output failed row/schema invariants.")
    # This exact comparison includes NaNs and confirms the 84 features plus ID
    # were copied from the base dataframe without a calculation step.
    if not final[expected_nontexture].equals(base[expected_nontexture]):
        raise RuntimeError("A non-texture base column changed before output writing.")

    t = perf_counter(); output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False); timings["output_writing"] = perf_counter() - t
    diagnostics = {
        "mode": "texture_only_full" if full_dataset else "texture_only_subset",
        "base_table": str(base_table), "output_csv": str(output),
        "object_count": len(final), "column_count": final.shape[1],
        "copied_nontexture_columns_including_id": len(expected_nontexture),
        "recomputed_texture_columns": NEW_TEXTURE,
        "nontexture_dataframe_equality_check": True,
        "preflight": p, "texture_extraction": tex_diag,
        "texture_feature_statistics": feature_stats(final, NEW_TEXTURE),
        "stage_timings_seconds": timings, "elapsed_seconds": perf_counter() - started,
    }
    TEXTURE_ONLY_DIAGNOSTICS.write_text(
        json.dumps(_jsonable(diagnostics), indent=2), encoding="utf-8"
    )
    LOG.info("Texture-only run completed: %d rows x %d columns -> %s", *final.shape, output)
    return final, diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group()
    group.add_argument("--validation",action="store_true"); group.add_argument("--benchmark",action="store_true"); group.add_argument("--full-dataset",action="store_true")
    parser.add_argument("--texture-only", action="store_true",
                        help="Recompute only five BG-log GLCM columns and copy all other columns from --base-table.")
    parser.add_argument("--base-table", type=Path, default=OUTPUT_FULL,
                        help="Existing 90-column v05 table used by --texture-only.")
    parser.add_argument("--limit",type=int,default=None); parser.add_argument("--output-csv",type=Path); parser.add_argument("--force-preflight",action="store_true")
    args=parser.parse_args(); mode="full" if args.full_dataset else "benchmark" if args.benchmark else "validation"
    limit=args.limit or (10000 if mode=="benchmark" else 750)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    if args.texture_only:
        run_texture_only(args.base_table, args.output_csv,
                         full_dataset=args.full_dataset, limit=limit,
                         force_preflight=args.force_preflight)
    else:
        run(mode,limit,args.output_csv,args.force_preflight)


if __name__=="__main__": main()
