"""Extract six immediate NTL spatial-context features from a saved Rook graph.

The script never reads labels, reference maps, optical variables, or model
predictions. A pilot writes a representative subset; --full-run writes every
object while preserving the canonical input-table row order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEGMENTATION_PATH = ROOT / "Object_based_fusion" / "Segmentation_output" / "eCog_segments.shp"
FEATURE_TABLE_PATH = ROOT / "Object_based_fusion" / "Segmentation_output" / "objects_features_final_full.csv"
GRAPH_PATH = ROOT / "Object_based_fusion" / "Segmentation_output" / "rook_adjacency_graph_v01.npz"
GRAPH_METADATA_PATH = ROOT / "Object_based_fusion" / "Segmentation_output" / "rook_adjacency_graph_v01_metadata.json"
OUTPUT_DIRECTORY = ROOT / "Object_based_fusion" / "Segmentation_output" / "Spatial_Context_Extraction_02"

site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
proj_data = site_packages / "rasterio" / "proj_data"
if proj_data.is_dir():
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_FIELD = "Object_ID"
SEGMENT_ID_FIELD = "Seg_id"
PAN_FIELD = "SDGSAT_Pan_mean"
EPSILON = 1e-12
RANDOM_SEED = 20261001
CONTEXT_COLUMNS = [
    "NTL_NeighborMean", "NTL_NeighborStd", "NTL_LocalContrast",
    "NTL_NormalizedContrast", "NTL_MeanAbsNeighborContrast", "NeighborCount",
]
CORRELATION_COLUMNS = [PAN_FIELD, *CONTEXT_COLUMNS]


class Progress:
    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(1, int(total))
        self.start, self.previous = time.perf_counter(), 0

    def update(self, completed: int):
        completed = min(completed, self.total)
        elapsed = time.perf_counter()-self.start
        fraction = completed/self.total
        eta = elapsed/completed*(self.total-completed) if completed else np.nan
        eta_text = "--:--" if not np.isfinite(eta) else f"{int(eta)//60:02d}:{int(eta)%60:02d}"
        filled = round(24*fraction)
        message = (f"\r{self.label:<28} [{'#'*filled}{'-'*(24-filled)}] "
                   f"{100*fraction:6.2f}% ETA {eta_text}")
        print(message+" "*max(0,self.previous-len(message)),end="",flush=True)
        self.previous=len(message)
        if completed == self.total: print()


def polygonal_part(geometry):
    if geometry is None or geometry.is_empty: return None
    if geometry.geom_type in {"Polygon","MultiPolygon"}: return geometry
    if geometry.geom_type == "GeometryCollection":
        parts=[part for part in geometry.geoms
               if part.geom_type in {"Polygon","MultiPolygon"} and not part.is_empty]
        return unary_union(parts) if parts else None
    return None


def output_paths(directory: Path, pilot: bool):
    suffix="_PILOT" if pilot else ""
    return {
        "context":directory/f"Spatial_Context_Features_02{suffix}.csv",
        "merged":directory/f"objects_features_with_spatial_context_02{suffix}.csv",
        "spearman":directory/f"Spatial_Context_Spearman_02{suffix}.csv",
        "neighbor_histogram":directory/f"Spatial_Context_NeighborCount_Histogram{suffix}.png",
        "correlation_heatmap":directory/f"Spatial_Context_Correlation_Heatmap{suffix}.png",
        "diagnostics":directory/f"Spatial_Context_Extraction_02_Diagnostics{suffix}.json",
        "report":directory/f"Spatial_Context_Extraction_02_Report{suffix}.md",
    }


def load_inputs(segmentation_path: Path, feature_path: Path):
    if not segmentation_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError(f"Missing segmentation={segmentation_path} or feature table={feature_path}")
    header=pd.read_csv(feature_path,nrows=0).columns.tolist()
    missing={ID_FIELD,PAN_FIELD}.difference(header)
    if missing: raise KeyError(f"Feature table missing columns: {sorted(missing)}")
    requested=[ID_FIELD,PAN_FIELD]+[c for c in CONTEXT_COLUMNS if c in header]
    signal=pd.read_csv(feature_path,usecols=requested)
    if signal[ID_FIELD].isna().any() or not signal[ID_FIELD].is_unique:
        raise ValueError("Feature-table Object_ID must be non-null and unique.")
    object_ids=signal[ID_FIELD].to_numpy(dtype=np.int64)
    if not np.array_equal(signal[ID_FIELD].to_numpy(),object_ids):
        raise ValueError("Feature-table Object_ID is not exactly representable as int64.")
    radiance=signal[PAN_FIELD].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(radiance)):
        raise ValueError("SDGSAT_Pan_mean contains missing or non-finite values.")
    if np.any(radiance < 0):
        raise ValueError("SDGSAT_Pan_mean contains negative radiance values.")

    polygons=gpd.read_file(segmentation_path,columns=[SEGMENT_ID_FIELD])
    if polygons.crs is None or SEGMENT_ID_FIELD not in polygons:
        raise ValueError("Segmentation needs CRS metadata and Seg_id.")
    if polygons[SEGMENT_ID_FIELD].isna().any() or not polygons[SEGMENT_ID_FIELD].is_unique:
        raise ValueError("Segmentation Seg_id must be non-null and unique; ID is not used.")
    segment_ids=polygons[SEGMENT_ID_FIELD].to_numpy(dtype=np.int64)
    if len(signal)!=len(polygons) or set(object_ids)!=set(segment_ids):
        raise ValueError("Feature-table Object_ID and segmentation Seg_id do not match one-to-one.")
    return polygons,signal,object_ids,radiance,header


def degree_from_positions(n,left,right):
    return np.bincount(np.concatenate((left,right)),minlength=n).astype(np.int64)


def validate_graph_arrays(graph,expected_ids):
    required={"src_object_id","dst_object_id","neighbor_count"}
    if not required.issubset(graph.files):
        raise ValueError(f"Graph lacks arrays {sorted(required.difference(graph.files))}")
    src=graph["src_object_id"].astype(np.int64,copy=False)
    dst=graph["dst_object_id"].astype(np.int64,copy=False)
    object_ids=(graph["object_id"].astype(np.int64,copy=False)
                if "object_id" in graph.files else np.asarray(expected_ids,dtype=np.int64))
    stored_degree=graph["neighbor_count"].astype(np.int64,copy=False)
    if len(object_ids)!=len(expected_ids) or len(np.unique(object_ids))!=len(object_ids):
        raise ValueError("Graph object_id count/uniqueness is invalid.")
    if set(object_ids)!=set(expected_ids): raise ValueError("Graph Object_ID coverage differs from segmentation.")
    if len(src)!=len(dst) or np.any(src>=dst): raise ValueError("Edges are not canonical src<dst pairs.")
    if len(np.unique(np.column_stack((src,dst)),axis=0))!=len(src):
        raise ValueError("Graph contains duplicate undirected edges.")
    index=pd.Index(object_ids)
    left=index.get_indexer(src); right=index.get_indexer(dst)
    if np.any(left<0) or np.any(right<0): raise ValueError("Graph edge endpoint is absent from object_id.")
    degree=degree_from_positions(len(object_ids),left,right)
    if len(stored_degree)!=len(object_ids) or not np.array_equal(stored_degree,degree):
        raise ValueError("Stored neighbor_count disagrees with reconstructed degree.")
    return object_ids,src,dst,left,right,degree


def build_graph(polygons,graph_path,metadata_path,chunk_size=10000):
    print("Saved graph missing/invalid; rebuilding strict first-order Rook graph once.")
    working=polygons[[SEGMENT_ID_FIELD,"geometry"]].copy()
    invalid=~working.geometry.is_valid
    repaired=int(invalid.sum())
    if repaired:
        working.loc[invalid,"geometry"]=[polygonal_part(make_valid(g))
                                          for g in working.loc[invalid,"geometry"]]
    unusable=working.geometry.isna()|working.geometry.is_empty|~working.geometry.is_valid
    if unusable.any(): raise ValueError(f"{int(unusable.sum())} unusable geometries after repair.")
    geometries=working.geometry.to_numpy()
    tree=shapely.STRtree(geometries)
    left_parts=[]; right_parts=[]; candidates=points=overlaps=0
    progress=Progress("Rebuilding Rook graph",len(geometries))
    for start in range(0,len(geometries),chunk_size):
        stop=min(start+chunk_size,len(geometries))
        pairs=tree.query(geometries[start:stop],predicate="intersects")
        left=pairs[0].astype(np.int64)+start; right=pairs[1].astype(np.int64)
        keep=left<right; left=left[keep]; right=right[keep]; candidates+=len(left)
        if len(left):
            relations=shapely.relate(geometries[left],geometries[right])
            rook=np.fromiter((r[0]=="F" and r[4]=="1" for r in relations),bool,len(relations))
            points+=sum(r[0]=="F" and r[4]=="0" for r in relations)
            overlaps+=sum(r[0]!="F" for r in relations)
            left_parts.append(left[rook]); right_parts.append(right[rook])
        progress.update(stop)
    left=np.concatenate(left_parts); right=np.concatenate(right_parts)
    ids=working[SEGMENT_ID_FIELD].to_numpy(dtype=np.int64)
    src=np.minimum(ids[left],ids[right]); dst=np.maximum(ids[left],ids[right])
    order=np.lexsort((dst,src)); src,dst=src[order],dst[order]
    index=pd.Index(ids); lp=index.get_indexer(src); rp=index.get_indexer(dst)
    degree=degree_from_positions(len(ids),lp,rp)
    np.savez_compressed(graph_path,src_object_id=src,dst_object_id=dst,
                        object_id=ids,neighbor_count=degree)
    metadata={"graph_version":"v01","created_at":datetime.now().astimezone().isoformat(),
              "number_of_objects":len(ids),"number_of_unique_undirected_edges":len(src),
              "adjacency_definition":"Strict first-order Rook; non-zero shared boundary segment; point contacts excluded",
              "id_field":"Object_ID","source_polygon_id_field":SEGMENT_ID_FIELD,
              "source_segmentation_file":str(SEGMENTATION_PATH.resolve()),
              "geometry_repair":{"method":"shapely.make_valid with polygonal-part retention","repaired_count":repaired},
              "topology_diagnostics":{"candidate_intersecting_pairs":candidates,
                  "point_only_contacts_excluded":int(points),"overlapping_polygon_pairs_excluded":int(overlaps)}}
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return ids,src,dst,lp,rp,degree


def load_or_build_graph(polygons,expected_ids,graph_path,metadata_path,chunk_size):
    started=time.perf_counter(); reused=False; failure=None
    if graph_path.is_file():
        try:
            with np.load(graph_path,allow_pickle=False) as graph:
                arrays=validate_graph_arrays(graph,expected_ids)
            reused=True
        except Exception as error:
            failure=f"{type(error).__name__}: {error}"
    if not reused:
        arrays=build_graph(polygons,graph_path,metadata_path,chunk_size)
    metadata={}
    if metadata_path.is_file():
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    return (*arrays,reused,failure,metadata,time.perf_counter()-started)


def compute_context(radiance,left,right,degree):
    started=time.perf_counter(); n=len(radiance)
    neighbor_sum=np.bincount(left,weights=radiance[right],minlength=n)
    neighbor_sum+=np.bincount(right,weights=radiance[left],minlength=n)
    neighbor_sq=np.bincount(left,weights=radiance[right]**2,minlength=n)
    neighbor_sq+=np.bincount(right,weights=radiance[left]**2,minlength=n)
    difference=np.abs(radiance[left]-radiance[right])
    abs_sum=np.bincount(left,weights=difference,minlength=n)
    abs_sum+=np.bincount(right,weights=difference,minlength=n)
    valid=degree>0
    mean=np.full(n,np.nan); std=np.full(n,np.nan); mac=np.full(n,np.nan)
    mean[valid]=neighbor_sum[valid]/degree[valid]
    variance=np.full(n,np.nan)
    variance[valid]=neighbor_sq[valid]/degree[valid]-mean[valid]**2
    std[valid]=np.sqrt(np.maximum(variance[valid],0))
    mac[valid]=abs_sum[valid]/degree[valid]
    contrast=np.where(valid,radiance-mean,np.nan)
    normalized=np.where(valid,contrast/(radiance+mean+EPSILON),np.nan)
    context=pd.DataFrame({"NTL_NeighborMean":mean,"NTL_NeighborStd":std,
                          "NTL_LocalContrast":contrast,"NTL_NormalizedContrast":normalized,
                          "NTL_MeanAbsNeighborContrast":mac,"NeighborCount":degree})
    return context,time.perf_counter()-started


def graph_statistics(degree,edge_count):
    return {"objects":len(degree),"unique_undirected_edges":int(edge_count),
            "isolates":int((degree==0).sum()),"minimum_degree":int(degree.min()),
            "q25":float(np.quantile(degree,.25)),"median":float(np.median(degree)),
            "mean":float(degree.mean()),"q75":float(np.quantile(degree,.75)),
            "q95":float(np.quantile(degree,.95)),"maximum_degree":int(degree.max())}


def morans_i(radiance,left,right,degree):
    n=len(radiance); valid=degree>0; s0=float(valid.sum())
    centered=radiance-radiance.mean(); denominator=float(centered@centered)
    product=centered[left]*centered[right]
    numerator=float(np.sum(product*(1/degree[left]+1/degree[right])))
    observed=(n/s0)*numerator/denominator
    return {"N":n,"S0":s0,"observed_I":observed,
            "expected_I":-1/(n-1),"analytical_z_score":None,
            "note":"Analytical z-score omitted to avoid mixing normality/randomization assumptions; no permutation test requested."}


def select_validation_objects(radiance,degree,count=15):
    rng=np.random.default_rng(RANDOM_SEED); chosen=[]
    dq=np.quantile(degree,[.25,.75,.95]); rq=np.quantile(radiance,[.10,.90])
    groups=[np.flatnonzero(degree<=dq[0]),np.flatnonzero((degree>dq[0])&(degree<=dq[1])),
            np.flatnonzero(degree>=dq[2]),np.flatnonzero(radiance<=rq[0]),
            np.flatnonzero(radiance>=rq[1])]
    for candidates in groups:
        available=np.setdiff1d(candidates,np.asarray(chosen,dtype=int),assume_unique=False)
        if len(available):
            take=min(3,len(available)); chosen.extend(rng.choice(available,take,replace=False).tolist())
    if len(chosen)<count:
        available=np.setdiff1d(np.arange(len(radiance)),np.asarray(chosen,dtype=int))
        chosen.extend(rng.choice(available,count-len(chosen),replace=False).tolist())
    return np.asarray(chosen[:max(10,count)],dtype=np.int64)


def validate_features(object_ids,radiance,context,left,right,degree):
    nodes=np.concatenate((left,right)); neighbors=np.concatenate((right,left))
    order=np.argsort(nodes,kind="stable"); nodes,nbr=nodes[order],neighbors[order]
    counts=np.bincount(nodes,minlength=len(object_ids)); offsets=np.r_[0,np.cumsum(counts)]
    row_weights=np.concatenate((1/degree[left],1/degree[right]))
    row_nodes=np.concatenate((left,right))
    weight_sums=np.bincount(row_nodes,weights=row_weights,minlength=len(object_ids))
    valid=degree>0
    weight_error=float(np.max(np.abs(weight_sums[valid]-1))) if valid.any() else 0.0
    targets=select_validation_objects(radiance,degree)
    max_errors={"NeighborCount":0.0,"NTL_NeighborMean":0.0,
                "NTL_LocalContrast":0.0,"NTL_MeanAbsNeighborContrast":0.0}
    records=[]
    for node in targets:
        values=radiance[nbr[offsets[node]:offsets[node+1]]]
        expected={"NeighborCount":len(values),"NTL_NeighborMean":values.mean(),
                  "NTL_LocalContrast":radiance[node]-values.mean(),
                  "NTL_MeanAbsNeighborContrast":np.mean(np.abs(radiance[node]-values))}
        for name,value in expected.items():
            error=abs(float(context.at[node,name])-float(value))
            max_errors[name]=max(max_errors[name],error)
        records.append({"Object_ID":int(object_ids[node]),"Degree":int(degree[node]),
                        "Radiance":float(radiance[node]),"Explicit_Neighbor_Count":len(values)})
    identity_error=float(np.nanmax(np.abs(context.NTL_LocalContrast-
                                          (radiance-context.NTL_NeighborMean))))
    normalized_min=float(np.nanmin(context.NTL_NormalizedContrast))
    normalized_max=float(np.nanmax(context.NTL_NormalizedContrast))
    if weight_error>1e-12 or identity_error>1e-15 or max(max_errors.values())>1e-12:
        raise RuntimeError("Independent spatial-context numerical validation failed.")
    if normalized_min < -1-1e-10 or normalized_max > 1+1e-10:
        raise RuntimeError("Normalized contrast exceeds its non-negative-radiance bounds.")
    return {"random_seed":RANDOM_SEED,"sample_size":len(targets),
            "sample_records":records,"maximum_explicit_neighbor_list_errors":max_errors,
            "maximum_row_weight_sum_error":weight_error,
            "maximum_local_contrast_identity_error":identity_error,
            "normalized_contrast_minimum":normalized_min,
            "normalized_contrast_maximum":normalized_max,
            "status":"passed"}


def descriptive(frame):
    rows=[]
    for column in CONTEXT_COLUMNS:
        values=pd.to_numeric(frame[column],errors="coerce")
        valid=values[np.isfinite(values)]
        rows.append({"Feature":column,"Valid_Count":len(valid),
                     "NaN_Count":int(values.isna().sum()),"Zero_Count":int(valid.eq(0).sum()),
                     "Zero_Fraction":float(valid.eq(0).mean()) if len(valid) else np.nan,
                     "Minimum":valid.min(),"Q25":valid.quantile(.25),"Median":valid.median(),
                     "Mean":valid.mean(),"Q75":valid.quantile(.75),"Q95":valid.quantile(.95),
                     "Maximum":valid.max(),"Std":valid.std(ddof=0)})
    return pd.DataFrame(rows)


def strongest_correlations(correlation,count=8):
    rows=[]; columns=list(correlation.columns)
    for i in range(len(columns)):
        for j in range(i+1,len(columns)):
            rows.append((columns[i],columns[j],float(correlation.iloc[i,j])))
    return sorted(rows,key=lambda row:abs(row[2]),reverse=True)[:count]


def compare_existing(signal,new_context):
    present=[c for c in CONTEXT_COLUMNS if c in signal.columns]
    result={"existing_columns":present,"behavior":"append" if not present else "replace_after_validation",
            "maximum_absolute_difference":{}}
    for column in present:
        old=signal[column].to_numpy()
        new=new_context[column].to_numpy()
        if column=="NeighborCount": difference=float(np.max(np.abs(old.astype(float)-new.astype(float))))
        else: difference=float(np.nanmax(np.abs(old.astype(float)-new.astype(float))))
        result["maximum_absolute_difference"][column]=difference
    legacy_epsilon=max(np.finfo(np.float64).eps,
                       1e-12*float(np.max(np.abs(signal[PAN_FIELD].to_numpy(dtype=float)))))
    result["legacy_epsilon_from_version_01_method"]=legacy_epsilon
    result["version_02_epsilon"]=EPSILON
    result["normalized_contrast_note"]=(
        "Version 02 follows the requested fixed epsilon=1e-12. Version 01 used "
        "max(machine epsilon, 1e-12*max_abs_radiance), explaining the small normalized-contrast change."
    )
    return result


def write_merged(input_path,output_path,context_by_id,header,pilot_ids=None,chunk_size=25000):
    base_columns=[column for column in header if column not in CONTEXT_COLUMNS]
    final_columns=[*base_columns,*CONTEXT_COLUMNS]
    first=True; total=0; seen=[]
    pilot_set=set(map(int,pilot_ids)) if pilot_ids is not None else None
    for chunk in pd.read_csv(input_path,chunksize=chunk_size):
        if pilot_set is not None:
            chunk=chunk[chunk[ID_FIELD].isin(pilot_set)]
            if chunk.empty: continue
        chunk=chunk.drop(columns=[c for c in CONTEXT_COLUMNS if c in chunk.columns])
        merged=chunk.merge(context_by_id,on=ID_FIELD,how="left",sort=False,validate="one_to_one")
        if merged[CONTEXT_COLUMNS].drop(columns="NeighborCount").isna().any().any():
            # NaN is allowed only for graph isolates.
            bad=merged[merged.NeighborCount.gt(0)][CONTEXT_COLUMNS[:-1]].isna().any(axis=1)
            if bad.any(): raise RuntimeError("Context join produced NaN for a non-isolated object.")
        merged[final_columns].to_csv(output_path,index=False,mode="w" if first else "a",header=first)
        first=False; total+=len(merged); seen.extend(merged[ID_FIELD].astype(np.int64).tolist())
    if first: raise RuntimeError("Merged output received no rows.")
    if len(seen)!=len(set(seen)): raise RuntimeError("Merged output contains duplicate Object_ID.")
    return total,set(seen),final_columns


def make_figures(frame,correlation,paths):
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.labelsize":11})
    degree=frame.NeighborCount.to_numpy()
    bins=np.arange(degree.min()-.5,degree.max()+1.5)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.hist(degree,bins=bins,color="#287c8e",edgecolor="white",linewidth=.3)
    ax.set(title="First-order Rook neighbor count",xlabel="NeighborCount",ylabel="Number of objects")
    ax.grid(axis="y",color=".9"); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(paths["neighbor_histogram"],dpi=250,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    image=ax.imshow(correlation.to_numpy(),cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_xticks(range(len(correlation)),correlation.columns,rotation=45,ha="right")
    ax.set_yticks(range(len(correlation)),correlation.index)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            ax.text(j,i,f"{correlation.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8,
                    color="white" if abs(correlation.iloc[i,j])>.55 else "black")
    fig.colorbar(image,ax=ax,label="Spearman rho",shrink=.82)
    ax.set_title("NTL spatial-context Spearman correlations")
    fig.savefig(paths["correlation_heatmap"],dpi=250,bbox_inches="tight"); plt.close(fig)


def markdown_table(frame,digits=6):
    headers=list(frame.columns); lines=["| "+" | ".join(headers)+" |",
                                       "| "+" | ".join("---" for _ in headers)+" |"]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if pd.isna(value): values.append("NA")
            elif isinstance(value,(float,np.floating)): values.append(f"{value:.{digits}g}")
            elif isinstance(value,(int,np.integer)): values.append(f"{int(value):,}")
            else: values.append(str(value).replace("|","\\|"))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def write_report(args,paths,input_info,graph_info,stats,moran,correlation,validation,
                 existing_comparison,timings,full_object_count,output_rows):
    mode="PILOT" if args.pilot else "FULL DATASET"
    strong=strongest_correlations(correlation)
    strong_text="\n".join(f"- `{a}` vs `{b}`: rho = `{rho:.6f}`" for a,b,rho in strong)
    interpretation=("positive" if moran["observed_I"]>.1 else
                    "negative" if moran["observed_I"]<-.1 else "weak")
    graph=pd.DataFrame([graph_info])
    comparison_lines=("No pre-existing context columns were present."
                      if not existing_comparison["existing_columns"] else
                      "Pre-existing same-name columns were independently recomputed and replaced in the new output. "
                      "Maximum absolute differences: "+", ".join(
                          f"{k}={v:.3g}" for k,v in existing_comparison["maximum_absolute_difference"].items())+
                      f". The normalized-contrast difference follows from the required fixed epsilon=1e-12 "
                      f"versus the version-01 scale-dependent epsilon={existing_comparison['legacy_epsilon_from_version_01_method']:.3g}.")
    output_text="\n".join(f"- {key}: `{path.resolve()}`" for key,path in paths.items())
    conclusion=("The pilot successfully validates formulas, graph reuse, Moran diagnostic, and output joins; "
                "the full-population output remains to be run manually."
                if args.pilot else
                "All six specified context features were successfully generated for the full object population. "
                "No implementation issue remains before classification, subject to the documented pre-existing-column replacement.")
    text=f"""# Spatial Context Extraction 02 — {mode}

## A. Inputs

- Segmentation: `{args.segmentation.resolve()}`
- Canonical feature table: `{args.feature_table.resolve()}`
- Saved adjacency graph: `{args.graph.resolve()}`
- Objects: `{full_object_count:,}`; output rows in this run: `{output_rows:,}`
- Identifier: segmentation `Seg_id` preserved as feature-table `Object_ID`; source `ID` was never used.
- Base NTL signal: `{PAN_FIELD}` only. No optical value, class/reference/training/test label, or prediction was read.

The canonical table was selected because `Feature_Redundancy_Analysis_Final_Report.md` explicitly identifies `objects_features_final_full.csv` as the final canonical input. Input checks confirmed equal row and unique-ID counts and an exact one-to-one match with all segmentation objects.

## B. Adjacency definition

Strict first-order Rook adjacency means two polygons have disjoint interiors and share a boundary segment of non-zero length; point-only corner contact is excluded. The undirected edge list is binary, and neighbors receive equal row-standardized weight `w_ij=a_ij/d_i`. The saved graph was **{'reused' if graph_info['saved_graph_reused'] else 'rebuilt'}**.

{markdown_table(graph)}

## C. Mathematical definitions

For `r_i={PAN_FIELD}`, degree `d_i=Σ_j a_ij`, and `w_ij=a_ij/d_i`:

- `NTL_NeighborMean = M_i = Σ_j w_ij r_j`.
- `NTL_NeighborStd = S_i = sqrt[Σ_j w_ij(r_j-M_i)^2]` (population standard deviation).
- `NTL_LocalContrast = C_i = r_i-M_i`.
- `NTL_NormalizedContrast = NC_i = (r_i-M_i)/(r_i+M_i+1e-12)`.
- `NTL_MeanAbsNeighborContrast = MAC_i = Σ_j w_ij |r_i-r_j|`.
- `NeighborCount = d_i` (integer).

For isolates, degree is zero and the five radiometric context values are NaN, never zero.

## D. Output statistics

{markdown_table(stats)}

## E. Spatial autocorrelation

Global row-standardized Moran's I for `{PAN_FIELD}` is **{moran['observed_I']:.8f}**; the random-spatial expectation is **{moran['expected_I']:.8f}** (`N={moran['N']:,}`, `S0={moran['S0']:.0f}`). This indicates **{interpretation} spatial dependence** in object-level PAN mean radiance. It is diagnostic only and is not an object-level feature. No significance claim is based solely on the large sample size; an analytical z-score and permutation test were intentionally omitted.

## F. Redundancy diagnostics

Strongest absolute Spearman associations in this run:

{strong_text}

No feature was removed. {comparison_lines}

## G. Validation

- Saved edge endpoints, explicit graph Object_ID coverage, reconstructed degree, and stored NeighborCount agree exactly.
- Canonical edges satisfy `src_object_id < dst_object_id` and contain no duplicate undirected pair.
- All non-isolate row-standardized weight sums equal one within maximum error `{validation['maximum_row_weight_sum_error']:.3g}`.
- `{validation['sample_size']}` objects were selected with fixed seed `{validation['random_seed']}` across low/medium/high degree and low/high radiance strata. Explicit neighbor-list calculations passed; maximum errors: `{validation['maximum_explicit_neighbor_list_errors']}`.
- Global identity `LocalContrast = PAN mean - NeighborMean` passed with maximum error `{validation['maximum_local_contrast_identity_error']:.3g}`.
- With non-negative radiance, normalized contrast remained within `[{validation['normalized_contrast_minimum']:.9g}, {validation['normalized_contrast_maximum']:.9g}]`, consistent with the expected [-1, 1] range.
- Context table and merged table contain `{output_rows:,}` unique Object_ID values in input-table order for this run; no ID was lost.

## H. Runtime

| Stage | Seconds |
|---|---:|
| Input loading and one-to-one checks | {timings['input_loading']:.3f} |
| Graph loading/building | {timings['graph']:.3f} |
| Feature computation | {timings['feature_computation']:.3f} |
| Total | {timings['total']:.3f} |

## I. Deviations

- The canonical input already contained all six same-name context fields from the earlier workflow. Duplicate column names would be invalid, so version 02 independently recomputed them, compared them, and replaced them only in the new merged output; the original table was untouched.
- {'This is a representative pilot output, not the requested final full-population table.' if args.pilot else 'No population-scope deviation; every object was written.'}
- Analytical Moran inference was omitted because it was optional and assumption-sensitive; observed and expected I are reported.

## J. Final research summary

{conclusion} The six-feature family remains compact and interpretable: surrounding brightness, neighbor variability, signed and normalized local contrast, absolute discontinuity, and topology.

## Output files

{output_text}
"""
    paths["report"].write_text(text,encoding="utf-8")


def run(args):
    total_started=time.perf_counter(); timings={}; args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=output_paths(args.output_dir,args.pilot)
    existing=[p for p in paths.values() if p.exists()]
    if existing and not args.overwrite: raise FileExistsError("Outputs exist; use --overwrite: "+str(existing[0]))
    print("Resolved inputs:")
    print(f"  segmentation = {args.segmentation.resolve()}")
    print(f"  feature table= {args.feature_table.resolve()}")
    print(f"  Rook graph   = {args.graph.resolve()}")

    stage=time.perf_counter()
    polygons,signal,input_ids,input_radiance,header=load_inputs(args.segmentation,args.feature_table)
    timings["input_loading"]=time.perf_counter()-stage
    (graph_ids,src,dst,left,right,degree,reused,graph_failure,metadata,
     timings["graph"])=load_or_build_graph(
         polygons,input_ids,args.graph,args.graph_metadata,args.graph_chunk_size)
    graph_signal=signal.set_index(ID_FIELD).loc[graph_ids]
    radiance=graph_signal[PAN_FIELD].to_numpy(dtype=np.float64)

    context, timings["feature_computation"]=compute_context(radiance,left,right,degree)
    context.insert(0,ID_FIELD,graph_ids)
    new_values=context[CONTEXT_COLUMNS].copy()
    comparison=compare_existing(graph_signal.reset_index(),new_values)
    graph_stats=graph_statistics(degree,len(src))
    graph_stats.update({"saved_graph_reused":reused,"graph_validation_failure_before_rebuild":graph_failure})
    moran=morans_i(radiance,left,right,degree)
    validation=validate_features(graph_ids,radiance,new_values,left,right,degree)

    context_input_order=pd.DataFrame({ID_FIELD:input_ids}).merge(context,on=ID_FIELD,how="left",validate="one_to_one")
    if args.pilot:
        positions=np.unique(np.linspace(0,len(input_ids)-1,args.pilot_size).round().astype(int))
        output_context=context_input_order.iloc[positions].copy()
        pilot_ids=output_context[ID_FIELD].to_numpy(dtype=np.int64)
    else:
        output_context=context_input_order; pilot_ids=None
    output_context.to_csv(paths["context"],index=False)
    context_by_id=context[[ID_FIELD,*CONTEXT_COLUMNS]]
    merged_rows,merged_ids,final_columns=write_merged(
        args.feature_table,paths["merged"],context_by_id,header,pilot_ids,args.chunk_size)
    if merged_rows!=len(output_context) or merged_ids!=set(output_context[ID_FIELD]):
        raise RuntimeError("Merged output row/ID validation failed.")
    if len(output_context)!=output_context[ID_FIELD].nunique():
        raise RuntimeError("Context output contains duplicate Object_ID.")
    if list(output_context.columns)!=[ID_FIELD,*CONTEXT_COLUMNS]:
        raise RuntimeError("Context output schema is not exactly the required seven columns.")

    diagnostic_frame=pd.DataFrame({PAN_FIELD:signal.set_index(ID_FIELD).loc[
        output_context[ID_FIELD],PAN_FIELD].to_numpy()})
    for column in CONTEXT_COLUMNS: diagnostic_frame[column]=output_context[column].to_numpy()
    stats=descriptive(output_context)
    correlation=diagnostic_frame.corr(method="spearman")
    correlation.to_csv(paths["spearman"])
    make_figures(output_context,correlation,paths)
    timings["total"]=time.perf_counter()-total_started
    diagnostics={"mode":"PILOT" if args.pilot else "FULL_DATASET","inputs":{
        "segmentation":str(args.segmentation.resolve()),"feature_table":str(args.feature_table.resolve()),
        "graph":str(args.graph.resolve()),"Object_ID_field":ID_FIELD,"base_NTL_variable":PAN_FIELD},
        "input_object_count":len(input_ids),"output_row_count":len(output_context),
        "input_column_count":len(header),"merged_output_column_count":len(final_columns),
        "graph":graph_stats,"graph_metadata":metadata,"epsilon":EPSILON,"morans_i":moran,
        "feature_statistics":stats.to_dict(orient="records"),"existing_context_comparison":comparison,
        "validation":validation,"timings_seconds":timings}
    paths["diagnostics"].write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    write_report(args,paths,{},graph_stats,stats,moran,correlation,validation,comparison,
                 timings,len(input_ids),len(output_context))
    print(f"Completed {len(output_context):,} output objects; graph reused={reused}; "
          f"Moran's I={moran['observed_I']:.6f}")
    print(f"Report: {paths['report'].resolve()}")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot",action="store_true")
    mode.add_argument("--full-run",action="store_true")
    parser.add_argument("--segmentation",type=Path,default=SEGMENTATION_PATH)
    parser.add_argument("--feature-table",type=Path,default=FEATURE_TABLE_PATH)
    parser.add_argument("--graph",type=Path,default=GRAPH_PATH)
    parser.add_argument("--graph-metadata",type=Path,default=GRAPH_METADATA_PATH)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--pilot-size",type=int,default=5000)
    parser.add_argument("--chunk-size",type=int,default=25000)
    parser.add_argument("--graph-chunk-size",type=int,default=10000)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if args.output_dir is None:
        args.output_dir=Path(str(OUTPUT_DIRECTORY)+( "_pilot" if args.pilot else ""))
    if args.pilot_size<10 or args.chunk_size<1 or args.graph_chunk_size<1:
        parser.error("pilot-size >=10 and chunk sizes must be positive.")
    run(args)


if __name__=="__main__": main()
