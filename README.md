# Paper reproducibility code

This folder contains the research scripts that most directly support the methods and main experiments in the **2 October 2026 manuscript `Manuscript_1405_7_10.docx`**, *Object-Based Urban Built-Up Area Mapping Using High-Resolution SDGSAT-1 Nighttime Light Imagery and Optical-Guided Segmentation: A Case Study of Tehran*. The older file named `Manuscript_final.docx` is an earlier, different draft. No imagery, labels, intermediate feature tables, predictions, or model outputs are distributed here.

The scripts are procedural research code, copied from the executed project. Their scientific calculations have not been redesigned. Input/output path defaults point to folders **relative to this README**; scripts with `--input`, `--output`, or other path flags can be configured on the command line. Create the named local data directories or edit the path constants near the top of the script. Run scripts from this folder with Python 3.11 or later. Install `requirements.txt` in a geospatial Python environment. All full runs require substantial disk space and memory.

## Workflow and required inputs

| Order | Script | Required input | Main output | Manuscript connection |
| --- | --- | --- | --- | --- |
| 1a | `salt_pepper_removal_sdgsat_3.py` | One raw SDGSAT-1 DN band, passed as `--input` | A filtered DN GeoTIFF passed as `--output`; repeat separately for R, G, B, PL, PH | SDGSAT-1 noise mitigation |
| 1b | `sdgsat_correction.py` | Filtered DN rasters in `output_corrected_bands/`; local calibration metadata and `NTL_data/moon_data.csv` as referenced by the script | Calibrated band radiance GeoTIFFs in `output_corrected_bands/` | Radiometric calibration; see limitations below |
| 1c | `sdgsat_coregisteration_sentinel.py` | Sentinel-2 B2/B3/B4/B8 in `optic_data/`; PL and PH radiance in `output_corrected_bands/` | Aligned and co-registered PL/PH radiance GeoTIFFs | PCA/phase-correlation registration; run with `--band PL` and `--band PH` |
| 1d | `HDR_ntl.py` | Co-registered PH and PL radiance; paths may be overridden with `--ph`, `--pl`, `--output` | `output_corrected_bands/Pan_HDR_Radiance.tif` | Equal-weight HDR PAN synthesis |
| 1e | `rgb_sdgsat_pansharpening.py` | 40 m RGB radiance and 10 m HDR PAN in `output_corrected_bands/` | 10 m pan-sharpened RGB radiance GeoTIFFs | HSV/histogram-matched pan-sharpening |
| 1f | `NTL_indices_calculation.py` | PAN/RGB radiance, LST and Sentinel-2 indices in `optic_data/` | `NTL_indices/*.tif` | 9 NTL composite indices, including NBDI and NLCSI; see index input discrepancy below |
| 2a | `DW_GT_generation.py --full-run` | Four 2023 Dynamic World probability rasters in `GTs/`; Tehran Province boundary in `Shapefile/` | Three-class confidence-filtered reference raster, threshold diagnostics and report under `GTs/` | Figure 3, Figure 4, confidence-filtered labels |
| 2b | `Tehran_ThreeClass_Test_Block_Candidates_05km.py` | Three-class reference raster and province boundary | 5 km candidate block GeoPackage/CSV and loss audit | Spatial independent-test block design; human selection and manual interpretation remain external |
| 3a | `Object_based_fusion/Feature_extraction_04.py --full-dataset --output-csv Object_based_fusion/Segmentation_output/objects_features_v04_full.csv` | Externally produced eCognition `eCog_segments.shp`; corrected NTL and Sentinel-2 rasters; NTL indices | v04 object table with 39 NTL, 39 optical, 6 geometry and 5 conventional GLCM columns | Object radiometry, geometry and Table 2 texture |
| 3b | `Object_based_fusion/Spatial_Context_Extraction_01.py` | eCognition segments and v04 full table | `objects_features_v04_spatial_context.csv` | Six strict Rook neighbor descriptors in Table 3 |
| 3c | `Object_based_fusion/Persist_Rook_Adjacency_Graph_01.py` | Same segments, v04 table and spatial-context diagnostics | `rook_adjacency_graph_v01.npz` and metadata | Reusable Rook graph for train/test exclusion and context |
| 3d | `Object_based_fusion/Feature_extraction_05.py --full-dataset` | v04 full table, corrected optical rasters, segmentation and pre-registration PAN HDR | v05 corrected optical table; run again with `--texture-only --full-dataset` for `objects_features_v05_texture_only_full.csv` | Corrected spectral table and BG-log GLCM features used by ablation |
| 3e | `Build_objects_features_final_full.py` | v04 spatial-context and v05 corrected optical tables | `objects_features_final_full.csv` | Final conventional-GLCM table used for redundancy diagnostics |
| 3f | `Spatial_Context_Extraction_02.py --full-run` | Final feature table, segments and saved Rook graph | `Spatial_Context_Extraction_02/objects_features_with_spatial_context_02.csv` | Validated context table used by classifiers |
| 4a | `Object_Labeling_From_DW_01.py --full-run` | eCognition segments, three-class DW raster, manually labeled `GTs/Test_Points_1000.shp`, Rook graph | Master object labels and leakage-excluded training labels | Pixel-to-object purity/coverage filter; independent test exclusion |
| 4b | `Object_based_fusion/Feature_Redundancy_Analysis_Final.py` | `objects_features_final_full.csv` | Spearman within/between-family diagnostics and figures | Figure 6, Figure 7, Table 6 |
| 5a | `Object_Based_Ablation_XGBoost_01.py --full-run` | Context and texture tables, training labels, manual test points, segments and graph | Feature manifest and three-class XGBoost baseline artifacts | Upstream manifest required by binary experiments; three-class analysis |
| 5b | `Binary_Stratified_Object_Ablation_XGBoost_01.py --full-run` | Previous feature manifest/predictions and the same tables/labels/segments | Stratified binary train, primary test and full test CSVs; ablation outputs | Common binary cohort, independent-test metrics, paired bootstrap |
| 5c | `Pixel_Based_Spectral_Sensor_Comparison_XGBoost_01.py --full-run` | Aligned NTL/Sentinel-2 rasters, DW reference, manual points, segments, graph and binary memberships | Pixel sensor performance, full-test predictions and paired bootstrap CSVs | Table 4, pixel-based independent test |
| 5d | `Spectral_Sensor_Comparison_XGBoost_01.py --full-run` | Corrected context table, binary memberships and feature manifest | Object sensor performance and paired bootstrap CSVs | Table 5, object-based optical/NTL comparison |
| 5e | `Reduced_NTL_Object_Ablation_XGBoost_01.py --full-run` | Corrected context table, BG-log texture table, binary memberships, graph and feature manifest | Training-only reduction, Table 7 performance, Figure 8 values and paired bootstrap CSV | Reduced NTL/geometry/texture/context ablation |

Launch scripts from this folder using their listed relative paths; Python resolves sibling imports from each script's directory. The saved Rook graph is needed before object labeling and classification. Some scripts default to pilot mode and require the stated full-run flag. Outputs are intentionally excluded from version control by `.gitignore`.

### Minimal input layout

```text
paper_reproducibility/
  GTs/
    Tehran_Built_Probability_2023_Int16_compressed.tif
    Tehran_Veg_Probability_2023_Int16_compressed.tif
    Tehran_Bare_Probability_2023_Int16_compressed.tif
    Tehran_Scrub_Probability_2023_Int16_compressed.tif
    Test_Points_1000.{shp,shx,dbf,prj,...}
  Shapefile/Tehran_True_Province.{shp,shx,dbf,prj,...}
  optic_data/                    # Sentinel-2 inputs for preprocessing and NTL indices
  optic_data_codex/              # corrected 10 m Sentinel-2 bands/indices for extraction
  output_corrected_bands/        # corrected SDGSAT-1 radiance rasters
  NTL_indices/                   # computed NTL index rasters
  Object_based_fusion/Segmentation_output/
    eCog_segments.{shp,shx,dbf,prj,...}
```

For exact filenames of all 10 Sentinel-2 bands, three optical indices, and PAN/RGB/texture rasters, use the constants in `Feature_extraction_04.py`, `Feature_extraction_05.py`, and `Pixel_Based_Spectral_Sensor_Comparison_XGBoost_01.py`. The v05 BG-log texture step also expects `Texture_Method_Audit_02_outputs/PAN_HDR_PreCoreg_Radiance.tif`; generate the same equal-weight HDR raster from the pre-registration PH/PL radiance with `HDR_ntl.py --ph ... --pl ... --output ...` before running it. The input rasters must have their original CRS, grid and NoData conventions. Model scripts validate expected schemas and many alignment properties.

## Data access and redistribution

Obtain SDGSAT-1 GLI scenes and calibration metadata from the licensed data provider, Sentinel-2 imagery/derived products and Dynamic World probability rasters from their providers, and the Tehran boundary from its source. Redistribution rights vary by source and must be checked before sharing any file. The manually interpreted test point labels, eCognition segmentation, corrected raster layers, feature tables, and prediction outputs are project-derived artifacts and are **not included**; obtain permission before redistributing them. Do not upload raw imagery, calibration XML/scene metadata, manually labeled point files, large rasters, shapefile bundles, local reports with absolute paths, or any private credentials/URLs to GitHub.

## Reproducibility limits that require attention

1. **Source raster preparation and segmentation:** the repository contains no identified code for the three-month Dynamic World probability compositing or for producing all corrected 10 m Sentinel-2 bands. The 336,109 objects were created in external eCognition Multiresolution Segmentation from B2/B3/B4/B8; there is no exported setting or script proven to regenerate this exact polygon layer. The older SLIC script is unrelated and is intentionally omitted. Exact end-to-end regeneration requires the source rasters, the original `eCog_segments` data, and the segmentation parameters.
2. **Manual test reference:** block-candidate code exists, but the final selection and visual interpretation of the 996 independent test objects were manual. Their labels cannot be recreated from code alone.
3. **NTL indices:** the executed `NTL_indices_calculation.py` associates the `NDVI`, `NDWI`, and `NDBI` variable names with the wrong filename stems. This copy preserves the executed assignment for result traceability. Check the source rasters and regenerate affected indices before claiming that their published formulas match the model inputs.
4. **Lunar correction and calibration:** `sdgsat_correction.py` uses hard-coded gain/bias values and does not actually subtract an albedo-based lunar contribution, even though it reads lunar information. The prose description of metadata-parsed gain/bias and moonlight subtraction is not supported by this executed function.
5. **Registration and denoising provenance:** the available scripts show the filters and PCA/phase registration method, but not a complete run record for every RGB/PL/PH raster. Pixel-exact reconstruction of all corrected inputs cannot be certified from the available scripts alone.
6. **Feature counts:** the full conventional table checks for 96 columns including `Object_ID` (95 features: 39 NTL, 39 optical, 6 geometry, 5 GLCM, 6 context). The manuscript says 94 features and lists five geometry/context variables in one passage, while other passages list six. The reduced ablation deliberately uses five geometry and five context candidates, then retains four each after reduction.
7. **Classifier description:** one methods paragraph says Random Forest, XGBoost and SVM were compared; another says Random Forest importance was used. The reported Table 4, Table 5 and Table 7 results trace to XGBoost only. Those other classifier claims need manuscript correction or additional evidence.

See `CODE_MANUSCRIPT_MAP.md` for the claim-level trace and `REVIEW_REPORT.md` for inclusion/exclusion decisions and sharing status.
