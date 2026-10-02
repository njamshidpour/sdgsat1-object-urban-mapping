# Reproducibility review report

## 1. Files included

Twenty-one copied research scripts, `README.md`, `CODE_MANUSCRIPT_MAP.md`, `requirements.txt`, this report and `.gitignore`. The copied scripts are listed with their inputs and outputs in the README. Their scientific feature definitions, thresholds, sampling, tuning, metric and bootstrap calculations were preserved. Changes in this copy only remove private machine paths, allow the same denoiser and co-registration function to be called for each band, make HDR inputs configurable, and use a portable font fallback.

## 2. Files excluded and why

| Excluded source | Reason |
| --- | --- |
| `Object_based_fusion/Segmentation_01.py`, `vectorized_Segments.py`, `Sentinel-2-Geo-SAM.py`, `SAM.ipynb`, `segment-anything-2/`, SAM checkpoint | SLIC/SAM workflow does not generate the reported eCognition Multiresolution Segmentation objects. |
| `Object_based_fusion/Feature_extraction_02.py`, `_03.py`, `Feature_extraction_06.py` | Superseded feature extraction or Scenario-3 structural texture not used by Table 7. |
| `Object_based_fusion/Feature_Redundancy_Analysis_01.py`, texture audit/consistency scripts | Diagnostic or earlier redundancy versions; final report uses `Feature_Redundancy_Analysis_Final.py` and reduced ablation's own training-only reduction. |
| `Object_based_fusion/Strategy1_*`, `Strategy2_*`, `Main_Pipeline_05.py`, `Data_fusion_codes/`, `Phase_one_data_fusion_ML_2.py` | Incomplete, pilot, CNN, or alternate classifier experiments outside the reported XGBoost result chain. |
| `Object_FusedSpectral_Geometry_Context_XGBoost_01.py`, `Pixel_Based_13x13_Spectral_Fusion_XGBoost_01.py`, other exploratory comparison scripts | Extra feature configurations and patch experiments not reported in the main Tables 4, 5, or 7. |
| `Representative_Comparison_Patches_01.py`, `Representative_Comparison_TestBlocks_02.py`, `Selected_TestBlocks_Figure_03.py`, map/plot helpers | Figure assembly/visualization; not required to calculate the core methods or accuracy claims. |
| Other salt-and-pepper/coregistration variants, moonlight/albedo studies and one-off statistics | Alternate or unverified versions, or diagnostics that do not feed the proven final result chain. |
| Original imagery, shapefiles, raster products, CSVs, notebooks, reports, figures, archives, checkpoints, virtual environment and caches | External data, generated intermediates, large assets, or potentially private metadata; all are deliberately omitted. |

## 3. Results without confidently identified code

- The exact 336,109 eCognition Multiresolution Segmentation polygons cannot be regenerated from a found executable script or stored parameter record.
- The three-month Dynamic World probability compositing and all corrected Sentinel-2 10 m band derivations are upstream input-production steps without confidently identified final code.
- The visual interpretation and final selection of 996 independent manual test objects is external to the code. The candidate-block script identifies potential blocks only.
- Figure 5's scripts were identified (`Representative_Comparison_TestBlocks_02.py` and `Selected_TestBlocks_Figure_03.py`), but its example imagery, auxiliary inputs and full prediction maps remain external; the figure-only scripts were excluded from the minimal analysis folder.
- A complete provenance record for all five corrected SDGSAT-1 bands is absent; the retained preprocessing functions are the closest matching implementation, but the archived raster sequence cannot be certified band by band.

## 4. Manuscript and implementation discrepancies

- The manuscript's lunar-reflectance subtraction is not performed by `sdgsat_correction.py`; metadata parsing returns fixed calibration constants and the moon variables are read but not applied.
- `NTL_indices_calculation.py` assigns NDVI, NDWI and NDBI variables to filename stems of other indices. This affects VANUI, TVANUI and ENUI. The source assignment is preserved here so the historical model inputs can be traced; it needs scientific resolution before revised outputs are described as manuscript-formula indices.
- The final conventional feature table has 96 columns including ID, hence 95 predictors. The manuscript states 94 in one paragraph and lists five geometry and five context descriptors there, whereas the full table uses six of each. The reduced ablation uses five candidates in each group.
- Methods prose mentions Random Forest and SVM model comparison and Random Forest index importance, while the traced main experimental tables are XGBoost outputs. These claims should be removed, qualified, or backed by separate results.
- Methods prose says phase correlation plus feature matching/affine transform for registration; the retained registration script uses PCA, reprojection, phase correlation and an affine warp. Explicit feature matching is not visible in that script.
- Table 7's caption describes a spectral-source comparison although its values are the reduced NTL/object-feature ablation; the figure discussion also refers to Table 6 where the main sensor comparison is Table 5.
- The manuscript says the independent samples are from reserved blocks. The experiment scripts read a manually labeled point shapefile and perform test-object and Rook-neighbor exclusion; exact correspondence of the point geometries with the selected candidate blocks needs the source test data for confirmation.
- Figure 5 was rendered from separately refitted three-class pixel/object models (`Representative_Comparison_TestBlocks_02.py`), whereas Tables 4 and 5 report binary Urban/NonUrban experiments. The figure therefore should not be described as a direct map of those tabulated binary models.

## 5. External data needed

Original SDGSAT-1 GLI DN bands and calibration metadata; Sentinel-2 B2/B3/B4/B8 and the further corrected 10 m bands/NDVI/NDBI/NDWI; Dynamic World urban/vegetation/bare/scrub probability rasters for the study period; LST for TVANUI; Tehran Province boundary; original eCognition segmentation polygons and segmentation settings; manually interpreted test points with class labels; and intermediate corrected radiance/HDR/feature tables if the preprocessing gaps are not independently resolved. All rasters must align according to each script's validation rules.

## 6. Do not upload

Raw or licensed satellite scenes, scene XML, private/manual test point labels without permission, full segmentation and intermediate raster/feature/prediction data, large model/checkpoint files, original result reports containing local absolute paths, virtual environments, cached outputs, credentials, API keys or private links. The supplied `.gitignore` excludes the expected data/output directories but should not replace a manual review of any eventual Git commit.

## 7. Ready to share with reviewers?

**Ready as an honest code-only companion for methodological review, but not ready to claim complete end-to-end reproducibility.** It contains the code that generated the main XGBoost comparison and ablation results, with dependencies and paths made portable. Before claiming full reproduction, resolve the segmentation and manual-reference availability, verify/correct the NTL index input mapping and lunar-correction prose, and reconcile feature counts and unused-classifier statements in the manuscript. No dataset or sensitive project file is included in this folder.

## Verification performed

All 21 retained Python files parse successfully; all 14 copies without intentional portability edits are byte-identical to their source. Eleven argument-parsing scripts completed `--help` successfully. The resulting folder contains only 21 `.py` files plus five small documentation/configuration files (about 650 KB total), with no imagery, shapefiles, CSV outputs, caches or checkpoints. A static scan found no machine-specific source path, credential string or private URL in the retained files. Full scientific runs were not attempted because the data are intentionally absent; the inspection environment also lacks `scikit-learn`, OpenCV, XGBoost and Skyfield, all of which are listed in `requirements.txt`.
