# Code to manuscript map

Source manuscript: `Manuscript_1405_7_10.docx` (modified 2 October 2026). Section headings below follow the manuscript wording because several headings are unnumbered.

| Manuscript section | Main claim or output | Script(s) responsible | Main generated output |
| --- | --- | --- | --- |
| SDGSAT-1 NTL Data preprocessing: raw noise mitigation | Percentile threshold, connected components and local median replacement | `salt_pepper_removal_sdgsat_3.py` | Filtered DN GeoTIFFs |
| SDGSAT-1 NTL Data preprocessing: radiometric calibration and lunar correction | DN-to-radiance conversion | `sdgsat_correction.py` | Calibrated radiance rasters; no lunar subtraction is implemented |
| SDGSAT-1 NTL Data preprocessing: spatial co-registration | Sentinel-2 PCA reference and phase-correlation registration | `sdgsat_coregisteration_sentinel.py` | PL/PH co-registered radiance rasters |
| SDGSAT-1 NTL Data preprocessing: RGB pan-sharpening | Equal-weight PAN HDR and HSV pan-sharpened RGB | `HDR_ntl.py`; `rgb_sdgsat_pansharpening.py` | `Pan_HDR_Radiance.tif`, 10 m R/G/B radiance rasters |
| NTL Indices; Table 1 | NTL composite indices including NBDI and NLCSI | `NTL_indices_calculation.py` | `NTL_indices/*.tif`; NDVI/NDWI/NDBI input names are misassigned in the script |
| Training and Test data sets; Figure 3 and Figure 4 | Three-class Dynamic World probability merge, 0.70 sum filter, confidence-weighted entropy cutoff | `DW_GT_generation.py` | Three-class reference raster, cutoff diagnostic CSV/figures |
| Spatially Independent Test Samples; Figure 2 | 5 km test-block candidate selection subject to class retention constraints | `Tehran_ThreeClass_Test_Block_Candidates_05km.py` | Candidate block tables and geometries; manual labels external |
| Object-based data fusion | Sentinel-2 B2/B3/B4/B8 eCognition Multiresolution Segmentation, 336,109 objects | **No confidently identified repository script** | External `eCog_segments.shp` input |
| Object-level reference labeling | Majority, purity ≥0.90, coverage ≥0.50, at least five labeled pixels; remove test objects/neighbors | `Object_Labeling_From_DW_01.py`; `Persist_Rook_Adjacency_Graph_01.py` | Master and training object-label CSVs |
| Object-Level Features; Table 2 | 39 NTL, 39 optical, 6 geometry and five conventional PAN GLCM descriptors | `Object_based_fusion/Feature_extraction_04.py`; `Object_based_fusion/Feature_extraction_05.py`; `Build_objects_features_final_full.py` | `objects_features_final_full.csv`; BG-log texture in `objects_features_v05_texture_only_full.csv` |
| Spatial-Context features; Table 3 | Six strict Rook adjacency variables based on PAN mean radiance | `Object_based_fusion/Spatial_Context_Extraction_01.py`; `Object_based_fusion/Persist_Rook_Adjacency_Graph_01.py`; `Spatial_Context_Extraction_02.py` | Rook graph; validated context feature table |
| Feature redundancy analysis; Figure 6, Figure 7, Table 6 | Within- and between-family Spearman correlations, active-object diagnostic | `Object_based_fusion/Feature_Redundancy_Analysis_Final.py` | Correlation tables and heat maps |
| Pixel-level Data Fusion; Table 4 | Pixel NTL, Sentinel-2, and fused XGBoost independent-test comparison | `Pixel_Based_Spectral_Sensor_Comparison_XGBoost_01.py` | `Pixel_Spectral_Sensor_Primary_Performance.csv`, paired bootstrap CSV |
| Object-based data fusion; Table 5 | Object NTL, Sentinel-2, and fused XGBoost independent-test comparison | `Object_Based_Ablation_XGBoost_01.py`; `Binary_Stratified_Object_Ablation_XGBoost_01.py`; `Spectral_Sensor_Comparison_XGBoost_01.py` | `Spectral_Sensor_Primary_Performance.csv`, paired bootstrap CSV |
| Ablation study; Table 7, Figure 8 | Training-only Spearman cluster reduction; independently tuned NTL + geometry/texture/context XGBoost models | `Reduced_NTL_Object_Ablation_XGBoost_01.py` with manifests/memberships from the two upstream ablation scripts | `Reduced_NTL_Object_Ablation_Performance.csv`, reduction manifest, paired bootstrap CSV |
| Performance Evaluation | Independent manual-object metrics, primary and full test evaluations | Pixel, binary object, spectral object, and reduced ablation XGBoost scripts above | Confusion/performance/prediction CSVs |
| Results visual comparison; Figure 5 | Five illustrative reserved test blocks | `Representative_Comparison_TestBlocks_02.py` and `Selected_TestBlocks_Figure_03.py` in the source project, deliberately excluded from this minimal analysis folder | Separate three-class refit rasters and final comparison PDF; external imagery required |
