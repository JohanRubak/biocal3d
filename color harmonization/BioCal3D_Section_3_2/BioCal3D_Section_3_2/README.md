# BioCal3D manuscript characterization

## Run from Anaconda Prompt

Activate your environment, navigate to this extracted code folder and install dependencies:

```bat
conda activate biocal3d
python -m pip install -r requirements.txt
```

Run on one line:

```bat
python run_sections_3_1_3_2.py --raw-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs_reoriented" --corrected-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs_reoriented_harmonized"
```

Results default to the results folder beside the code. --output selects another location. The raw exploration reuses valid per-scan caches on subsequent runs. New analysis summaries and figures are regenerated.

The full-model example uses these defaults:

- Raw full scans in C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected
- Harmonized full scans in C:\Users\au662213\repos\biocal3d\data\Color-harmonized Data

Override them using --raw-model-root and --corrected-model-root when needed. The example is BCG3T1-1, partial cleaning, across all four scanners.

## Recommended manuscript figures

| Figure | Location under results | Placement |
|---|---|---|
| Raw CIELAB distributions, five rows by three channels | section_3_2_lab/manuscript_figures/01_CIELAB_before | Section 3.1 before geometry |
| Harmonized CIELAB distributions in the same layout and scales | section_3_2_lab/manuscript_figures/01_CIELAB_after | Section 3.2 |
| Cleaning ΔE00 before/after median overlay | section_3_2_lab/manuscript_figures/02_cleaning_before_after_overlay | Section 3.2 |
| SWD ratios as paired horizontal bars | section_3_2_rgb/manuscript_figures/03_SWD_ratios_fresh_mature | Section 3.2 |
| BCG3T1-1 full models and ROIs before/after | specimen_comparison/04_BCG3T1_1_full_model_and_ROI | Section 3.2 or appendix |

Every figure is saved as PNG and PDF. The RGB analysis now generates one figure, the SWD horizontal-bar comparison. It has fresh baseline and clean in the top row and mature baseline, partial and clean in the lower row. All five panels share the same ratio axis and panel width. Gray and teal bars show before and after. The dotted line is a ratio of one.

The CIELAB recommendation uses the harmonized-only figure in Section 3.2 and the original raw version in Section 3.1. Both preserve the same stage/channel arrangement and axis limits. The cleaning overlay retains dashed gray before, solid scanner colors after and interquartile shading. Separate individual-specimen cleaning figures remain available as supporting outputs.

Outputs from earlier package versions can remain in an existing results folder. The table above identifies the current recommended figures.

To redraw just the SWD figure using an existing run:

```bat
python manuscript_figures.py "results\section_3_2_rgb\roi_cohort_before_after_summary.csv"
```

## Supporting tables

In results/section_3_2_rgb:

- Tables_for_manuscript.html contains five compact tables, one per cohort/stage, which can be opened in a browser and copied into Word.
- Table_RGB_compact.csv contains the same tables as formatted values.
- Table_RGB_absolute_metrics.csv contains full precision mean RGB, RGB SD and SWD estimates before and after, paired changes, 95% CIs and percentage reductions.
- Table_SWD_references_and_ratios.csv contains the absolute SWD estimates, fixed reference, ratios and sample counts.
- roi_cohort_before_after_summary.csv contains the full cohort-specific summaries.
- roi_before_after_summary.csv retains the pooled summaries.
- roi_paired_distances.csv and roi_raw_within_scanner.csv document individual comparisons and reference calculations.

The compact tables show each metric before/after plus the SWD paired change with 95% CI and reduction percentage. The original raw geometry analysis and its report remain under section_3_1_raw. Additional CIELAB and cleaning-transition summaries remain under section_3_2_lab.

## Region and metric definitions

Every valid pixel of the existing extracted ROI contributes to the analysis. Raw and harmonized ROI maps must share the same projection grid and valid region. Input roots must mirror sample-folder paths and contain the approved geometry.npz, maps.npz and metadata files.

CIELAB uses sRGB/D65/2° and the original visible surface-area weights. Curves average normalized scan histograms with equal weight per specimen. Cleaning changes compare scan-mean Lab colors within the same physical specimen and scanner. The before and after cleaning trajectories each use their own condition's baseline.

RGB metrics use equal projected-pixel weights with RGB in [0,1]. Euclidean distances compare the three-channel mean and standard-deviation vectors. Sliced Wasserstein distance averages absolute differences between sorted projections of sampled RGB colors. The same projection directions are used throughout, and raw/harmonized versions of each scan reuse the same pixel indices.

The raw within-scanner reference is recalculated separately for each cohort and cleaning stage. It averages distances between distinct physical specimens from one scanner. A scanner-pair reference is the average of its two scanner references. That fixed raw reference is used for both before and after ratios. This cohort-specific denominator makes the fresh and mature panels independently interpretable. A zero or unavailable reference gives an undefined ratio, reported as a blank.

Stage-specific across-scanner estimates average matched specimens. Pooled results average eligible stages within each specimen and then average specimens. Paired-change confidence intervals resample physical specimens. RGB within-scanner references and projected-pixel sampling differ from the thesis's original vertex/pseudo-label analysis. The new estimates describe the full extracted ROI.

## Full-model and ROI example

The script uses the recorded raw mesh source when available and searches for BCG3T1-1 in mesh filenames and enclosing folders. This supports scanner-generated names inside specimen folders such as BCG3T1-1_ITERO. It supports colored PLY/VTP and single-texture OBJ/PLY. Keep texture files with the meshes. Full models are rendered without lighting using the same camera, framing and scale for geometrically matched pairs. The camera is chosen from the raw full-model geometry. ROI panels use the existing unlit maps.

The renderer checks geometric agreement independently of vertex order, then tries rigid alignment if needed. If agreement is insufficient, it labels the full-model panels as independent views and records that status in full_model_status.csv. The default camera is determined independently for each scanner, so inspect the resulting full-model views. The ROI views retain the approved reorientation.

If files are missing, ambiguous, uncolored or unreadable, the ROI-only figure is still saved and full_model_status.csv records the exact issue. The run reports the pending example. For explicit file selection, supply --models-manifest with a CSV containing scanner,raw_model,corrected_model and one row for each scanner. Paths may be absolute or relative to that CSV.

Run only the example using:

```bat
python specimen_comparison.py --raw-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs_reoriented" --corrected-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs_reoriented_harmonized"
```

## Figure captions

CIELAB distributions. Scanner-specific L*, a* and b* distributions by cohort and cleaning stage after harmonization. Histograms are weighted by visible surface area within each scan and averaged with equal weight per specimen. The axes match the corresponding raw-color figure.

Cleaning response. Median within-specimen ΔE00 from baseline before (dashed gray) and after (solid scanner colors) harmonization. Shading denotes interquartile ranges. Baseline is defined within each condition. All panels share the same vertical scale.

SWD ratios. Mean matched between-scanner sliced Wasserstein distances before and after harmonization, divided by the mean raw within-scanner reference for that scanner pair, cohort and cleaning stage. Bars show fresh baseline and clean and mature baseline, partial and clean. The dotted line indicates equality with the raw within-scanner reference.

Example. BCG3T1-1 after partial cleaning, shown across the four scanners before and after harmonization. Rows identify the scanners. Columns show the full model before and after harmonization, followed by the ROI before and after harmonization. Lighting is disabled. Geometrically matched full models share camera and framing. Panels labeled independent views use separate cameras. ROI panels use the approved unlit color projections.

## Manuscript draft and reported thesis results

Results_3_2_draft.md contains placeholders for the computed outcomes. The included thesis_baseline_tables.csv and baseline_RGB_benchmark figures reproduce the reported T0 values from Speciale_afhandling-8.pdf, Tables 5–12. The source-defined ROI uses the original pseudo-labels. These historical values remain separately labeled from the full extracted-ROI measurements.
