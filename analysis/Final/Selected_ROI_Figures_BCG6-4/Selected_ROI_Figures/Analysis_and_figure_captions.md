# Review of ROI exploration run 20261022_122749

The completed run analysed 512 scans from 48 independent specimens, with no failed scans. Each scanner contributed 128 scans. Fresh specimens (G1–G2; n=16) had baseline and clean stages; mature specimens (G3–G6; n=32) also had a partial-cleaning stage. All findings below describe the uncalibrated exported representations. No plaque annotations were available.

## Main findings

1. **Scanner colour differences remain after cleaning.** At mature baseline, median paired CIEDE2000 differences between scan-mean CIELAB vectors ranged from 4.59 (TRIOS3–TRIOS5) to 15.34 (iTERO–LABscanner). After cleaning they ranged from 4.84 (iTERO–TRIOS3) to 11.39 (iTERO–LABscanner). These are paired scan-level differences, not pixelwise differences or colour-accuracy measurements.

2. **The colour response to cleaning is scanner dependent.** Median paired baseline-to-clean differences were:

| Scanner | Fresh ΔE00 | Mature ΔE00 | Fresh change in L* | Mature change in L* |
|---|---:|---:|---:|---:|
| iTERO | 8.70 | 8.24 | +7.61 | +7.23 |
| TRIOS3 | 6.12 | 6.88 | +4.91 | +5.96 |
| TRIOS5 | 7.66 | 9.34 | +6.21 | +7.88 |
| LABscanner | 2.94 | 2.93 | +0.53 | +0.06 |

Mean a* decreased after cleaning for all 48 specimens on each scanner. LABscanner shows substantially less global colour change and almost no net lightness shift, consistent with the user's visual impression that its colour may provide weaker plaque contrast. Whole-specimen colour change does not establish local plaque–clean separability.

3. **Mesh sampling differs substantially.** At mature baseline, median triangle densities were 40.70, 9.99, 10.10 and 153.91 triangles/mm² for iTERO, TRIOS3, TRIOS5 and LABscanner, respectively. Median geometric edge lengths were 239.65, 498.80, 485.71 and 123.90 µm, respectively. The LABscanner's meshes are more densely sampled; this is not proof of greater geometric accuracy. Triangle density decreased after cleaning: paired clean/baseline median ratios were 0.660, 0.687, 0.575 and 0.548, respectively. Mesh density therefore also varies with experimental stage and should not be treated as a fixed scanner characteristic.

4. **Local surface detail and whole-surface height amplitude tell different stories.** Mature baseline median native face slopes were 1.11°, 0.57°, 0.75° and 3.73° for iTERO, TRIOS3, TRIOS5 and LABscanner, respectively. LABscanner slopes decreased to 1.79° after cleaning. In contrast, whole-surface RMS was often larger on iTERO, which also had the largest outliers (native RMS up to 46.64 µm). Greater RMS does not imply better plaque visibility: it also reflects broad surface shape, reconstruction differences and possible artefacts.

5. **The LABscanner's stage-associated topographic change persists at a shared sampling scale.** At 100 µm cells, its paired median RMS change from baseline to clean was −1.35 µm in fresh specimens (15/16 decreased) and −1.44 µm in mature specimens (28/32 decreased). The corresponding changes were −0.23/−0.66 µm for TRIOS3, −0.50/−0.81 µm for TRIOS5 and +1.45/−0.09 µm for iTERO (fresh/mature). These are differences of paired specimen values, not differences between group medians. Common grids standardise calculation scale but do not restore detail absent from a coarser source mesh or make scanner resolution equal.

Taken together, colour and geometry respond differently across the scanners. This motivates an annotation-based comparison of colour and geometric features, but does not yet demonstrate superior plaque discrimination or improved model performance from geometry.

## Recommended figure selection

Use three figures for this part of the manuscript if space permits, with final numbering assigned in the manuscript:

- **Main_1_Colour_differences:** paired scanner differences and the scanner-dependent cleaning response, combining the main message of original figures 03 and 04.
- **Main_2_Mesh_and_topography:** triangle density, native face slope and common-grid RMS, combining selected information from original figures 05–07. Separate raw vertex counts and multiple correlated height summaries add little to the main narrative.
- **Main_3_BCG6-4_Colour_and_topography:** one partially cleaned specimen, viewed as colour, neutral-shaded topography and detrended height. This illustrates the rationale for examining both modalities; it does not identify plaque without labels.

If only two main figures fit, retain Main_1 and Main_2 and move Main_3 to the supplement. If a specimen/stage colour overview already appears earlier in the paper, do not repeat it in the main text.

Keep full CIELAB distributions (original 01), all-stage scanner-pair colour matrices (original 03), common-grid sensitivity plots (original 07) and the BCG6-4 stage montage in the supplement. Place raw vertex/face counts, triangle sizes and the remaining height statistics in a supplementary table. Original 02, 08 and 11 are useful working outputs but need not each become a manuscript figure; they largely repeat messages shown more clearly in the selected figures.

## Figure captions

**Main_1 — Scanner-dependent colour characteristics and cleaning-associated changes.** (A, B) Median paired CIEDE2000 colour differences (ΔE00) between area-weighted scan-mean CIELAB vectors at baseline for fresh (n=16) and mature specimens (n=32). The same specimen and stage were paired between scanners. (C) Baseline-to-clean ΔE00 calculated within each specimen and scanner. Small points represent specimens; large symbols show medians and error bars show 95% bootstrap confidence intervals. Open circles represent fresh specimens and filled squares represent mature specimens. CIELAB conversion assumed sRGB with D65 illumination and a 2° observer. These measures describe differences between exported colours without an independent colour reference.

**Main_2 — Scanner-dependent mesh sampling and surface topography across experimental stages.** Fresh specimens (A–C; n=16) were scanned at baseline and after cleaning; mature specimens (D–F; n=32) additionally underwent partial cleaning. Panels show triangle density per unit mesh surface area (A, D), area-weighted median native face slope relative to the fitted plane (B, E), and plane-detrended RMS height after averaging into 100 µm cells (C, F). Small points represent scans; connected symbols and error bars show medians and interquartile ranges. Triangle density and RMS height use logarithmic axes. Shared cell width standardises the analysis scale without equalising scanner resolving power. These measurements describe unfiltered scanner-derived topography rather than reference-measured material roughness.

**Main_3 — Colour and geometric appearance of specimen BCG6-4 after partial cleaning.** Columns show the four scanners. Rows show (A) the saved unlit RGB projection, (B) neutral shading of the saved 50 µm height grid with identical illumination and 20-fold vertical exaggeration, and (C) detrended surface height. Every panel spans a 13.4 × 13.4 mm field of view. Height maps share a colour range of ±20.29 µm, derived from the 98th percentile of absolute heights across the 512 report maps; values outside the range are saturated. Neutral views are derived from the common height grid and do not show the original mesh triangles. Colour and height panels use their saved image frames; cross-scanner frames are not registered for pointwise comparison. These images illustrate differing colour and topographic representations without assigning plaque locations. RGB panels use the report's 320 × 320 cached previews; regenerate from the full-resolution maps for final publication if needed.

**Supplement — Colour appearance of specimen BCG6-4 across experimental stages.** Columns show scanners and rows show baseline, partial cleaning and complete cleaning. Panels display saved, uncalibrated, unlit RGB projections over the same 13.4 × 13.4 mm field of view. Cross-scanner orientation has not been registered for pointwise comparison. The RGB panels use cached report previews.

## Interpretation and outstanding checks

- The run used **all specimens**, with no training/test partition supplied. Treat these as descriptive full-cohort outputs. Check any data-dependent choices affecting preprocessing or model evaluation using the training subset; do not state that this run was training-only.
- iTERO metadata reports texture-derived colour in 47 scans and vertex-derived colour in 81 scans. LABscanner reports texture-derived colour in all 128 scans; TRIOS3 and TRIOS5 report vertex-derived colour in all 128 scans. Verify whether this mixture reflects intended source availability, and whether texture sampling or colour fallback contributes to the iTERO appearance. Do not attribute all observed differences specifically to scanner hardware.
- No colour-calibration experiment, calibration target or before/after harmonisation comparison is included in this run. Colour differences motivate calibration, but cannot demonstrate its effectiveness. No clinical acceptability threshold is applied to these ΔE00 values.
- No labelled plaque–clean comparison has been completed. Keep the sixth analysis step explicitly planned in the Methods draft until annotations and analysis are available.
- There is no independent geometric or roughness reference in these results. Neutral shaded fine structure can include reconstruction noise; mesh density, slope and height are not accuracy measurements.

## Files and reproducibility

PNG and vector-text PDF versions of the four selected figures are included, along with exact paired summary tables and the BCG6-4 measurements. `make_selected_figures.py` recreates these figures directly from the exploration report tables and cache:

```
python make_selected_figures.py "C:\path\to\ROI_Exploration_20261022_122749" "Selected_ROI_Figures"
```

For native mesh-edge panels and full-resolution RGB/Lab maps, rerun the original `investigate_approved_rois.py` on the original Approved ROIs with `--examples BCG6-4` and a new output folder. The uploaded report does not contain the original `geometry.npz` and `maps.npz` files, so the native mesh panel cannot be reconstructed here.
