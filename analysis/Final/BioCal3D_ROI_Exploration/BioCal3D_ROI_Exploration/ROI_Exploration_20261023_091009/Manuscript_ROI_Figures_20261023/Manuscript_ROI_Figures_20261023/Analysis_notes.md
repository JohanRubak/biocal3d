# Analysis of the reoriented Approved ROI dataset

The 20261023_091009 run used Approved ROIs_reoriented. All 512 scans from 48
independent specimens were analysed; no scans failed. Fresh specimens (G1–G2,
n=16) had baseline and clean stages. Mature specimens (G3–G6, n=32) additionally
had a partial-cleaning stage. Each scanner contributed 128 scans. These are
uncalibrated exported representations. Plaque annotations were not used.

## 1. Cleaning-related colour change depends strongly on scanner

Median paired baseline-to-clean CIEDE2000 differences between area-weighted
scan-mean CIELAB vectors were:

| Scanner | Fresh ΔE00 (95% bootstrap CI) | Mature ΔE00 (95% bootstrap CI) |
|---|---|---|
| iTERO | 8.70 (7.10–9.95) | 8.24 (7.26–9.80) |
| TRIOS3 | 6.12 (5.76–6.34) | 6.88 (6.27–7.37) |
| TRIOS5 | 7.66 (6.20–8.72) | 9.34 (8.60–9.88) |
| LABscanner | 2.94 (2.48–3.47) | 2.93 (2.51–3.18) |

The LABscanner has the smallest colour response in both cohorts. Median paired
lightness increases were only +0.53 and +0.06 for the LABscanner (fresh/mature),
versus +4.91 to +7.88 for intraoral scanners. Mean a* decreased after cleaning
for all 48 specimens on every scanner. This suggests that the direction of
change is shared but its magnitude and colour balance depend on the exported
scanner representation. ΔE00 here is a difference between scan means, not mean
pixelwise ΔE00, calibrated colour error or a clinical acceptability test.

## 2. Mesh density is scanner dependent and stage dependent

At mature baseline, median triangle densities were 40.70, 9.99, 10.10 and
153.91 triangles/mm² for iTERO, TRIOS3, TRIOS5 and LABscanner, respectively.
The LABscanner therefore has approximately 15 times the median density of the
TRIOS scanners in this comparison. The median clean/baseline paired density
ratios were 0.660, 0.687, 0.575 and 0.548, respectively. Mesh density declines
with cleaning and should not be treated as an invariant scanner property.
Counts and edge lengths describe mesh sampling; they do not independently
measure optical resolution or geometric accuracy.

## 3. Local slope and height amplitude provide complementary information

For mature specimens, the LABscanner's median native face slope decreased from
3.73° at baseline to 1.79° after cleaning. Its median paired decrease was 1.98°
(95% bootstrap CI 1.72–2.08°). For comparison, the other scanners' baseline
medians ranged from 0.57° to 1.11°. Native-face slopes depend on acquisition and
mesh reconstruction, so a larger slope does not by itself establish better
biological detail.

At the common 100 µm cell width, LABscanner RMS height decreased in 15/16 fresh
and 28/32 mature specimens. Median paired decreases were 1.35 µm (95% CI
0.85–2.43) and 1.44 µm (95% CI 0.92–2.01), respectively. Mature TRIOS3 and
TRIOS5 scans also tended to decrease (median changes −0.66 and −0.81 µm).
iTERO changes were less consistent: +1.51 µm in fresh and −0.28 µm in mature
specimens, with bootstrap intervals spanning zero for both cohorts. Its larger
height outliers remain visible in the logarithmic RMS panels.

These are paired differences; they need not equal the difference between the
two stage medians. Plane-detrended RMS still captures broad shape variation,
not solely small surface features. Common cell width standardises calculation
scale without restoring detail missing from a coarser mesh.

## 4. What the examples support

BCG6T1-4 and BCG3T1-1 demonstrate that scanner representations differ in colour,
faceting, mesh sampling and fine surface structure. LABscanner looks relatively
uniform in colour while its neutral geometry shows much finer variation. This
is consistent with investigating geometric features as a complement to colour.
Without plaque/clean annotations and a reference surface, the maps cannot tell
whether each fine feature is plaque, substrate structure or reconstruction
noise. They do not demonstrate superior plaque discrimination or accuracy.

For the main manuscript, prioritise the mesh/surface summary, the colour-change
plots and one BCG6-4 colour/geometry example. The second example, detailed CIELAB
maps and full channel-distribution figures can support the supplementary
material if the main article becomes crowded. Keep all nine exports available
until the final figure selection is made.

## 5. Remaining interpretation limits

- iTERO metadata identifies texture-derived colour in 47 scans and vertex-derived
  colour in 81 scans. Verify this mixture before attributing all appearance
  differences specifically to scanner acquisition. LABscanner is texture-derived
  throughout; TRIOS3 and TRIOS5 are vertex-derived throughout.
- No before/after colour-calibration experiment is present. These results
  motivate harmonisation but cannot establish that it improves segmentation.
- No plaque–clean labels were used. Whole-specimen stage changes do not establish
  local plaque contrast or segmentation performance.
- This is descriptive full-cohort exploration. Use training data for decisions
  that could affect preprocessing, model design or evaluation.

## 6. Manuscript exports and use in Word

All nine figures are 170 mm wide. Main lettering is 8 pt and tick lettering is
7 pt at that width. PNG and compressed RGB TIFF files are exported at 600 dpi;
PDF files retain vector text and quantitative plot lines with embedded fonts.
Use PNG for convenient insertion into Word and PDF/TIFF as separate artwork
files. Insert a figure as a picture, preserve its aspect ratio, and avoid Word
image compression. Copy the matching caption from Figure_captions.txt beneath
it; assign final manuscript figure numbers when arranging the article.

The colour bar has a dedicated strip. All four scanner columns have the same
panel positions in every example row, including the bottom height row. Each
CIELAB channel uses one common scale across scanners and both examples.

Quantitative plots were redrawn from the supplied tables/cache. Actual RGB,
neutral-native-geometry and native-edge panels were preserved from embedded
source PDF images, with their positions recovered from the original axes.
Height panels were redrawn from the saved numerical height grids using the
original shared colour limit. No specimen measurements, colours, mesh triangles
or inferred plaque annotations were changed. Exporting at 600 dpi does not add
detail to the source images or to scanner geometry.

The original uploaded results are unchanged. The updated code renders these
layouts directly from the native data on future runs. Its optional
reformat_results.py recreates this package from a completed version 1.1 report
without accessing the original mesh files.

Formatting references: Elsevier artwork types and artwork sizing guidance:
https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-types
https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
