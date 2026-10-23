# BioCal3D: Approved ROI exploration — version 1.1.0

This program implements the agreed, unlabelled-data analyses for Materials and
Methods section 2.4. It covers colour distributions, paired scanner and stage
differences, mesh sampling, and scanner-derived surface topography/roughness.
It creates an HTML figure report, CSV tables, and PNG/PDF figures.

**No plaque annotations are required or inferred.** Plaque–clean separability
(step 6) is deliberately deferred. Stage is an experimental descriptor, not a
pixel label. All G1–G6 data are eligible; no group is reserved or excluded.

## 1. Install and run on your Windows PC

Extract this folder outside the data directory. Open **Anaconda Prompt**, then:

```bat
conda activate biocal3d
cd "C:\path\to\BioCal3D_ROI_Exploration"
python -m pip install -r requirements.txt
python investigate_approved_rois.py --data-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs"
```

You can alternatively run `Run_Exploration.bat` from the activated prompt. It
asks for the dataset folder and uses your known AU path as the default. Paste a
folder path without surrounding quotation marks at that prompt.

The default run describes **all data**. Output goes into a new timestamped
folder **inside the code folder, beside the Python script and launcher**, for
example `BioCal3D_ROI_Exploration\ROI_Exploration_20261022_120000`. This does not
depend on the current working directory or where the dataset is stored. Open its
`report.html` in a browser. The report is local and needs no internet connection.

The program reads the source dataset without modifying it. It uses Matplotlib's
noninteractive Agg backend, so it does not require Qt, PyVista or Open3D and does
not trigger the earlier Qt platform-plugin problem.

Python 3.11 or newer is required. It uses CPU processing; no GPU is needed.

## 2. Training-only exploration for preprocessing/model decisions

Full-cohort description is useful for documenting the dataset. Use a separate
training-only run when findings could influence feature selection, calibration,
preprocessing or model design:

```bat
python investigate_approved_rois.py --data-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs" --scope train --splits "C:\path\to\specimen_splits.csv"
```

The split CSV must have **one row per independent specimen**, covering every
specimen, using your actual experiment split:

```csv
physical_id,split
BCG1-1,train
BCG1-2,val
BCG1-3,test
```

Those rows illustrate the format only: do not use them as a new split.
IDs such as `G1_S1` are also accepted. Split names are `train`, `val`, `test`.
All stages and scanners for a specimen stay together. A supplied CSV takes
precedence over preparation metadata; this matters if your later 3D experiments
used a newer split. Without `--splits`, the preparation metadata/index assignments
are used. Training-only mode stops if assignments are missing or conflicting.
The program never silently generates or changes your split.

Keep all-data and training-only runs in different output folders. Resuming
cannot change the dataset, selected scans or analysis scope.

## 3. Expected input structure

```text
Approved ROIs/
  run_config.json
  dataset_index.json                  (optional, for split metadata)
  samples/
    TRIOS5/
      G1/
        BCG1T0-1/
          geometry.npz
          maps.npz
          metadata.json
          roi.ply / roi.vtp           (not needed by this analysis)
```

`geometry.npz` must contain `vertices` (N×3) and triangular `faces` (M×3).
`maps.npz` must contain the corresponding `vertices`, `faces`, `rgb`, `valid`,
`height_mm`, `pixel_surface_area_mm2`, and `fov_mm` arrays, as exported by your
preparation/reviewed-data rebuild. Original JPGs, original full models and label
files are not needed: the saved RGB maps already incorporate texture sampling.

The map geometry is checked against the approved mesh. A stale map or explicitly
pending/flagged ROI is rejected, not silently substituted. This is a consistency
check, **not another manual ROI-quality review**. Missing geometry review status
is recorded, so older exports remain readable; use the approved rebuilt folder.
If maps are missing or stale, regenerate the reviewed snapshot with your existing
rebuild tool. The script does not rebuild or edit the dataset.

Coordinates must be in millimetres, matching the scanner export convention.
The script reads each scan's saved FOV instead of hardcoding 13.4 mm, and lists
all observed FOVs in the output configuration. The common FOV in your present
dataset should be 13.4 × 13.4 mm.

G1–G2 are fresh: T0 baseline, T1 clean. G3–G6 are mature: T0 baseline, T1 partial,
T2 clean. Fresh and mature comparisons are kept separate. LABscanner is treated
as a laboratory scanner.

## 4. Figures and the story they support

Each figure is saved as a PNG (300 dpi by default) and a PDF. The HTML report
includes an interpretation guide and an editable text caption for each output
in `figures.json`. These captions describe the analysis, not invented findings.

The default run produces **nine PNG/PDF figure pairs** when both cohorts and
both requested specimens are present:

| File prefix | Figure | Layout |
|---|---|---|
| `01_Lab_distributions_fresh` | Fresh CIELAB distributions | Stages in rows; L*, a*, b* in columns; four scanners overlaid |
| `01_Lab_distributions_mature` | Mature CIELAB distributions | Same layout, including partial cleaning |
| `02_within_specimen_colour_changes_fresh` | Fresh within-specimen colour changes | Four scanner panels; each specimen followed from its own baseline |
| `02_within_specimen_colour_changes_mature` | Mature within-specimen colour changes | Same layout, including partial cleaning |
| `03_mesh_surface_metrics_fresh_mature` | Three selected mesh/surface metrics | Fresh/mature rows; triangle density, median native slope, 100 µm-grid RMS columns |
| `04_colour_geometry_mesh_BCG6-4_partial` | BCG6T1-4 colour and native geometry | Four scanners; RGB, neutral native geometry, native mesh zoom, height map |
| `04_colour_geometry_mesh_BCG3-1_partial` | BCG3T1-1 colour and native geometry | Same layout |
| `05_Lab_maps_BCG6-4_partial` | BCG6T1-4 CIELAB maps | Four scanners; L*, a*, b* rows |
| `05_Lab_maps_BCG3-1_partial` | BCG3T1-1 CIELAB maps | Same layout |

The three geometric metrics describe complementary aspects: **triangle density**
measures sampling, **median native face slope** measures local inclination, and
**RMS height at 100 µm cells** measures height amplitude at a common calculation
scale. A denser mesh does not prove greater accuracy. The RMS and density panels
use log scales so all observations, including outliers, remain visible. The
same metric has the same vertical limits in both cohort rows. The colour-change
figures also share vertical limits across scanners and both cohorts. These
trajectories show ΔE00 from each specimen's own baseline, not cumulative change
or plaque percentage.

Full-resolution saved RGB projections and original native triangles are used
for the example figures. Both default examples are mature specimens shown at
**partial cleaning**: physical specimen BCG6-4 corresponds to scan BCG6T1-4,
and BCG3-1 corresponds to BCG3T1-1. They are explicitly requested illustrative
examples, not automatically selected median-profile representatives. The
profile rankings remain in the CSV for reference. If a default specimen is
absent from a subset, it is omitted with a console message; no other specimen
is substituted. An explicitly requested missing specimen stops with an error.

To choose different specimen examples:

```bat
python investigate_approved_rois.py --data-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs" --examples BCG6-4 BCG3-1 --dpi 600
```

Pass `--examples` with no IDs to omit example figures. Examples must belong to
the selected analysis scope. For custom mature specimens, the partial stage is
used when available; otherwise the baseline stage is used. Fresh custom
examples use baseline. No additional stage montage, colour-centroid scatter,
paired-scanner heatmap or redundant six-metric geometry plots are generated.
The corresponding numerical analyses remain available in the CSV tables.
 Shared physical display limits, height
colour limits, lighting and geometry exaggeration are used. Neutral geometry
uses **20× vertical exaggeration for display only**, explicitly labelled. Use
`--relief-exaggeration 1` for unexaggerated views. Numeric measurements always
use the original coordinates. `--height-limit-um 30` fixes height colours to
±30 µm instead of the shared automatic limit. Native mesh zooms default to a
4 × 4 mm square and can be changed with `--mesh-zoom-mm`.

## 5. Measurement definitions

### Colour

- CIELAB uses scikit-image `rgb2lab`: **assumed sRGB, D65 white, 2° observer**.
  Scanner colour characterisation is not known. Results describe exported
  appearance and must not be presented as calibrated colour accuracy.
- All valid RGB-map pixels are used by default, weighted by their saved visible
  3D surface-area weights. No geometry-only grey fallback and no background
  pixels enter the analysis. `--colour-stride 2` optionally samples every second
  row and column for faster runs; the setting is recorded.
- Per-scan means, SDs, medians, P5/P95 and histograms are saved for L*, a*, b*.
  Group histograms first normalise each scan, then give each specimen equal
  weight. A high-resolution mesh or larger ROI therefore does not automatically
  dominate a group distribution.
- Paired ΔE00 is calculated **between the two scan mean Lab vectors** with unit
  parametric weights. It is not mean pointwise ΔE00 and can miss differences in
  distribution shape. Paired per-channel histogram Wasserstein distances are
  provided to complement it.
- Signed paired differences are scanner B minus scanner A, or later stage
  minus earlier stage. No interscanner pointwise colour differences are made.

### Mesh sampling

Encoded vertices, exact-coordinate unique geometric vertices, faces, duplicates,
surface area, counts/mm², median/P10/P90 triangle areas and unique geometric-edge
lengths are recorded. Exact coordinate duplicates are welded **for analysis
only**, to avoid counting a texture seam as additional geometric resolution.
No tolerance-based welding, mesh decimation or smoothing is applied to inputs.
Counts and edge lengths are sampling descriptors, not measurements of trueness
or independently established scanner resolution.

### Native geometry and roughness descriptors

An area-integrated covariance of the triangular surface defines its best-fit
plane. Translation and global tilt are removed. Heights are measured normal
to that plane, in millimetres, and reported in micrometres where indicated.

- **RMS height:** exact integral of squared linear triangle height divided by
  total surface area, square-rooted. It does not use a vertex-count average.
- **Mean absolute height and P95–P5:** deterministic, uniform-area samples over
  the native mesh (50,000 by default). These are Monte Carlo estimates; the
  per-scan deterministic seed and sample count are reproducible.
- **Surface slope:** triangle inclination relative to the fitted plane,
  summarised with surface-area-weighted median and P95.

These are **unfiltered, plane-detrended scanner-derived topography descriptors**.
They are not independently validated biological roughness or fully specified
standardised Sa/Sq measurements. No standard roughness-band cutoff is imposed.
Geometry differences can reflect acquisition, reconstruction, sampling or the
specimen; this analysis alone cannot establish which scanner is accurate.

### Common physical scales

The saved height maps are surface-area-weighted box-averaged into cells of
**0.05, 0.10 and 0.20 mm** by default. Each grid is re-levelled and its mean
absolute height, RMS height, P95–P5 and median finite-difference slope are saved.
Cells need at least 75% projected coverage; holes are never filled. Slopes need
finite neighbouring heights. These values describe the cell-averaged surface,
not the native mesh and not a standard Gaussian roughness filter.
Fractional pixel/cell overlaps are used, so a noninteger ratio between pixel
spacing and cell width does not create a checkerboard of false missing cells.
Levelling uses weighted source-sample coordinates to preserve affine planes.

The grids provide common analysis scales; they do not restore detail absent
from a scanner or establish its resolving power. Pitches finer than the source
projection spacing are rejected. Use `--pitches-mm 0.05 0.1 0.2` to set them. Include **0.1 mm** when generating
figures: the focused RMS panel uses that exact width and will not silently
substitute another scale. Table-only runs can use other widths.
At a 13.4 mm FOV and 1024 pixels, source grid spacing is about 13.1 µm; that
spacing is distinct from native scanner resolution.

### Paired comparisons and uncertainty

Scanner comparisons match physical specimen and stage. Stage comparisons match
physical specimen and scanner and include baseline→clean, plus baseline→partial
and partial→clean for mature specimens. Scanner-response differences compare
these within-specimen changes between scanner pairs (difference of changes).

Summary tables report medians, IQRs, specimen counts and percentile bootstrap
95% intervals for medians. Bootstrap units are physical specimens, not pixels
or vertices. Default repetitions: 1,000; a single specimen has no estimated
interval. No p-values, independent-pixel tests or causal claims are produced.
Missing paired scans are not imputed; actual paired counts are reported.

## 6. Tables

- `inventory.csv`, `inventory_counts.csv`: all discovered scans and their scope.
- `selected_scans.csv`: scans selected for the requested scope.
- `scan_measurements.csv`: one row per successfully processed scan.
- `common_scale_measurements.csv`: one row per scan and physical cell width.
- `paired_scanner_differences.csv`: one row per specimen/stage/scanner pair.
- `paired_stage_changes.csv`: one row per specimen/scanner/stage transition.
- `scanner_stage_response_differences.csv`: scanner differences in paired changes.
- `*_summary.csv`: specimen-level medians, IQRs and bootstrap intervals.
- `group_*_summary.csv`: corresponding summaries stratified by experimental
  group, for checking whether pooled fresh/mature patterns vary by group.
- `representative_candidates.csv`: specimen profiles and selected example IDs.
- `failures.csv`: processing failures with reasons. An empty file means no failures.

The CSVs can be opened directly in Excel. No spreadsheet add-in is required.

## 7. Reruns, performance and failure handling

Processing is one scan at a time; valid per-scan caches allow resuming after an
interruption. Cache keys use source sizes/mtimes, analysis settings and script
content. Strict export-hash checks can additionally be enabled with
`--verify-hashes` (extra disk reads). These fast cache keys are not substitutes
for the map/mesh correspondence checks, which are always performed on new scans.

```bat
python investigate_approved_rois.py --data-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs" --output "C:\path\to\previous_output" --resume
```

For this version update, use a **new default output folder**. The changed
script invalidates the previous version's per-scan caches. The original input
data are unchanged, and the main measurements use the same definitions.

Keep the same scope and split when resuming. Changes to measurement settings
invalidate per-scan caches; script changes also invalidate caches; changing only command-line example
selection or figure settings regenerates figures without changing measurements. Use a new
output folder after changing the specimen split, dataset or scope. The cache
is an analysis output, not a replacement for the prepared dataset.

For a fast table-only first pass add `--tables-only`. Source data on a local
drive will generally avoid shared-drive transfer overhead. Do not benchmark
performance from the included tiny synthetic fixtures.

Processing errors are logged per scan, and remaining scans continue. Exit code
0 means no scan failed; 2 means the report was created with processing failures;
1 means analysis stopped. **Review failures and actual paired counts before
using a partial analysis.** No missing scan is silently filled with zeros.

## 8. Tests and technical references

```bat
python -m unittest discover -s tests -v
```

Tests cover analytic geometry, area weighting, texture-seam duplicates, colour
conversion/ΔE00, physical-scale filtering, stage pairing, split handling,
source-map consistency, caching and read-only source behaviour. Synthetic test
data have no biological interpretation and are not manuscript results.

Colour implementation: https://scikit-image.org/docs/stable/api/skimage.color.html

Colour encoding assumption: https://registry.color.org/rgb-registry/srgb

Surface metrology context (detrending/filtering):
https://physics.nist.gov/VSC/jsp/About.jsp

Consult `run_config.json`, `figures.json` and this document when writing the
methods. The complete 512-scan dataset is processed on your PC; validation of
this delivered program does not imply that your entire dataset was run here.
