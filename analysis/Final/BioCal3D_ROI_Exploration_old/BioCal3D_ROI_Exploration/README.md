# BioCal3D: Approved ROI exploration

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
sibling folder, for example `data\ROI_Exploration_20261022_120000`. Open its
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

| File prefix | Figure | Question it addresses |
|---|---|---|
| `01` | CIELAB distributions by scanner and stage, separately for fresh/mature | Do scanners represent the same stages with different colour distributions? |
| `02` | Scan mean a*–b* scatter | Does the dominant colour variation follow scanner or stage? |
| `03` | Paired scanner ΔE00 heatmaps | How different are scanner colour centroids for the same specimen/stage? |
| `04` | Within-specimen colour trajectories from baseline | How does colour change with cleaning within each scanner? |
| `05` | Mesh sampling summaries | How do encoded/geometric vertices, face counts, density and triangle sizes differ? |
| `06` | Native surface topography and roughness descriptors | How do height variation and face slopes differ across scanners/stages? |
| `07` | Roughness at shared physical cell widths | Do scanner differences persist when analysed at comparable spatial scales? |
| `08` | Paired scanner-specific stage responses | Do scanners represent the same cleaning transitions differently? |
| `09` | Matched specimen stage montage | What do the same specimen and stages look like across scanners? |
| `10` | RGB, neutral geometry, mesh zoom and height map | How do colour, geometric structure and mesh sampling differ in one matched specimen? |
| `11` | Matched L*, a* and b* maps | Which colour channels account for visible scanner differences? |

For an initial manuscript story, start with `01`, `03`, `05`, `06` or `07`, and
one matched `10` panel. Use `04`/`08` when discussing changes with cleaning.
The supplied data determine the findings; this code does not presume LABscanner
will have better geometric plaque contrast.

By default, one complete specimen from each cohort is selected with the smallest
robust distance to the cohort median feature profile. This is a deterministic
selection rule, not an attempt to find the most striking example. Candidate
rankings and selected IDs are recorded in `representative_candidates.csv`.
If a cohort has no specimen complete across the available scanners/stages,
automatic representative figures are omitted for that cohort; the numeric
analysis still runs. Examples can be selected explicitly:

```bat
python investigate_approved_rois.py --data-root "C:\Users\au662213\repos\biocal3d\data\Approved ROIs" --examples BCG1-2 BCG6-4 --dpi 600
```

An example must belong to the selected analysis scope. Representative structure
views use the partial stage for mature specimens and baseline for fresh ones.
RGB montages show all available stages. Shared physical display limits, height
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
projection spacing are rejected. Use `--pitches-mm 0.05 0.1 0.2` to set them.
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
- `representative_candidates.csv`: representative selection audit.
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

Keep the same scope and split when resuming. Changes to measurement settings
invalidate per-scan caches; figure-only changes regenerate figures. Use a new
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
