# BioCal3D: approved ROI 3D investigation results, 20 October 2026

## Main interpretation

The current point-cloud models classify acquisition stage very well when RGB
is available. Geometry-only results are much weaker under the tested models
and sampling protocols. Training with shuffled RGB retains high scores, so
the task can be solved without preserving detailed colour-position association.
This is compatible with a strong colour-distribution signal, but does not prove
geometry is unused: shuffled inputs still retain the original coordinates and
normals. A stronger colour-only histogram/quantile baseline would help isolate
the incremental value of geometry.

## What this export contains

- 512 approved scans, 48 physical specimens, 4 scanners, no missing/excluded scans.
- Train: 384 scans / 36 specimens. Validation: 64 scans / 6 specimens.
  Reserved test: 64 scans / 6 specimens.
- Completed: 23 original investigations, 36 colour-pathway fits and 9
  PointNeXt interior fits. No failures in the completed baseline/colour suites.
- Concerto interior geometry seed 42 was still training at epoch 8 at export.
  No completed CV, scanner-holdout or final-test evaluations were included.
- All completed fits share the same validation specimens. Partial-stage
  validation comprises 16 scans from only 4 physical specimens.
- Scores below are validation specimen-balanced accuracy; means/SDs summarize
  3 fitting seeds, not independent subjects or independent dataset splits.
- Recomputed metrics from predictions for all 68 completed runs. Prediction
  IDs matched the validation cohort; training/validation specimens were
  disjoint; reserved-test specimens were absent from their run cohorts.

## Slide 1 — RGB enables high stage-classification accuracy

- PointNeXt-S: **99.3% with RGB vs 45.6% geometry-only**, using matched
  encoder input width and classifier architecture.
- Concerto: **95.1% with RGB vs 50.2% geometry-only** under the matched protocol.
- Partial-stage recall is **97.9% for PointNeXt-S and 93.8% for Concerto**,
  averaged over three RGB fits.

Figure: `01_geometry_vs_rgb.png` (or editable SVG).

Presenter note: these are specimen-held-out validation results on scanners
represented in training. Do not describe them as unseen-scanner generalization
or final-test accuracy. The original four-model geometry benchmark is provided
as `backup_geometry_four_models.png`: PointNeXt 48.8%, TSGCNet 53.5%, Concerto
53.2%, Sonata 57.6%. Original and matched geometry heads differ, so keep these
two experiment families labelled separately.

## Slide 2 — Detailed spatial colour arrangement is not essential here

- Shuffled RGB retains accuracy: **99.3% PointNeXt-S and 95.4% Concerto**,
  compared with 99.3% and 95.1% for their full RGB counterparts.
- Six global RGB means/maxima reach only **73.6–73.8%**; the encoder captures
  useful information beyond this limited summary.
- Models retrained on grayscale still reach **90.7–93.1%**, consistent with
  a substantial brightness/intensity contribution; a chromatic contribution
  remains plausible from the higher RGB scores.

Figure: `02_colour_pathways.png`, showing all six prespecified pathways.

Presenter note: permutation breaks sampled colour-position association but
keeps each scan's marginal RGB distribution, geometry and normals. It does
not establish that all geometric or spatial information is irrelevant.
The shuffled and ordinary models are separately fitted. These observations
support a colour-distribution interpretation; they do not validate plaque
boundaries or segmentation.

## Slide 3 — The interior retains performance; pose robustness needs work

- Restricting inputs to the inner **80% of approved ROI radius** preserves
  high PointNeXt-S RGB accuracy: **99.3% → 97.9%**, a 1.4 percentage-point drop.
- High performance therefore persists when the outer annulus is removed
  from sampling; the crop edge is not necessary for high scores in this check.
- A 90° inference rotation reduces seed-42 RGB accuracy from **100.0% to
  85.4% for PointNeXt-S**, and **95.1% to 85.4% for Concerto**.

Figure: `03_interior_and_rotation.png`.

Presenter note: left panel contains three matched-seed interior/full fits;
right panel is a single-seed inference perturbation. Do not combine these as
one population estimate. Interior normals/centering still follow the original
full-mesh convention; this is not complete removal of every possible boundary
influence. Concerto interior results remain pending. Removing an annulus may
remove real plaque as well as artifacts.

## Slide 4 — Mesh density is associated with stage and scanner

- Baseline meshes contain more triangles than paired clean meshes in
  **168/168 development specimen/scanner pairs**.
- Scanner-specific median baseline/clean triangle ratios are **1.48–1.85×**.
- **Scanner identity + triangle count reaches 71.5%** validation accuracy;
  scanner alone and count alone each achieve 33.3% in these controls.

Figure: `04_mesh_density_and_controls.png`.

Presenter note: these are 42 physical development specimens × 4 correlated
scanner observations, not 168 independent biological specimens. This flags
stage-associated reconstruction/density information. It does not distinguish
biological surface changes from reflectance, scanner processing or other
acquisition effects. Concerto's seed-42 geometry score also changes from
52.8% with area sampling to 73.6% with native face-center sampling, making
sampling sensitivity an additional useful investigation.

## Optional final slide — Which conclusions remain to be tested?

- Complete Concerto interior runs and the planned specimen CV / scanner holdout.
- Run mature-only comparisons: the partial class occurs only in G3–G6.
- Compare against richer colour-only histograms/quantiles before attributing
  the RGB models' advantage over six statistics to 3D geometry.
- Test rotation augmentation and common-resolution sampling/remeshing as
  separate development experiments, keeping reserved test untouched until
  model/protocol selection is fixed.

Figure suggestion: a compact completion table with completed stages versus
pending stages. Use a qualitative sensitivity image only as a backup; it is
not evidence of a segmentation result.

## Backup diagnostics and interpretation

- `backup_focus_PointNeXt_TRIOS3_BCG3T1-5.png` is a correctly classified
  partial sample. Its probability is saturated near 100% and its occlusion
  map is almost flat. A flat map here does not demonstrate a biologically
  meaningful focus or absence of useful regions.
- `backup_error_Concerto_iTERO_BCG4T1-5.png` shows a partial sample called
  baseline at 89.3% confidence. This is a useful failure-analysis figure.
  The map explains the predicted baseline class, not plaque annotation.
- Seed-42 Concerto errors are TRIOS3 BCG3T2-5 (clean → partial), iTERO
  BCG4T1-5 (partial → baseline), and iTERO BCG4T2-5 (clean → partial).
  Do not relabel automatically based on model disagreement.
- Inference-only grayscale causes much larger drops (PointNeXt 100.0% →
  33.3%; Concerto 95.1% → 38.9%) than retraining on grayscale. This is a
  distribution-shift diagnostic, not a contradiction of Slide 2.
- PointNeXt RGB stage-order scores have no observed reversals across the
  three reference seeds in this validation set. This complements the high
  stage accuracy; it is not an independent validation or plaque-coverage measure.
- Do not show the perfect model's bootstrap [100%,100%] interval as evidence
  of certainty. Resampling an all-correct small validation set cannot expose
  unobserved errors or checkpoint/model-selection uncertainty.

## Slide footnote

"Validation specimen-balanced accuracy; 64 scans from 6 specimens (4 with
partial-stage scans). Mean ± SD across 3 fitting seeds where shown. Validation
used for checkpoint selection; scanner-holdout and final-test results pending."

## Files and reproducibility

- Four main figures are supplied as high-resolution PNG and editable SVG.
- One additional original geometry benchmark is supplied as PNG/SVG.
- Two exact original focus/error PNGs are included without alteration.
- `source_tables/` contains the numeric tables used in the charts.
- `make_figures.py /path/to/extracted/review` reproduces the figures.
- Source: `3d_approved_review_20261020_140827_821306.zip`, exported 20 October
  2026. All conclusions concern the data available in that export; ongoing
  training may have completed additional jobs subsequently.
