# BioCal3D targeted 3D follow-up

**Use these follow-up scripts before the original `04_full_cv_loso.cmd`.** The old default full suite adds outer assessment only for area-geometry PointNeXt-S and Concerto. It does not assess the RGB winners or Sonata by default.

This package tests the questions raised by your completed 23-run comparison: how much the RGB models use global colour versus spatial features; whether they transfer to new specimens/scanners; what their RGB predictions depend on; and whether mesh reconstruction, outliers and pose may explain results.

Your existing models/results are read, not edited. New outputs go to a separate folder. No reserved-test fitting, evaluation, QC figures or model predictions are performed.

## Start here

Extract `BioCal3D_3D_followup` under your project's `3D branch` directory. Open Anaconda Prompt, activate `biocal3d`, and `cd` into the extracted folder. All launchers work directly in Command Prompt/Anaconda Prompt; PowerShell is not required.

Run the first stage:

```bat
01_audit_and_plan.cmd
02_preflight.cmd
10_run_tests.cmd
03_existing_rgb_diagnostics.cmd
04_colour_pathways.cmd
```

Open:

```text
runs\3d_followup_20261013\followup_comparison.html
```

Review the colour-pathway results before spending GPU time on the longer stages. Then run:

```bat
05_targeted_group_cv.cmd
06_targeted_loso.cmd
```

Each launcher resumes interrupted training and reuses completed jobs if data/settings/provenance match. Do not run multiple launchers simultaneously into the same output folder: training/report files are updated incrementally.

Defaults point to your exact snapshot:

```text
C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261013_101905_315169
```

Original results/checkpoints are read from:

```text
C:\Users\au662213\repos\biocal3d\3D branch\BioCal3D_3D_investigations_v1\BioCal3D_3D_investigations\runs\3d_prepared_20261013
```

If moved, pass `--original-output "C:\path\to\3d_prepared_20261013"` to each launcher. If official backends are not detected, pass `--backends-root "C:\path\to\third_party"` to GPU launchers. The existing third_party repositories and original pretrained checkpoints are reused; there are no new backbone downloads or replacement CUDA installations.

## What each stage does

| Launcher | Work | Default size |
|---|---|---:|
| `01_audit_and_plan.cmd` | CPU: match full 512-scan cohort and specimen split; export plans, mesh-count pairs, roughness outliers, named sample inspection and nuisance/colour controls | No neural training |
| `02_preflight.cmd` | GPU: test all six pathways on both models plus Sonata geometry, repeatability and pathway gating | 13 model/pathway checks |
| `03_existing_rgb_diagnostics.cmd` | GPU: load original seed-42 RGB PointNeXt-S/Concerto; reproduce saved validation predictions; export focus for fixed examples and confident errors; test pose/order/colour perturbations | No training |
| `04_colour_pathways.cmd` | Train six matched pathways on PointNeXt-S and Concerto, seeds 42/43/44 | 36 jobs |
| `05_targeted_group_cv.cmd` | Five outer specimen folds, independent inner validation, four selected configurations | 20 jobs |
| `06_targeted_loso.cmd` | Same five outer folds, four held-out scanners, four configurations | 80 jobs |
| `07_mature_only_colour.cmd` | Optional: repeat colour pathways using only G3–G6 in a separate output | 36 jobs |
| `08_refresh_report.cmd` | Rebuild the follow-up HTML/CSV without Torch/backbones | No training |
| `09_make_review_zip.cmd` | Create a small upload ZIP, omitting weights and caches | No training |
| `10_run_tests.cmd` | Synthetic software checks; includes Torch routing test if Torch is installed | No official-backbone training |

The 512 scans remain 384 train / 64 validation / 64 reserved test under the original reference split. Only the 448 development scans supply new model inputs or QC. Outer folds repartition these development subjects; global test stays reserved. All scanners/stages belonging to a physical specimen remain together.

## Six colour pathways

The new matched comparison deliberately gives every PointNeXt pathway a **nine-channel encoder architecture**, and every pathway within each model the **same full-width classifier head**. Missing information pathways are zeroed rather than changing head dimensions. Encoder initialization and classifier initialization use the same seed within each model's comparison.

| Pathway | Information available to the model | Purpose |
|---|---|---|
| `geometry_matched` | XYZ + normals; zero RGB; zero direct colour-statistics path | Geometry-only control with matched architecture/head width |
| `stats_only` | Exact six global colour values; encoder pooled vector zeroed | Can global colour explain the classification without geometry? |
| `rgb_encoder_only` | XYZ + normals + per-point RGB through encoder; direct colour statistics zeroed | Is encoder output sufficient without the explicit global shortcut? |
| `rgb_combined` | RGB encoder features plus six direct colour values | Controlled counterpart of the original successful RGB configuration |
| `rgb_shuffled` | RGB deterministically shuffled among positions within each scan, during both fitting and evaluation; direct statistics retained | Does spatial colour–geometry association matter beyond the colour distribution? |
| `gray_combined` | Luminance replicated to three channels; encoder and direct intensity statistics | Does chromatic information add value beyond brightness? |

The six direct values are the same RGB means and maxima used by the original model, computed from the sampled surface points after its 2*x−1 convention. The same LayerNorm → Linear(128) → GELU → Dropout(0.25) → Linear(3) head is used, with the original training defaults and validation specimen-balanced accuracy checkpoint selection.

PointNeXt's `stats_only` encoder is unused/frozen, so fitting uses cached six-statistic embeddings padded to the same head input width. It does not secretly use geometry. Sonata/Concerto remain frozen pretrained feature extractors; PointNeXt's encoder is trained for the encoder-using variants. Head parameter counts differ across backbone families because their feature widths differ; comparisons are matched **within each family**, not claimed to have identical total parameter counts across families.

**Do not compare the new `geometry_matched` score directly as if it were an identical rerun of the old `area_geometry` job.** The old PointNeXt geometry model had six encoder input channels, and the old geometry heads lacked the extra six positions. That difference is precisely why the controlled comparison is rerun here. Existing weights are not repurposed to initialize these new experiments.

Colour shuffling is fixed per scan and sampling seed, independent of class labels. It preserves that scan's marginal RGB distribution and original geometry/normals but breaks position–colour association. A shuffled model retaining high accuracy suggests local colour arrangement is not needed for this **stage-classification task**; it does not establish that spatial structure is irrelevant to segmentation.

## Which configurations enter CV and scanner holdout?

Both stages include:

1. PointNeXt-S `rgb_combined`.
2. PointNeXt-S `stats_only` with the matched nonlinear head.
3. Concerto `rgb_combined`.
4. Sonata `legacy_geometry`, preserving the original geometry-baseline architecture.

The cheap scanner-only, face-count-only, scanner+face-count, roughness and colour-statistics logistic-regression controls are also fitted separately in every protocol/fold. Scalers/categories/classifiers are fitted using that fold's training subjects only. Their predictions are pooled separately from learned models.

CV/LOSO uses seed 42 by default; three-seed colour results estimate initialization variability on the reference split. These address different uncertainty sources. Add `--include-concerto-stats` if you also want a matched Concerto statistics-only head in outer assessment (CV becomes 25 jobs, LOSO becomes 100).

LOSO trains on three scanners, validates on different specimens at those three scanners, and assesses on the held-out scanner **using outer-assessment specimens absent from fitting at every scanner**. This prevents same-specimen cross-scanner leakage.

`followup_out_of_fold.csv` and `control_out_of_fold.csv` mark partial suites incomplete, check scan IDs against the full development cohort, and reject duplicate predictions within a model/pathway/protocol/seed. Seed predictions are never counted as additional independent scans. Reference validation is used for checkpoint/model selection and is not an independent final estimate.

## Existing RGB diagnostics

`03_existing_rgb_diagnostics.cmd` loads `best.pt` and `config.json` from the original seed-42 RGB jobs. It validates dataset identities, original runner/backend fingerprints and encoder initialization, then checks **all 64 reproduced validation probabilities and predicted classes against the original CSV**. It refuses interpretation if agreement fails.

It exports:

- Actual scanner-colour mesh rendering, 4×4 region-removal sensitivity and overlay for the fixed stage/scanner examples.
- Up to six high-confidence errors per RGB model, in addition to the fixed examples.
- 90°, 180° and 270° surface-axis rotation checks, point-order permutation, within-scan RGB shuffling and grayscale inference.
- Original-checkpoint hashes and prediction consistency audit.

Output: `existing_rgb_diagnostics\<model>\`, and `existing_rgb_diagnostics\perturbations.csv`.

The old training/checkpoint/prediction files are never written. New surface caches and diagnostics go into the follow-up output. If original files/backends/checkpoints have moved, specify their paths; do not bypass a prediction mismatch.

These perturbations at inference can produce distribution shift; they are diagnostics, not substitutes for the retrained shuffled/grayscale variants. Occlusion removes a region and refills to the same point count, affecting sampling density and pooled colour as well as local geometry. Maps explain the predicted class, which can be wrong, and are not plaque segmentation or proof of causality.

## Geometry QC and named samples

The CPU stage reuses original geometry metadata only when scan IDs and geometry hashes agree; otherwise it recomputes development metadata from the prepared meshes. It exports:

- Paired baseline/clean triangle counts and scanner-specific ratios.
- The largest surface-roughness outliers.
- Cross-scanner RGB/height inspection for BCG2T0-1, BCG4T2-5, BCG4T0-5, BCG4T1-2, BCG6T0-7, BCG4T1-5 and BCG5T0-8, restricted to development subjects.

No scan is automatically excluded or relabeled. Review raw source exports, ROI boundary, registration and acquisition notes before changing the cohort. This package **does not automatically remesh or repair original geometry**: common-resolution remeshing, interior-ROI retraining and reference-phantom scans should follow the QC findings under a separately specified preprocessing protocol. It likewise does not add unvalidated rotation augmentation to existing trained checkpoints; it measures pose sensitivity first.

## Mature-only comparison

Partial-stage scans occur only in G3–G6. `07_mature_only_colour.cmd` keeps only those groups in fitting and validation, preserving each specimen's original assignment, and writes to `runs\3d_followup_mature_20261013` by default. This is a useful group-confounding check, but its validation contains only four specimens. Do not combine its metrics with the full-cohort results as if the populations were the same.

To assess mature-only models in outer folds:

```bat
05_targeted_group_cv.cmd --mature-only
06_targeted_loso.cmd --mature-only
08_refresh_report.cmd --mature-only
```

For the upload ZIP of those outputs:

```bat
09_make_review_zip.cmd --output runs\3d_followup_mature_20261013
```

## Short pilot / selective execution

Every launcher forwards extra arguments. For example:

```bat
REM Four CV jobs / 16 scanner-holdout jobs for outer fold index 0:
05_targeted_group_cv.cmd --fold-indices 0
06_targeted_loso.cmd --fold-indices 0

REM Run remaining folds later using the default complete command:
05_targeted_group_cv.cmd
06_targeted_loso.cmd

REM Limit the reference study to PointNeXt-S, or add Concerto later:
04_colour_pathways.cmd --models pointnext_s
04_colour_pathways.cmd --models concerto

REM Planning only, without opening meshes for QC or importing Torch:
python followup.py --action colour --plan

REM Use a different original result location:
01_audit_and_plan.cmd --original-output "C:\path\to\3d_prepared_20261013"
```

Keep data/sampling/head settings and code unchanged when resuming. Changed experiment identities require a fresh `--output`. The original test partition is never used by this follow-up; final testing should happen only after the development protocol and selected configuration are fixed.

## Review outputs

Open `followup_comparison.html`. It contains pathway mean/seed-SD comparisons, outer-assessment results and completion counts, nuisance controls, RGB focus figures, perturbation results and geometry QC. Per-run folders retain the original runner's histories, checkpoints, probability/error CSVs, class/scanner figures and whole-specimen bootstrap intervals.

The report also retains a copy of your original aggregate comparison as `original_comparison.csv` for context. Old and new matched controls are not silently pooled. Surface/feature caches, not model weights, dominate many output folders; use `09_make_review_zip.cmd` to export the small review bundle. If necessary use `09_make_review_zip.cmd --no-images` for numeric results first.

## Validation and dependencies

Use the same working CUDA-enabled `biocal3d` environment as your original run; official backends/checkpoints are required for GPU actions. No torch/torchvision/CUDA packages are installed or changed by the launchers. Only install missing utilities from `requirements.txt` if necessary, after other jobs have finished.

See `VALIDATION.json` for checks actually run. The delivery environment has NumPy/scikit-learn/Matplotlib but no Torch or GPU: synthetic CPU tests and integration checks were run; actual pretrained/scratch model training, existing-checkpoint reproduction and Windows batch execution require the included local preflight/tests. No performance outcome is promised by these scripts.
