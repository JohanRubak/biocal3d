# BioCal3D: training and investigations on the October 13 prepared snapshot

This package extends your existing four-model benchmark using the actual official
PointNeXt-S, TSGCNet, Sonata and Concerto adapters recovered from your source package.
It reads the new prepared snapshot directly:

`C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261013_101905_315169`

The task is baseline / partial / clean **stage classification**, using experimental
stage labels. No segmentation labels are required. No segmentation result is claimed.

## Start here

1. Extract the `BioCal3D_3D_investigations` folder beside the existing `third_party`
   directory, or anywhere under the existing benchmark project. Keep all five `.py`
   modules together. The old training scripts and runs are not modified.
2. Open your working Anaconda prompt and activate the environment where the four
   original model preflight checks passed (for example `conda activate biocal3d`).
3. Change into the extracted folder and run:

```bat
python biocal3d_train_3d_investigations.py --plan
python biocal3d_train_3d_investigations.py --audit-only
python biocal3d_train_3d_investigations.py --suite core --preflight
python biocal3d_train_3d_investigations.py --suite investigations --resume --live
```

The default data folder is already set to your October 13 snapshot. A different
folder can be supplied with `--data-root "C:\...\_development_..."`.

If automatic backend discovery fails, add the exact existing backend directory:

```bat
python biocal3d_train_3d_investigations.py --preflight --suite core --backends-root "C:\Users\au662213\repos\biocal3d\3D branch\biocal3d_3d_benchmark\biocal3d_3d_benchmark\third_party"
```

The included `01_check_data.cmd`, `02_preflight.cmd`, `03_train_investigations.cmd`
and `04_full_cv_loso.cmd` are convenient wrappers. Run them from the activated
environment. Double-click only if your default Python is the correct environment.
They pause after completion/errors so messages remain visible.

The report is `runs\3d_prepared_20261013\comparison.html` relative to the working
directory; the full resolved path is printed. `--output` selects another location.
Training logs and checkpoints are saved every completed epoch. Ctrl+C refreshes
the report and stops the suite; use the identical command with `--resume` to continue.
Interrupting within an epoch resumes from the previous completed epoch.

To inspect an interrupted model's best completed epoch without continuing training:

```bat
python biocal3d_train_3d_investigations.py --suite core --models pointnext_s --seeds 42 --resume --evaluate-only
```

Use the model/seed that has a `best.pt`; this exports validation predictions,
robustness checks and focus maps. Afterwards normal `--resume` continues training.

## What is included

### Cohort and provenance audit

- Reads only `dataset_index.json` and each indexed `sample_dir/geometry.npz`.
  It never scans an older `_processed_ROI` tree or the raw data as a fallback.
- Expects 512 scans: G1–G2 have 8 specimens × 2 stages × 4 scanners; G3–G6 have
  8 specimens × 3 stages × 4 scanners. All 48 physical specimens are checked.
- Validates scan identities, scanner names, timepoint/stage agreement, duplicate
  records, triangle indices, finite geometry and saved geometry hashes when present.
- Missing or failed scans stop the full-cohort run. `--allow-incomplete` is an
  explicit override after reviewing the missing-scan message; no scan is quietly
  discarded. `cohort_audit.json` records differences on an allowed incomplete run.
- A changed snapshot/split cannot reuse the same output directory. Cache signatures
  contain the source geometry hash, sampling implementation/configuration, model
  state, and adapter/runner hashes. Changed code requires a new output directory.

Prepared index schema, based on your previous preparation loader:

```json
{"scans":[{"scan_id":"TRIOS3__BCG3T1-6","scanner":"TRIOS3","group":3,
"physical_id":"BCG3-6","stage":"partial","status":"OK",
"sample_dir":"TRIOS3/G3/BCG3T1-6","geometry_sha256":"optional saved SHA256"}]}
```

`geometry.npz` needs `vertices` (N×3 in mm) and integer `faces` (M×3).
`vertex_rgb` (N×3, [0,1] or [0,255]) provides real sample colour. Normals are
recomputed consistently from triangles. This uses the prepared canonical ROI as
saved; it does not reinterpret a full raw mesh or apply an unknown ROI mask.
If the actual new preparation format differs, the audit reports the mismatch
before training; it cannot be verified against your Windows folder from here.

### Reference splits and reserved test

- Preserve complete specimen-safe `split`/`partition` fields if saved in the index,
  or pass `--split-json` with `physical_ids: {train: [...], val: [...], test: [...]}`.
- Otherwise create one fixed group-stratified specimen split. With 8 specimens per
  group this is 36 train / 6 validation / 6 test specimens: 384 / 64 / 64 scans.
  Four independent partial specimens are represented in validation.
- Every scanner and timepoint from a specimen stays in one partition.
- Training seeds 42, 43 and 44 vary training, not the fixed reference split.
- Reserved test geometry is not used for QC feature extraction, baseline fitting,
  training, checkpoint selection, cross-validation, or scanner-holdout experiments.
  Geometry hashes may be read for integrity checks; reserved test features are
  opened only by explicit `--final-evaluate`.

### Main model comparison

- PointNeXt-S: official scratch encoder + trained stage head.
- TSGCNet: official geometric streams with the existing BioCal3D pred3 feature
  adapter + trained stage head. It is a stage-classification adaptation, not a
  reimplementation of the paper's segmentation evaluation.
- Sonata / Concerto: official frozen pretrained encoder + trained stage head.
- This comparison combines architecture and adaptation; do not label differences
  as architecture-only effects. No replacement network is used on dependency failure.
- CUDA is required unless `--allow-cpu` is explicitly passed.

### Nuisance and geometric baselines

Fitted independently within every reference/CV/LOSO protocol, on training rows only:

- Scanner identity only.
- Log triangle count only.
- Scanner identity + log triangle count.
- Plane-detrended geometric features: Rq, Ra, height percentiles/range, normal
  variation, area and dimensions.
- Colour-only reference: area-weighted RGB mean and standard deviation, using
  genuine saved colours. This is a simple colour-statistics baseline, not DINO.

These are fixed, balanced multinomial logistic regressions (C=1); scalers and
one-hot categories are fitted on training data. No assessment/test data fits them.
An unavailable RGB baseline is marked unavailable, not filled with fake black RGB.
Use `--controls-only` for these investigations without loading GPU models.

### Representation and adaptation investigations

On PointNeXt-S and Concerto by default, using the first training seed:

1. Area-uniform points within triangles (reference).
2. Area-weighted face centres (previous representation).
3. XYZ without normal channels (normal channels zeroed with the same architecture).
4. Voxel-averaged point density at `--surface-grid-mm .15`, refilled to fixed N.
5. Plane-aligned geometry without changing the physical height scale.
6. Geometry + true RGB, with the original small direct RGB pooling route.

TSGCNet requires native triangle streams; its reference inputs remain sampled
native faces/centres. It cannot create new triangle detail through point interpolation.
Voxel averaging is a point-encoder control and is not applied to TSGCNet.
Point density matching cannot recover details absent from a coarse scanner mesh.
`sampling_qc.csv` records unique point/face counts, effective cells and RGB status.

Also included: frozen random PointNeXt features + trained head versus the scratch
trainable PointNeXt baseline. This isolates encoder training within PointNeXt.
It does **not** isolate Concerto pretraining. There is no unverified random-weight
Sonata/Concerto construction API in this package.

`--finetune` optionally adds a Concerto low-LR encoder fine-tuning run. It uses
batch size 1, encoder LR 1e-5, and retains encoder eval-mode normalisation/dropout
while enabling gradients. Head LR remains 1e-3. This can use considerably more GPU
memory than frozen features. The official model must support backward on your machine.

Every evaluated run also measures prediction sensitivity to a 90° in-plane rotation
and a fixed translation. These are evaluation perturbations, not training augmentations.

### Robust evaluation: full suite

```bat
python biocal3d_train_3d_investigations.py --suite full --resume --live
```

The default complete full suite has **73 model jobs**:

- 12 reference model/seed runs.
- 10 representation ablations + 1 frozen-random PointNeXt control.
- 10 outer cross-validation jobs: 5 folds × 2 ablation models.
- 40 scanner-holdout jobs: 5 folds × 4 held scanners × 2 ablation models.

The ordinary investigations suite has **23 jobs**. `--suite core` has **12 jobs**.
Optional fine-tuning adds one job. Altering model/seed/fold selections changes totals;
`--plan` writes the exact queue before any model is loaded.

CV and LOSO use development specimens only. Outer assessment specimens never
select checkpoints. Each outer training pool has a separate inner validation
specimen from each group. In LOSO, the held scanner is excluded from both training
and checkpoint-selection validation; assessment uses only that scanner on disjoint
outer assessment specimens. In a completed suite each development scan is assessed
once per model/seed/protocol; repeated seeds are not counted as new scans.

CV/LOSO use the first requested training seed by default to limit the job count.
Use a different `--seeds` first value and a separate output for repeated full CV/LOSO.
The groupwise reference seed is independently controlled by `--split-seed`.

For a shorter pilot:

```bat
python biocal3d_train_3d_investigations.py --suite full --fold-indices 0 --resume --live
```

With the same output and settings, later omit `--fold-indices 0` to complete remaining
folds. A partial suite is identified by `completed_assessment_jobs` in the pooled CSV;
do not present its partial assessment as completed cross-validation.

### Reproducible focus maps and actual colour

- Fixed row RNG contexts control Python, NumPy and torch serialisation randomness
  and restore caller RNG states. The same encoder instance/state is used for cached
  and direct inference; no freshly loaded explanation encoder is substituted.
- The direct unoccluded full probability vector must reproduce the saved CSV vector
  within `--probability-tolerance .0005` and have the same predicted class.
  A failed check rejects the explanation and records the failure.
- Geometric centring and full-surface voxel origin remain fixed during occlusion.
  Removing a patch does not silently recalculate the whole surface's origin.
- Model geometry-only RGB channels remain zero, while `display_rgb` independently
  retains original sample colour. The figure shows a 2D rendering of actual scanner
  colours on the full prepared mesh, its sensitivity map and an RGB overlay.
  This is a rendered surface view, not an acquired camera photograph.
- Defaults to the same 12 validation examples across baseline models: one per
  scanner/stage. Partial examples are selected first. High-confidence mistakes are
  listed separately in `errors_val.csv`; examples do not change to favour a model.
- `--inspect-samples BCG3T1-6` makes an additional cross-scanner RGB/height QC gallery.
  If that specimen is reserved test, inspection is skipped and recorded; do not use
  the test set to debug model choices.
- Occlusion removes a 4×4 spatial patch and refills remaining samples to fixed N.
  Maps show model sensitivity, not plaque segmentation or thickness. Changing
  neighbourhoods/occupancy can still influence the result. Spatial agreement with
  plaque must be evaluated once dentist surface labels exist.

### Reports, confidence intervals and failures

`comparison.html` links to cohort QC, the exact experiment plan, controls, results,
training curves, sampling checks, scanner metrics, focus plots and failed jobs.
`comparison.csv` stores individual results. `assessment_pooled.csv` pools completed
outer-fold predictions within model/variant/seed without duplicating scans.

Per-run outputs include `best.pt`, `last.pt`, `config.json`, `history.csv`,
`predictions_val.csv`, class recall/confusion in `metrics_val.json`, per-scanner
metrics, confidence intervals, error lists and rigid-motion sensitivity.
Assessment files are separate from validation. A saved result remains exploratory
if you used its validation set for architecture selection.

Confidence intervals resample entire specimens within fresh/mature strata, preserving
all scanners and stages. They are withheld when any class has fewer than three
independent specimens. They are conditional on the fitted model/split and do not
capture retraining or model-selection uncertainty. Fresh/mature strata avoid the
degenerate bootstrap that would result from one validation specimen per G group.

Failures are recorded individually and the suite proceeds to other jobs. Missing
RGB blocks colour experiments; missing model dependencies block that model. A
nonzero exit code indicates failures; inspect `failures.json` and each `error.txt`.

## Final test evaluation

Only after choosing the model/representation using development experiments:

```bat
python biocal3d_train_3d_investigations.py --suite core --models concerto --resume --final-evaluate
```

Use the same saved output, backend/checkpoint settings, seeds and hyperparameters.
This loads existing reference best checkpoints, performs no training and writes
`metrics_test.json` / `predictions_test.csv`. Do not repeatedly select architectures
from test performance. To evaluate a chosen ablation, include the same investigations
suite/model/ablation options used for its saved run; this evaluates matching saved
reference jobs, not CV/LOSO jobs. Check the printed plan first.

## Dependencies and validation

Keep your existing working CUDA torch and official backend dependencies. Do not
replace CUDA torch with a CPU wheel. If needed, install the non-torch utilities:

```bat
python -m pip install -r requirements.txt
python -m unittest -v test_study
```

The code is syntactically checked and the automated tests exercise a synthetic
512-scan prepared snapshot: provenance errors, duplicate/split leakage, area sampling,
colour separation, native face streams, voxel controls, rigid transforms, confidence
intervals, CV/LOSO coverage, baseline fitting and report generation.

The build environment has no torch, official repositories or access to your Windows
snapshot. Actual four-backend forward/backward GPU execution has **not** been tested
here. `--preflight` must pass in your working environment before the long run.
The optional fine-tuning job additionally requires a successful real backward pass.

The source adapters retain the existing narrow TSGCNet compatibility patches. If
its official source revision changes, the adapter fails explicitly rather than
silently applying a different model.
