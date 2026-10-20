# BioCal3D 3D investigations v2 — approved ROIs

One package merges the relevant parts of `BioCal3D_3D_investigations_v1` and
`BioCal3D_3D_followup_v1`. It includes the training runner, official-backend
adapters, original model/geometry comparisons, controlled colour pathways,
specimen CV, scanner holdout, geometry QC, nuisance controls and reports.
It adds matched interior-ROI retraining and within-specimen stage-order checks.
The previous rotation/point-order/RGB perturbation diagnostics now operate on
newly trained checkpoints from the approved cohort.

## Installation and first run

1. Extract this folder into `biocal3d\3D branch\BioCal3D_3D_investigations_v2`.
2. Put the **whole rebuilt approved snapshot** in `biocal3d\data\Approved ROIs`.
   Directly inside that folder there must be `dataset_index.json`,
   `geometry_rebuild.json`, and the `samples` directory with each scan's
   `geometry.npz`. Keep the other rebuilt files with it. A folder containing
   only `annotations\_roi_review` is an approval record, not the rebuilt data.
3. Use your existing working CUDA `biocal3d` environment and existing
   `third_party` repositories. No training code from the old investigation
   folders is needed. Keep the benchmark's installed backends/checkpoints.
4. Open Anaconda Prompt in this extracted folder and run:

```bat
conda activate biocal3d
00_check_and_plan.cmd
01_audit_data.cmd
02_preflight.cmd
```

The launchers also attempt to activate `biocal3d` when double-clicked. They
leave the window open after completion. `00` validates approval inventory,
geometry file hashes and cohort identities, creates the split and writes plans;
it does not open mesh arrays or import Torch. `01` computes development-set
geometry/RGB QC, interior retention and cheap controls. `02` probes all four
backends and the colour/interior pathways (22 checks by default); it does not
train models. Correct any failed audit/preflight before a long run.

Then run the merged reference investigations:

```bat
03_train_original_investigations.cmd
04_train_colour_pathways.cmd
```

Open `runs\approved_rois_v2\comparison.html`. After inspecting those results,
the additional stages are:

```bat
05_train_interior_roi.cmd
06_specimen_cv.cmd
07_scanner_holdout.cmd
```

Run one launcher at a time. All stages use the same split/output. Rerunning a
launcher resumes the last saved completed epoch and reuses completed jobs.
Changing data, split, implementation or fitting/sampling settings requires a
new output directory. A machine sleeping or shutting down does not continue
GPU training.

## What is run

| Launcher | Default work | Neural training jobs |
|---|---|---:|
| `00_check_and_plan.cmd` | Validate new snapshot, create partitions and all plans | 0 |
| `01_audit_data.cmd` | Mesh/RGB QC, face-count pairs, roughness, nuisance controls, interior retention | 0 |
| `02_preflight.cmd` | 22 backend/pathway probes including interior sampling | 0 |
| `03_train_original_investigations.cmd` | Original four-model geometry study and ablations | 23 |
| `04_train_colour_pathways.cmd` | Six matched pathways × two models × three seeds | 36 |
| `05_train_interior_roi.cmd` | Three pathways × two models × three seeds on inner ROI | 18 |
| `06_specimen_cv.cmd` | Five outer specimen folds × four fixed configurations | 20 |
| `07_scanner_holdout.cmd` | Five specimen folds × four held-out scanners × four configurations | 80 |
| `08_mature_only_colour.cmd` | Optional G3–G6 comparison in a separate output folder | 36 |
| `09_refresh_report.cmd` | Refresh all summaries/HTML without Torch or input data | 0 |
| `10_export_review.cmd` | Small review ZIP excluding weights and caches | 0 |
| `11_run_software_tests.cmd` | Synthetic regression tests | 0 |

The full default sequence `03`–`07` contains **177 training jobs**. These are
separate stages so you can inspect the reference comparison before committing
time to the longer protocols. An optional Concerto fine-tuning reference job
is retained via `03_train_original_investigations.cmd --finetune`.

## Cohort and partitions

The expected current cohort is 512 scans from **48 physical specimens**: six
groups × eight specimens; four scanners; G1–G2 have T0/T1 and G3–G6 have
T0/T1/T2. For the default grouped seed-42 split, that gives:

| Partition | Scans | Physical specimens | Specimens with a partial stage |
|---|---:|---:|---:|
| Train | 384 | 36 | 24 |
| Validation | 64 | 6 | 4 |
| Reserved test | 64 | 6 | 4 |

All timepoints and scanners for a specimen stay together. A new split file
is created from this cohort; no old results are required. The same seed and
algorithm can reproduce an earlier specimen assignment. Source partitions
copied during the ROI rebuild are not implicitly inherited. To preserve a
complete saved source split, pass `--split-policy source` consistently to
every stage; alternatively pass `--split-json "...\reference_split.json"`.
This matters for a strictly paired comparison with previous results.

Reserved-test files are hashed to establish dataset identity; their mesh
arrays are not opened for QC, feature extraction, model selection or prediction.
The package deliberately has no final-test launcher. CV and scanner holdout
use only the 448 development scans. Outer-assessment specimens are excluded
from training and inner validation at **every** scanner. Scanner holdout
additionally excludes its scanner from both training and inner validation.
OOF summaries mark partial fold suites incomplete and reject duplicate scans.

Only rebuilt, approved, complete, hash-consistent snapshots are accepted.
No silent scan deletion or raw-data fallback is performed. Moving a snapshot
is supported because geometry comes from relative `sample_dir` paths and RGB
is embedded in `geometry.npz`. Old absolute source PLY/JPG paths are provenance
only and need not exist on the new PC.

## Preserved experiments

The original reference suite trains PointNeXt-S and TSGCNet from scratch and
uses frozen pretrained Sonata/Concerto plus a stage head. It includes area
sampling, native face centers, XYZ-only inputs, voxel averaging, plane
alignment, RGB and a frozen random PointNeXt control. The preprocessing
defaults remain 2,048 inputs, 10 mm scale, and .01 scaled voxel size.

The controlled colour study uses `geometry_matched`, `stats_only`,
`rgb_encoder_only`, `rgb_combined`, `rgb_shuffled`, and `gray_combined`.
PointNeXt/Concerto pathways share encoder input width and head architecture
within each backbone/seed. Absent pathways are zeroed. Colour shuffling is
deterministic within each scan and applied during fitting and evaluation.
Statistics-only freezes/does not use encoder features, but still constructs
the official encoder to match the head width. It therefore needs its backend.
The direct colour path contains sampled RGB means/maxima, as in v1.

CV and scanner holdout retain the four prespecified follow-up configurations:
PointNeXt `rgb_combined`, PointNeXt `stats_only`, Concerto `rgb_combined`, and
Sonata `legacy_geometry`. Add `--include-concerto-stats` for a fifth
configuration (CV 25 jobs; scanner holdout 100). Validation reference results
select checkpoints and are not a final independent accuracy estimate.

Nuisance controls include scanner identity, face count, their combination,
roughness and global colour statistics; preprocessing and classifiers are
fitted on each protocol's training subjects. Mature-only runs check the
confounding between partial stage and G3–G6. The original geometry heads
and new matched controls have different widths and are labelled separately.

## New investigations

**Interior ROI:** train `geometry_matched`, `stats_only` and `rgb_combined`
for PointNeXt/Concerto at seeds 42/43/44. All triangle vertices must lie
inside 80% of the approved extraction radius. The circle uses the stored
approved center/tilt, preserving source face indices and the original mesh.
Sampling point count, normals and centering convention remain matched to
the full surface; normals/centering are derived from the approved full mesh.
This is an interior-sampling experiment, not complete removal of every
geometric influence of the annulus. `qc\interior_roi.csv` reports retained
face counts/area. `interior_comparison.csv` pairs independent full/interior
fits by model, pathway and seed; it remains explicitly unmatched if the
corresponding full fit has not completed. Existing full-ROI nuisance controls
are not refitted or presented as interior controls.

Removing the outer ring removes possible artifacts **and real plaque**.
Accuracy changes alone cannot establish which explains the effect. This
experiment measures dependence on the annulus for acquisition-stage
classification; it is not a segmentation test or a remeshing experiment.

**Stage order:** from each completed run's validation/outer predictions,
calculate `P(partial) + 2*P(clean)`. Compare baseline→partial, partial→clean
and baseline→clean for the same specimen/scanner when available. Export
individual pairs and reversal rates. Scanner pairs are averaged within
physical specimen before reporting rates; folds/seeds remain separate.
This is a prediction-consistency diagnostic, not measured plaque coverage
or removal percentage.

**Input sensitivity:** retain rotation/translation checks for each fit and add
180°/270° rotation, point ordering, RGB shuffling and grayscale inference for
first-seed reference RGB fits. These inference shifts complement the retrained
colour pathways and are summarized in `input_sensitivity.csv`.

**Focus figures:** first-seed reference geometry and RGB runs export actual
scanner colour mesh renderings next to occlusion-sensitivity maps and an
overlay, plus up to six high-confidence errors per model. Focus is for the predicted class, which can be wrong; it does not
replace plaque annotation.

## Paths and shorter runs

The default input is detected relative to the nearest repository parent
with a `data` directory. No username or old snapshot path is used by the
merged entry point. All launchers forward extra arguments:

```bat
00_check_and_plan.cmd --data-root "D:\repos\biocal3d\data\Approved ROIs_v2"
03_train_original_investigations.cmd --backends-root "D:\path\to\third_party"
04_train_colour_pathways.cmd --models pointnext_s
06_specimen_cv.cmd --fold-indices 0
07_scanner_holdout.cmd --fold-indices 0
```

For custom input/output paths used repeatedly, set these in Anaconda Prompt:

```bat
set "BIOCAL_DATA=D:\repos\biocal3d\data\Approved ROIs_v2"
set "BIOCAL_OUTPUT=D:\biocal_runs\approved_rois_v2"
```

Or pass `--data-root`/`--output` to every launcher. Model/fold subsets and
seeds may be added later without changing the experiment settings. Keep
`--epochs`, learning rates, point count, split policy/seed and other fitting
settings consistent throughout one output folder. A low-epoch pilot must use
a separate output. `--interior-fraction .7` similarly needs a new output.
To check all merged CLI options, run `python followup.py --help`.

Mature-only colour has a separate default output:
`runs\approved_rois_v2_mature`. Mature CV/holdout are available using
`06_specimen_cv.cmd --mature-only` and `07_scanner_holdout.cmd --mature-only`.
Refresh/report/export the same output using `--mature-only` or explicit
`--output` as appropriate.

## Outputs and restart behaviour

Start with `runs\approved_rois_v2\comparison.html`. It links to the detailed
colour/CV/holdout report and shows original model comparisons, interior
differences, stage-order diagnostics and actual-colour/focus figures.
Reports update after each completed/failed job. Histories, predictions,
checkpoints and per-run plots remain under `runs\<job>`. The original live
epoch plot/report option `--live` is retained for additional training feedback.
`reference_split.json`, `scan_manifest.csv`, `plan_*.csv` and `partitions\`
make each protocol inspectable before training. Caches use the new geometry
hashes/configuration; old caches/checkpoints are not copied or reused.

An output lock blocks two launchers writing into the same output. After a
forced process kill/power loss, confirm Python training has stopped and delete
`runs\approved_rois_v2\.active_run.lock` before resuming. Ordinary errors and
Ctrl+C release the lock. Epoch checkpoints permit resuming completed epochs;
an interrupted partially processed epoch is repeated.

`10_export_review.cmd` prints the path of a small ZIP containing reports,
metrics and images, omitting weights and surface/feature caches. Upload that
ZIP for the next review. Use `--no-images` for a smaller numeric-only bundle.

## Software and validation

Use the installed working CUDA environment. These launchers do not install
or replace Torch, CUDA, backend repositories or pretrained weights. Missing
CPU utilities are listed in `requirements.txt`. The old checkpoint-reproduction
diagnostic was intentionally left out: old predictions/checkpoints correspond
to different ROI geometry and cannot be interpreted as new-data reruns.

Run `11_run_software_tests.cmd` for regression tests. See `VALIDATION.json`
for exactly what was executed during packaging. Actual neural training and
Windows launcher execution require your training PC; neither a GPU nor your
actual approved dataset was available in the delivery environment.
