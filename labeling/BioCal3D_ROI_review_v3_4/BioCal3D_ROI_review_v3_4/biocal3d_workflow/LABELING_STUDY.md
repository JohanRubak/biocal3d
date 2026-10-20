<<<<<<< HEAD
# Two-dentist calibration, agreement and consensus workflow

Use your **existing prepared snapshot**. No ROI extraction or PLY conversion
needs to be rerun. The scripts use its approved `annotation_inventory.csv`,
with `roi.ply` when available and `roi.vtp` for older snapshots. Queue files
select cases from the same frozen geometry; they do not move/copy/reorder meshes.
Existing NPZ annotations remain attached to their original input files.

## 1. Generate the plan once

Double-click `create_labeling_plan.bat`. It activates `biocal3d` and uses the
last prepared snapshot (or the latest discoverable development snapshot).
For an explicit existing snapshot and customized subset sizes:

```bat
create_labeling_plan.bat --root "C:\path\to\_development_TIMESTAMP" --calibration 4 --agreement 8 --seed 42
```

Counts mean **physical specimens**, not individual scans. All available
scanners and time points for a specimen stay in one study subset. By default,
75% of selected calibration/agreement specimens are drawn from those with
partial-stage scans, when available. Remaining selections supply control
specimens; selection is balanced across group/split strata. The sizes and
partial fraction can be changed before starting labels:

```bat
create_labeling_plan.bat --root "C:\path\to\snapshot" --calibration 2 --agreement 4 --partial-fraction 0.75
```

The plan directory must be new/empty, preventing silent reshuffling of an
existing study. Inspect `labeling_plan\assignments.csv` and `plan.json`, then
freeze the plan. Each dentist is assigned approximately half the remaining
physical specimens; within that constraint the allocation balances estimated
labeling workload (partial-stage scans weight 3, other stages weight 1).

| Subset | Dentist 1 | Dentist 2 | Purpose |
| --- | --- | --- | --- |
| Calibration | Same selected cases | Same selected cases | Discuss definitions/tools; excluded from primary agreement analysis |
| Agreement | Same overlap cases | Same overlap cases | Independent ratings before comparison or consensus |
| Production | Assigned half | Other assigned half | Label the remaining dataset once |
| Consensus | Joint reconciliation | Joint reconciliation | Separate final labels for agreement cases after both ratings are complete |

The combined `all_dentist_1.csv` and `all_dentist_2.csv` include each dentist's
calibration, agreement and production assignments. Together they cover all
approved scans; their overlap is exactly calibration + agreement. The existing
model train/validation/test split is retained as metadata; this annotation
allocation does not redefine those splits.

## 2. Calibration

In `SNAPSHOT\labeling_plan`, open:

- `open_calibration_dentist_1.bat`
- `open_calibration_dentist_2.bat`

Agree on plaque/clean boundaries and uncertain regions using these cases.
Calibration can involve joint discussion. Review samples across scanners,
including low color contrast, boundaries and missing surface. Do not infer a
surface label solely from baseline/partial/clean stage metadata.

## 3. Independent agreement labels

Use `open_agreement_dentist_1.bat` and `open_agreement_dentist_2.bat`.
Each saves to its own `SNAPSHOT\annotations\dentist_1` or `dentist_2` tree.
The study launchers lock the annotator dropdown to reduce accidental viewing
or editing of the other rating. Keep the ratings independent until **both are
complete**. This is a workflow control, not filesystem access security; use
separate copies/accounts or operational separation if needed for blinding.

The ordinary `start_labeling.bat` remains a general editor with a dropdown
and New annotator action. Do not use that to inspect the other dentist's
agreement labels during independent rating. Normal number shortcuts and all
view controls remain available in the study queues.

**Auto-save PLY too** is on by default. Every draft/save writes a matched
binary labeled PLY and authoritative NPZ, including original scanner color.
Only **Save & Next** marks the NPZ complete. Drafts retain -2 for unreviewed
faces and -1 for uncertainty. A PLY alone is not evidence of completion.

## 4. Measure agreement before consensus

Run:

```bat
analyse_agreement.bat --root "C:\path\to\snapshot"
```

Or double-click it when the last snapshot is the intended dataset.
Outputs go to `labeling_plan\agreement_results`:

- `agreement_by_scan.csv`: plaque Dice/IoU, area agreement, area-weighted
  nominal kappa, jointly reviewed coverage, both raters' plaque burden,
  uncertainty and unreviewed fractions.
- `agreement_by_scanner.csv`: scanner summaries.
- `agreement_summary.json`: all-stage and partial-stage summaries, with
  means and 95% confidence intervals from resampling physical specimens.
- `agreement_missing_or_failed.csv`: cases lacking two completed ratings
  or having a geometry/file mismatch. A nonzero exit code highlights these.

Only the designated **agreement subset** is used; calibration, production
and consensus are excluded. Metrics compare clean/plaque on jointly reviewed
ROI faces, weighted by triangle area. Coverage and excluded fractions must
be considered alongside agreement: uncertainty must not be used to hide
hard regions. Both-empty plaque Dice/IoU are undefined and reported as such,
with a `both_empty_plaque` indicator; clean-clean agreement is still measurable.
Kappa is undefined when the marginal class distribution is degenerate.

Summaries first average defined scan metrics within a physical specimen,
then average specimen means. Confidence intervals resample those specimen
means, keeping scanner/time-point dependence within a specimen. They are
pilot estimates, not confidence from millions of independently sampled mesh
faces. Partial-stage results are reported separately because full/clean
cases can be less informative about boundary agreement.

## 5. Consensus without overwriting independent ratings

After both dentists have completed the overlap queue:

```bat
seed_consensus.bat --root "C:\path\to\snapshot"
```

This creates **consensus drafts** seeded from dentist_1 for convenience.
They are not automatically accepted ground truth. The command requires both
independent ratings to be complete and keeps any existing consensus files.
Original dentist_1/dentist_2 files are not modified.

Open `labeling_plan\open_consensus.bat`. The dropdown lets you compare the
independent masks in the same view. Dentist_1/2 annotations are read-only
in this review mode; select **consensus** to edit the reconciled mask.
Discuss disagreements with both dentists, document uncertain decisions in
notes, then use Save & Next to record completion. Neither averaging, union,
nor intersection of two masks is automatically declared consensus.

## 6. Production labeling

Use `open_production_dentist_1.bat` and `open_production_dentist_2.bat` for
the disjoint remainder. The `open_all_dentist_*.bat` launchers are available
for reviewing each person's full assignment, but keep the calibration →
independent agreement → consensus order for the pilot.

## Batch export

The annotator's **Export all…** button exports completed annotations for the
current queue and annotator. Choose an export folder outside the working
`annotations` folder. Each bundle contains original-color PLY, matching NPZ,
and a copied texture when applicable, plus a manifest/summary.

For all dentists and consensus at once:

```bat
python annotator\export_annotations.py --root "C:\path\to\snapshot" --dest "C:\path\to\label_export" --annotators dentist_1 dentist_2 consensus
```

Incomplete drafts are skipped by default. `--include-drafts` explicitly exports
them with NPZ completion status retained. Keep working NPZs: they preserve
source-relative paths, geometry fingerprints, labels, rater, notes, timestamp
and completion. Exports are portable snapshots; their manifest pairs PLY/NPZ
files. A consumer can check the labeled PLY geometry against the NPZ fingerprint.

## Updated view controls

Brightness, contrast, gamma and saturation adjust display only, for both
vertex RGB and UV textures. **Reset colors** restores the scanner appearance.
Exports always preserve source RGB/texture; adjustments are recorded in the
annotation's view-state metadata for audit, not applied as scanner calibration.

The 3D TOP axis and green TOP / orange SIDE / red BOTTOM status follow camera
rotation. **Paint from top only** blocks brushing from bottom and near edge-on
views, including the apparent reverse side of a zero-thickness surface.
**R** returns to the top. Prepared canonical exports use reference +Z;
otherwise direction is inferred from plane/mesh normals. Inspect it and use
**Flip top/bottom** if needed; that correction is saved per sample and shared
across annotators. The guard classifies the camera hemisphere, not anatomical
plaque or top/bottom eligibility of individual triangles. It preserves the
existing ROI mask; inspect unusual/closed meshes before assigning labels.

## Verification and local limits

Synthetic tests cover grouping, no production overlap, reproducible assignment,
consensus readiness/non-overwrite, agreement metrics, batch paired exports,
source color preservation, rater switching, bottom-view guard and read-only
consensus review. Windows launchers and your real scanner dataset still need
local validation. The workspace could not access your C: drive, so no actual
specimen assignments or labeling agreement results are included in this ZIP.
=======
# G5 calibration and agreement with approved ROIs

Apply update v3.9 to the existing app, then use the successfully rebuilt approved
snapshot as input. Do not use the original `_development_...` snapshot. The builder
requires a completed `geometry_rebuild.json`, rejects an incomplete rebuild, and
checks that each selected mesh still matches its approved review. A geometry edit
after rebuilding requires another rebuild first.

## Create the portable folders

From the app folder containing `start_labelling.bat`, run this in Windows CMD.
Replace the input path if the successful rebuild has a different name:

```bat
create_labeling_plan.bat --root "C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected\Approved ROIs_v2" --output "C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected\G5_labeling_study_approved"
```

Or double-click `create_labeling_plan.bat` and paste the rebuilt snapshot path.
With no `--output`, the default is `G5_labeling_study` beside that snapshot.
**Existing nonempty study folders are never overwritten.** Use a new name if an
older G5 study already exists. Your original snapshot, reviews and labels are
not changed. No preprocessing or ROI review needs to be repeated.

The launcher in `biocal3d_workflow` works too. The earlier helper name
`prepare_pilot_study.bat` is provided there as an alias. All use the app's existing
`.labeling-env`; no extra environment is needed.

| Folder | Physical specimens | Scans with four scanners and T0/T1/T2 |
| --- | --- | --- |
| calibration | 2 from G5 | 24 |
| agreement | Remaining 6 from G5 | 72 |

All scanners/timepoints for a physical specimen stay together. Calibration and
agreement specimens do not overlap. Selection uses seed 42; assignments and input
hashes are recorded in `assignments.csv` and `study_plan.json`. The command prints
the actual counts and blocks selected specimens with scans missing from the
annotation inventory rather than silently dropping those scans. It does not
create production assignments. Existing train/validation/test metadata stays
attached to the scans; these study subsets do not redefine model splits.

Each subset has `samples`, `annotations/_roi_review`, `annotations/bruna`,
`annotations/katrin`, manifests, and a short README. Meshes, RGB/UV textures,
geometry/maps and shared ROI approval records are copied. The ROI geometry and
review versions stay unchanged; scanner textures resolve locally after moving
the folder. Prior plaque labels, rater exports and caches are excluded.
The original full scanner models are not bundled: ROI labeling is portable,
while restoring additional raw surface would require access to the raw models.

## What Bruna and Katrin open

1. Put the complete new study folder on the shared drive.
2. Start the normal annotator with `start_labelling.bat`.
3. Use **Open folder** and select the `calibration` folder itself.
4. Choose their own annotator name, **bruna** or **katrin**.
5. Label calibration scans, discuss boundaries and uncertain regions, and agree
   on the approach before starting the agreement set.
6. Open the `agreement` folder and label independently under the same personal
   name. Do not inspect or edit the other dentist's agreement ratings.
7. Use **Approve Label & Next** when each scan is complete. Drafts autosave,
   but only approval marks the label complete. ROI badges should already say
   approved; do not change the frozen ROI geometry during the study.

No per-dentist BAT files are required. Autosaved NPZ and both PLY exports go into
the person's `annotations/NAME` tree. Keep the entire `annotations` tree when
moving or returning a subset. Both dentists must use exactly the same new study
inputs, and should not label copies from the older study. Selecting a personal
name is a workflow convention, not filesystem blinding.

## Agreement after both have finished

From the app root, use its existing labeling Python and the **agreement** folder
as the dataset root. Adjust the path if the study was moved to a shared drive:

```bat
.labeling-env\python.exe biocal3d_workflow\analyse_labeling_agreement.py --root "C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected\G5_labeling_study_approved\agreement" --rater-1 bruna --rater-2 katrin
```

The supplied `agreement/labeling_plan` is recognized even after moving the folder.
Results go to `agreement/labeling_plan/agreement_results`: scan and scanner
metrics, specimen-level summaries and bootstrap confidence intervals, and an
incomplete/failed list. Only agreement cases are included; calibration is
excluded. Both-empty plaque Dice/IoU are undefined, and uncertainty/coverage are
reported separately. Partial-stage results are reported separately. The six
specimens form a pilot; individual faces are not treated as independent samples.
Preserve both independent ratings before any subsequent consensus work.

## Other existing planning mode

The queue-only planner remains available for the broader calibration/agreement/
production workflow via `python biocal3d_workflow/labeling_study.py plan --root ...`.
The `plan` command still has its original all-inventory defaults (4 calibration,
8 agreement specimens). The standard `create_labeling_plan` launcher now calls
**folders** with the G5 pilot defaults (2 and 6). To customize the portable builder,
pass `--group`, `--calibration`, `--agreement`, `--seed`, or `--output` to that launcher.

## Validation limits

Automated synthetic tests cover 96 G5 scans across four scanners and three
stages, disjoint specimen selection, approvals, exact geometry/source IDs/RGB/UV,
byte-preserved review payloads, empty plaque labels despite preexisting ratings,
folder relocation with the source unavailable, independent saves and the existing
agreement analysis. Incomplete/pending/changed inputs and existing study outputs
are rejected. Your actual dataset and Windows shell are not available here;
real file paths, filesystem permissions and the expected counts must be checked
when you run the command.
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
