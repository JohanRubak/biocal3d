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
