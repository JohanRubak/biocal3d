# Portable G5 pilot folders for a shared drive

Extract all files in this add-on into your existing `biocal3d_workflow` directory.
The current annotator GUI and preprocessing are preserved. The updated helper
creates actual copies of the selected ROI data in calibration/agreement folders.
There are no profile-specific GUI launchers; each dentist uses the normal
annotator's Open folder button and selects their own name.

## Create the data folders once

In a command prompt opened in your `biocal3d_workflow` directory:

```bat
prepare_pilot_study.bat --root "C:\path\to\prepared_snapshot"
```

Use the real `_development_TIMESTAMP` snapshot containing
`annotation_inventory.csv`, not the raw-data parent folder. The default output
is `G5_labeling_study` beside that prepared snapshot. The helper prints its exact
path. To choose a local or shared-drive destination directly:

```bat
prepare_pilot_study.bat --root "C:\path\to\prepared_snapshot" --output "Z:\BioCal3D\G5_labeling_study"
```

Replace Z: with your real shared-drive path. The destination must not already
exist; the helper will not overwrite existing data or labels. It copies and
verifies scans before publishing the completed study folder. Source files remain
unchanged, and existing source annotation files are not copied.

If `SNAPSHOT\labeling_g5_pilot\plan.json` exists, its selected physical IDs are
reused automatically. This preserves the selection made by the previous helper.
For a plan stored elsewhere, supply `--plan "C:\path\to\existing_plan"`.
The inventory must still match that plan. Without an existing plan, the default
is two G5 specimens for calibration and the other six for agreement, with seed 42.

## Contents and use

| Folder | Default contents | How to use |
| --- | --- | --- |
| `calibration` | 2 G5 specimens × 3 stages × 4 scanners = 24 scans | Discuss/practice together |
| `agreement` | Remaining 6 G5 specimens × 3 stages × 4 scanners = 72 scans | Bruna and Katrin label independently |

Each phase folder contains its own `samples` tree, textures, orientation geometry,
QC/metadata where available, inventory, plan and `annotations` directory. Only one
ROI mesh is copied per scan; alternate PLY/VTP versions are not duplicated. Vertex
colors, texture coordinates, ROI masks and face order are preserved. Texture
references are local, so these folders work after moving to a different drive or
computer, with the original snapshot unavailable.

1. Copy the complete `G5_labeling_study` folder to the shared drive, if it was not
   created there directly.
2. Start your usual annotator normally.
3. Click **Open folder** and select `G5_labeling_study\calibration`.
4. Select **bruna** or **katrin** in the annotator dropdown. Both names, and the
   separate experimental **johan** profile, are predefined.
5. After calibration and freezing the rules, open
   `G5_labeling_study\agreement` and again confirm your own name.

Select the phase folder itself, not its `samples` child or the parent study folder.
The annotator stores each person's files in that phase's `annotations\NAME`
directory, with matching relative sample paths. Keep the phase folder and its
`annotations` together when copying it. Each rater's work is separate, even on the
same shared data folder. Complete agreement independently without inspecting the
other profile's masks. Johan's experimental labels are excluded from comparison.

The `selection.csv` and `study_plan.json` document the allocation. The remaining
416 scans of 40 physical specimens are listed in `reserved_for_later.csv`; they
are not copied or assigned for immediate labeling in the portable folders.

## Calibration and pilot interpretation

Use contrasting calibration examples: a clear/patchy partial surface and one with
fainter boundaries or artifacts, plus the baseline/clean references. The helper
cannot judge their visual difficulty, so review the chosen pair before starting.
Every stage and scanner of a physical specimen belongs to one phase, and
calibration/agreement IDs are disjoint.

For explicit IDs in a new plan, omit `--plan` and pass counts/IDs:

```bat
prepare_pilot_study.bat --root "C:\path\to\snapshot" --output "C:\path\to\new_study" --calibration-ids G5_S1 G5_S2 --agreement-ids G5_S3 G5_S4 G5_S5 G5_S6 G5_S7 G5_S8
```

These IDs are syntax examples, not claims about your specimens' appearance.
Changing the allocation before freezing it is possible with
`--calibration 3 --agreement 5`, for example. All agreement specimens remain G5
unless `--agreement-g5 K` explicitly requests a mixed design. Selected specimens
need approved baseline, partial and clean scans from all four scanners; QC
exclusions cause a clear error rather than silent substitution from other groups.

The initial 2/6 allocation is a manageable methods pilot, not a powered final
agreement study or a guarantee of narrow confidence intervals. Its primary
analysis uses 24 partial-stage paired scans from **six independent specimens**,
averaging scanners within each specimen. Baseline/clean results are secondary so
easy surfaces do not dominate the main Dice estimate. Review agreement, confidence
intervals, coverage and variation before deciding the next phase.

Scanner/group/stage remain visible. Stage is context, not an automatic face label.
Freeze the labeling rules after calibration. Score agreement before any optional
reconciliation; preserve the original independent masks. A separate consensus
subset is not needed. If the rules change after the pilot, distinguish validation
of the new rules from the initial scores.

## Analyze the shared-drive agreement folder

After both dentists complete the agreement queue, run this from your
`biocal3d_workflow` program directory:

```bat
analyse_shared_agreement.bat --root "Z:\BioCal3D\G5_labeling_study\agreement"
```

The analyser automatically finds the portable plan inside the agreement folder
and reads its `annotations\bruna` and `annotations\katrin` files. It does not need
the original prepared snapshot or its absolute path. Results are written to
`agreement\agreement_results`. Calibration and Johan are excluded. Labels are
checked against the copied meshes' geometry fingerprints.

Reports include area-weighted Dice/IoU, area agreement, nominal kappa,
reviewed/uncertain coverage, per-scanner results and specimen-level bootstrap
confidence intervals. Empty/empty plaque Dice is undefined, not automatically 1.
Missing/incomplete or mismatched labels are reported and produce a nonzero exit
code. Interpret six-specimen intervals and variance estimates as exploratory.

Both batch files explicitly use
`%LOCALAPPDATA%\anaconda3\envs\biocal3d\python.exe`.
Set `BIOCAL3D_STUDY_PYTHON` to the correct executable if installed elsewhere.
The folder exporter uses the existing numpy, VTK and plyfile dependencies; it does
not start PySide/Qt or require an annotator launcher. The GUI remains your existing
working installation.

## Timing and optional future sample-size planning

Fill `specimen_timing_to_complete.csv` from the experiment log: only some G5
specimens matured +2h, so actual hours are blank and no specimen is automatically
marked 26h. G5 is retained as G5. This pilot cannot certify agreement on correctly
matured G3/G4/G6 alone.

The included calculator defaults to G5 when supplied with this pilot's plan:

```bat
plan_agreement_size.bat --from-results "Z:\BioCal3D\G5_labeling_study\agreement\agreement_results\agreement_by_scan.csv" --plan "Z:\BioCal3D\G5_labeling_study\agreement" --cohort G5 --output "C:\path\to\new_precision_report.json"
```

It estimates SD across partial-stage specimen means, not scan/face counts. The
future precision target remains a 95% CI half-width 0.05. Conditional Student-t
planning solves `t(0.975,n-1) × SD/sqrt(n) <= 0.05`; hypothetical SDs
0.05/0.10/0.15 require 7/18/38 specimens. These are not measured BioCal3D values.
The normal-model SD upper sensitivity scenario is not a distribution-free
precision guarantee. Small or skewed pilots need cautious interpretation, and
transferring G5 variance to standard groups requires justification. Zero pilot SD
does not certify a tiny sample size. Planning never modifies the data folders.

## Validation

Automated tests cover G5-only selection, reuse of an existing plan, whole-specimen
separation, unchanged originals, label/profile isolation, folder discovery,
textures and orientation for PLY/VTP/NPZ, removal of original source files,
relocation to a different directory and agreement analysis after relocation.
Legacy analysis and the precision calculator remain tested. Windows batch launch
and actual shared-drive permissions still require local verification.

Methodological references:
- https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
- https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm
- https://arxiv.org/abs/2307.10926
- https://www.equator-network.org/wp-content/uploads/2012/12/GRRAS-checklist-for-reporting-of-studies-of-reliability-and-agreement.pdf
