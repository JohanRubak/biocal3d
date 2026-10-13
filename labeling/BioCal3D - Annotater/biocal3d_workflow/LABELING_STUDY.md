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
