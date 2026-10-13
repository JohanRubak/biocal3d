# G5-only initial labeling pilot

Extract this add-on directly into your existing `biocal3d_workflow` directory.
It uses approved scans in an existing prepared snapshot and preserves the current
annotator GUI, preprocessing, meshes and existing annotations.

## Initial design

| Phase | G5 physical specimens | All-stage scans, four scanners |
| --- | --- | --- |
| Calibration | 2 | 24, discussed together |
| Independent agreement | Remaining 6 | 72 per dentist |
| Other groups, reserved for later | 40 | 416 total, with future queues prepared |

Every scanner and all three stages of a physical specimen stay in one phase.
Calibration and agreement IDs are disjoint. Bruna and Katrin receive identical
agreement cases under separate profiles; Johan has a separate experimental
profile excluded from the comparison. Scanner, group and stage remain visible.
There is no separate consensus subset.

Two calibration specimens provide two contrasting partial-stage examples plus
baseline/clean references across four scanners. Six agreement specimens preserve
most of the limited G5 pool for independent comparison. This is a manageable
methods pilot and preliminary variance estimate, **not a powered definitive
agreement study or a guarantee of a narrow confidence interval**. Complete the
pilot and inspect results before deciding how much more labeling is necessary.

The primary agreement result uses the partial stage: 6 specimens × 4 scanners =
24 paired scans. Average scanner results within each specimen, then report the
mean and uncertainty across the six independent physical specimens. All 72 scan
pairs are analyzed, with baseline/clean results secondary so easy cases do not
dominate the primary Dice summary. Small-sample intervals and the estimated
between-specimen SD remain exploratory.

## Create the pilot

From a command prompt in your existing `biocal3d_workflow` folder:

```bat
prepare_pilot_study.bat --root "C:\path\to\prepared_snapshot"
```

Use the actual `_development_TIMESTAMP` folder containing
`annotation_inventory.csv`, not the raw-data parent directory. Defaults already
select 2 calibration and 6 agreement specimens, all from G5. The plan is written
to `SNAPSHOT\labeling_g5_pilot`. The folder must be new/empty, so an existing
study is not accidentally reshuffled.

Every selected specimen needs approved baseline, partial and clean scans from all
four scanners. Missing QC coverage produces an error instead of automatically
substituting another group or reducing the agreement sample size. Counts may need
an explicit change after QC review. Selection is reproducible (seed 42).

The script cannot assess actual plaque appearance. Before the study starts,
inspect the two proposed calibration specimens: aim for one clear/patchy partial
surface and one faint/ambiguous or artifact-containing example. If the selected
pair does not cover this, choose explicit IDs using a new output folder:

```bat
prepare_pilot_study.bat --root "C:\path\to\snapshot" --output "C:\path\to\new_g5_plan" --calibration-ids G5_S1 G5_S2 --agreement-ids G5_S3 G5_S4 G5_S5 G5_S6 G5_S7 G5_S8
```

These IDs demonstrate syntax; they are not claims about your specimens' appearance.
Use IDs from your approved inventory. Two calibration specimens are a starting
allocation: if the rules remain unclear, resolve training first. Reserving three
G5 calibration specimens and five agreement specimens is possible with
`--calibration 3 --agreement 5` before freezing the pilot, acknowledging less
information for independent agreement.

The full dataset is 512 scans of 48 specimens: G1/G2 have two stages, while
G3/G4/G5/G6 have three, with eight specimens per group and four scanners. Reserving
all G5 for this pilot leaves 416 scans of 40 specimens in the future production
queues. Their launchers are prepared, but proceed with those assignments after
reviewing the pilot and deciding the next design.

## Annotate and compare

1. Open `open_calibration_bruna.bat` for joint discussion and optionally
   `open_calibration_katrin.bat` for practice. Agree on plaque/clean definitions,
   uncertain surfaces, artifacts, backside avoidance and permitted color tools.
   Calibration masks are excluded from the agreement analysis.
2. Freeze the instructions. Bruna uses `open_agreement_bruna.bat` and Katrin uses
   `open_agreement_katrin.bat`. Each labels all 72 designated scans independently,
   without seeing the other masks or discussing the agreement cases.
3. Run the plan's `analyse_agreement.bat` after both queues are complete, before
   any reconciliation. It compares Bruna and Katrin only.
4. Inspect partial-stage Dice/IoU, per-scanner behavior, uncertainty/unreviewed
   coverage, failed cases and between-specimen variation. Decide whether rules
   need revision, whether another independent pilot is needed, or whether to
   proceed to a larger standard-group agreement study and production labeling.

Generated launchers lock the chosen annotator profile. `open_experiment_johan.bat`
uses independent Johan files. Existing labels are not moved or overwritten. There
is no automatic consensus; any later reconciliation is saved separately after
scoring, preserving both independent masks.

Stage is context, not an automatic face label. Both dentists have the same
permitted metadata. High agreement alone does not establish biological accuracy.
If rules change after reviewing agreement results, keep the original pilot scores
and distinguish any later validation of the revised rules.

## G5 timing and analysis

Fill `specimen_timing_to_complete.csv` from the experimental log. Actual growth
hours are blank: only some G5 specimens matured +2h. The helper never calls every
G5 specimen late and keeps all original G5 identifiers. The pilot evaluates G5;
it does not establish agreement or variability for correctly matured G3/G4/G6.

Analysis exports `agreement_by_scan.csv`, `agreement_by_scanner.csv`,
`agreement_missing_or_failed.csv` and `agreement_summary.json` inside the plan's
`agreement_results` directory. The summary includes partial-stage and cohort
results. Reports use triangle-area-weighted Dice/IoU and agreement, nominal kappa,
reviewed coverage and uncertainty. Empty/empty plaque Dice is undefined rather
than 1; inspect undefined metrics and coverage alongside scores.

Confidence intervals bootstrap physical-specimen means. Missing/incomplete
ratings and geometry mismatches produce a nonzero exit code and a failure report.
With only six independent specimens, interpret intervals and variance estimates
as exploratory; repeated scanners, stages and mesh faces do not increase n to 72.

## Optional precision planning after the pilot

The included calculator retains the future target of a 95% confidence interval
with half-width 0.05 for mean specimen-level partial-stage Dice. Run:

```bat
plan_agreement_size.bat --from-results "C:\path\to\snapshot\labeling_g5_pilot\agreement_results\agreement_by_scan.csv" --plan "C:\path\to\snapshot\labeling_g5_pilot" --cohort G5 --output "C:\path\to\new_precision_report.json"
```

With `--plan`, the calculator now defaults to the plan's cohort (G5 here).
Without a plan, it defaults to `standard_groups`. It filters partial stage,
requires defined Dice from all specified scanners for each included specimen,
reports exclusions, and estimates SD across specimen means. Zero SD is not treated
as proof that a tiny final sample is adequate.

Planning solves `t(0.975, n-1) × SD / sqrt(n) <= 0.05` under independent,
approximately normal specimen scores. Hypothetical SDs 0.05/0.10/0.15 require
7/18/38 independent specimens (84/216/456 all-stage scans per dentist). These are
conditional scenarios, not observed BioCal3D results. A sensitivity calculation
also uses a one-sided normal-model SD upper bound; bounded/skewed Dice can violate
that model. The upper scenario uses the population SD bound 0.5 for scores in
[0,1], unless the observed sample SD is already larger. It does not guarantee
achieved precision. Review assumptions and consider simulation for a final study.

G5 variability can inform feasibility, but transferring it to G3/G4/G6 requires
justification or additional standard-group pilot data. Specify the intended
population and final sampling design. Do not repeatedly add cases until an
ordinary confidence interval first becomes narrow and then claim its usual 95%
coverage. Queue sizes are never changed automatically by the calculator.

For later explicitly requested designs, `--agreement N --agreement-g5 K` sets
agreement size and composition, with G3/G4/G6 providing N-K. The default K is N,
so changing only the count keeps a G5-only study. Always use a new output folder
and preserve the original pilot and its labels.

## Environment and verification

Batch files use `%LOCALAPPDATA%\anaconda3\envs\biocal3d\python.exe`.
Set `BIOCAL3D_STUDY_PYTHON` to your exact environment executable if needed.
Prep needs standard Python only; analysis uses existing numpy/VTK helpers and
planning uses scipy. GUI launchers use your existing
`annotator\start_annotator.bat`; supply `--app-launcher` if it is elsewhere.

Tests verify the synthetic 512-scan dataset, G5-only 24/72/416 allocation,
whole-specimen separation, identical independent queues, profile isolation,
partial-stage reporting, QC validation, reproducibility, other explicit designs,
legacy compatibility and precision aggregation. Windows batch execution and real
sample appearance/QC still need local checking.

Methodological sources:
- NIST, Confidence Limits for the Mean:
  https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
- NIST, Confidence Limits for Standard Deviation:
  https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm
- El Jurdi et al., Confidence Intervals for Performance Estimates in Brain MRI
  Segmentation: https://arxiv.org/abs/2307.10926
- GRRAS reporting checklist:
  https://www.equator-network.org/wp-content/uploads/2012/12/GRRAS-checklist-for-reporting-of-studies-of-reliability-and-agreement.pdf
