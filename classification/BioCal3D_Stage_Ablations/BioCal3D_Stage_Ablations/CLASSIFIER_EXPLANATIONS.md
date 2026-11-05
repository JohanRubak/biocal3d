# What drives the actual stage-classifier decision?

Use `explain_stage_classifier.py` for this question. The PCA/similarity inspector
does not use logistic-regression weights. Its maps characterize the representation
and user-selected similarities, not the fitted stage decision. Earlier
`make_stage_heatmaps.py` maps already used the head weights; the new script adds
ranked channel contributions, feature distributions, deletion controls, and
optional interventions that rerun the encoder.

## The classifier being explained

For a scan, the existing pipeline concatenates mean and max pooled local features,
standardizes that vector using development training means/scales, and applies the
saved logistic-regression head. The script loads that exact saved pipeline.

For target class A versus comparison B, the margin is:

`margin = (intercept_A - intercept_B) + sum_j ((weight_Aj - weight_Bj) * standardized_feature_j)`

This margin equals `log(P(A)/P(B))` for the multinomial logistic model. Positive
means preference for A over B, even when a third class has the highest probability.
An edit can reduce this pairwise margin while increasing P(A) if it also changes
the third class; inspect the exported probabilities as well as the contrast.
The feature contribution is the product of its standardized value and the
coefficient difference. Coefficient magnitude alone does not explain an individual
scan. All terms plus the intercept are checked against `decision_function`.

PCA instead finds feature directions with the largest variation. A large source
of variation can be irrelevant to stage; a smaller one can be discriminative.
This is why visually striking rim/PCA patterns do not establish the classifier's
decision mechanism. A latent channel is not automatically a named biological cue.

## Install and start with the cached analysis

Copy `explain_stage_classifier.py` and this guide into your existing stage-ablation
folder. It also uses the included `make_feature_pseudomasks.py` and
`make_stage_heatmaps.py` for validated rendering/projection helpers. Keep these
files beside `stage_ablation.py`, `models_2d.py`, and the other original modules.

From your existing folder:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --scan iTERO__BCG4T1-2 --target partial --against clean --out runs/stage_pooled_v3/explain_partial_vs_clean_v1
```

It processes DINOv3 RGB and Concerto RGB together from cached features without
loading the encoders. Use `--modes dinov3_rgb dinov3_gray concerto_geometry concerto_rgb`
for all four conditions, or select a subset. `--grid 8` is the default; output
directories must be new. The local `--spatial` caches are required.

By default the script reads heads from
`locked_test/all_groups_three_stage/*_all_development_head.joblib`. If only the
development fit exists, use `--head-source development_cv`. Both locations contain
heads fitted on all development specimens, not fold-held-out heads. A development
scan explanation is therefore in-sample for this head. The figures/metadata mark
this. Use a test scan ID to investigate a held-out specimen's actual decision.
No model is retrained and no class label is used to fit a new explainer.

## More examples: batch selection

Start with up to 12 partial-stage **test scans**, rotating across available scanners:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --partition test --stage partial --against clean --limit 12 --out runs/stage_pooled_v3/explain_test_partial_batch_v1
```

Open `index.html` in the output folder to browse the figures, filter by scanner or
recorded stage, and open the enhanced mean/max views. `batch_decisions.csv`
combines the decisions/margins from every successful scan/model. Each scan has
its own subfolder. A single explicitly selected scan retains the original flat
output layout. Batch size counts scans, not independent specimens.

If local test features are missing, repeat your original extraction with
`--partition test --spatial`. To inspect existing development spatial features
instead, change `--partition test` to `--partition development`; those figures
are marked as head-training specimens.

Other selections:

```bat
REM Up to 12 test scans across all stages and scanners
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --partition test --limit 12 --out runs/stage_pooled_v3/explain_test_stages_v1

REM All stages/scanner views of the development specimen from your example
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --partition development --specimen BCG4-2 --all --out runs/stage_pooled_v3/explain_specimen_BCG4_2_v1

REM All partial-stage test scans from two scanners
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --partition test --stage partial --scanner TRIOS5 iTERO --all --out runs/stage_pooled_v3/explain_trios5_itero_v1

REM List available IDs without loading features or writing results
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py cached --run runs/stage_pooled_v3 --partition test --list-scans
```

You can also supply several exact IDs after `--scan`. Without explicit IDs,
selection defaults to test, with up to 12 scans unless `--all` is used. Explicit
IDs retain their supplied order and may select either partition. Filters that
conflict with explicit IDs are rejected rather than silently omitting requested
scans. Input-occlusion mode accepts the same batch selectors but requires separate
encoder inference per scan and is slower than cached analysis.

The default target is each scan's recorded stage. With no explicit `--against`,
baseline and partial targets are compared with clean; clean targets are compared
with partial. This keeps mixed-stage batches valid. For a fixed comparison use
`--target partial --against clean` and, normally, `--stage partial`. The chosen
contrast is recorded in every figure and CSV.

If one scan fails, other selected scans are still processed. `failures.csv` lists
errors and the command returns a failure code after saving successful results.
Choose a new output directory for another batch.

## What to open

### 1. `classifier_evidence.png`

Per model: original, combined exact spatial allocation, mean contribution,
max-pooling winner allocation, and the effect of deleting cached regions and
repooling through the same saved head. Red supports the target-vs-comparison
contrast; blue supports the opposite direction. Colourbars use logit units.

The mean and max maps share a scale within a model so sparse max contributions
can be compared with distributed mean contributions. Allocation maps are centered
at the development scaler means; their sum plus intercept reproduces the margin.
Spatial allocation is a convention for tracing the pooled score, not a unique
causal attribution to raw input locations. Max terms are assigned to their winning
locations and split across ties, which can naturally make hotspot patterns.

### Enhanced contrast: `<mode>/allocation_detail.png`

This additional figure displays combined, mean and max allocations with a separate
99th-percentile magnitude scale for each panel, a grayscale underlay and stronger
overlay opacity. It makes weak mean-pooling spatial structure easier to see. The
underlying values, model, predictions and CSV contributions are unchanged.

Each title reports the sum of native contributions; each colourbar shows the raw
numeric range. A bright red mean map in this separate-scale view is not necessarily
as large a contribution as a bright red max map. The original shared-scale overview
remains available for magnitude comparisons. Net summed contributions can cancel
between positive and negative regions, so a small sum alone does not imply an
uninformative branch.

### 2. `<mode>/decision_features.png`

Left: the largest positive/negative feature-channel contributions. The title
reports the intercept, all omitted contributions and total margin. Labels distinguish
mean versus max pooling and zero-based latent channel IDs.

Right: remove locations with the largest positive allocated contributions and
compare with random location removal. Features of the remaining locations are
held fixed; mean/max are recalculated. The random band shows the range across ten
repeats. Requested removal fractions are 5%, 10%, 20%, 40%; the CSV records actual
support fractions because patch coverage is discrete. Controls are approximately
matched in support fraction, not in spatial shape. This is an internal test of
information dependence in the cached representation, not raw input removal.

### 3. `<mode>/top_channel_locations.png`

Where the strongest mean-channel terms are allocated, and where max-channel
winners occur. These maps locate the feature representation's evidence. They do
not name the underlying visual cue, and contextual features can encode distant
input regions. A rim winner does not prove a local rim pixel alone caused it.

### 4. `<mode>/feature_stage_distributions.png`

The six most influential pooled features for this scan, shown across development
scans from the same scanner. This asks whether the selected feature values shift
with stage. Distributions are descriptive and contain repeated stages of each
specimen; they are not independent-sample significance tests. Features were
selected for the displayed scan, not as a universal ranking for the dataset.

### Numeric exports

- `decision_summary.csv`: probabilities, margin, mean/max branch contributions,
  partition and head/cache fingerprints.
- `feature_contributions.csv`: every channel's standardized value, coefficient,
  exact contribution, and margin after replacing that pooled feature by its
  training mean. These replacements can be outside the natural feature distribution.
- `frozen_feature_region_deletion.csv`: each region's support fraction, post-removal
  probabilities and margin drop.
- `frozen_feature_deletion_controls.csv`: all ranked/random removal results.
- `feature_stage_distributions.csv`: source values and specimen identities.
- `exact_evidence.npz`: native allocations, channels, coordinates and region effects.

## Rerun the encoder after editing original inputs

The stronger next diagnostic alters one spatial region at a time and recomputes
the complete frozen encoder output, pooling, standardization and head prediction.
It verifies that an unedited encoder pass reproduces the saved prediction first.
It checks checkpoint checksum and repository commit; a baseline mismatch stops
the analysis rather than silently explaining a different model.

### DINOv3: appearance replacement

Run in the original DINO environment:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python explain_stage_classifier.py input --run runs/stage_pooled_v3 --scan iTERO__BCG4T1-2 --target partial --against clean --encoder dinov3 --mode dinov3_rgb --repo vendor/dinov3 --checkpoint weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --out runs/stage_pooled_v3/explain_dino_input_v1
```

Each tile's valid ROI pixels are replaced, first with the scan's median ROI colour,
then with neutral RGB 127. Background and ROI pooling support stay fixed. All
encoder features are recomputed, allowing global/contextual responses to change.
For `dinov3_gray`, the input comes from the original cached grayscale render.

### Concerto: appearance replacement

Run in the original Concerto environment:

```bat
conda run --no-capture-output -n biocal3d-concerto python explain_stage_classifier.py input --run runs/stage_pooled_v3 --scan iTERO__BCG4T1-2 --target partial --against clean --encoder concerto --mode concerto_rgb --repo vendor/concerto --checkpoint weights/concerto_small.pth --out runs/stage_pooled_v3/explain_concerto_input_v1
```

This deterministically recreates the original sampled point order and checks it
against the spatial cache. Points/normals/geometry remain fixed; selected points'
colours are replaced with global median RGB or neutral 127/255 before applying
the original model colour transform. The encoder and head run again.
Appearance replacement in `concerto_geometry` should have no effect because that
condition suppresses RGB input; it can be used as a control.

### Concerto: point removal

For an explicitly different intervention add `--intervention remove-points` and
choose a new output directory. Selected points, their normals and colours are
removed, preserving the original coordinate normalization and voxel origin.
This is a point-cloud deletion experiment, not a physically cleaned/remeshed surface.

On WSL, use the same source-root mapping as the original extraction via
`--snapshot "/mnt/c/path/to/Approved ROIs_reoriented"` and appropriate paths for
the run, repo and checkpoint. If the geometry/weights/settings differ, the baseline
check will reject the run. GPU extraction is performed in the relevant original
environment, while cached explanations can run in the DINO environment alone.

### Input-intervention outputs

- `input_occlusion.png`: margin drops from actual re-encoding. Red means editing
  that region reduced support for the target over the comparison; blue means it
  increased it. Both appearance fills share a numeric scale.
- `input_top_regions.png`: locations with the largest absolute margin effects,
  with the corresponding probability changes.
- `input_occlusion.csv`: each intervention, edited support fraction, margin and
  all three probabilities.
- `input_baseline.json`: cached/recomputed baseline probabilities and margins,
  model and feature fingerprints.

Default 8x8 tiles cover the original full field of view, so at 16 mm each tile is
2 mm wide. Increase `--grid` for finer tiles at the cost of more encoder passes.
Each tile is tested independently against the unedited baseline, not cumulatively.
Edge tiles may contain fewer ROI pixels/points than central tiles; compare the
recorded edited fraction before concluding that one region is more important.

Input edits can introduce artificial boundaries or out-of-distribution inputs.
Agreement across median/neutral fills is more reassuring than one fill alone.
These tests show dependence on a specified intervention; they do not prove that
the affected cue is biological plaque. Geometry, staining, cleaning-related changes,
scanner reconstruction or other correlated factors may contribute.

## How to interpret the combination

1. Strong channel contributions and stage-separated values explain **how the head
   reads the representation**.
2. Frozen-feature deletion tests whether the head needs those cached locations
   after pooling; it cannot remove their influence on other contextual features.
3. Input edits followed by re-encoding test whether the full model depends on the
   original visible regions under those interventions.
4. To name the signal as colour, texture, surface relief, rim shape or acquisition
   artifacts, use targeted controlled interventions and independent reference labels
   across multiple held-out specimens/scanners. A heatmap alone cannot name it.

Good stage discrimination can therefore coexist with poor plaque segmentation.
The target is global stage identity, and a few distributed or indirect cues can
separate stages without recovering the complete plaque boundary.

## Validation status

```bat
conda run --no-capture-output -n biocal3d-dinov3 python -m unittest test_stage_explanations -v
```

Synthetic checks verify exact margins, spatial reconstruction, feature replacement,
regional repooling, figure exports and input-intervention bookkeeping with a mocked
re-encoder, including that DINO pixel edits preserve the unedited pixels and ROI mask.
Real BioCal3D caches and checkpoints were not available for implementation.
The real DINOv3/Concerto re-encoding paths must be run locally; they are guarded by
baseline reproduction checks but have not been GPU-tested here. The script does
not report causal biological conclusions or segmentation accuracy.
