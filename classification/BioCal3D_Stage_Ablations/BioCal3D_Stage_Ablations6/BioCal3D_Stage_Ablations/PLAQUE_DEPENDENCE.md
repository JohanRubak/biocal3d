# Connect the mean-pooling map to the visible scan

The question is: **does information in visually plaque-like regions actually affect
the fitted stage classifier more than comparable control regions?**

This workflow uses your current frozen encoder and saved logistic-regression head.
No classifier is retrained. It performs a controlled input-intervention experiment
and compares its effects with the existing mean/max allocation values.

## The first experiment

Use a partial-stage specimen and keep the contrast **partial versus clean**.

1. Select a conservative visible plaque-like region on the original RGB image.
2. Select a clean-like region of comparable size and distance from the rim.
3. Edit each region independently, rerun the encoder, and obtain the same head's
   probabilities and partial-versus-clean margin.
4. Compare with several random regions matched in size and edge distance.
5. Repeat using median-colour and neutral-colour replacement.

For an initial technical pilot, use a few different held-out physical specimens,
not just different scanner views of the same specimen. This pilot checks workflow
and consistency; it does not establish a statistically sufficient study size.
Choose regions before examining their attribution maps when feasible, to reduce
the tendency to select locations that already support an interpretation. Your
earlier exposure to some scans is still relevant to describing this exploration.

## Install

Copy the new `test_plaque_dependence.py` and updated `explain_stage_classifier.py`
into your current stage-ablation folder. Keep the accompanying original modules
beside them. `PLAQUE_DEPENDENCE.md` is this guide; the optional checks are in
`test_region_dependence_checks.py` and use the existing synthetic test fixtures.
The original extraction/classification Python modules and saved heads are unchanged.

You need existing spatial caches, original renders, the original checkpoints and
the corresponding DINOv3/Concerto environments. The previous inspector already
uses these renders/caches. No new library installation is required.

## 1. Select visible regions

Run from your existing `BioCal3D_Stage_Ablations` folder:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python test_plaque_dependence.py select --run runs/stage_pooled_v3 --scan TRIOS5__BCG4T1-1 --out selections/TRIOS5_BCG4T1_1.json
```

The example uses the held-out TRIOS5 specimen you showed. Check the exact scan ID
with the existing `explain_stage_classifier.py cached --list-scans` command if needed.
Use your actual existing run directory if it differs.

The selection window shows **only the original RGB and your region circles**:

- Click **Plaque-like** (P), then a conservative plaque-like location.
- Click **Clean-like** (C), then a nearby clean-like location at a comparable
  distance from the rim. Both circles in a pair use the same radius.
- The radius defaults to 0.5 mm. Set the slider before the first click in a pair;
  the radius then locks for that pair. Use smaller circles to avoid mixed regions
  or larger circles if too few sampled points are available.
- Choose **Next pair** (N) if more distinct appearances need sampling.
- **Undo** (U) reverses the last selection action; **Save** (S) saves the JSON.
- Close the window after saving. Existing selection files are not overwritten.

Start with one pair per specimen. Avoid regions touching the ROI boundary when
possible. The circles must not overlap. These labels describe your visual
assessment; they are not independent ground truth for plaque segmentation.

For reproducible coordinate-based selection without a window, supply both
`--plaque X Y --clean X Y --radius-mm 0.5`. Coordinates refer to the native render
with origin at top left, not the screenshot's resized display coordinates.

## 2. Run the DINOv3 experiment

```bat
conda run --no-capture-output -n biocal3d-dinov3 python test_plaque_dependence.py run --run runs/stage_pooled_v3 --selection selections/TRIOS5_BCG4T1_1.json --encoder dinov3 --repo vendor/dinov3 --checkpoint weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --target partial --against clean --out runs/stage_pooled_v3/region_tests/TRIOS5_BCG4T1_1_dino
```

This edits RGB pixels, keeping the ROI mask and geometry fixed, then reruns the
complete DINOv3 feature extraction, pooling, scaler and fitted head. The default
mode is `dinov3_rgb`; `--mode dinov3_gray` uses the original grayscale encoder input
at the same selected positions.

## 3. Run the same selection with Concerto

```bat
conda run --no-capture-output -n biocal3d-concerto python test_plaque_dependence.py run --run runs/stage_pooled_v3 --selection selections/TRIOS5_BCG4T1_1.json --encoder concerto --repo vendor/concerto --checkpoint weights/concerto_small.pth --target partial --against clean --out runs/stage_pooled_v3/region_tests/TRIOS5_BCG4T1_1_concerto
```

This recreates the same sampled points and changes selected point RGB values.
Coordinates, normals and ROI sampling remain fixed. The encoder and fitted head
run again. In WSL, use the same paths and `--snapshot` mapping as your original
extraction. Use the actual checkpoint/repository paths from your project.

The default mode is `concerto_rgb`. `--mode concerto_geometry` is a useful negative
control for RGB editing because that model suppresses colour input. The exported
snapshots contain the raw inputs before the adapter's colour transform/suppression;
their RGB is ignored by the geometry-only condition. They are not the final
normalized GPU tensors.

For a separate geometry-availability intervention, Concerto also accepts
`--intervention remove-points`. This removes selected sampled points and their
associated normals/colours while preserving the original voxel origin and scale.
It changes the point cloud and can introduce holes; it is not a physical cleaning
simulation. Keep these results separate from the appearance-replacement experiment.

## Controls and checks

- The JSON is tied to the cohort, mesh, original RGB, ROI mask and render layout.
  A changed image or mismatched scan is rejected.
- The original checkpoint and repository commit are checked. The unedited encoder
  pass must reproduce the cached feature vector, class probabilities and margin
  within the existing numerical tolerances before any experiment is saved.
- Plaque/clean circles are converted to original encoder input units: ROI pixels
  for DINO, sampled points for Concerto.
- The smaller circle's available unit count is used for both regions. If one region
  has more units, the closest units to its center are retained. Thus actual edited
  masks can be trimmed slightly relative to the drawn circle. Saved masks identify
  exactly what was changed. The edit count is identical within each model's pair.
- Five random controls are selected by default. Each edits exactly the same count
  and has a similar mean distance from the rim. They avoid the selected plaque/clean
  masks; different random controls may overlap each other and are not independent
  experimental observations. Random regions may also contain plaque.
- Default rim tolerance is 0.5 mm. The script checks both the clean reference and
  random controls against the plaque reference using projected ROI edge distance.
  This is an exploratory matching tolerance, not a validated biological threshold.
  If matching fails, select a better-positioned control, smaller circles, fewer
  random controls (`--random-controls`), or explicitly document a different
  `--rim-tolerance`. It does not silently relax matching constraints.
- Each region is edited against the unmodified original, not cumulatively. The same
  selected masks are reused across median/neutral fill styles.
- Median replacement uses the whole scan's median ROI RGB; neutral uses RGB 127.
  Concerto applies the original colour normalization after replacement. Uniform
  replacement removes local colour/texture information and introduces a boundary;
  it does not isolate plaque as a biological material.

Counts are matched **within a model**. DINO's projected pixel area and Concerto's
sampled surface support are related but are not identical physical quantities.
Selection on a Concerto preview refers to an XY column through sampled points.
For multilayer/overhanging geometry it may select more than one surface layer;
inspect native PLY/NPZ inputs. The current flat specimens are the intended pilot.

## What to open first

Inside each experiment, open `pair_01/pair_01_median.png`, followed by the neutral
version. Each figure shows:

1. Original RGB with the actual selected pixels/points marked gold and cyan.
2. The mean-pooling allocation map for the same class contrast.
3. Encoder input after changing the plaque-like region.
4. Encoder input after changing the clean-like region.
5. Margin drops for plaque, clean and random edits underneath.

Bars show the complete classifier's margin drop. Black circles show the mean
branch's change and black crosses show the max branch's change. Those two changes
sum exactly to the total. This helps distinguish a region with a plaque-like mean
map from a region whose input edit mainly affects max-pooling features.

Concerto edited images are approximate nearest-point XY previews; PLY/NPZ files
preserve the true native point positions, normals and raw RGB. DINO edited PNGs
are the exact uint8 images passed to the original preprocessing function.

The key quantities are:

`plaque effect = original margin - margin after plaque-region edit`

`clean effect = original margin - margin after clean-region edit`

`paired excess = plaque effect - clean effect`

For partial versus clean, a **positive plaque effect** means the edit reduced
support for partial relative to clean. A **positive paired excess** means its effect
was larger than the clean edit. Compare the random controls as well.

These signs must be read together: a positive paired excess can occur even if
both edits increase the target margin. That is not evidence that removing the
plaque-like region weakened the target. The CSV records these separate checks.

More supportive evidence would be a plaque edit that weakens partial-versus-clean,
more than both matched clean and random edits, across fill styles and multiple
physical specimens. This supports dependence on information in visually identified
deposit regions under those edits. It does not prove that the model recognizes
biological plaque rather than staining, texture, boundary artifacts or correlated
acquisition changes. Both fills are artificial and can produce different responses.

## Numeric and native outputs

- `region_effects.csv`: every region, exact edit count/fraction, rim distance,
  mean/max allocation, original/edited margin, the change through each pooling
  branch, and all three class probabilities.
- `paired_effects.csv`: plaque-minus-clean and plaque-minus-random effects, paired
  mean/max allocation differences, plus explicit sign checks.
- `pair_01/native_edit_masks.npz`: exact intervention masks, shared across fills,
  and source pixel positions of the input units.
- `original_encoder_input.*`, `plaque_median_encoder_input.*` etc: original and
  edited inputs. DINO saves PNG; Concerto saves NPZ plus world-coordinate PLY.
  Random controls are reproducible from their saved native masks and fill style;
  full native input copies are saved only for the plaque and clean regions.
- `experiment.json`: baseline reproduction checks, selected contrast, input/model
  fingerprints and control settings.

The displayed mean-map allocation is a different quantity from the observed
input-intervention effect. For DINO, region allocation uses the fraction of each
native patch covered by selected input pixels. For Concerto it sums the selected
native point allocations. The encoder's context means these allocations need not
equal the effects of editing those original inputs. Their relationship is exactly
what this experiment investigates.

## Several specimens and a combined summary

Create a separate selection JSON for each new scan, then repeat `run` in the
appropriate environment. Use new output directories. To summarize a directory
containing experiments from both models:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python test_plaque_dependence.py summarize --results runs/stage_pooled_v3/region_tests --out runs/stage_pooled_v3/region_test_summary_v1
```

Open `dependence_summary.png`:

- First panel: plaque-minus-clean input effect by physical specimen.
- Second: plaque-minus-random input effect by physical specimen.
- Third: does a larger plaque-minus-clean mean-map allocation accompany a larger
  actual input-effect difference in the mean-pooling branch?

ROI pairs are averaged within each scan, then repeated scanner/stage views are
averaged within each physical specimen. Modes, class contrasts, fill styles and
intervention types stay separate. Each dot is one specimen, not one scan or one
random edit. Duplicate experimental rows are rejected to avoid accidental double
counting of repeated output directories.

Descriptive specimen bootstrap intervals are drawn only with at least three
specimens; that minimum is a display rule, not a claim of sufficient sample size or
reliable inferential coverage with such small samples. Intervals are conditional
on the chosen regions, fits and perturbations. Do not interpret the random-control
rank fraction as a statistical p-value. `by_specimen.csv`, `summary.csv` and the
source list provide the aggregated values and provenance.

## Verification status

```bat
conda run --no-capture-output -n biocal3d-dinov3 python -m unittest test_region_dependence_checks -v
```

Synthetic checks exercise equal-size/rim matching, overlap rejection, annotation
replay, stale-render/baseline guards, both native input formats, image/point exports
and specimen aggregation. Re-encoder behavior is mocked for these checks because
the BioCal3D scans, GPU environments and weights are on your PC. Actual DINOv3 and
Concerto inference must be tested locally; baseline reproduction is required.
