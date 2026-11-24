# BioCal3D paper-style pseudo-label pilot

Extract the files in this ZIP into the existing `BioCal3D_Train_Test_Study` folder, alongside `run_train_test_on_this_pc.py`. Use the latest train/test study and a completed baseline run. Its example gallery supplies the mean-pooling maps and spatial caches for the paired examples. If it predates the threshold viewer, regenerate it with the baseline launcher and `--gallery-only`.

## Run from Anaconda Prompt

The launcher defaults to the three-stage classifier. If it is already fitted, rebuild its example gallery without retraining or scoring the full test set:

```bat
python run_paper_pilot_on_this_pc.py --compare-only
```

If Windows cannot write a file in the existing gallery, export to a fresh folder:

```bat
python run_paper_pilot_on_this_pc.py --compare-only --examples-out ".\paper_examples_3stage"
```

Open `paper_examples_3stage/index.html` in the study folder. This option reuses the fitted models and writes the figures, viewer pages, and mask agreement CSV to the new folder.

To score the fitted three-stage head on the full test split without rebuilding the gallery:

```bat
python run_paper_pilot_on_this_pc.py --metrics-only
```

If you are starting a new three-stage pilot, run:

```bat
python run_paper_pilot_on_this_pc.py
```

This fits missing three-stage heads, scores them, and builds the example gallery. Add `--smoke` if you are using the completed baseline smoke run. Pass `--classifier binary` for the binary pilot or `--classifier both` to run both. The existing model folders are recognised. Use `--run` for a baseline at a custom path and add `--out` with a single classifier if its models were saved in a custom folder.

The launcher uses the existing DINOv3 and Concerto environments and checkpoints. It reuses any available spatial caches. Otherwise it runs the corresponding frozen encoder to recover patch or point features. The full pilot trains for 20 epochs by default using up to 256 locations per training scan. The smoke pilot uses three epochs and up to 128 locations. Each input condition has its own fitted attention head. Test scoring uses all eligible locations in each held-out scan and shares encoder extraction across both heads.

## Performance

Open `runs/matched_raw_train_test/paper_pilot_performance/index.html` for the full report. The smoke path starts with `runs/matched_raw_smoke`. The report includes accuracy, balanced accuracy, macro F1, macro precision, per-class metrics, row-normalised confusion matrices with counts, and paired changes in balanced accuracy from adding colour. Metrics and matrices cover all scanners pooled, each scanner separately, and mature specimens pooled across scanners. Colour effects use the same group-stratified specimen bootstrap and 95% intervals as the baseline study.

The binary report compares the binary head, the three-class head and the mean-pooling logistic baseline on the same two labels. For three-class models, the binary decision uses the sum of the baseline and partial probabilities with a fixed threshold of 0.5. The three-stage report compares the three-class head and the mean-pooling logistic baseline on the three recorded stages. These are scan-level metrics based on cleaning-stage labels.

## Model and mask generation

1. Train a linear → ReLU → linear attention MLP on each scan's patch or point features. Softmax normalizes its scores across the scan. The attention-weighted pooled features feed a linear scan classifier. DINOv3 patch scores use ROI coverage as a prior. Concerto uses uniform point support.
2. The binary head uses class-weighted binary cross-entropy and groups baseline and partial as plaque stage. The three-class head uses class-weighted categorical cross-entropy and keeps baseline, partial and fully cleaned separate.
3. Project the learned attention weights to the same ROI grid as the baseline. Scale the projected map using its 1st and 99th percentiles. Blend with input darkness at weight 0.75 for attention and 0.25 for darkness. DINOv3 grayscale and RGB both use the corresponding luminance, and geometry-only Concerto uses attention alone.
4. Refine the blended score with a local four-neighbour mean-field graph. The intensity-enabled modes use luminance differences to reduce smoothing across edges. The geometry-only mode uses spatial neighbours. A clean prediction yields an empty candidate mask. Other predictions use the default pixel threshold of 0.5.

This adapts the paper's attention MLP, softmax pooling, and blend-and-threshold pseudo-label stage. Its gland classifier used binary presence labels. The graph refinement is a lightweight local approximation to the paper's dense CRF. The encoders stay frozen and the pilot ends at pseudo-label generation. [Wen et al., weakly supervised colorectal gland segmentation](https://doi.org/10.1038/s41598-026-36256-0)

## Compare candidate masks

Under the baseline run folder, open `paper_pseudo_label_pilot/examples/index.html` for the binary gallery or `paper_pseudo_label_3stage_pilot/examples/index.html` for the three-class gallery. Both use the same held-out example list, including all four BCG3T1-1 scanner views. Each example has side-by-side threshold sliders for the existing mean-pooling score and the refined learned-attention score, with PNG mask download for each. The static panel also shows the learned attention map before refinement. The default baseline threshold is 0 logit units and the default paper-style threshold is 0.5. Each method retains its own score scale.

### Three-stage attention views

- The cyan and gold contours outline the highest-weight patches or points carrying at least 50% and 80% of the native attention mass. Ties at the cutoffs are kept together. The percentages are calculated before projection. DINOv3 displays patch footprints and Concerto interpolates the point memberships onto the flat scan.
- The signed evidence overlay defaults to partial versus fully cleaned, with a baseline versus fully cleaned option. Red favours the selected stage and blue favours fully cleaned. At each native location, the contribution is its attention weight times the dot product between its normalized feature vector and the difference of the two classifier weight vectors. Summed native contributions plus the classifier bias difference recover the logit contrast. The overlay uses a symmetric colour scale set per map.
- Select either Concerto input to rotate or zoom the coloured native point cloud. The same attention-mass groups are highlighted in 3D.
- The `*_attention.png` figure shows the contours, partial-versus-clean evidence, and the candidate mask for all four inputs. The existing mask comparison PNG and threshold sliders remain available. The corresponding `*_maps.npz` contains the native attention weights, evidence values, and bias terms.

Use a threshold from the viewer for the CSV and static exports:

```bat
python run_paper_pilot_on_this_pc.py --compare-only --mean-threshold 0.002
```

Add `--smoke` to regenerate the smoke comparison. To fit a different pilot configuration, select a single classifier and use a new `--out` folder.

`mask_agreement.csv` reports both mask coverages, IoU and Dice between the two methods on their common scored ROI. IoU and Dice are blank when both masks are empty. It also gives the predicted class and the scan-level class probabilities. Photo-backed masks are needed to measure plaque localisation accuracy.

## DINOv3 and Concerto feature maps

For the most comparable view, run from the study folder:

```bat
conda run -n biocal3d-dinov3 python paper_feature_maps.py --encoder both
```

Open `matched_feature_maps/BCG3T1-1_matched_features.png`. The rows show the original scan, DINOv3 grayscale, DINOv3 RGB, Concerto geometry only, and Concerto geometry + RGB.

The figure places both encoders on the DINOv3 patch grid. It averages the Concerto point embeddings within each grid cell and applies the same bilinear interpolation to all four feature rows, within the cells supported by all four inputs for that scan. The interpolated colours blend neighbouring cell values for display. The original scan row shows the full ROI. Each encoder fits one PCA on the shared cells for both of its input conditions and all displayed scanner views. Compare the shapes and locations of the spatial patterns across encoders. Compare the PCA colours within an encoder.

For the standalone DINOv3 figure, run:

```bat
conda run -n biocal3d-dinov3 python paper_feature_maps.py
```

Open `dinov3_feature_maps/BCG3T1-1_dinov3_features.png`. The figure shows the original scan, frozen DINOv3 features from RGB input, and frozen DINOv3 features from grayscale input for each cached scanner view of BCG3T1-1.

For Concerto, run:

```bat
conda run -n biocal3d-dinov3 python paper_feature_maps.py --encoder concerto
```

Open `concerto_feature_maps/BCG3T1-1_concerto_features.png`. Its rows show the original scan, Concerto features from geometry + RGB, and Concerto features from geometry only. The exporter projects the colours of the nearest sampled surface points into the same top-down view as the original scan. Both commands use saved spatial embeddings. The DINOv3 environment supplies the plotting dependencies for both figures. Add `--smoke` for the smoke run, `--run` for a custom run, or `--sample` for another held-out specimen-stage ID with spatial caches.

Each encoder has its own three-component PCA shared across its displayed scanner views and both input conditions. All three modes use shared 2nd-to-98th percentile colour scaling within each encoder. The standalone views fit PCA on up to 2,048 ROI patches or points per panel. The colours describe feature variation. To reproduce the paper's before-and-after comparison, extract both rows from the same input images using the original encoder checkpoint and a second checkpoint fine-tuned on BioCal3D.
