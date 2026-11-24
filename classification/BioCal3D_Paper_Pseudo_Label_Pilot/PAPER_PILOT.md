# BioCal3D paper-style pseudo-label pilot

Extract the files in this ZIP into the existing `BioCal3D_Train_Test_Study` folder, alongside `run_train_test_on_this_pc.py`. Use the latest train/test study and a completed run. Its example gallery supplies the saved mean-pooling maps and test patch or point features. If the gallery predates the threshold viewer, regenerate it with the baseline launcher and `--gallery-only`.

## Run from Anaconda Prompt

For the completed baseline smoke run:

```bat
python run_paper_pilot_on_this_pc.py --smoke
```

For the completed full train/test run:

```bat
python run_paper_pilot_on_this_pc.py
```

The launcher uses the existing DINOv3 and Concerto environments and checkpoints. It reuses any available training spatial caches. Otherwise it extracts a bounded sample of training locations with the same frozen encoder and keeps those bags in memory. The full pilot defaults to 20 epochs and up to 256 patches or points per training scan. The smoke pilot uses three epochs and up to 128 locations. Each of the four conditions has its own fitted attention head. The same saved specimen split and scanner inputs are used for the baseline and the pilot. The pilot uses the original held-out example list, including all four BCG3T1-1 views.

## Model and mask generation

1. Train a linear → ReLU → linear attention MLP on each scan's patch or point features. Softmax normalizes its scores across the scan. The attention-weighted pooled features feed a binary linear scan classifier.
2. Use all three training stages. Baseline and partial scans are weak positives and final-clean scans are weak negatives. DINOv3 patch scores use ROI coverage as a prior. Concerto uses uniform point support. The training objective is class-balanced binary cross-entropy.
3. Project the learned attention weights to the same ROI grid as the baseline. Scale the projected map using its 1st and 99th percentiles. Blend with input darkness at weight 0.75 for attention and 0.25 for darkness. DINOv3 grayscale and RGB both use the corresponding luminance, and geometry-only Concerto uses attention alone.
4. Refine the blended score with a local four-neighbour mean-field graph. The intensity-enabled modes use luminance differences to reduce smoothing across edges. The geometry-only mode uses spatial neighbours. A scan prediction below 0.5 for the plaque-stage class yields an empty candidate mask. Otherwise the default pixel threshold is 0.5.

This adapts the paper's attention MLP, softmax pooling, and blend-and-threshold pseudo-label stage. The graph refinement is a lightweight local approximation to its dense CRF. The encoders stay frozen and the pilot ends at pseudo-label generation. The paper also fine-tuned its encoder and trained a segmentation decoder. [Wen et al., weakly supervised colorectal gland segmentation](https://doi.org/10.1038/s41598-026-36256-0)

## Compare

Open `runs\matched_raw_train_test\paper_pseudo_label_pilot\examples\index.html` for the full run. The smoke path starts with `runs\matched_raw_smoke`. Each example has side-by-side threshold sliders for the existing mean-pooling score and the refined learned-attention score, with PNG mask download for each. The static panel also shows the learned attention map before refinement. The default baseline threshold is 0 logit units and the default paper-style threshold is 0.5. Each method retains its own score scale.

Use a threshold from the existing viewer for the CSV and static exports:

```bat
python run_paper_pilot_on_this_pc.py --compare-only --mean-threshold 0.002
```

Add `--smoke` to regenerate the smoke comparison. To fit a different pilot configuration, use a new `--out` folder. The original study outputs are read from the completed run.

`mask_agreement.csv` reports both mask coverages, IoU and Dice **between the two methods** on their common scored ROI. IoU and Dice are blank when both masks are empty. The scan-level plaque-stage probability is the binary head prediction. It is not a pixel-level plaque probability. The models learn from stage labels, and overlap between their pseudo-labels measures agreement. Photo-backed masks are needed to measure plaque localisation accuracy.
