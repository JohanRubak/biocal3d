# BioCal3D: train and compare two 3D stage classifiers

This kit trains a **TSGCNet-derived mesh-face classifier** and a **portable PointNeXt-S classifier** from scratch. It reads the `geometry.npz` files from the BioCal3D preparation snapshot, compares them on the same held-out physical specimens, and generates class-specific 3D evidence maps and overlays on the prepared RGB images.

The default task is **baseline / partial / clean experimental-stage classification**. Stage names are read from the preparation index, not inferred from plaque polygons. Stage classification does not establish pixel-level plaque accuracy. No segmentation labels or `checked` flags are needed for this task.

## Start in your existing environment

Extract this folder into, for example:

`C:\Users\au662213\repos\biocal3d\data\BioCal3D_preprocessing_and_labeling\BioCal3D_3D_classification`

In **Anaconda Prompt / CMD**:

```bat
conda activate biocal3d
cd /d "C:\Users\au662213\repos\biocal3d\data\BioCal3D_preprocessing_and_labeling\BioCal3D_3D_classification"
python -m pip install -r requirements.txt
```

This uses the existing environment. The kit does not import PyVista, VTK, Open3D, or compiled point-cloud extensions. It needs Python 3.10+ and PyTorch 2.3+.

If PyTorch is already installed and works, keep it. For **CPU PyTorch**, if needed:

```bat
python -m pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, use the [official PyTorch installer selector](https://pytorch.org/get-started/locally/): Windows / Pip / Python / a CUDA build compatible with your driver. Install `torch` using that command through `python -m pip`; this kit needs neither torchvision nor torchaudio. The CPU command above installs a CPU build, which cannot train on CUDA.

Check the interpreter and device:

```bat
python biocal3d_train_3d.py check-env
```

A local end-to-end check is optional:

```bat
python smoke_test.py --output "C:\Users\au662213\repos\biocal3d\runs\3d_smoke_test"
```

Use a fresh path for the smoke test. It generates synthetic data, trains all four combinations for two epochs, checks gradients and mappings, and writes `VALIDATION.json`. Its accuracy has no scientific meaning.

## Train all four combinations and make the reports

With the snapshot from your preparation run, paste this **single line** into Anaconda Prompt. If you create another snapshot later, change the `--index` path and use a new output directory.

```bat
python biocal3d_train_3d.py run --index "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261007_100936_877172\dataset_index.json" --output "C:\Users\au662213\repos\biocal3d\runs\3d_stage_v1"
```

Alternatively, after installing the dependencies, double-click `run_stage_classification.cmd`. It invokes your existing environment's Python directly and uses those snapshot/output paths. Edit its three path settings for a new dataset. It includes `--resume`, so repeating it continues interrupted runs or reuses completed ones.

Default runs:

| Folder prefix | Architecture | Input |
|---|---|---|
| `tsgcnet_geometry` | TSGCNet-derived classifier | Triangle corners XYZ + corner normals |
| `pointnext_s_geometry` | Portable PointNeXt-S | Surface-point XYZ + interpolated normals |
| `tsgcnet_geometry_rgb` | TSGCNet-derived classifier, with colour stream | Geometry + original corner RGB |
| `pointnext_s_geometry_rgb` | Portable PointNeXt-S, with colour features | Geometry + original interpolated RGB |

All use seed 42, the same specimen split, 100 maximum epochs, validation-based early stopping (patience 20), AdamW, initial learning rate 0.001, weight decay 0.01, and cosine learning-rate decay. Training runs sequentially. CUDA is selected automatically if available; otherwise CPU is used. CPU training is supported, but pure PyTorch farthest-point sampling can be slow at 1,024 points. There is no reliable runtime estimate without your hardware.

For a short first run, append `--epochs 10 --patience 5 --explain-count 3` and use an output such as `3d_stage_pilot`. For a faster plumbing check on your real files, use `--faces 256 --points 256 --epochs 2`; do not treat that check as the final experiment. On CUDA, `--amp` enables mixed precision. If GPU memory is tight, lower `--batch-size` or point/face counts. At least 32 points/faces are required.

For repeated initializations, append `--seeds 42 43 44`. They reuse the same split and evaluation sampling. Each seed gets its own checkpoint and metrics. The plot shows individual runs rather than choosing the best test seed.

In PowerShell, the same `python ...` commands work after activating the environment; change directory with `Set-Location "..."` instead of `cd /d`. Do not add a dash before `python` or `pip`.

## What to open afterwards

Open `3d_stage_v1\comparison.html` in your browser. It links to:

- `comparison.png` / `comparison.csv`: test balanced accuracy, macro F1, accuracy, cross-entropy, ROC AUC where defined, and 95% specimen-bootstrap intervals for balanced accuracy and F1.
- `paired_differences.csv`: differences between runs using identical bootstrap draws of whole specimens. Positive means run A scored above run B.
- Each run's `learning_curves.png`, `confusion_matrix.png`, `test_predictions.csv`, and results by scanner, group, and growth condition.
- `best.pt`: the checkpoint selected using validation balanced accuracy; ties use validation macro F1, then validation cross-entropy.
- `encoder.pt`: encoder weights and configuration for later segmentation development.
- `last.pt`: optimizer, scheduler, mixed-precision scaler, model and random-state checkpoint for resuming.
- Each run's `explanations\index.html`: original image, 2D evidence overlay, links to rotating 3D surfaces, and feature-ablation results.

`train` runs training and comparison without explanation; `run` also explains six shared test scans by default. Their selection is deterministic across class/scanner buckets, independent of prediction or heatmap quality.

You can regenerate the comparison without training:

```bat
python biocal3d_train_3d.py compare --output "C:\Users\au662213\repos\biocal3d\runs\3d_stage_v1"
```

For resuming an interrupted run, repeat the original training command with **identical training settings**, appending `--resume`. Completed model runs are reused; interrupted runs restart at the next saved epoch. If no first epoch was saved, that model starts again. Changing the snapshot, labels, source code, split, or training settings requires a new experiment directory. Atomic checkpoint writes preserve the preceding checkpoint if a write is interrupted. Resuming on another device need not reproduce the same numerical trajectory.

## Specimen splits and experimental design

The index already identifies each physical specimen as `G1_S1`, etc. **Every timepoint and scanner for one physical specimen stays together.** Scan names, scanner IDs, stage, group, timepoint, and filenames are not model input features.

Default split: shuffle physical IDs separately within each experimental group, using split seed 20261008. Approximately 15% of specimens in each group go to test and 15% to validation, with at least one in each; the remainder train. For your uploaded index this gives:

| Partition | Physical specimens | Baseline scans | Partial scans | Clean scans | Total scans |
|---|---:|---:|---:|---:|---:|
| Train | 36 | 144 | 96 | 144 | 384 |
| Validation | 6 | 24 | 16 | 24 | 64 |
| Test | 6 | 24 | 16 | 24 | 64 |

Thus the independent test cohort has **six specimens**, not 64 independent samples. Bootstrap intervals resample entire specimens with all their stages/scanners. Bootstrap draws that omit an entire class are excluded and their usable count is reported. Intervals do not account for changes in the training cohort. With six test specimens they can be unstable; use the scores as exploratory evidence. Repeated model seeds do not enlarge the independent cohort.

Both the training and validation partitions must contain every requested class. Class weights are inverse training scan frequencies. These address class imbalance; they do not make correlated scans independent. Training treats each scan as an observation; test uncertainty treats specimens as clusters.

Use `make-splits` with the same `--index` and `--output` to write and inspect splits before training. `splits.json` stores physical IDs; `split_assignments.csv` lists every eligible scan. Reuse it for 2D comparisons, using physical IDs to map scans. A custom split has this structure:

```json
{"schema_version": 1, "physical_ids": {"train": ["G1_S1"], "val": ["G1_S2"], "test": ["G1_S3"]}}
```

The example only illustrates the schema. A real split must assign **every eligible physical ID exactly once**, with every class represented in each partition. Import using `--splits "path\splits.json"`. Keep any specimens used to tune your annotation instructions out of the final segmentation test set when establishing that later experiment. The supplied default stage split happens to place all six previously selected calibration specimens in training/validation, but check this again if changing splits.

For five-fold experiments, use separate outputs with `--fold 0`, ..., `--fold 4` (`--folds 5`). Within each group, test uses that fold, validation the next, and training the other three. Each specimen is tested once across the five experiments. The current comparison command intentionally compares runs from **one shared split**; it does not aggregate different folds or allow mixing their predictions. Keep fold reports separate or aggregate out-of-fold predictions in a later analysis, without choosing settings on those test predictions.

Primary metrics are balanced accuracy and macro F1. In group/growth slices that lack a class, the fixed-class macro metric includes zero recall for that absent class; `observed_class_balanced_accuracy` and `classes_present` provide the interpretable within-slice metric. Inspect scanner-specific results and errors, not only the pooled score. Choose settings on validation data; do not select a winning seed/hyperparameter by repeatedly inspecting the test scores.

For a scanner transfer experiment, e.g. after PRIMESCAN arrives, append `--exclude-train-scanners PRIMESCAN`. This excludes it from training and validation but retains it in the held-out-specimen test set. The scanner CSV then distinguishes known-scanner and unseen-scanner results, while specimen separation remains intact. PRIMESCAN is not hardcoded: any scanner in a new index is recognized.

When adding PRIMESCAN scans of existing specimens, regenerate preprocessing and run training with the new index, a new output, and `--splits` pointing to the previous specimen split. It is valid to reuse the same physical split even though the snapshot hash changes. If adding **new physical specimens**, explicitly regenerate or extend the specimen split. The script refuses silent assignment of unseen specimens.

## Input preparation and exact architecture choices

The original ROI preprocessing is not rerun here. `geometry.npz` provides canonical XYZ in mm, vertex normals, original vertex RGB, and faces. The geometry hash in the index is checked against every eligible geometry file before training. Failed scans, unknown stages and unlisted custom labels are excluded and reported. Missing/malformed files or changed hashes cause an error, rather than partial training.

Coordinates are divided by **one fixed 6.7 mm constant**, corresponding to half the common 13.4 mm image FOV. There is no per-scan height standardization, unit-sphere scaling, or deletion of a roughness scale. Change `--scale-mm` consistently if your acquisition units/FOV differ. Normals are unit length; RGB is shifted by 0.5 without per-scanner normalization. These are assumptions to validate against scanner units. Random rotations around canonical Z augment training only, with normals rotated consistently. No coordinate jitter, height rescaling, colour augmentation, or image-display brightness settings are applied.

| Component | Implementation and deliberate departures |
|---|---|
| TSGCNet-derived encoder | Three graph stages (64, 128, 256) per stream; channel-wise attention aggregates coordinate edges, neighbour max pooling aggregates normals and optional RGB. Multiscale features concatenate and map to 512 channels per stream, then fuse to 512 local channels. |
| Face representation | Nine XYZ corner coordinates and nine corner-normal values per face; optional nine RGB values. Up to 1,024 nondegenerate faces are sampled uniformly without replacement. All faces are used for meshes below that cap. Selected face IDs remain traceable to the exported mesh. |
| Face graph | 12 nearest face centroids in canonical space, built on the selected faces; reused across stages. This is a spatial face graph, **not native vertex-sharing connectivity** after mesh decimation. Face sampling changes the graph resolution. |
| Mesh classification head | Surface-area-weighted mean plus max pooling of local features; LayerNorm/dropout MLP 1,024 → 512 → 256 → classes. Equal-probability face sampling and area weights reduce triangulation-density effects; max pooling and graph resolution can still reflect scanner density. |
| TSGCNet departures | New classification head, spatial sampled-face graph, simplified attention implementation, no learned coordinate/normal transform, no original DentalPSAM adaptive stream gate. Encoder normalization uses per-mesh moments in **both training and inference**, as in DentalPSAM's per-mesh normalization path; microbatch-one training is therefore valid. This is a TSGCNet-derived baseline, not the published original segmenter or a checkpoint-compatible DentalPSAM branch. |
| Point surface sampling | 1,024 points drawn uniformly in mesh surface area: choose a triangle proportional to area and sample barycentric coordinates. Interpolate XYZ, unit-normal direction and optional vertex RGB. The mesh is not reduced to a random subset of vertices. |
| PointNeXt-S topology | Six stages, `[1,1,1,1,1,1]` blocks and `[1,2,2,2,2,1]` strides; widths 32, 64, 128, 256, 512, 512. The first stage is a linear point stem; four stages use FPS, radius neighbourhoods, two-layer residual set abstraction and neighbour max aggregation; the final stage globally aggregates. The S classification configuration has no extra inverted-residual blocks because each block count is one. |
| Point neighbourhoods | Initial radius 0.15 in scaled coordinates (1.005 mm at the default scale), ×1.5 after each downsampling stage; up to 32 nearest points within radius. If fewer exist, the closest valid point is repeated. FPS starts at the point farthest from the centroid. These deterministic pure PyTorch operators replace official CUDA kernels; boundary/order choices can differ. |
| PointNeXt input/head changes | Six input features (XYZ + normals) or nine (+RGB), rather than the four ScanObjectNN input features. Global 512-dimensional encoding → 512 → 256 → classes, with LayerNorm rather than head BatchNorm for small batches. Encoder layers use standard running-stat BatchNorm, trained with actual multi-scan batches. It is a portable implementation of the S topology with declared domain-specific changes, not official pretrained-weight compatibility. |

Both share optimizer settings, augmentation, target definitions, splits and evaluation policy. They cannot have identical inputs: one uses face corners and a sampled face graph, the other area-sampled points and a hierarchy. Report parameter counts and sampling choices from each run's config when writing your methods. Parameter counts differ because the face model has independent streams and a larger pooled input.

`--batch-size 4` means four scans in a PointNeXt forward; TSGCNet accumulates gradients over four separate variable-size meshes. The default final global PointNeXt layer has 64 points before global aggregation. Evaluation sampling uses a fixed independent seed (1234), and training sampling changes deterministically per epoch. Models do not receive segmentation label colours. LABscanner RGB here is vertex RGB sampled from its texture by preprocessing; interpolating it on the surface cannot recover texture detail lost within a triangle.

## Explain a specific prediction

The method is a **Grad-CAM adaptation to local 3D features**: backpropagate a selected class logit, average its feature gradients over local elements, form a gradient-weighted feature sum, then retain positive contributions. Each scan's positive map is divided by its own maximum for visualization. Raw signed scores are saved too.

- TSGCNet: direct evidence per sampled face, after stream fusion and before global pooling. Unsampled faces are `NaN`/grey, not zero evidence. Increase `--faces` during training if you want denser direct face maps; changing it only at explanation time would change the evaluated model input.
- PointNeXt-S: direct evidence at its **last local point stage**, before global abstraction. At 1,024 input points this has 64 points. Three-nearest inverse-distance interpolation supplies a display on the original faces. Saved point locations and scores are the direct result; the dense surface is an approximate visualization.
- If `maps.npz` and `rgb.png` are present and their provenance matches, the existing exact pixel → triangle-ID mapping produces the RGB overlay. This does not project arbitrary points by their XY position. TSGCNet pixels over unsampled faces remain unsupported; the PointNeXt pixel map is based on the interpolated face display.

Example: explain the true-stage evidence for a particular scan:

```bat
python biocal3d_train_3d.py explain --checkpoint "C:\Users\au662213\repos\biocal3d\runs\3d_stage_v1\tsgcnet_geometry_rgb_seed42\best.pt" --index "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261007_100936_877172\dataset_index.json" --scan-ids "TRIOS3__BCG3T1-1" --target true
```

`--target predicted` is the default. You can also use a class name, e.g. `--target partial`, to compare explanations of competing classes. Use `--output` to a separate directory to keep multiple targets for the same scans; otherwise that scan's previous explanation is replaced. Arbitrary explicitly selected scans may be from train/validation; the summary flags whether each was in the held-out test set.

For each scan:

- `surface.html`: offline interactive original-colour mesh and CAM mesh; PointNeXt also shows its direct coarse-point CAM. Plotly is embedded, so no internet is needed.
- `overlay.png` and `pixel_cam.npz`: overlays using the preparation mappings, when present.
- `attribution.npz`: direct local positions/scores, full face display, selected original face IDs, geometry identity and checkpoint hash.
- `attribution.vtp`: canonical ROI mesh with face `CAM`, signed scores and original exported face IDs, openable in ParaView/PyVista. PointNeXt face values are interpolated; TSGCNet unsampled faces are NaN.
- `feature_ablation.json`: suppress the top 10% of local CAM feature vectors, repeat with 10 equal-sized random selections, and compare target probability drops. Coordinates remain in place. This checks **internal-feature sensitivity**, not the causal effect of physically removing a region. A negative drop is possible; do not interpret that case as supportive evidence.

CAM has no plaque/clean boundary semantics. A model can correctly predict cleaning stage using colour, scanner artifacts, texture, or edges rather than plaque itself. A low/zero CAM is not “clean”, and a warm CAM is not a segmentation label. Check misclassifications and scanner differences. Later, compare these maps against independent dentist masks and use region perturbations / parameter-randomization checks before making localization claims. Scale is relative within each map, so two heatmaps' colours do not compare attribution magnitude across models. Feature-ablation and classifier-head dependence are useful checks, but do not validate biological plausibility.

## Reusing encoders later

`encoder.pt` contains the same run configuration and all weights except `classifier.*`. Construct the matching architecture from `models3d.py`, load those weights with `strict=False`, and add a dense decoder. Inspect the missing keys: they should only be the classification head when loading into the current classifier. Preserve this input normalization, point/face conventions and neighbourhood settings, or retrain intentionally.

The face encoder already produces one feature vector per sampled face. The point encoder provides a hierarchy; its classification forward exposes the final local stage. A segmentation decoder must retain/interpolate earlier-stage features to the required mesh/point resolution. These checkpoints are **not** plug-in weights for the released DentalPSAM TSGCNet or its final multimodal mesh decoder. The existing geometry/maps and raw IDs remain available for projecting your later dentist labels; this training script never rewrites those preparation files.

## Alternative class definitions

If you later want another scan-level label, provide a UTF-8 CSV with `scan_id,label`, plus `--labels-csv "labels.csv" --classes class_a class_b`. The CSV is copied into the experiment and its effective targets are fingerprinted. Unlisted scans are excluded, so you must inspect `excluded_scans.csv`. Do not automatically declare baseline=plaque or clean=no-plaque without an independently justified definition. All physical specimen grouping rules still apply.

## Sources and verification

Architectures were independently reimplemented from the described designs, with departures listed above:

- [TSGCNet, CVPR 2021](https://github.com/ZhangLingMing1/TSGCNet)
- [DentalPSAM 3D branch implementation](https://github.com/HKU-HealthAI/DentalPSAM/blob/main/dentalpsam/branch3d/model.py)
- [PointNeXt paper](https://arxiv.org/abs/2206.04670)
- [Official PointNeXt-S classification configuration](https://github.com/guochengqian/PointNeXt/blob/master/cfgs/scanobjectnn/pointnext-s.yaml)
- [OpenPoints encoder implementation](https://github.com/guochengqian/openpoints/blob/master/models/backbone/pointnext.py)
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)

The delivered `VALIDATION.json` records CPU checks performed on synthetic geometry. No actual BioCal3D training accuracy has been measured here because the provided uploads contain the preparation index/manifests, not your complete mesh snapshot. CUDA execution and Windows execution were not available for validation here. Use `check-env` and the smoke test in your existing environment before a long run.
