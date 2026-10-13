# BioCal3D: four-model 3D classification benchmark

This trains classifiers for **baseline / partial / clean** using your processed
ROI surfaces. It does not train plaque segmentation or treat experimental stages
as per-face labels. The comparison screens potential 3D branches for a future
SegDINO + 3D fusion model.

## Models and exact provenance

| Model | Encoder | What trains |
|---|---|---|
| PointNeXt-S | Official PointNeXt/OpenPoints S encoder, using the ScanObjectNN S configuration | Encoder and a new stage head, from scratch |
| TSGCNet | Official two-stream mesh code, with compatibility patches | Geometric feature extractor and a new stage head, from scratch |
| Sonata | Official pretrained PTv3 encoder | A new stage head; encoder stays frozen |
| Concerto | Official pretrained PTv3 encoder, default `concerto_small` | A new stage head; encoder stays frozen |

These are practical transfer-learning comparisons, not controlled claims about
architecture alone: prior training data, model capacity, and compute differ.
The released Concerto small/base/large variants are described by the repository
as pre-release variants of subsequent work. The checkpoint identity and exact
state hash are recorded, so do not describe `concerto_small` as an exact
reproduction of every setting in the original Concerto paper.

TSGCNet is a classification **adaptation**, not the paper's original segmentation
task. It consumes each triangle's three corner positions plus center (12 channels)
and corner/face normals (12 channels). Local features are taken after `pred3`.
The runtime patches the author's hard-coded CUDA device, 16,000-face reshapes,
and division near zero. The original source is left unchanged on disk.
The source hash and commit are recorded. Original data normalization is replaced
by a common fixed physical scale, preserving thin surface-height variations.

PointNeXt uses the official encoder's local output, followed by mean/max pooling.
The head differs from the original benchmark classification head. Its sampled
positions are coarser than the input surface. Sonata/Concerto features are mapped
through their encoder hierarchy back to sampled input faces before pooling.
All heads use LayerNorm, a 128-unit hidden layer, dropout, and three outputs.

With RGB inputs, PointNeXt and pretrained encoders receive color within the
backbone; TSGCNet retains geometric streams. Every classifier also gets the same
direct pooled RGB route. Thus every model has access to color, but color enters
their spatial architectures differently. The geometry-only condition isolates
this route, though feeding zero color to pretrained Sonata is a distribution
change, not a geometry-only pretrained checkpoint.

## Installation

Your existing Windows environment may already run TSGCNet. The other backbones
need model-specific GPU operators. For running all four, use a separate Linux or
WSL2 CUDA environment. Do not replace a working labeling environment's PyTorch.
The official repositories remain the authority for supported GPU builds.

1. Extract this folder, then fetch the official sources:

```bash
python setup_backends.py
```

This clones HTTPS repositories and records their commits. It does not install
packages, pull changes into existing repositories, or download your data.
Git is required. Official model weights download on first use unless local
checkpoint arguments are supplied.

2. Install a compatible CUDA-enabled PyTorch and torchvision pair. One upstream
Sonata/Concerto combination is Python 3.10, PyTorch 2.5.0, CUDA 12.4. For that
combination, in a fresh environment:

```bash
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install "numpy<2" spconv-cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

The scatter wheel must match the actual PyTorch/CUDA versions. FlashAttention is
disabled by default; install it only if desired and add `--flash`. Sparse GPU
operators still remain necessary. A driver reporting CUDA 13 does not imply that
the installed PyTorch was built for CUDA 13.

3. Install OpenPoints dependencies and GPU operations. From the extracted folder,
use the installation guide in `third_party/PointNeXt/README.md`. The upstream now
offers `pointnext_official` / `openpoints` packages as well as a source installation.
For this source-based adapter, ensure its OpenPoints dependencies and CUDA
extensions are installed. A matching CUDA toolkit (`nvcc`) and C++ compiler are
needed when building operators. Do not blindly run an upstream installation
script that targets a different GPU/CUDA combination. The checked source adapter
imports `third_party/PointNeXt/openpoints` and reads its S YAML configuration.

If incompatible dependencies require separate environments, run subsets using
`--models` with the same data, split, input settings, and output folder. The final
report combines the saved runs. Copy the same `third_party` checkouts to each
environment, preserving commits. Do not mix CPU and GPU timing comparisons.

4. Confirm CUDA and validate the actual models:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python biocal3d_train_3d_models.py --data-root "C:/Users/au662213/repos/biocal3d/data/02-09-2026-Data collected" --preflight
```

Under WSL, use `/mnt/c/Users/au662213/...` for Windows files. Data-loader source
paths inside `geometry.npz` may be Windows paths; when they are needed for texture
recovery, regenerate the ROI RGB/metadata or use paths that exist in that system.
Preflight performs a real forward pass for every requested model. Missing models
are recorded as failures; a lightweight look-alike is never substituted.

## Run all models

PowerShell, one line, from the extracted folder:

```powershell
python biocal3d_train_3d_models.py --data-root "C:/Users/au662213/repos/biocal3d/data/02-09-2026-Data collected" --output "runs/3d_models" --live
```

The default `Path.home()/repos/biocal3d/data/...` also works if that is your folder.
Use `--roi-root` when your processed folder has another name. Default inputs are
geometry + RGB, 2,048 sampled faces, batch size 4, 100 maximum epochs, patience 15,
one seed (42). The program prints the actual device and refuses an automatic CPU
fallback. Explicit CPU testing: `--device cpu`; pretrained sparse backends may
still require CUDA.

Sonata/Concerto extract frozen features once and cache them across epochs/seeds.
Only their heads train. This intentionally avoids recomputing a frozen encoder
on every epoch. Full encoder fine-tuning is not implemented in this version.
Training phases differ substantially in speed; inspect feature-extraction time
as well as head-training time before comparing cost.

Run three seeds once the first run works:

```powershell
python biocal3d_train_3d_models.py --data-root "YOUR_DATA_ROOT" --output "runs/3d_models" --seeds 42 43 44 --resume --live
```

Run a geometry-only experiment in a new output folder:

```powershell
python biocal3d_train_3d_models.py --data-root "YOUR_DATA_ROOT" --inputs geometry --output "runs/3d_geometry_only" --live
```

## Data and splits

Reads `roi.vtp` under `_processed_ROI/<scanner>/G<group>/BCGxTy-n/`.
Never reads `qc.png` as quantitative appearance. Uses actual RGB or UV+texture;
when necessary, recovers source colors using the saved original-to-canonical
transform with strict vertex-matching checks. Missing colors fail explicitly.
An alternate `roi.npz` may store vertices (V,3), faces (F,3), and optional rgb (V,3).
`geometry.npz` alone is never assumed to contain your ROI mesh.

All backbones receive the same deterministic, area-weighted sample of triangle
centers and normals. TSGCNet additionally receives the corresponding triangle
corners and corner normals. Sampling is without replacement when enough faces
exist, with replacement otherwise. Point count is a compute budget, not a claim
that every small plaque island is represented. Inspect `data_qc.csv`.

Coordinates must be in millimeters. Centering uses the area-weighted surface
centroid, and all XYZ axes use the same fixed divisor of 10 mm by default.
No per-axis normalization or independent Z rescaling is performed. Default
pretrained voxel size .01 corresponds to .1 mm under this scale. Adjust point
count and grid size on training/validation data, preserving biologically relevant
detail. No augmentation is used in this first comparison, so the cached probes
and scratch models see the same fixed surface samples.

If `_dino_preparation_80_10_10/scan_manifest.csv` exists, its split is reused
automatically. Alternatively provide `--split-manifest PATH`. The split must
cover every discovered ROI; incomplete coverage fails rather than silently
changing cohorts. Otherwise a deterministic group-stratified specimen split
is made: approximately 80/10/10, rounded to at least one val and one test specimen
per group (8 specimens/group becomes 6/1/1, i.e. 75/12.5/12.5).
Use `--make-split` explicitly to generate a new split despite an existing DINO
split. All stages and scanner acquisitions of a physical specimen stay together.

For custom naming or stage assignments, supply `--manifest` with columns:

```csv
mesh_path,scanner,sample_name,physical_id,group,stage,split
C:/path/roi.vtp,TRIOS3,BCG3T1-4,BCG3-4,3,partial,train
```

Relative mesh paths resolve against the manifest's directory. `split` is
optional. Stage names are baseline, partial, clean. Group/timepoint inference
supports G1-G2 (T0 baseline/T1 clean) and G3-G6 (T0 baseline/T1 partial/T2 clean).
These are experimental-stage targets; observed plaque can remain at a clean
timepoint. The exported manifest is the authoritative record for this run.

## Stop, resume, and compare early

Press Ctrl+C once. A partially completed epoch is discarded; the last completed
epoch and best validation checkpoint are retained. If at least one epoch completed,
validation predictions and the report are exported before stopping the queue.
The report also includes other models that already finished.

```powershell
python biocal3d_train_3d_models.py --data-root "YOUR_DATA_ROOT" --output "runs/3d_models" --resume --live
python biocal3d_train_3d_models.py --output "runs/3d_models" --report-only
```

Resume restores optimizer and RNG state. Exact data, backend state/configuration,
split, input sampling, device, and hyperparameter settings must match. Increasing
`--epochs` is allowed; an already early-stopped run remains early-stopped.
Runs already present require `--resume`; use a new directory for changed settings.
Do not run multiple writers into the same output at once.

## Outputs

- `comparison.html`: local HTML dashboard with curves, confusion matrices, ranking,
  optional focus examples, and model-load failures.
- `comparison.csv`: per-model/seed validation results, specimen-bootstrap intervals,
  training time, feature-extraction time, and parameter counts.
- `comparison_by_model.csv`: mean and SD across seeds; one seed has no SD estimate.
- `comparison.png`: specimen balanced-accuracy comparison.
- `per_scanner_comparison.png`: model-by-scanner performance heatmap.
- `runs/<model>__seed<seed>/history.csv`, `best.pt`, `last.pt`, `config.json`.
- `predictions_val.csv`, `metrics_val.json`, `per_scanner_val.csv/json` per run.
- `scan_manifest.csv`, `experiment.json`, `data_qc.csv` and cached surface samples.
- `failures.json` and each failed run's `error.txt`.

Selection uses **specimen-balanced accuracy**: calculate each specimen's accuracy
within a stage, average specimens within that stage, then average stages. This
limits domination by specimens with more scanner acquisitions. Ordinary scan BA,
accuracy, macro F1, and one-vs-rest macro ROC-AUC are also reported. Scanner BA is
undefined (`null`) if any stage is absent. Bootstrap resamples whole specimens.
With only a few validation specimens the intervals are weak evidence; they do
not include checkpoint-selection or training uncertainty. Validation ranking is
exploratory because the same validation set selects the best epoch. A common
learning rate/epoch budget does not establish optimal hyperparameters for every
architecture. This is a first benchmark, not a final statistical model-selection
study. Cross-validation and leave-one-scanner-out are not implemented here.

The held-out test meshes are not opened by default. After choosing the protocol,
use `--resume --evaluate-test` to evaluate saved validation-selected checkpoints
on test specimens. Test predictions/metrics are saved separately; the dashboard
continues to rank validation scores. Avoid repeated test-guided tuning.

## Optional focus and local features

`--focus-examples 6` produces region-removal sensitivity maps for up to six
validation examples per model. They are slower because every region needs a
forward pass. Red indicates a probability decrease after removal; blue an
increase. Removal plus deterministic resampling changes geometry/density, so
maps are diagnostic perturbation effects, not causal localization or plaque
annotations. Examples cover scanner/stage combinations deterministically.

`--export-local-features` retains Sonata/Concerto per-sampled-face descriptors,
coordinates and original face indices in their feature-cache NPZ files. Scratch
model checkpoints retain their local encoders for later segmentation/fusion work;
this script does not yet supply a trained dense decoder or bidirectional fusion.

Offline weights: use `--sonata-checkpoint PATH` and `--concerto-checkpoint PATH`.
Pretrained checkpoint state hashes and source commits are stored. No generated
performance numbers are provided: actual BioCal3D training occurs on your machine.

## Validation performed while creating this package

The package includes `--self-test` for head learning, group-leakage rejection,
identity parsing, and class-missing metrics. Additional development checks use
synthetic ROI meshes to exercise loading, an actual official TSGCNet forward/
backward pass, training, resume, report generation and texture recovery. These
checks establish code-path behavior, not performance on BioCal3D. The official
PointNeXt/Sonata/Concerto GPU kernels and pretrained weights need the real
`--preflight` check on your CUDA machine.

## Official sources

- https://github.com/guochengqian/PointNeXt
- https://github.com/guochengqian/openpoints
- https://github.com/ZhangLingMing1/TSGCNet
- https://github.com/facebookresearch/sonata
- https://github.com/Pointcept/Concerto

Source repositories and weights are fetched from the authors; they are not
redistributed in this package. Follow their respective licenses.
