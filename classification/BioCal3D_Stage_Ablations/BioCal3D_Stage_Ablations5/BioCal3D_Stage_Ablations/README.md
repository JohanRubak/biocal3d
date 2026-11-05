# BioCal3D stage ablations

This pipeline trains independent scan-level classifiers from the main **Approved ROIs_reoriented** export. Each encoder and input condition has its own head. Pooling combines patch or point features within a single scan and encoder. No features, logits or predictions from the two encoders are combined.

| Frozen backbone | Inputs | Pooled feature |
|---|---|---|
| DINOv3 ViT-S/16 | Raw RGB and grayscale from the same orthographic render | ROI-weighted patch mean and covered-patch max |
| Concerto Small | XYZ and mesh normals with RGB absent, and the same geometry with raw RGB | Mean and max over the same area-sampled points |

The four heads use the same formulation, train-only feature standardization followed by a regularized linear classifier (scikit-learn logistic regression, L2 C=1.0 and balanced class weights). There is no head hyperparameter search. This is a learned head on the encoder embeddings. No separate hand-engineered colour-summary model is fitted. DINOv3's official evaluation includes linear and logistic-regression heads for image classification. Concerto's released ScanNet head predicts semantic classes at the point level. Train new scan-level heads on BioCal3D's stage labels.

## Data and split

Point `--approved-root` at the main BioCal3D Studio orientation export **Approved ROIs_reoriented** containing `dataset_index.json` and per-scan `geometry.npz` files. If Studio created a collection, select the main all-groups dataset within it. Keep calibration and agreement subsets out of this input. The pipeline reads the indexed mesh and vertex RGB and renders a standardized orthographic image. For texture-based scans, this is the RGB baked into the exported vertices. The code never writes into the input tree or changes ROI approval. A raw directory of standalone meshes is not a Studio orientation export.

The single task is **pooled three-stage classification** using all 48 specimens and 512 scans across G1–G6. G1–G2 contribute baseline and final-cleaning scans, and G3–G6 also contribute partial-cleaning scans. Report per-stage recall alongside balanced accuracy. Partial-cleaning recall reflects G3–G6, where that stage was collected. The pooled model could use growth-cohort cues to help rule out partial cleaning. There are no separate fresh, mature or binary classification runs.

The default split reserves **12 test specimens (25%)**, two per group, and uses the other **36 development specimens (75%)** in three-fold cross-validation. Each development fold has **24 training, 12 validation and 12 locked test specimens**, or **50/25/25** of the full dataset (256/128/128 scans for a complete cohort). Validation rotates across the development specimens. The test specimens are the same in each fold and are evaluated separately, once, after the settings are settled. All stages and scanner views of a specimen stay together. The split is recorded in `specimen_split.csv`, with scan details and geometry hashes in `manifest.csv` and `cohort.json`.

Run **all four ablations on identical folds**. Cache frozen encoder features per scan, then refit each condition's feature scaler and classifier head within each training fold. Compare the conditions using paired out-of-fold predictions. For the final test, refit each head on all 36 development specimens and evaluate the same four conditions once on the locked test.

Existing split fields from the export are ignored for the new study. A new split gives an internal held-out assessment. If these data were already used to select the pipeline, a different seed does not make them independently collected validation data. Use `--cv-only` for an explicitly internal CV study without a holdout, or `--split-json` only when deliberately preserving a reviewed holdout. The shared split evaluates unseen specimens from represented groups and scanners.

## Setup on Windows

Extract the ZIP, open Anaconda Prompt, and change to the folder containing this README. Git for Windows must be on PATH. Run `git --version` and `nvidia-smi` to check Git and the NVIDIA driver.

Create two dedicated Conda environments. Skip a create command if that named environment already exists with the required Python version.

```bat
conda create -n biocal3d-dinov3 python=3.11 -y
conda create -n biocal3d-concerto python=3.10 -y
conda run --no-capture-output -n biocal3d-dinov3 python setup_model_env.py dinov3
conda run --no-capture-output -n biocal3d-concerto python setup_model_env.py concerto
```

The helper installs matching CUDA-enabled PyTorch and torchvision wheels and the shared pipeline dependencies. For DINOv3 it also installs the upstream repository requirements and checks that the PyTorch Hub entry point imports. It clones the two official repositories into `vendor/`, verifies CUDA access, and checks the CUDA scatter operation in the Concerto environment. DINOv3 uses PyTorch 2.7.1 with CUDA 12.6. Concerto uses PyTorch 2.5.0 with CUDA 12.4 and the matching torch-scatter and spconv wheels. The two CUDA wheel builds use the same NVIDIA driver. The Concerto adapter runs with FlashAttention disabled.

The Concerto setup downloads and checks the official `weights/concerto_small.pth`. For DINOv3, request model access through [Meta’s DINOv3 repository](https://github.com/facebookresearch/dinov3#pretrained-models), download the **ViT-S/16 LVD-1689M PyTorch Hub .pth checkpoint**, and place it at `weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`. Keep a checkpoint you already downloaded by passing its actual path to the smoke command below. The helper also accepts `--checkpoint` for an existing official file. Source weights and dependency references are in the [DINOv3 repository](https://github.com/facebookresearch/dinov3), [Concerto repository](https://github.com/Pointcept/Concerto) and [Concerto Small model download](https://huggingface.co/Pointcept/Concerto/blob/main/concerto_small.pth).

If a previous DINOv3 setup reports `No module named 'torchmetrics'`, repair that environment from the extracted folder:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python -m pip install -r vendor/dinov3/requirements.txt
```

Then retry the DINOv3 smoke command below.

### Concerto on WSL2 if native Windows fails

Concerto’s reference environment targets Linux. If its CUDA extension or encoder smoke check fails in Windows, use Ubuntu on WSL2 for Concerto. Check WSL2 with `wsl -l -v` in PowerShell. If Ubuntu is missing, use `wsl --install -d Ubuntu` from an elevated PowerShell and follow Windows' prompts. Inside Ubuntu, use `nvidia-smi` to confirm GPU access through the Windows NVIDIA driver. Install Git and a Linux Conda distribution there, then run the Concerto create and setup commands above in Ubuntu. The helper supports the same Python and CUDA wheel combination on Linux x64. With the project and dataset on a drive mounted in WSL, use the same project and run folder for both models. Add `--snapshot "/mnt/c/path/to/Approved ROIs_reoriented"` to the Concerto extraction command to translate the Windows paths saved by `prepare`. The pipeline checks the relocated geometry files against the locked hashes.

## Run

From the extracted folder, replace the example ROI path. Prepare one shared study run. The smoke commands then try both input modes of each encoder on one real development scan.

```bat
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py prepare --approved-root "C:\path\to\Approved ROIs_reoriented" --run runs/stage_pooled_v3
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py extract --run runs/stage_pooled_v3 --encoder dinov3 --repo vendor/dinov3 --checkpoint weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --smoke
conda run --no-capture-output -n biocal3d-concerto python stage_ablation.py extract --run runs/stage_pooled_v3 --encoder concerto --repo vendor/concerto --checkpoint weights/concerto_small.pth --smoke
```

Remove `--smoke` from both extraction commands once they pass. Then run:

```text
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py fit --run runs/stage_pooled_v3
```

The two full extraction commands can run in separate environments, sequentially. Embeddings are cached per scan, with encoder and geometry provenance. Use `--spatial` during extraction to retain patch and point embeddings and the correspondence metadata for future fusion. This increases disk use. The geometry-only and colour conditions share all point coordinates, normals, voxel settings and model weights. Colour is omitted by zeroing the colour channels in encoder feature space. The 2D conditions share render, ROI mask, image dimensions and field of view. Grayscale is fixed RGB luma replicated to three channels before model normalization.

Defaults are 448 pixels, a 16 mm physical field of view, 8192 area-sampled points and a 0.10 mm Concerto voxel. The 16 mm field is the existing 2D benchmark convention. If using the manuscript's 13.4 mm view, set `--field-mm 13.4` on both DINOv3 extraction commands after verifying all ROIs fit. Geometry is plane-aligned and transformed with one fixed scale, preserving physical relief. Before committing the run, inspect the rendered face and check the reported occupied voxel count. If changing extraction settings after a full run, start a new run directory.

Results are under `development_cv/all_groups_three_stage/`. The saved heads in this directory are separately fit on all development scans for reuse. The reported development scores use fold-held-out predictions. Open `summary.csv` for pooled and per-scanner performance, `predictions.csv` for one held-out prediction per scan and ablation, and `paired_deltas.csv` for the colour effects on balanced accuracy and per-stage recall. `metrics.json` includes confusion matrices and paired whole-specimen bootstrap intervals. The intervals are descriptive and conditional on these splits and fits. Performance is per scan with specimen-disjoint assessment. The four scans of a specimen-stage do not constitute four independent specimens.

If you reserved a test set, repeat both extraction commands with `--partition test` after settling the study settings. Run `conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py fit --run runs/stage_pooled_v3 --test` once. The heads are refit on all development scans and the fixed assessment writes to `locked_test/all_groups_three_stage/`, including the four fitted heads. The default development workflow never extracts test features.

## Plot performance and head optimization

Install Matplotlib into the DINOv3 environment once:

```bat
conda run -n biocal3d-dinov3 python -m pip install matplotlib
```

Plot existing development results from their folder or use an exported ZIP as the first argument:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python plot_stage_ablation.py runs/stage_pooled_v3/development_cv/all_groups_three_stage --out runs/stage_pooled_v3/plots/development
```

This saves `performance.png` and `confusion_matrices.png`. For locked test plots, pass the `locked_test/all_groups_three_stage` folder and choose a different output folder.

To examine how the heads improved while fitting, add `--trace-run runs/stage_pooled_v3` to the development plotting command. It reads the existing development embeddings and replays the same scaler and logistic regression on the original folds with several increasing caps on optimizer iterations. It logs the converged iteration count per fold and saves training and validation log loss, validation balanced accuracy, and actual iteration counts in `optimization_replay.png` and `optimization_replay.csv`. This is a diagnostic replay from cached embeddings. The original fit did not log intermediate iterations. It also checks the final replayed validation probabilities against the originals. Keep the same scikit-learn and SciPy environment for that comparison. There is no need to re-extract embeddings or access the locked test.

The ROI loader, renderer, grayscale conversion and DINO adapter come from your earlier 2D benchmark. The Concerto adapter uses the pretrained encoder and hierarchy mapping from your earlier 3D investigations. Synthetic end-to-end checks used the real split, render and fitting code with simulated encoder embeddings. GPU extraction requires your local official model environments and weights.


## Local-feature pseudo masks

See [PSEUDO_MASKS.md](PSEUDO_MASKS.md) for the new DINOv3/Concerto local-feature visualizations and exploratory plaque pseudo masks. Start with development spatial caches, fit `make_feature_pseudomasks.py fit`, then use `predict` for held-out development previews. The original encoder and stage-classification Python modules are unchanged.


## Repository-style PCA and cosine similarity inspector

See [FEATURE_MAPS.md](FEATURE_MAPS.md) for `inspect_feature_maps.py`: an interactive desktop viewer for per-specimen PCA, clicked reference cosine maps, cross-scan comparisons and prompted candidate masks for DINOv3 and Concerto. It uses spatial caches directly and does not need the stage-association pseudo-model fit.


## Explain the fitted stage-classifier decisions

See [CLASSIFIER_EXPLANATIONS.md](CLASSIFIER_EXPLANATIONS.md) for `explain_stage_classifier.py`: exact logistic-regression feature contributions, mean/max spatial allocation, feature deletion with random controls, and optional original-input interventions with encoder re-inference. Use this workflow to investigate why stage classification works; PCA/similarity maps alone do not explain the fitted head.


The classifier explanation script now accepts batch filters (`--partition`, `--stage`, `--scanner`, `--specimen`, `--limit`, `--all`, multiple `--scan` IDs), writes a local HTML gallery and combined CSV, and saves `allocation_detail.png` with separate mean/max contrast scales. See CLASSIFIER_EXPLANATIONS.md for commands.
