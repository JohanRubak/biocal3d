# BioCal3D attention ablation study

## Run on the BioCal3D Windows PC

Extract the ZIP into `C:\Users\au662213\repos\biocal3d\classification`. In Anaconda Prompt, open the extracted folder containing `run_study_on_this_pc.py` and run:

```bat
python run_study_on_this_pc.py --smoke
```

Open `runs\smoke\results\index.html`. This checks the whole pipeline on one training, one validation, and one test specimen. The smoke scores are pipeline checks.

For the full experiment, run:

```bat
python run_study_on_this_pc.py
```

Open `runs\full\results\index.html`. Run the same command again to resume with completed feature caches and fitted heads. Use a new `--run` folder if the data or encoder settings change. A plain `--prepare-only` validates the full dataset and writes its split before extracting features. Add `--smoke` to prepare the smoke subset.

To upgrade a completed run to the longer schedule, replace the package scripts in place and run:

```bat
python run_study_on_this_pc.py --retrain-heads
```

This reuses the split and cached encoder features, refits heads with a different training schedule, and refreshes the report. To change the defaults, pass `--epochs` or `--patience`. Add `--smoke` to refit a completed smoke run, which still defaults to a three-epoch cap.

After a full run has completed, regenerate the figures with the fitted heads:

```bat
python run_study_on_this_pc.py --figures-only
```

For the six-condition report in `results_harmonized` (including raw and harmonised RGB), specify the harmonised ROI export when regenerating figures:

```bat
python run_study_on_this_pc.py --run runs/full_fov13p4 --figures-only --harmonized-roi
```

This regenerates the harmonised report from the locked test predictions and fitted heads, including the selected and all-configuration qualitative figures. It does not retrain the attention heads.

Add `--smoke` to refresh a completed smoke run. The launcher replays native encoder features for the additional partial-stage examples when selected and reuses the primary example caches. The same model checkpoints and encoder environments are needed for replay. Figures can also be regenerated with checkpoints from the earlier 50- and 120-epoch packages.

The launcher uses the existing `biocal3d-dinov3` and `biocal3d-concerto` Conda environments. By default it reads `%USERPROFILE%\repos\biocal3d\data\Approved ROIs_reoriented` and checks the approved index, mesh identities and orientation status. It searches the current package and prior BioCal3D study folders for the official local encoder checkouts and checkpoints. Supply `--roi`, `--dinov3-repo`, `--dinov3-checkpoint`, `--concerto-repo`, or `--concerto-checkpoint` to override a path. The repositories and checkpoints are separate from this ZIP. Extraction uses CUDA in the two existing environments. The attention heads are fitted on the CPU.

If an encoder environment is missing, run the corresponding setup commands in Anaconda Prompt. DINOv3 requires the official ViT-S/16 checkpoint from your earlier study.

```bat
conda create -n biocal3d-dinov3 python=3.11 -y
conda run -n biocal3d-dinov3 python setup_model_env.py dinov3
conda create -n biocal3d-concerto python=3.10 -y
conda run -n biocal3d-concerto python setup_model_env.py concerto
```

## Export a dataset of harmonized approved ROIs

In Anaconda Prompt, from this study folder, run:

```bat
conda run --no-capture-output -n biocal3d python export_harmonized_rois.py
```

The defaults use `%USERPROFILE%\repos\biocal3d\data` with `Color-harmonized Data` and `Approved ROIs` as inputs and a new `Approved ROIs_harmonized` as output. Use `--harmonized`, `--approved` or `--output` with quoted paths to override them. To match the Attention Study's orientation, pass `--approved "%USERPROFILE%\repos\biocal3d\data\Approved ROIs_reoriented"`. Its default output is `Approved ROIs_reoriented_harmonized`. The converter uses NumPy, SciPy, scikit-learn, Pillow, plyfile and VTK in the BioCal3D environment.

The script matches scanner and BCG scan ID regardless of the group-folder wording or scanner suffix. It verifies vertex and triangle correspondence before transferring RGB to the already approved surfaces. It preserves the indexed scan folder names, vertices, triangle order, orientation, raster footprint and split metadata. It writes harmonized geometry, mesh and raster files using the template's available formats, refreshes the index and manifests, and records the match method in `harmonized_roi_audit.csv`. The export contains the approved analysis surfaces. Raw texture images, historical previews and annotation working sessions are outside this export.

Use `--check-only` to check matching without an output. A missing, duplicate or ambiguous match stops the conversion. An export appears at the requested output path only after every scan succeeds.

If Windows blocks the final rename, close applications or terminals using the staging or destination folder. The exporter retries briefly and keeps the completed staging folder if the rename is still blocked. To publish it without reprocessing, run the exporter with `--publish-staged "C:\path\to\biocal3d_harmonized_...\export"`, using the staging path from the error. If an earlier exporter already removed the temporary folder, rerun the conversion.

## Experiment

The 13.4 mm field of view changes the DINOv3 feature inputs. For an existing study with
16 mm caches, use a fresh output folder. For example, for all six conditions:

```bat
python run_study_on_this_pc.py --run runs/full_fov13p4 --harmonized-roi
```

The new folder regenerates features and fits classifiers using the same deterministic
specimen split. The previous run remains available in its original folder.

| Encoder | First condition | Second condition |
|---|---|---|
| DINOv3 ViT-S/16 | Grayscale render | Raw RGB render |
| Concerto Small | Geometry and surface normals | Geometry, surface normals and raw RGB |

The fixed specimen split is **30 training, 6 validation, 12 test** (62.5%, 12.5%, 25%). Each of the six groups contributes five, one and two specimens respectively. This retains the previous study's seeded two-specimen holdout per group and gives the per-scanner test estimates two specimens per group. The 512 scans split into 320, 64 and 128. Each specimen stays in one partition across every scanner and cleaning time. The full BCG3-1 specimen is reserved for test, including all four scanner views of BCG3T1-1. `split.csv` lists every scan and specimen assignment.

Both pretrained encoders are frozen. Each condition fits its own attention block and three-class linear head. The paper-inspired block applies a one-hidden-layer MLP to each patch or point feature, uses softmax over valid locations, and weights the scan embedding by learned attention. The three logits represent baseline, partial, and fully cleaned. This package implements the paper's attention-based classifier stage with BioCal3D inputs and labels.

DINOv3 uses aligned 448 × 448 renders at a 13.4 mm field of view and excludes patches below 5% ROI coverage from the classifier. Concerto encodes 8,192 reproducibly sampled surface points with 0.10 mm voxel spacing and normals in both arms. A deterministic cap of 512 locations per scan keeps the head caches compact. The cap and the sampling seed apply equally to paired conditions. All train, validation and test bags use the same selection and feature-normalization rules. Full native embeddings are saved only for BCG3T1-1 for the matched PCA illustration.

All heads use the same 128-unit ReLU attention MLP, batch size 16, AdamW with a constant learning rate of 0.0001 and weight decay 0.00001, and a maximum of 1,000 epochs. Every strictly lower validation loss sets a new best and resets patience. Training stops after 30 consecutive epochs without a new best. The trainer checks every epoch. The progress log reports every five epochs and shows the best loss, its epoch, and the number of epochs since that best. Class weights come from the training scans only. The loss figure uses the same class-weighted cross-entropy for train and validation, evaluated at the end of each epoch. Test metrics use the checkpoint with the lowest validation loss.

## Outputs

| Output | Contents |
|---|---|
| `loss_progress.png` | Train and validation loss for each configuration. The plot marks the lowest validation loss and shows whether training ended at the epoch limit or stopped early. |
| `balanced_accuracy.png` | All configurations, pooled and separately for each scanner. Balanced accuracy averages the three stage recalls. |
| `balanced_accuracy_bars.png` | Horizontal bars grouped by scanner and encoder, with the same pooled and per-scanner balanced accuracy and specimen-bootstrap intervals. |
| `roc_curves.png` | Pooled one-vs-rest ROC curves for each active encoder/input configuration. Each panel shows the three cleaning stages, class-specific AUCs and the macro-average AUC. Curves use the held-out scan probabilities and are descriptive of this test set. |
| `BCG3T1-1_qualitative_all_configurations.png` | Compact four-scanner layout showing attention, PCA features and proposed plaque masks for every active input configuration. With the harmonised report enabled, this includes all six configurations. |
| `partial_stage_selected_model_examples.png` | Unique selected partial-stage scans, each with attention, PCA features and proposed masks for every active input configuration. If one scan meets multiple selection rules, it appears once with all applicable criteria in its panel heading. |
| `confusion_*.png` | A matrix for each configuration in each scanner's figure, plus a pooled figure. Entries show counts and percentages within the observed stage. |
| `paired_colour_effects.png` | Within-encoder changes in balanced accuracy from adding RGB, with paired 95% intervals in the full run. The bootstrap resamples complete test specimens within each group. |
| `BCG3T1-1_learned_attention.png` | Learned weights overlaid on all four scanner views with their test predictions. The shared scale expresses attention relative to uniform weighting. A value of 1 is uniform. Concerto weights are interpolated from the selected surface points across the scan ROI. |
| `BCG3T1-1_matched_features.png` | Native features aggregated to one DINOv3 patch grid. All rows share the supported area, and all use the same bilinear display interpolation. PCA colours are shared within each encoder. |
| `examples/*.png` | Four primary scan panels and up to six additional partial-stage examples. Each shows the original scan and, for every model input, its predicted stage, learned attention, matched PCA, and proposed plaque mask. |
| `examples/*.html` | Offline plaque threshold viewers for the same scans, with a shared slider and a live mask area for each model. |
| `confidence_examples.json` | Selected scans, observed stages, per-model predictions and top probabilities, correct model counts, combined scores and the ranking rules. |
| CSV, JSON and NPZ files | Per-scan test predictions, metrics by stage, confusion counts, loss history, paired effects and raw learned attention weights for the primary example. |

The additional examples come from held-out scans with an observed partial cleaning stage, excluding the entire primary BCG3-1 specimen in the full run. Highest and lowest confidence use the median top-class probability across the active models, with mean top probability and scan ID breaking ties. Most and fewest correct use the number of models predicting partial. Highest and lowest combined score use the sum of the correct models' confidence scores divided by the total number of models. Incorrect predictions contribute zero. Ties in correct count or combined score favour scans without an assigned role, then follow the confidence ranking in the same direction. A scan selected by several rules gets a shared panel. The default smoke run shows the four primary views. The PCA basis and colour scaling are fitted to the primary example and reused for the additional scans.

The plaque masks adapt the paper's attention blending, thresholding and CRF pipeline. For image-enabled inputs, the overlay blends 75% scaled attention and 25% original scan darkness. Concerto geometry-only uses attention and spatial affinities. A fully connected, Gaussian pairwise CRF refines the proposals on the DINOv3 patch grid, and the result is projected back to the scan. The CRF starts from a seed cutoff of 0.50. The static masks use a final threshold of 0.10. A model prediction of fully cleaned produces an empty proposed mask. The cyan overlays represent unvalidated pseudo-labels. [Paper](https://doi.org/10.1038/s41598-026-36256-0)

From `index.html`, open an **Adjust plaque threshold** link and move the slider from 0 to 1. It applies the same final cutoff to every model's refined score map and shows the percentage of the scan ROI covered by each proposed mask. The default 0.10 matches the static figures. Run `--figures-only` to regenerate the viewers and masks for a completed study.

## Compare the exported harmonized RGB with the raw scan colours

After exporting `Approved ROIs_reoriented_harmonized`, run from this study folder:

```bat
python run_study_on_this_pc.py --harmonized-roi
```

The default harmonized input is `%USERPROFILE%\repos\biocal3d\data\Approved ROIs_reoriented_harmonized`. For a different location, supply its quoted path after `--harmonized-roi`. Use the same `--run`, `--epochs` and `--patience` as the existing raw run if you overrode those defaults.

The launcher pairs each harmonized scan with the locked raw cohort and checks that the vertices and faces match exactly. It adds DINOv3 harmonized RGB and Concerto geometry + harmonized RGB. The four existing conditions supply the comparisons, and the launcher reuses their compatible feature caches and heads. All six conditions use the same training, validation and test specimens. A separate report in `runs\full\results_harmonized\index.html` contains the usual figures, predictions and plaque viewers, including paired harmonized-versus-raw RGB differences for each encoder and scanner. The heads use the supplied harmonized colours as fixed preprocessing.

## Use a fitted colour harmoniser directly

1. Use `split.csv` to fit the colour harmoniser using only specimens assigned to `train`. Use a fitted artefact that has kept all validation and test specimens outside its fitting process.
2. Edit `colour_harmonization.py`. Implement `load_transform(artifact)` to load the fitted weights and return a callable `transform(rgb, scanner)`. Its input and output are ordered vertex RGB arrays of shape `(n_vertices, 3)` in `[0,1]`. The scanner argument is the canonical scanner name. The same adapter is loaded in both encoder environments.
3. In the study folder, run:

```bat
python run_study_on_this_pc.py --harmonization-artifact "C:\path\to\fitted_model"
```

This applies the transformation to vertex colours before rendering or surface colour interpolation. It adds DINOv3 harmonised RGB and Concerto geometry + harmonised RGB to the same split, alongside the four initial conditions. The launcher reuses the raw feature caches and fitted raw-input heads in `runs\full`, and writes the six-condition report to `runs\full\results_harmonized`. Each new colour arm gets its own attention head. The report compares harmonised RGB with raw RGB within each encoder. Use a new run directory to test a different fitted harmoniser or adapter implementation.
