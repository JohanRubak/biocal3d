# BioCal3D fixed-split matched study

## Run on the Windows research PC

Extract this ZIP into `C:\Users\au662213\repos\biocal3d\classification`. The ZIP contains the `BioCal3D_Fixed_Split_Study` folder. In Anaconda Prompt run:

```bat
cd /d "%USERPROFILE%\repos\biocal3d\classification\BioCal3D_Fixed_Split_Study"
python run_fixed_study_on_this_pc.py
```

The launcher uses `C:\Users\au662213\repos\biocal3d\data\Approved ROIs_reoriented`, the installed `biocal3d-dinov3` and `biocal3d-concerto` environments, and the existing pretrained models. It looks first in the previous `BioCal3D_Stage_Ablations` installation, then the old matched-study installation and the new folder. It also searches `classification` and `Downloads` for checkpoint files. A checkpoint elsewhere can be passed with `--dinov3-checkpoint` or `--concerto-checkpoint`.

The run is stored in `runs\matched_fixed_split` under the new folder. A completed `matched_raw` study supplies the specimen assignments. This study extracts its own feature caches with the added harmonised condition. Restart the same command after an interruption to reuse finished results and compatible feature caches.

The default command runs training, validation, test and the test gallery in order. To inspect validation first, add `--validation-only`. Run the original command afterwards to score test with the saved training models and make the gallery.

## Study design

| Partition | Physical specimens | Scans | Use |
| --- | ---: | ---: | --- |
| Training | 24 | 256 | Fit the scanner map and all stage classifiers |
| Validation | 12 | 128 | Score the frozen training models |
| Test | 12 | 128 | Score the same models and select spatial examples |

All scanners and stages from a physical specimen travel together. The earlier saved development group 0 becomes the validation set once. The other development specimens form the training set. The existing test specimens are used for test, including all stages and scanner views of BCG3-1. If the previous cohort is unavailable, the launcher makes one new stratified assignment with two validation and two test specimens per growth group. Each stage classifier is fit once on training. Each boosting condition also fits one mask booster. Validation and test share those fitted models. Every condition pools the training scans from all four scanners into one classifier.

| Model | No colour | Raw colour | Harmonised colour |
| --- | --- | --- | --- |
| GMM | Local lightness contrast | Local darkness and redness | Same rule on harmonised RGB |
| Gradient boosting | Local lightness contrast | Local Lab contrast | Same learner on harmonised RGB |
| DINOv3 | Grayscale from the raw render | Raw RGB render | Harmonised RGB render |
| Concerto | XYZ and normals | XYZ, normals and raw RGB | XYZ, normals and harmonised RGB |

### How the mask methods work

**GMM** fits a two-component mixture separately on each scan. The RGB rule looks for a component with higher local darkness and redness and keeps pixels above 0.95 membership. The grayscale rule uses local darkness. The mask is computed from the scan alone.

**Gradient boosting** learns local pixel scores on the training scans. Its supervision comes from comparing earlier and later cleaning stages for the same specimen and scanner. The mask uses a cutoff learned from final-clean training scans. The mask booster and the second-stage logistic regression both fit on the fixed training specimens. The regression uses the booster masks from that training set, so validation and test are the independent checks of the resulting pipeline.

**Stage prediction from either mask** uses three values: the ROI fraction covered by the mask, the fraction covered by its largest connected region, and the log-transformed count of regions. The logistic stage classifier predicts baseline, partial or clean. DINOv3 and Concerto use the same classifier family on pooled frozen embeddings.

### Colour harmonisation

By default, the launcher computes per-channel RGB quantiles on area-sampled surfaces from the **training specimens only**. It pairs the same specimen and stage across scanners and fits a regularised affine colour map to TRIOS5. This is scanner-relative distribution matching. The fitted mapping is frozen, applied to mesh RGB, and used by all four models. Spatial coordinates, normals, and render geometry follow the same settings across conditions. The raw and harmonised conditions are evaluated separately to measure how this specific map affects classification.

For an independent colour chart instead, pass a CSV with `scanner,patch_id,r,g,b` columns. Use one 0–255 sRGB patch measurement per scanner and patch, with at least six matched, varied swatches. Use a separate run folder for a second mapping, for example:

```bat
python run_fixed_study_on_this_pc.py --run runs\matched_fixed_chart --swatches scanner_swatches.csv
```

## Open the results

All paths below are relative to `runs\matched_fixed_split`.

| Output | File or folder |
| --- | --- |
| Exact scan membership | `fixed_split.csv` |
| Colour transform and provenance | `colour_mapping.json` |
| Validation results | `matched_results\validation\` |
| Test results | `matched_results\test\` |
| Balanced accuracy and macro F1 overview | `matched_results\test\figures\performance.png` |
| Raw colour gain and harmonisation gain | `matched_results\test\figures\paired_colour_deltas.png` |
| Per-scan scores and predicted stage | `matched_results\test\predictions.csv` |
| Manuscript and failure examples | `matched_results\test\examples\` |

Both evaluation partitions report the same metrics for all 12 conditions, pooled and for each scanner. The paired plots show raw colour minus no colour and harmonised colour minus raw colour. Their 95% intervals resample whole specimens within growth group. The test estimate is an internal holdout on the represented scanners.

The test gallery exports **all four views of BCG3T1-1**, with TRIOS5 listed first, plus up to three other examples. It seeks correct predictions, model disagreements and errors when they occur. Each PNG shows the recorded stage, the predicted stage and model probability in each condition. The same scan and projection are used across the panel. Its companion NPZ contains the projected masks, signed attribution arrays and valid ROI. The selection reasons and predictions are recorded in the two gallery index files.

Yellow overlays show GMM and boosting candidate masks. Red and blue overlays show signed contributions of DINOv3 or Concerto features to a common stage logit contrast within each scan. The colours can describe which regions drove the stage prediction. Independent spatial labels are needed to measure plaque segmentation accuracy.

## Recommendation for attention and pseudo-labels

Use these frozen classifiers and their signed maps for the main matched comparison. Wen and colleagues used a trained attention pooling block, additional domain adaptation and a segmentation decoder. Changing pooling in the main benchmark would introduce a second experimental change alongside colour. Keep learned attention as a separate localization experiment with the fixed specimen assignment. Compare candidate masks with the photo-backed specimens using Dice, precision, recall and plaque coverage error before choosing a segmentation approach. [Colorectal gland segmentation paper](https://doi.org/10.1038/s41598-026-36256-0)
