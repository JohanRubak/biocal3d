# BioCal3D matched cleaning-stage study

The benchmark predicts baseline, partially cleaned or final clean from one scan. Four model families share one specimen split, recorded stage labels and scoring code.

| Family | No colour | Raw colour | Harmonised colour |
| --- | --- | --- | --- |
| GMM | Grayscale, local darkness | RGB, local darkness and redness | Same rules on harmonised RGB |
| Gradient boosting | Grayscale, local lightness contrast | RGB, local Lab contrast | Same ranking model on harmonised RGB |
| DINOv3 ViT-S/16 | Grayscale from the raw render | Raw RGB render | Harmonised RGB render |
| Concerto Small | XYZ and normals | Same XYZ and normals with raw RGB | Same XYZ and normals with harmonised RGB |

The GMM and booster convert their candidate masks to projected ROI fraction, largest connected-region fraction and log(1 + region count). The two encoders use pooled frozen features. The mixture itself is fitted to each scan individually without stage labels. Every condition fits the same type of balanced, standardised multinomial logistic regression head. The colour-free GMM drops redness and the colour-free booster drops chromatic features and the redness gate. These are luminance-only adaptations.

## Cohort and calibration

Use the main **Approved ROIs_reoriented** export. For a complete cohort the benchmark expects 48 physical specimens and 512 scans. It reserves 12 specimens and 128 scans for test. Three development folds rotate across 36 specimens and 384 scans, with 24 specimens used for training and 12 for validation in each outer fold. The booster also uses cross-fitting within each outer training set to train its stage head on held-out mask predictions. Each fold uses its own booster fits and clean thresholds. Test boosters are refit using all 36 development specimens.

Pin physical specimen **BCG3-1**, including all 12 scanner-stage scans, to test. **BCG3T1-1** is its partial-cleaning scan on each scanner. The earlier default split recipe would put BCG3-1 in development. Document any previous use of the specimen and describe the new evaluation as an internal holdout. All scanners are represented in training. Scanner-specific scores measure generalisation to held-out specimens on those scanners.

This package applies one independently fitted scanner-to-reference colour map at the mesh vertices, before orthographic rendering or 3D point colour interpolation. XYZ, normals, sample locations, ROI and projection stay fixed. Use the same map in both encoder extractions. For a reproducible baseline, supply paired patch medians from an independently scanned colour chart as a CSV with columns `scanner,patch_id,r,g,b`. Use one median per scanner and patch on the 0–255 sRGB scale. Include at least six diverse matched patches covering light substrate and pink or red tones. The fitter uses a regularised affine map to TRIOS5 and clips mapped RGB to the valid range. Reference patch values come from TRIOS5, so absolute reference colours are optional for this relative fit.

If a project harmoniser was fitted on BioCal3D specimens that overlap this study, use an independent calibration fit or adapt the pipeline to fit it inside every training partition. The packaged affine mapper assumes independent calibration data. Freeze the mapping and confirm its colour behaviour on calibration data before evaluating the test set. If calibration data are pending, omit `--harmonization` from both extraction commands to run the eight raw/grayscale/geometry development conditions in a separate pilot run.

## Run

Use [the existing environment setup](STAGE_ABLATIONS.md#setup-on-windows) for the official checkpoints and the two Conda environments. The same document covers the `--snapshot` path translation if Concerto runs in WSL. The commands below assume the original environment names. Start a **new run directory** for the matched study. Replace the approved export path and swatch CSV with the actual paths.

```text
conda run -n biocal3d-dinov3 python -m pip install matplotlib
conda run -n biocal3d-dinov3 python scanner_harmonization.py --swatches scanner_swatches.csv --out scanner_colour_map.json
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py prepare --approved-root "C:/path/to/Approved ROIs_reoriented" --run runs/matched_colour --require-test-specimen BCG3-1
```

A reviewed previous split can be supplied with `--split-json` if it already holds out BCG3-1. Otherwise the command above pins BCG3-1 while retaining two test specimens per group. The manifest lists every scan, its physical specimen, partition, fold and geometry hash.

```text
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py extract --run runs/matched_colour --encoder dinov3 --repo vendor/dinov3 --checkpoint weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --harmonization scanner_colour_map.json
conda run --no-capture-output -n biocal3d-concerto python stage_ablation.py extract --run runs/matched_colour --encoder concerto --repo vendor/concerto --checkpoint weights/concerto_small.pth --harmonization scanner_colour_map.json
conda run --no-capture-output -n biocal3d-dinov3 python matched_study.py --run runs/matched_colour --partition development
conda run --no-capture-output -n biocal3d-dinov3 python matched_figures.py --run runs/matched_colour --partition development
```

Review the development output, then freeze the settings. Extract test embeddings using the same arguments, including any field-of-view changes. Use `--spatial` to retain the local features needed by the example gallery. Spatial caches can take several GB.

```text
conda run --no-capture-output -n biocal3d-dinov3 python stage_ablation.py extract --run runs/matched_colour --encoder dinov3 --repo vendor/dinov3 --checkpoint weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --harmonization scanner_colour_map.json --partition test --spatial
conda run --no-capture-output -n biocal3d-concerto python stage_ablation.py extract --run runs/matched_colour --encoder concerto --repo vendor/concerto --checkpoint weights/concerto_small.pth --harmonization scanner_colour_map.json --partition test --spatial
conda run --no-capture-output -n biocal3d-dinov3 python matched_study.py --run runs/matched_colour --partition test
conda run --no-capture-output -n biocal3d-dinov3 python matched_figures.py --run runs/matched_colour --partition test --examples
```

Use the same `--trees` and `--pixels-per-scan` in development and test if you change the defaults of 80 and 1,024. The test runner checks the development settings and requires a calibration map for the full benchmark.

## Outputs and interpretation

- `matched_results/<partition>/summary.csv` reports balanced accuracy, macro F1, accuracy, macro precision and support. `per_stage_metrics.csv` includes precision, recall, F1 and stage support. Results cover pooled scans, each scanner and a pooled mature-only sensitivity analysis because partial cleaning was only collected for mature specimens.
- `paired_deltas.csv` reports raw colour minus no colour and harmonised colour minus raw. The primary endpoint is balanced accuracy. Each 95% interval uses 2,000 paired bootstrap resamples of whole physical specimens, stratified by growth group. The intervals are descriptive and conditional on the fitted models and internal split.
- `figures/` contains common performance and paired-difference plots plus row-normalised confusion matrices with counts. The gallery shows all four scanner views of BCG3T1-1 and up to three additional test examples, covering correct cases, disagreements and errors where they exist. Selections are logged in `gallery_predictions.csv`.
- Yellow shows a GMM or boosted candidate mask used by the stage head. Red and blue show contributions to a stated stage logit contrast for the frozen encoder's actual stage head. Every neural panel for a scan uses the recorded stage against the same alternative stage, prioritising the most frequent wrong prediction when present. Within each neural model family and scan, colour conditions share the display scale. The 3D point map is projected into the same scan frame. All panels show the predicted stage and its model probability. Localization quality requires independent spatial reference labels.

For the main benchmark, keep the frozen encoders and linear stage heads. They let the input comparisons reuse the same pooling and head and give a direct decomposition of the existing classifier. The separately trained [feature-cluster pseudo masks](PSEUDO_MASKS.md) are useful as exploratory localization baselines. They measure local feature enrichment by stage.

The colorectal gland paper by [Wen et al.](https://doi.org/10.1038/s41598-026-36256-0) adds domain-specific DINOv2 adaptation, learned attention pooling, pseudo-mask post-processing and a segmentation decoder. For a later plaque-localization pilot, try only the attention pooling component first. Freeze DINOv3 or Concerto and fit a plaque-present multiple-instance head on local tokens or points using baseline and final-clean stage proxies. Use partially cleaned scans to test localization against photo-backed annotations. Compare against the current candidate masks, stage-associated clusters and signed attributions on the same specimens. For segmentation, use Dice and precision/recall, with a separate projected coverage error. Document image-to-scan correspondence, use the physical specimen as the evaluation unit and keep annotation-backed evaluation scans out of threshold and model selection. If the eight photo-backed specimens all serve as test, choose settings using development data or obtain a separate photo-backed development subset.
