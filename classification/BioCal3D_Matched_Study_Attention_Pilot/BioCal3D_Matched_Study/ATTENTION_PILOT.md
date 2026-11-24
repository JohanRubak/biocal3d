# BioCal3D frozen-feature attention pilot

The archive adds two entry points to the matched study.

| Entry point | Spatial input | Available conditions |
| --- | --- | --- |
| `run_dinov3_attention.py` | DINOv3 ViT-S/16 patch tokens and ROI coverage | Grayscale, raw RGB, harmonised RGB |
| `run_concerto_attention.py` | Concerto Small features at sampled surface points | Geometry, geometry + raw RGB, geometry + harmonised RGB |

## Reuse the existing run

Unzip the package and run from its `BioCal3D_Matched_Study` folder. Point `--run` at the **same** run folder used by the matched study. The scripts read its `cohort.json` and encoder caches. They keep all scans of each physical specimen together, use the original development folds, and check that the four scanner views of **BCG3T1-1** and their physical specimen **BCG3-1** belong to the locked test partition.

If the original run already has spatial caches in both partitions, skip extraction. If it only has pooled embeddings, repeat its original `stage_ablation.py extract` commands with `--spatial`, once for development and once with `--partition test`. Use the same checkpoint, environment, colour map and preprocessing flags as the existing extraction. See [the matched study setup](README.md#run) for preparation and checkpoint commands. A new run needs the approved reoriented ROI export and `prepare --require-test-specimen BCG3-1`. Reuse an existing locked cohort if one is available.

The attention scripts use the existing `requirements.txt` and can run on CPU without the encoder packages or checkpoint files. The checkpoints and encoder environments are needed to build or upgrade the caches.

## Run the heads

In the examples below, `runs/matched_colour` is a placeholder for the actual matched run. Run each development command before its corresponding test command. Keep the same options and output root for both. Review the development scores before opening the test results.

```text
python run_dinov3_attention.py --run runs/matched_colour
python run_concerto_attention.py --run runs/matched_colour

python run_dinov3_attention.py --run runs/matched_colour --partition test
python run_concerto_attention.py --run runs/matched_colour --partition test
```

By default each script runs its raw and no-colour modes, plus harmonised when the encoder config includes a fixed colour map. Use `--modes dinov3_rgb` or `--modes concerto_rgb` in both commands for a raw-only first pass. Raw-only evaluation follows the same split as the main benchmark and is a separate pilot. For an alternate configuration, use a fresh `--out` root per encoder and keep it for both partitions. The training defaults use all eligible patches in the standard DINOv3 grid and a deterministic sample of up to 512 points per Concerto training scan. `--max-locations 0` uses all locations for training and can increase memory substantially. Inference uses every eligible patch or point.

Test exports use the same raw orthographic render as the matched study. If the Concerto run has no renders, it can generate them from the original ROI meshes. Supply `--snapshot PATH` if the meshes moved or the run was transferred between Windows and WSL. Concerto exports retain the original world coordinates and mesh correspondence in addition to the flat projection.

## What the models compare

- **Logistic** refits the existing matched head on cached mean-plus-max features. It uses the same standardisation, class balancing and fixed regularisation. If completed matched results are present, the script checks the refitted probabilities against the saved predictions.
- **Uniform** uses coverage-weighted mean pooling for DINOv3 and uniform point pooling for Concerto. It has the same stage and endpoint heads, feature normalisation and training settings as learned attention.
- **Attention** uses the pooling architecture in [Wen et al., Stage 2](https://doi.org/10.1038/s41598-026-36256-0). A trainable linear → ReLU → linear MLP scores each patch or point. Spatial softmax turns the scores into weights, and the weighted sum feeds a linear classifier. The pilot defaults to 32 hidden units to keep training on cached features practical. Set `--attention-hidden-dim` to try another width. The paper used 2,048 units. BioCal3D uses its own frozen encoder features and a coverage prior for DINOv3 patches.

The same attention MLP receives gradients from both heads. Both losses use scan-level labels.

| Training stage | Three-class stage head | Auxiliary plaque-presence head |
| --- | --- | --- |
| Baseline | Baseline class | Weak positive |
| Partially cleaned | Partial class | Excluded from this loss |
| Final clean | Clean class | Weak negative |

The paper trained a single gland-presence classifier and derived pseudo-masks by refining its attention maps. The pilot adds the three-class head for the matched classification comparison and exports the attention map alongside the plaque-like mask.

The candidate mask thresholds the auxiliary head’s **local logit** at the default 95th percentile of local scores on clean training scans. Each clean scan contributes equal weight, with patch coverage or a uniform point prior within scans. The threshold is fit separately for every mode and fold using the training locations and refit on all development specimens for test. Softmax attention is relative within a scan, so even clean scans can have highly attended patches. The gallery displays attention gain and the local plaque-like score in separate panels.

## Outputs

Results go to `runs/matched_colour/attention_results/<encoder>/<partition>/` unless `--out` is set.

| Output | Contents |
| --- | --- |
| `summary.csv`, `per_stage_metrics.csv`, `confusion_matrices.json` | Classification results for the same scans and stage order, pooled, per scanner and pooled mature-only |
| `paired_deltas.csv` | Attention minus logistic and attention minus uniform, with 2,000 paired whole-specimen bootstrap draws stratified by growth group |
| `predictions.csv` | Recorded stage, predicted stage and all three class probabilities for every assessment scan |
| `settings.json`, `fit_manifest.json`, `models/` | Run settings, specimen IDs, thresholds, fitted heads and loss histories |
| `mask_summary.csv`, `masks/` | Test candidate masks, ROI support, projected area and coverage, native maps and point-to-mesh correspondence |
| `gallery_index.csv`, `masks/**/example.png` | All four scanner views of **BCG3T1-1** plus up to three other test examples chosen to illustrate agreement, errors or disagreements where available |

`candidate_mask.png` is binary. Use `covered_mask.png` to identify where it was evaluated and `valid_mask.png` for the ROI. The projected coverage uses all valid ROI pixels as the denominator, and projected area is in mm² of the image plane. Concerto’s preview assigns the nearest sampled point in XY within 0.35 mm. DINOv3 uses nearest-patch projection and excludes patches with less than 5% ROI coverage. The native file preserves the threshold, candidate labels, local logits, attention weights and the signed contributions to the predicted-stage versus runner-up logit margin. Summing these contributions with the saved margin bias reconstructs that margin.

These are stage-supervised candidate plaque masks. Compare them with GMM and boosted masks using the same scan frame. Photo-backed spatial annotations are needed to measure mask accuracy. Classification performance alone does not establish plaque localisation quality.
