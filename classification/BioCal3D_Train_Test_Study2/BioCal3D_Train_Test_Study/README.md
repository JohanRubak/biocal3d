# BioCal3D raw-colour train/test study

## Smoke check on the Windows research PC

Extract the ZIP into `C:\Users\au662213\repos\biocal3d\classification`. The ZIP already contains the `BioCal3D_Train_Test_Study` folder. In Anaconda Prompt run:

```bat
cd /d "%USERPROFILE%\repos\biocal3d\classification\BioCal3D_Train_Test_Study"
python run_train_test_on_this_pc.py --smoke
```

The launcher uses the approved ROI export, both installed Conda environments, and the checkpoints and encoder checkouts from the previous Stage Ablations study. It reuses the prior test assignment and finds the study files relative to the launcher.

The smoke check trains on 12 scans of one mature training specimen and predicts 12 scans of held-out BCG3-1. It uses the normal image size, point count and encoder checkpoints, with two booster trees and fewer training pixels. It runs the full prediction and rendering path for all eight conditions, including the pooled-to-spatial upgrade. The outputs include all four scanner views of BCG3T1-1.

Open `runs\matched_raw_smoke\matched_results\test\examples\` and check a PNG before starting the full study. The smoke scores are diagnostics from a two-specimen run.

## Full study

```bat
python run_train_test_on_this_pc.py
```

All 36 development specimens become training, and the same 12 previously held-out specimens become test. This gives **384 training scans and 128 test scans**. Each physical specimen stays together across scanners and stages. There is a single fit for each condition, with no validation split, cross-validation, or colour harmonisation. The run uses its own feature caches. Restart the same command after an interruption to reuse completed stages and compatible caches.

| Model | No colour | Raw colour |
| --- | --- | --- |
| GMM | Local lightness contrast | Local darkness and redness |
| Gradient boosting | Local lightness contrast | Local Lab contrast |
| DINOv3 | Grayscale render | Raw RGB render |
| Concerto | XYZ and normals | Same XYZ and normals with raw RGB |

The GMM fits a mixture on each scan and uses the candidate mask's coverage, largest-region coverage and log region count for stage classification. The booster learns a pixel score from the cleaning order in the training scans. It thresholds scores using training clean scans and uses the same three mask features for stage classification. DINOv3 and Concerto use a logistic head on pooled frozen features. All stage classifiers learn from the same training scans.

## Results

The full run writes to `runs\matched_raw_train_test\matched_results\test\`.

| Output | Location |
| --- | --- |
| Interactive mean-pooling threshold viewer | `examples\index.html` |
| Matched scan panels and map arrays | `examples\` |
| BCG3T1-1 example from TRIOS5 | `examples\TRIOS5__BCG3T1-1.png` |
| Overall and per-scanner performance | `figures\performance.png` |
| Paired change from adding colour | `figures\paired_colour_deltas.png` |
| Stage confusion matrices | `figures\confusion_*.png` |
| Stage and model probabilities per scan | `predictions.csv` |
| Stage-specific and pooled metrics | `per_stage_metrics.csv` and `summary.csv` |

The full gallery includes all four BCG3T1-1 scanner views and up to three other test examples selected to show correct predictions, errors and disagreements where they occur. Yellow overlays show candidate plaque masks from GMM and gradient boosting. DINOv3 and Concerto each show combined mean-plus-max and mean-only allocations of the same contrast, the average baseline and partial logits minus the clean logit. Each row shares its colour scale between inputs, with separate scales for mean-only and combined maps. The matching NPZ contains the projected masks, contribution arrays and valid ROI.

Open `examples\index.html` in your browser, select a scan and choose DINOv3 or Concerto. The slider uses the mean-pooling map and sets one threshold across both input conditions. Yellow marks candidate plaque above threshold. Teal marks candidate clean at or below threshold. Each panel can download a mask as a PNG with transparent unscored pixels. These are model-derived candidate labels. Spatial accuracy can be evaluated against independent plaque annotations.

For a completed run, replace the program files with the updated ZIP and refresh its gallery from the same folder:

```bat
python run_train_test_on_this_pc.py --gallery-only
```

Add `--smoke` to refresh a completed smoke run.

The initial prediction pass picks example scans. The launcher then saves spatial features only for those scans. The final neural predictions and maps are computed from the same final cache. The exact pooling operations are shared between extraction and gallery replay.
