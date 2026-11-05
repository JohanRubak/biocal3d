# PCA, similarity and prompted segmentation for BioCal3D

Use `inspect_feature_maps.py` for the repository-style feature inspection workflow.
It uses local features from both DINOv3 and Concerto. It does not use the earlier
cluster-to-cleaning-stage association or the scan-level stage classifiers.

## What changed after the first results

The earlier iTERO example showed leading PCA components and clusters dominated by
position/rim structure, while stage association produced broad uncertain regions.
That result does not establish plaque localization. This inspector separates
three things:

| Output | Meaning |
| --- | --- |
| Per-specimen PCA | Which directions explain feature variation within this scan |
| Cosine similarity | How similar each native feature is to a clicked reference feature/region |
| Prompted candidate segmentation | Whether a feature is more similar to plaque-like or clean-like clicked references |

Unsupervised feature clusters are also exported, with arbitrary class IDs. They
do not automatically acquire plaque/clean semantics.

## Relationship to the official examples

- [Concerto PCA demo](https://github.com/Pointcept/Concerto/blob/main/demo/0_pca.py)
  performs PCA within a point cloud and mixes the first six components into RGB.
  The inspector offers that blend with `--pca-style concerto`, plus separate
  PCs 1-3 and 4-6. Axes are fit to raw cached features using scikit-learn rather
  than Torch; colours/signs may differ. Brightness is 1 rather than the demo's
  boosted brightness. These are methodological adaptations, not an exact visual replica.
- [Concerto similarity demo](https://github.com/Pointcept/Concerto/blob/main/demo/1_similarity.py)
  uses L2-normalized features and dot products. This inspector does the same with
  your full-dimensional cached features, with an optional mean feature for a
  small clicked region. It supports within-scan and cross-scan comparisons.
- [DINOv3](https://github.com/facebookresearch/dinov3) shows cosine similarities
  between a selected patch and other patches; the same operation is used here.
- [Concerto semantic-segmentation demo](https://github.com/Pointcept/Concerto/blob/main/demo/2_sem_seg.py)
  loads a separately trained ScanNet linear head. The [DINOv2 notebook](https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb)
  loads pretrained VOC2012/ADE20K heads and also demonstrates Mask2Former. These
  have scene/object classes, not a plaque class. Their head weights are not used
  here. The transferable idea is applying a local classifier to frozen spatial
  features; the existing optional seed-based fit in `PSEUDO_MASKS.md` offers a
  small BioCal3D-specific version once appropriate spatial seeds are available.

Concerto's local/global crop demo reruns its encoder on crops. This inspector
compares features already cached from whole scans; it does not re-encode a click
region as a separately cropped input. Cross-scan comparisons retain the original
whole-scan context of each feature.

## Installation

Copy `inspect_feature_maps.py`, `make_feature_pseudomasks.py` and this guide into
your existing folder beside `stage_ablation.py`. The optional checks use
`test_feature_inspector.py` and `test_feature_pseudomasks.py` too.
The original extraction/classification modules are unchanged. No new package
installation is required if the existing Matplotlib/scikit-learn environment works.
The desktop viewer uses Matplotlib with Tkinter, normally included in Conda Python.

**You still need local features cached using `--spatial`.** Reuse the development
extraction commands in `PSEUDO_MASKS.md` with your original settings if necessary.
Do not run `prepare` or retrain the stage heads for this inspector. It does not
require the previous `pseudo_models_v1` fit. It reads cached features directly.

## Open the example you showed

From your existing stage-ablation folder:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python inspect_feature_maps.py --run runs/stage_pooled_v3 --partition development --scan iTERO__BCG4T1-2 --out runs/stage_pooled_v3/feature_inspection
```

Both DINOv3 RGB and Concerto RGB appear in one window. If the scan ID differs,
list IDs with:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python inspect_feature_maps.py --run runs/stage_pooled_v3 --partition development --list-scans
```

To browse held-out development fold 0, omit the scan/partition arguments:

```bat
conda run --no-capture-output -n biocal3d-dinov3 python inspect_feature_maps.py --run runs/stage_pooled_v3 --out runs/stage_pooled_v3/feature_inspection
```

Use Next/Prev or arrow keys to browse. Save before changing specimens if you want
to keep prompts. Changing specimens clears them. `--partition test` is an explicit
option if test spatial caches exist. Prompting a test scan interactively is an
interactive assessment, not an automatic segmentation evaluation.

## Click workflow

1. Click **Plaque** (or press P), then click a visibly plaque-like location in
   either original image. The same physical image position is applied to both models.
2. Inspect the plaque-reference cosine map before adding any clean references.
3. Click **Clean** (or press C), then click a clean-like location. Prefer a similar
   distance from the specimen rim when possible, to test appearance rather than
   simply center-versus-boundary structure.
4. A prompted candidate mask appears, comparing the two kinds of reference.
5. Add more references if multiple appearances need representation. Undo/U removes
   the last; Clear removes all.
6. Press **Save** or S to export figures, raw maps and the reproducible prompt record.

The Radius slider selects a circular reference region in mm for **future clicks**;
zero uses the nearest single native patch or point. Each region averages its
normalized features into a normalized prototype. Multiple regions use the maximum
similarity over prototypes in that class. Opposite labels that select overlapping
native cells are rejected, including overlap caused by DINO's coarse patch grid.
The full feature vector is used for similarity; PCA is only a visualization and
does not discard channels before the similarity calculation.

The mask uses:

`margin = max cosine to plaque references - max cosine to clean references`

Gold means margin above the positive uncertainty threshold; clean-like means below
the negative threshold; blue is the uncertain interval. Default +/-0.02 is an
exploratory cosine margin, not a calibrated confidence. The slider changes that
margin for the whole view. Equally similar features remain uncertain. Both
reference classes are required before writing a candidate mask.

## PCA and display controls

- **First PC** switches between PCs 1-3, 2-4, ... up to 7-9. Saved output always
  includes PCs 1-3 and 4-6, plus the Concerto-style six-component blend. Later PCs
  may reveal variation hidden by position/rim components; this does not guarantee
  that they encode plaque or that removing early PCs would improve segmentation.
- `--pca-style concerto` shows the mixed first-six-component RGB view in the viewer.
  The First PC slider only affects the `components` display style.
- **Scale** toggles cosine display between each map's 1st/99th percentiles and fixed
  [-1,1]. The colourbars show actual cosine values. Percentile display can make tiny
  differences visible; it changes neither the raw values nor the mask. Do not
  interpret a bright map as strong plaque confidence. Different maps can use
  different display ranges. Fixed-range TIFF/NPZ outputs retain the numbers.
- PCA is fitted within each specimen, so RGB colours are not semantic labels and
  are not directly comparable between scans or models.

If PCA and similarity both keep following the rim despite several carefully chosen
references, that is evidence that this cached representation is not providing
useful local plaque separation in this example. Adjusting visual contrast cannot
fix absent spatial information. Layer-specific features, finer extraction and a
spatial head trained with independent labels would then be separate experiments.

## Cross-scan reference comparisons

Add `--reference ANOTHER_SCAN_ID` to select references in a separate reference
window. The five-column view remains the target scan. Reference/target use the
same feature channels and normalization within each model. No geometric
registration is needed for feature similarity; this is not a physical change map.
Prefer training reference specimens when assessing transfer to held-out scans.

For a final-clean counterpart, for example, select clean-like reference points
there rather than assuming every point on that scan is clean. Do not select
positive/negative points in the target window when a separate reference is active.

## Files written by Save

Each save uses a new numbered directory so earlier results remain available.

- `comparison.png`: current interactive comparison, including numeric cosine scales.
- `reference_marked.png`, `reference_original.png`, `target_original.png`.
- `prompts.json`: scan IDs, click coordinates/radii, selected PCA components,
  uncertainty margin, display settings and input fingerprints.
- Per model, `maps.png`: clean six-panel figure containing original, PCs 1-3,
  PCs 4-6, plaque cosine, clean cosine, and prompted mask.
- `pca_pc1_3.png`, `pca_pc4_6.png`, `pca_concerto_blend.png`.
- `positive_cosine.tiff`, `negative_cosine.tiff`: float32 cosine images. A regular
  image viewer may not display their numeric range correctly; use the plots or NPZ.
- `prompted_mask.png`: 0 clean-like, 128 uncertain, 255 plaque-like. Background and
  unsupported locations are also 128; use `valid_mask.png` and `covered_mask.png`.
- `unsupervised_clusters.png`: arbitrary clusters fitted within this scan; there is
  no stage-based semantic mapping.
- `native_maps.npz`: raw positive/negative cosines, margin and labels when available,
  PCA coordinates/explained variance, clusters, native point/patch mapping and hashes.
- Concerto exports PCA, each similarity and prompted-mask PLY point clouds, retaining
  native XYZ and scalar fields. Similarity PLY colours use fixed [-1,1], while the
  stored values preserve exact scores. Open PLY files in CloudCompare or Open3D
  to rotate the actual surface. The interactive Matplotlib view is orthographic 2D.

The DINO maps retain native patch resolution. Concerto previews use nearest-XY
point projection with a 0.35 mm distance limit; on overhanging/multilayer geometry,
inspect the native PLY rather than treating the preview as a visibility-correct
surface rendering. No smoothing is applied to fabricate fine boundaries.

## Reproduce a saved view without opening a window

```bat
conda run --no-capture-output -n biocal3d-dinov3 python inspect_feature_maps.py --run runs/stage_pooled_v3 --partition development --load-prompts runs/stage_pooled_v3/feature_inspection/YOUR_SAVED_FOLDER/prompts.json --headless --out runs/stage_pooled_v3/feature_replay
```

Use the same partition as the target. Changed cache fingerprints are rejected.
You can also supply repeated `--positive X Y` and `--negative X Y` flags, using
native render-pixel coordinates with origin at top left, for a headless export.

## Checks and limits

```bat
conda run --no-capture-output -n biocal3d-dinov3 python -m unittest test_feature_inspector -v
```

Synthetic checks verify cosine values, multiple references, PCA component selection,
physical reference regions, overlap rejection, callbacks and both model export
formats. Real BioCal3D feature caches and model weights were not available while
implementing this update; those checks do not establish actual plaque performance.
The desktop display must be exercised on your local PC. This implementation does
not automatically turn the natural-scene segmentation heads into plaque heads.
