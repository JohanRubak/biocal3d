# Saved scanner reference fits

These are the four original `reference_roi.npz` files supplied for the first ROI extraction stage. They are small configuration archives, rather than full scanner meshes, and are kept with the preprocessing code. The original file bytes are preserved; `manifest.json` records their SHA-256 values and fields.

Each scanner fit uses `BCG1T0-1`. The archives contain the fitted circle centre, normal and in-plane basis, fitted radius, crop radius, height offset, scanner identity and recorded reference-scan source path. They contain the fitted parameters, not the original sequence of perimeter clicks. Historical absolute source paths are retained as provenance; preprocessing discovers the current native scan separately.

From the project root, with preprocessing dependencies installed:

```bat
python preprocessing\preprocessing.py --raw-root "D:\BioCal3D\Data\raw_native" --output "D:\BioCal3D\initial_rois" --reference-root preprocessing\references
```

Use a fresh output folder to reuse these fits exactly. The command copies each saved fit into the working run's `_references/SCANNER/reference_roi.npz` when missing. Existing working fits are retained. Changes to the working copy do not edit the references stored here. Without `--reference-root`, preprocessing retains its existing behaviour of using the run's saved fits or opening manual reference selection.

These fits concern scanner-specific initial cropping. They are not the later cross-scanner orientation correction JSON used to align approved ROIs to TRIOS5.
