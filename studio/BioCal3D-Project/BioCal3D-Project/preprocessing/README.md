# Initial ROI preprocessing

`preprocessing.py` is the original workflow for generating the first unedited ROI crops from the native full-model PLY files and their companion textures. It uses a reference scan for each scanner, fits a circle to the manually selected specimen edge, registers the other scans from that scanner to the reference, and extracts the specimen surface. The historical registration and crop settings are kept in the script. The default input is `raw_native` under the shared data root, and the default output is a sibling folder named `initial_rois`. See the [project configuration](../README.md#configure-the-data-folder) to set the shared root.

## Run on Windows

From the project root, install the dependencies into a Python environment with desktop graphics access, such as a Python 3.11 environment.

```powershell
python -m pip install -r preprocessing/requirements.txt
python preprocessing/preprocessing.py --reference-root preprocessing/references
```

Use `--raw-root` and `--output` to override the paths. Use a fresh output folder for a new run. The four original scanner reference fits are included in [references/](references/README.md). With `--reference-root preprocessing/references`, missing working fits are copied into the output run's `_references` directory and checked against their manifest. Existing working fits are retained; use a fresh output directory to reproduce the supplied fits. Without saved fits or `--reference-root`, the script opens a 3D window for each scanner's reference scan, configured as `BCG1T0-1`. Left-click about 8–12 points around the true outer specimen edge. Use **U** to undo a point, **C** to clear the selection, and **Q** to finish.

## Inspect the outputs

- `roi.vtp` and `geometry.npz` for each exported crop.
- `qc.png` previews and `_contact_sheets/` for visual checks.
- `manifest.csv` with source paths, registration metrics and per-scan status. Inspect rows marked `CHECK` or `FAILED` before proceeding to ROI review.
- `_references/` with the saved scanner reference fits. Keep these files with the run for reproducibility.

These are initial crops. Approval, manual geometry corrections, cross-scanner reorientation, and colour harmonization take place in later steps. The archived approved ROI datasets in the [dataset handover](../docs/datasets.md) capture those later stages.

The [data dictionary](../docs/data-dictionary.md) distinguishes these initial crops from the approved/reoriented/harmonized release, explains mesh/array/provenance fields, and lists historical inputs that may be archived later.
