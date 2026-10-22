# BioCal3D: common orientation using TRIOS5

This tool establishes one manually reviewed, rigid in-plane rotation per scanner.
TRIOS5 stays unchanged at 0 degrees. Full original models supply the tooth/holder
direction; their existing `original_to_canonical` mapping places them in the same
coordinate frame as your extracted ROIs before orientation is estimated.

It does not crop, remesh, scale, deform, or establish correspondence between scanners.
It does not use dentists' plaque labels to choose rotations.

## Start

Extract the ZIP completely. Double-click **Start_Orientation.bat**. The launcher
looks for your existing `biocal3d` conda environment. Alternatively, in an Anaconda
Prompt:

```bat
conda activate biocal3d
cd /d "C:\path\to\BioCal3D_Scanner_Orientation"
python orient_scanners.py
```

If needed, run **Install_Missing_Dependencies.bat**. Existing compatible packages
are reused; PyVista and Qt are not required by this tool. Tkinter must be available
in the Python environment. For a custom Python installation, set
`BIOCAL3D_PYTHON` to its python.exe path before running the launcher.

## 1. Establish the rotations once

1. Click **Load Approved ROI dataset**. Select the FULL rebuilt approved snapshot
   containing `scan_manifest.csv`, `samples`, and usually `annotations`. Use the
   full dataset so its saved coordinate-frame inventory covers the later study subsets.
2. Click **Original full-model folder**. Select the original data root containing
   the scanner folders, for example `02-09-2026-Data collected`. Old absolute paths
   are resolved under this folder; raw PLY checksums are checked when recorded.
3. The left panel shows TRIOS5; the middle shows the selected scanner; the right
   overlays them. These are top views with physical millimeter axes. Start with
   matching specimen/stage acquisitions; the tool picks a common representative
   when one exists. Different specimens are also usable if the holder/teeth are unchanged.
4. Use the slider/numeric angle to align the tooth/holder direction. Alternatively,
   click **Match two landmarks**: click feature A then feature B on TRIOS5, then
   the SAME physical A then B in the other scanner. Choose well-separated stable
   tooth or holder features. The direction difference estimates the rotation;
   inspect and refine the result. Do not select plaque or a circular edge alone.
5. Click **Accept this scanner**. Select another representative to check the same
   angle on additional specimens/stages. Then repeat for TRIOS3, iTERO and LABscanner
   (and PRIMESCAN if present). LABscanner's small field may need zooming with the
   plot toolbar; the same physical tooth direction must be used.
6. Click **Save rotations** and save `scanner_rotations_TRIOS5.json`. A PNG of the
   currently displayed review is saved beside it. Keep the JSON: it contains every
   accepted angle, its reference models, and the original frame mappings.

Positive angles rotate counterclockwise in the displayed XY plane. The operation
rotates around the existing canonical origin, preserving each ROI's existing center
and height convention. Translation, tilt changes and ICP are deliberately omitted:
your scans are already aligned within each scanner, and this tool addresses the
remaining difference in in-plane direction between scanner frames. Two landmarks
estimate a direction, not a full surface registration.

## 2. Create an oriented Approved ROI copy

Click **Apply saved rotations to dataset copies…**:

1. Select your saved rotation JSON.
2. Select the original approved snapshot.
3. Choose a NEW sibling output folder, e.g. `Approved_ROIs_TRIOS5_oriented`.

Close annotators that are writing to the selected input while copying. Progress is
shown at the bottom of the window and in the console. On success, open the NEW
folder in the usual annotator/browser. Original files remain untouched.

## 3. Apply later to the dentists' work

After labeling finishes, use the **same JSON** with the calibration folder and
agreement folder. You can also select their common parent if each subfolder has
its own inventory. Each dataset is copied into a corresponding new output folder.
For parent-folder mode, open the resulting calibration/agreement subfolders, not
the parent, in the annotator. A failed collection has `INCOMPLETE_ORIENTATION.json`.

Do not calculate fresh angles for each dentist. All copies from the same scanner
must receive the same correction. The study copies must retain their original
`geometry.npz` mappings and scan identities. Do not run this on an already oriented
copy: it is detected and blocked.

## What is preserved and updated

- PLY/VTP vertices and normals rotate; connectivity, winding, RGB, UV coordinates,
  face labels, source IDs, and active-face masks are retained.
- `geometry.npz` canonical coordinates, normals, extraction plane settings and
  raw-to-canonical transforms are updated. Raw coordinates remain unchanged.
- All shared ROI-review payload versions rotate; their input fingerprints are
  updated. Approval status and review version IDs stay unchanged.
- Saved rater annotation fingerprints are verified against the original effective
  mesh and updated for the rotated mesh. Label arrays, notes, annotator identities,
  completion flags and review versions are checked unchanged.
- Saved cameras are cleared so the annotator opens in its common default top view.
  Their previous values are retained in `orientation_previous_camera`.
- If present, RGB images, valid masks and triangle/barycentric maps are rerendered
  in the new orientation. Portable texture identity and V convention are verified
  against the old RGB map. Existing image size/FOV is retained; clipping blocks the
  operation. Rebuild 2D training inputs from these NEW images and maps.
- Inventory/dataset hashes and internal dataset paths are updated. Split and study
  assignments stay unchanged. Existing scanner `references`, legacy `labels`,
  `predictions`, `x_anylabeling`, and old QC previews are clearly archived under
  `orientation_previous_*` rather than presented as current results. They are not
  transformed and must not be reused as inputs to the oriented dataset.
- Feature caches and archived `.geometry_history` label backups are omitted from
  the copy; they remain available in the original dataset. Recompute features,
  derived 2D polygon annotations, predictions and previous result figures.

The completed output includes `scanner_orientation.json` with per-scan angles,
checksums and verification counts, plus the exact rotation configuration used.
Partially written single-dataset outputs are removed if a check fails.

## Supported inputs and checks

Use an exported/rebuilt approved snapshot or its portable study copies. The tool
checks that the effective reviewed mesh agrees with the frozen exported mesh.
If review edits have not been rebuilt/exported, rebuild them with your existing
workflow first. A lone review-history folder containing only `surface.npz` is not
a complete input dataset.

New snapshots with different coordinate frames are blocked, even if their scan
names match. Establish new corrections from those original snapshot frames instead
of reusing this JSON. The tool supports the direct face-label annotation format
from BioCal3D ROI review v3_13 (bundled loaders). External custom annotation formats
and arbitrary derived arrays are not interpreted.

The reference convention aligns holders. It cannot recover a physical rotation of
a specimen relative to its holder. Inspect several matched partial-stage examples
to judge whether the orientation is adequate for your comparisons.

## Command line

```bat
python orient_scanners.py check --input "C:\data\Approved_ROIs" --config "C:\data\scanner_rotations_TRIOS5.json"
python orient_scanners.py apply --input "C:\data\Approved_ROIs" --config "C:\data\scanner_rotations_TRIOS5.json" --output "C:\data\Approved_ROIs_TRIOS5_oriented"
```

Run integration checks with `python -m unittest discover -s tests -v`.
See `VALIDATION.md` for what was verified before delivery.

## Included existing code

`vendor/` contains unchanged supporting modules from your
`BioCal3D_ROI_review_v3_13.zip`, including the existing annotation loader and
rasterizer. These allow format checks without installing or modifying the tool
inside your existing annotator project. PLY property-preserving edits use
[plyfile](https://python-plyfile.readthedocs.io/en/stable/usage.html).
