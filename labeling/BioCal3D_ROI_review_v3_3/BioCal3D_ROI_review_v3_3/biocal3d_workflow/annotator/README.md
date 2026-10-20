> Setup: use setup_labeling_environment.bat and start_labelling.bat at the
> package root. The standalone setup instructions below are historical; their
> launcher now delegates to the root launcher. Points are fixed yellow.

# BioCal3D local 3D annotator

A desktop surface-labeling app for colored triangle meshes. No login, server,
or MeshLab installation is required. Open a dataset once, label a sample,
then use **Save & Next** to move through the queue.

This update adds brightness, contrast, gamma and saturation for RGB/UV display,
annotator switching, shared top/bottom cues and a top-only painting guard.
Surface/points/wireframe, vertex/face/unlit shading and numeric shortcuts remain
available. Saves automatically create original-color labeled PLY alongside NPZ;
Export all bundles completed annotations. Existing annotation NPZ files remain
compatible when their original input meshes and paths are retained.

## Install on Windows

Keep this `annotator` folder inside the integrated `biocal3d_workflow` package.
Its launchers activate your existing **biocal3d** Conda environment. Install the
new dependencies once if needed, from Anaconda Prompt:

```bat
conda activate biocal3d
python -m pip install "PySide6>=6.6,<7" "plyfile>=1.0,<2"
```

Use the package's `start_labeling.bat` for your approved preparation inventory,
or `create_labeling_plan.bat` for the two-dentist study. See `../LABELING_STUDY.md`.
To try the synthetic demo, run `start_annotator.bat --demo` from this folder.
The demo contains synthetic colored disks for checking the controls.

For standalone folder discovery:

```bat
start_annotator.bat --root "C:\path\to\_development_TIMESTAMP" --annotator johan
```

Or run without arguments and click **Open dataset folder**. The optional filter
applies to the relative input path (for example `G3` or `T1`); set it before
opening the folder. The annotator dropdown switches IDs while a dataset is
open, saving the current annotation first and loading the selected rater
independently. New creates an ID. Study launchers lock this dropdown during
independent rating. General folder discovery bypasses the approved QC inventory.

Use **Sort by** above the file list to choose dataset order, sample A–Z/Z–A,
scanner, group/specimen, partial-stage first, or annotation status. Sorting keeps
the current sample, labels and undo history. **↻** reapplies the selected sort
using current statuses (draft → new → error → completed); the order stays stable
while painting/saving. Metadata comes from the approved queue and sample files.
Sorting changes navigation order within the same queue.

Uncheck **Scanner color** to show neutral gray geometry while keeping the label
overlay. This hides vertex RGB and UV textures; rechecking restores the adjusted
color display. Color sliders are disabled while scanner color is hidden.
Original colors/textures are preserved in PLY/NPZ exports. Vertex/face shading
can help inspect the gray geometry.

## Label a sample

1. Inspect the full colored surface. Hold **Space** and drag with the left
   mouse button to rotate; use the wheel to zoom and middle drag to pan.
2. Select **Plaque** and paint with left drag. Use **Clean** to correct any
   region you overpainted. **Uncertain** excludes ambiguous surface from the
   clean/plaque denominator. **Unreviewed** clears a region for later review.
3. Toggle the labels with **O** to see the original colors without an overlay.
4. After reviewing the whole ROI, click **Reviewed remainder → clean**.
   This changes only unreviewed faces; plaque and uncertain labels stay intact.
5. Press **N** or click **Save & Next** to mark this sample complete and
   load the next unfinished sample. Remaining unreviewed ROI faces block
   completion. Time point/stage never automatically supplies surface labels.

Edits autosave about one second after a stroke and every ten seconds while
dirty. Navigation and closing save the current annotation too. Autosave
creates a **draft**; only **Save & Next** records completion. Any label edit
to a completed sample turns it back into a draft. Click any queue item to
reopen it, or use **Next unfinished** to skip ahead while retaining a draft.

| Control | Action |
| --- | --- |
| Left drag | Paint, when Paint mode is checked |
| Space + left drag | Rotate, including immediately after adjusting sliders |
| Middle drag | Pan |
| Wheel | Zoom |
| Ctrl + wheel / `[` / `]` | Change brush radius |
| `1` / `2` / `3` / `4` | Plaque / clean / uncertain / unreviewed |
| Ctrl+Z | Undo annotation (up to 100 recent strokes; session only) |
| Ctrl+Y or Ctrl+Shift+Z | Redo annotation |
| Ctrl+S | Save draft or save current completed status |
| `N` | Complete and advance |
| `O` | Toggle overlay |
| `R` | Reset to best-fit surface view |
| `B` | Toggle painting/navigation mode |
| `S` / `P` / `W` | Toggle Surface / Points / Wireframe, matching the checkboxes |
| `A` / `D` | Previous / next sample in the sorted queue; save before moving |

Shortcuts apply across the app, including after clicking buttons or choosing
display options. When notes, filenames, or numeric text fields have focus,
their normal typing and editing shortcuts take priority. Both the keyboard
number row and numpad switch label classes. The brush radius is in screen
pixels, so zoom in for finer boundaries.

## Display options

**Surface**, **Points**, and **Wireframe** are independent toggles. Enable
surface + wireframe to inspect triangles, or disable surface and enable
points to inspect sample density. Point size is adjustable. S/P/W control
these same checkboxes; VTK’s separate built-in character shortcuts are blocked
to keep actor representations consistent with the controls.

A/D navigate the current sorted queue, including completed samples. They save
the current annotation without marking a draft complete, and stop at the first
or last sample. N remains the complete-and-advance shortcut.

- **None (original color)** disables lighting: recommended while judging
  plaque from scanner RGB/texture.
- **Vertex (smooth)** lights using area-weighted normals at shared vertices.
- **Face (flat)** lights each triangle with its face normal.

Shading, point size, and display modes change rendering only. Source RGB,
texture pixels, geometry, triangle order, and saved labels are unchanged.
Texture-seam duplicates share normals from the original vertices. Label
overlays remain unlit. In points-only/wireframe-only views the brush still
labels **faces**, using a solid surface internally to determine visibility;
these toggles do not change the annotation into per-point labels.

## Input files and scanner color

- **PLY**: triangular faces and vertex RGB, or vertex UVs, or face-corner
  `texcoord` arrays. PLY `comment TextureFile filename.jpg` is respected.
  Face-corner UVs are expanded only for rendering; label indices continue to
  refer to the original faces.
- **VTP**: triangle `PolyData` with RGB point data or texture coordinates.
- **geometry.npz**: `vertices` (N × 3), `faces` (M × 3), optional `rgb` or
  `colors`, `uv`/`tcoords`, and `roi_mask` (one Boolean per vertex) or
  `face_roi_mask` (one Boolean per face). Common key aliases are accepted;
  errors list NPZ keys if required geometry is missing. Pickled object arrays
  are not loaded.

For each prepared sample directory, discovery prefers `roi.vtp`, then
`roi.ply`, then `geometry.npz`, to avoid annotating the same prepared sample
more than once. Other PLY/VTP files are discovered recursively. Other NPZs
are ignored to avoid treating feature archives as meshes.

Textures are matched through the PLY comment, a NPZ texture path, a VTP
`TextureFile` field, or an unambiguous matching filename stem/prefix. An
unmatched texture can be selected once with **Choose texture**; that path is
saved and reused when reopening. The mesh must contain UV coordinates to
use a texture. A screenshot alone cannot be attached correctly to a mesh.

For TRIOS, use vertex RGB. For LAB/iTERO, use the actual texture+UV export,
or your already prepared vertex-colored ROI. Missing color or texture is
shown explicitly and blocks completion. If your prepared export has a
different schema, use its colored PLY/VTP or adapt the NPZ aliases in
`mesh_io.py`; no training scripts need to be changed to run this app.

Use **ROI exports** rather than raw holder scans. If a NPZ includes a vertex
ROI mask, only faces whose three vertices lie inside the ROI are editable.
Faces outside the ROI are saved with `active_faces=False` and excluded from
statistics. Without a mask the whole mesh is editable; the app does not
infer a disk boundary or crop your raw scan.

## Output and training use

By default output goes to:

```text
DATASET\annotations\ANNOTATOR\relative_sample_folder\roi.vtp.labels.npz
```

Use `--output "D:\BioCal3D_labels"` to choose another output location.
Each rater gets an independent annotation tree. A per-file lock prevents
two app instances from writing the same rater/sample simultaneously.

| `face_labels` value | Meaning |
| --- | --- |
| `-2` | Unreviewed; exclude from supervised training |
| `-1` | Uncertain/excluded; exclude from supervised training |
| `0` | Reviewed clean |
| `1` | Plaque |

The NPZ includes `face_labels`, `active_faces`, `face_indices`, `completed`,
`source_relative`, `fingerprint`, `annotator`, `notes`, timestamp, and view state.
Labels are saved with an atomic replacement; the original mesh is not
rewritten. A geometry-and-ROI fingerprint blocks reopening labels against
a changed/reordered mesh. Copy the same frozen input meshes to each rater.
Do not decimate, repair, reorder, or register the input meshes after labeling
without explicitly transferring labels.

```python
from pathlib import Path
import numpy as np
from mesh_io import load_mesh

root = Path(r"C:\path\to\dataset")
with np.load(root / "annotations/johan/sample/roi.vtp.labels.npz", allow_pickle=False) as a:
    mesh = load_mesh(root / str(a["source_relative"].item()))
    if str(a["fingerprint"].item()) != mesh.fingerprint:
        raise ValueError("Geometry/ROI changed")
    labels = a["face_labels"].copy()
    usable = a["active_faces"] & (labels >= 0)
    completed = bool(a["completed"])
# labels[i] is the label of mesh.faces[i].
# Train/evaluate on completed files and usable faces only.
```

Sampling a 3D point from a triangle inherits that triangle's label. Vertex
labels require an explicit boundary convention; do not silently average
class IDs. Rendering 2D masks from face labels is a later pipeline step and
is not included in this version.

**Export labeled mesh** offers **PLY binary** (default), **PLY ASCII**, and
**VTP**. PLY preserves original vertices, triangle order, normals, and RGB.
It stores `face_label` as a signed integer, a duplicate `quality` scalar for
MeshLab compatibility, `source_face_id`, and `roi_active`. Texture coordinates
are exported per face corner; the matching texture image is copied beside
the PLY and referenced by its `TextureFile` comment. Transfer both files.
Labels never replace the original RGB values. In MeshLab, visualize or select
the face quality values to inspect the annotation.

VTP exports `face_label`, `source_face_id`, and `roi_active` cell arrays. For
corner-UV meshes it uses duplicated display vertices, but preserves source
face IDs. Its external image is referenced by path rather than copied.
The NPZ remains the authoritative working annotation, including completion,
annotator, notes, and camera state. PLY/VTP exports are snapshots; exporting
does not mark the sample complete. Many applications ignore or discard
custom PLY fields, so retain the NPZ files when using other mesh software.

**Auto-save PLY too** is enabled by default and creates a matched binary PLY
with every NPZ save, including drafts. Disable it for NPZ-only working files.
The NPZ completion flag determines whether labels are ready for training.
**Export all…** exports completed cases in the current queue/rater as paired
PLY+NPZ files. The CLI export tool can include all raters and optional drafts.

Explicit `face_label` properties can seed a new annotation when a labeled
PLY/VTP is loaded without a matching NPZ; its state is a draft until reviewed.
Arbitrary face quality is not guessed to be a plaque label. Files ending in
`.labeled.ply` or `.labeled.vtp` are excluded from normal recursive dataset
discovery to avoid queueing export duplicates; rename/move an export to a
separate input dataset if you intend to use it as the source.

## Preparing a PLY annotation dataset

The recommended interchange format is **binary little-endian triangle PLY**.
Retain vertex RGB for TRIOS; for textured LAB/iTERO inputs retain UVs and the
original texture image. Baking textures down to vertex RGB is convenient,
but can lose texture detail between vertices. Your already vertex-colored
ROIs are supported too; preserve their original texture exports if needed
for later color comparisons.

Keep your existing registration and reference-circle ROI extraction. The
included converter operates on **already prepared ROI files/masks**; it
does not locate a sample disk in a raw holder scan. It exports only active
ROI triangles, removes unused vertices, and saves a mapping back to the
input vertices/faces. It does not smooth, decimate, remesh, rescale, or change
color values. Boundary triangles follow the existing mask convention; it
does not split triangles along a new analytic circle.

Before starting new annotations, you can create a separate frozen PLY set:

```bat
python prepare_roi_ply.py --root "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261007_100936_877172" --output "C:\Users\au662213\repos\biocal3d\data\annotation_rois_ply"
```

The output keeps relative sample folders, with a PLY, a `.mapping.npz`, and
a texture image when applicable. Inspect a few samples from each scanner,
then label this frozen dataset. The output directory must be empty to
prevent accidentally replacing meshes already used for annotation.
Each map stores source face/vertex IDs and the source geometry fingerprint.

**If you already started labeling the original VTP/NPZ inputs, keep using
those inputs.** This app update accepts them unchanged. Preparing a new PLY
dataset changes filenames and may compact vertices; existing annotations
are not automatically migrated. Format conversion is optional, not a
required extra step for the annotator.

Plaque percentage in the UI is **triangle-area weighted**, not vertex/face
count weighted. It uses plaque / (plaque + clean) within the active ROI;
unreviewed and uncertain surface is excluded. This number is provisional
while the sample is a draft. Areas use the coordinate units of the source
mesh; the app does not assume that every file uses millimeters.

## Implementation and current scope

The visible-face brush uses VTK's hardware selector, then a circular test
on projected triangle centroids. It includes the triangle under the cursor
even if its centroid is outside the circle. Hidden faces are excluded by the
depth-tested selection; outside-ROI faces are never edited. Boundaries have
triangle-level resolution. Very large triangles may require a denser source
mesh prepared **before** labeling.

This version includes manual brushing, undo/redo, texture display, autosave,
resume, per-rater storage, area statistics, and labeled PLY/VTP export. It does
not yet include lasso selection, automatic plaque proposals, agreement
analysis, or 2D-mask rendering.

Run core verification with `python -m unittest -v test_core.py`.
`python -m unittest -v test_brush.py` verifies visible-face selection and the
ROI guard with offscreen VTK. `python test_ui.py` exercises mouse brushing,
undo/redo, autosave, reopen, completion/advance, and editing completed labels
back to a draft on a working desktop display. All three checks passed on
synthetic data during development; the desktop check ran on Linux with a
virtual display and software OpenGL.
GUI verification uses the included synthetic samples; use `--demo` before
starting experimental labeling. Your Windows/GPU and actual prepared-file
schema still need a first local run. If loading fails, the dialog includes
the source path and error; keep those details for troubleshooting.

## Color, orientation and study review

Brightness, contrast, gamma and saturation change the display only, including
UV textures. Reset colors returns to the original appearance. Source assets
and exported PLY colors/textures are preserved. TOP/BOTTOM/SIDE status and
a 3D TOP axis follow rotation; Paint from top only prevents bottom/edge-on
brushing. R resets the top view; Flip top/bottom corrects inferred direction
and saves it for this mesh across raters. This is a camera-side guard, not an
automatic per-triangle anatomical classifier. In consensus review, switching
to a dentist shows that independent annotation read-only. See the workflow
package LABELING_STUDY.md for complete study and agreement-analysis commands.
