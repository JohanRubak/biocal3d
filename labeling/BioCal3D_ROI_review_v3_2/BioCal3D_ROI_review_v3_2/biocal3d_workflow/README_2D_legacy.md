# BioCal3D: prepare all scans, label images, project to 3D

This package adapts your actual `preprocessing.py` and `biocal3d_prepare_dino.py`.
It reruns raw scanner loading, scanner-specific reference alignment, full/holder
ICP, circular ROI extraction and QA. It then exports full triangle-rasterized
RGB images and the correspondence needed for later 3D labels. No model download,
DINO feature extraction or training is required to prepare or label this dataset.

## 1. Install and place the files

Extract the ZIP into a folder such as `biocal3d/segmentation/preparation`.
Keep **all three Python files in the same folder**. In your `biocal3d` conda
environment, install:

```powershell
python -m pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended for compatible Open3D wheels.
A GPU is not required. Reference selection uses an interactive PyVista window;
the optional context screenshots use VTK off-screen rendering.

## 2. Rerun the whole preprocessing pipeline

From the extracted script folder:

```powershell
python biocal3d_prepare_all.py prepare --data-root "C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
```

On your AU workstation, use the actual path (your earlier preprocessing script
used `C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected`).
The default now searches relative to the script for your project data folder, then
tries your current user profile under `repos/biocal3d/data`. Set `--data-root` for
a different collection folder. `BIOCAL3D_DATA_ROOT` can also set the default.

The command processes **all discoverable PLY scans**, including new sample
numbers and additional groups, and creates a new `_development_<timestamp>`
folder under the data root. It prints that folder at the end. Original inputs
and older preparation/annotation folders are retained.

The expected path contains one scanner directory (`LABscanner`, `iTERO`,
`TRIOS3`, `TRIOS5`, `PRIMESCAN`) and a sample identity such as `BCG3T1-9`. Group folders such
as `G1-OK`, mixed-case file extensions and additional nested folders are supported.
One authoritative PLY is allowed per scanner/sample/stage identity; duplicate
identities stop the run instead of overwriting each other. Additional experiments
with reused specimen numbers need separate roots. The supported design is
G1–G2: T0/T1; G3–G6: T0/T1/T2. G6 uses the same 24-hour growth protocol
as G3–G5, as confirmed by you. Any positive group number is discoverable.
For a later group without a configured protocol, original timepoint names are
retained (`timepoint_T0`, etc.) and growth is `unspecified`; geometry/color export
and labeling still work. Set its entry in `GROUP_PROTOCOLS` before creating a
growth-stratified split. Scanners other than the five listed require a loader/
identity configuration update.

References are copied from `DATA_ROOT/_processed_ROI/_references`, if present,
then updated with the newest per-scanner templates in earlier timestamped
`_development_*` snapshots. PRIMESCAN reference selection is therefore reused
on subsequent default runs. For custom output names, pass that snapshot's
`references` directory with `--references` on subsequent runs.
The matching reference **raw PLY must still exist** in the data tree. Saved
scanner-specific ROI templates are reused. If a template is absent, define the
circle once in the reference window using the existing interaction workflow.
For references stored elsewhere:

```powershell
python biocal3d_prepare_all.py prepare --data-root "C:\path\to\data" --references "C:\path\to\_references"
```

To choose a fixed output folder, add `--output "C:\path\to\new_prepared_data"`.
It must be empty/new. Image/map snapshots are deliberately not regenerated in
place after annotation: a tiny geometry change could invalidate existing labels.

If off-screen VTK screenshots fail, add `--no-context-qc`. RGB/coverage/height
QA images will still be produced; reference selection still requires a display
when no saved reference exists.

## 3. Review the output

Per scan, under `samples/<scanner>/G<group>/<sample_id>/`:

| File | Purpose |
|---|---|
| `roi.vtp` | Canonical 3D ROI with triangles, normals, RGB, original vertex/triangle IDs and UVs when available. |
| `geometry.npz` | XYZ, connectivity, normals, RGB, optional UV, source IDs, raw-coordinate XYZ and forward/inverse transforms. |
| `rgb.png` | 1024 × 1024 color image without lighting, shading, contrast adjustment or hole filling. |
| `valid_mask.png` | 255 for actual visible mesh pixels; 0 for the background/missing surface. This is **not** a plaque mask. |
| `maps.npz` | Triangle ID, barycentric coordinates, height, surface area per pixel, projection origin/basis and exact mesh arrays. |
| `metadata.json` | Identity, physical scale, color source, source paths/hashes, ROI/registration QC and image/map hashes. |
| `qc.png` | RGB, valid coverage and height overview. |
| `context_qc.png` | Optional context-versus-ROI screenshot to check the crop location on the holder. |

At the output root:

- `scan_manifest.csv` and `dataset_index.json`: successfully exported scans.
- `discovery_inventory.csv`: all identified inputs before preprocessing.
- `failures.csv`: discovery, reference, preparation and screenshot problems.
- `annotation_inventory.csv`: labeling images, identities, priorities and QC.
- `run_config.json`: locked resolution, physical FOV and script versions.
- `qc_overviews/`: contact sheets grouped by scanner for quick crop/coverage review.
- `classes.txt` / `class_map.json`: clean, plaque and ignore class names/values.
- `x_anylabeling/`: **only annotation images**, with unique scanner/sample names.

Do not assume that an empty error table proves correct registration. Review the
context crops, image colors, holes and edge coverage across **each scanner and
stage**, especially the new data. Weak-holder-registration `CHECK` scans are
exported for inspection but are excluded from the labeling folder by default.
Use `--include-check` only after planning to review those scans.

Embedded RGB takes precedence when available, including iTERO; UV textures are
used when embedded RGB is absent. Texture sampling interpolates UVs **inside each
triangle** before looking up the texture, preserving texture detail. `geometry.npz`
also includes sampled vertex RGB for convenient 3D display. The default flips
texture V to match image row coordinates, as in your previous script. If source
comparison proves a texture upside down, start a new run with `--no-flip-v`.

Coordinates are assumed to be **millimetres**, as in your earlier scripts. Verify
scanner export units. The common FOV defaults to the largest reference ROI
diameter plus 10%; it can be supplied with `--fov-mm`. An ROI exceeding the FOV
fails rather than being silently cropped. Holes remain invalid. Where the surface
folds over itself, only the upper visible surface is mapped.

## 4. Label in X-AnyLabeling

1. Open the output folder's `x_anylabeling` directory through X-AnyLabeling's
   **Open Dir** action. The images are named, for example,
   `TRIOS3__BCG3T1-9.png`, so corresponding scanners never share filenames.
2. Label regions with exactly `clean`, `plaque`, or `ignore`. Use polygons,
   including SAM-assisted polygons. Rectangles and circles are also supported.
   Lines, points and other annotation types are rejected by the projection command.
3. Save the native **JSON beside its image**. Do not change the image resolution,
   rotate/crop it externally, or rename it. No PNG-mask export step is needed.
4. For partial annotation, mark explicit clean and plaque regions. Unmarked pixels
   stay ignored, avoiding false clean training labels.
5. For complete plaque-only annotation, inspect the entire visible sample and
   **Mark as Checked**. Then use `--background clean` when projecting. This makes
   the remaining valid surface clean; `ignore` polygons still exclude uncertain
   regions. The JSON must contain `checked: true` or image flag `complete: true`.
   Older X-AnyLabeling versions without the checked action can use an image-level
   `complete` flag or explicit clean polygons instead.

No empty annotation JSONs are pre-created. An unlabelled image therefore does
not appear already annotated. `annotation_inventory.csv` records its initial
state; projection results are written to separate manifest files.

For intersecting shapes, the conversion precedence is **clean → plaque → ignore**,
independent of drawing order. Use clean surface polygons first conceptually,
then plaque islands and uncertain regions. To represent a clean hole within a
plaque region, edit/split that plaque polygon so the hole is excluded; overlapping
clean polygons do not erase plaque. `difficult` alone does not mean ignored:
label an uncertain region explicitly as `ignore`.

## 5. Project saved labels to the 3D model

Partial annotations, with unmarked regions ignored:

```powershell
python biocal3d_prepare_all.py project-labels --output "C:\path\to\_development_TIMESTAMP"
```

Complete, checked plaque-only annotations, with unmarked valid surface clean:

```powershell
python biocal3d_prepare_all.py project-labels --output "C:\path\to\_development_TIMESTAMP" --background clean
```

Use `--annotations "C:\path\to\json_folder"` if native JSONs were saved to a
separate directory. Keep the prepared annotation images in `x_anylabeling`; their
hashes, image identity, dimensions and mapping hashes are checked before conversion.
Only JSONs for known scan IDs are processed. Missing JSONs stay unlabelled.
This command can be rerun after editing annotations; it replaces derived label
files for each successfully projected scan and retains the preparation geometry.
Check `label_projection_failures.csv` after every run: older derived labels for
a scan that now fails are not a successful current projection.

Per annotated sample, the `labels/` subfolder contains:

| File | Purpose |
|---|---|
| `mask.png` | Single-channel integer labels: **0 clean, 1 plaque, 255 ignore/outside**. |
| `pixel_labels_3d.npz` | Labelled pixel coordinates, triangle IDs, barycentric weights, exact canonical/raw surface XYZ and area weights. |
| `mesh_labels.npz` | Vertex/face class votes, approximate labels, labelled-support fractions and geometry/IDs. |
| `roi_labelled.vtp` | Color ROI with point/cell `plaque_label` and support fractions; canonical coordinates. |
| `roi_labelled_raw.vtp` | Same labeled ROI in the original scanner model's coordinates. |
| `overlay.png` | Color overlay for checking label/image agreement. |
| `label_metadata.json` | Annotation hash, mapping identity and label-conversion settings. |

`mask.png` may look almost black in a normal image viewer because the class
values are 0 and 1. This is expected. Inspect `overlay.png` for visible colors.

The precise correspondence is:

```python
with np.load(sample_dir / "maps.npz", allow_pickle=False) as m:
    face_id = m["triangle_id"][row, col]  # -1 outside valid surface
    if face_id >= 0:
        corners = m["faces"][face_id]
        xyz = m["barycentric"][row, col] @ m["vertices"][corners]
        xyz_raw = m["barycentric"][row, col] @ m["vertices_raw"][corners]
```

Pixel surface points retain the exact geometry correspondence of the prepared
raster. Vertex/face labels are **approximations**, obtained by area-weighted class
votes (also barycentric-weighted for vertices). Mixed boundaries cannot be
represented exactly by one label per large triangle. Ties and unsupported labels
remain 255. The default requires at least 50% of a vertex/face's **visible** support
to be labelled; it does not claim unobserved surface coverage. Hidden surfaces are
not filled using nearest neighbours. Use `pixel_labels_3d.npz` when precise 3D
boundary placement matters, and support fractions when training on vertex labels.

`RawVertexId` refers to the input PLY point ordering. `RawTriangleId` refers to the
triangulated source mesh. In maps-only mode older preprocessing outputs may lack
these IDs; -1 is exported instead of inventing them, while inverse-transform XYZ
still supports overlay on the raw model.

## 6. Training preparation and optional splits

For a 2D segmentation model, load `rgb.png`, `labels/mask.png`, and the valid mask.
Use 255 as `ignore_index`. Keep geometric information in `geometry.npz` and
pixel/triangle mappings in `maps.npz` for multimodal/3D development.

Image size defaults to 1024 for annotation detail. DINOv2 uses 14-pixel patches,
so its model preprocessing may resize/pad the input to a suitable size; DINOv3
backbone requirements depend on the chosen model. Resize labels with nearest
neighbour if constructing training inputs and track that transform. **Always**
project annotation JSON from the original saved image coordinates. This workflow
does not write DINO embeddings or claim interchangeability with old feature files.

Optional specimen-level 80/10/10 splits:

```powershell
python biocal3d_prepare_all.py prepare --data-root "C:\path\to\data" --make-splits
```

All scanners and stages of `Gx_Sn` stay together. With 40 specimens, this produces
32 training, 4 validation and 4 test specimens, stratified by fresh/mature growth
condition. Small synthetic subsets cannot support that split and fail explicitly.
Without this option, scans are marked `unassigned`; assign specimen splits before
training. To preserve an existing experiment, use `--splits "C:\path\to\specimen_splits.csv"`.
New specimen IDs absent from the old split require an explicit new experiment
split; existing assignments are never silently changed.

The specimen identity assumption remains: group + sample number represents the
same physical specimen across all scanners/stages, and different groups contain
different specimens. Correct this convention before training if specimens were
reused. Per-scan plane alignment is **not** cross-stage or cross-scanner pixel
registration and must not be interpreted as local plaque-removal change maps.

## 7. Optional maps-only rerun

If the raw ROI extraction has already been done, you can export a new image/map
snapshot from one authoritative ROI tree:

```powershell
python biocal3d_prepare_all.py prepare --data-root "C:\path\to\raw_data" --processed-root "C:\path\to\_processed_ROI"
```

Each `roi.vtp` needs a sibling `geometry.npz` containing `original_to_canonical`
and `roi_radius`. Original external textures must remain available when needed.
The default command without this option reruns the complete raw preprocessing.

## Validation performed for this release

Synthetic tests cover five scanner paths, embedded RGB and external UV textures,
reference reuse (including automatic reuse from earlier snapshots), holder-based registration on translated scans, preservation of
raw vertex identities, triangle/barycentric image-to-surface round trips, class
mask conversion, raw-coordinate labeled exports and annotation safeguards.
The user's new raw data were not available here: inspect the generated QA and
failure tables during the first local run.

X-AnyLabeling native JSON and review-state behavior:
https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md

## PRIMESCAN and dependency repair (updated release)

PRIMESCAN is recognized case-insensitively as a scanner folder. Keep the same
BCG group/stage/sample structure, for example
`PRIMESCAN/G3/BCG3T1-1/scan.ply`. Empty or absent PRIMESCAN directories are harmless.
Once its scans are added, rerun `prepare`. The first run defines its own reference
circle, preferably on `BCG1T0-1`; if that sample is absent, the first available
baseline is selected. The reference is saved and reused in subsequent snapshots.
The registration and color pipeline supports PLY with embedded RGB or UV
coordinates plus an external JPG/PNG. Actual PRIMESCAN exports have not yet been
validated; formats without a color PLY need conversion or a loader extension.

The original attachment `preprocessing(1).py` was compared with the source used
for this package. All function definitions match; the data-root path differs.

Requirements now pin the locally validated pair: PyVista 0.49.0 and VTK 9.3.1.
If you see the `VTK_TRIQUADRATIC_PYRAMID` import error, repair them in the exact
interpreter used to run BioCal3D. In PowerShell on your AU workstation:

```powershell
$biocalPython = "C:\Users\au662213\AppData\Local\anaconda3\envs\biocal3d\python.exe"
& $biocalPython -m pip install --upgrade "pyvista==0.49.0"
& $biocalPython -m pip install --force-reinstall --no-deps "vtk==9.3.1"
& $biocalPython biocal3d_prepare_all.py check-env
```

Close any Python sessions that imported VTK before reinstalling its wheel. The
VTK reinstall leaves your NumPy/PyTorch dependencies alone. The PyVista upgrade
lets pip install any dependencies required by that version. `check-env` reports
the interpreter, installed versions, imported module paths and VTK symbol check,
without reading or modifying scans. It helps diagnose remaining shadowed or
mixed-installation modules rather than hiding the import error.

After the import check succeeds, rerun preparation with your actual data root.
Official supported-version information:
https://github.com/pyvista/pyvista/releases

## Additional groups (G6 update)

The original hard-coded G1–G5 discovery restriction has been removed. G6 is
configured as mature 24-hour growth, T0 baseline, T1 partial and T2 clean.
New positive group numbers are included automatically without assuming their
biological protocol. Unknown protocols cannot be used for an automatically
growth-stratified split until configured; imported specimen splits are supported.
Existing snapshots are retained. Extract the updated package, replace the three
Python scripts together and rerun `prepare` to export all data, including G6.
Check `scan_manifest.csv` for the new group and `failures.csv` for any failed scans.
Older snapshots may list rejected G6 scans as `FAILED_DISCOVERY`.
