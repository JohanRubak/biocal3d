# BioCal3D: raw scans → registered ROI PLY → direct 3D labels

This updates the existing BioCal3D preprocessing package. Its scanner-specific
reference/holder registration and circular ROI extraction module are retained.
Preparation now exports a portable **binary triangle ROI PLY** for every
successful scan and a dedicated 3D labeling inventory. The bundled annotator
uses this inventory and writes separate per-rater labels.

## Place the package

Extract `biocal3d_workflow` into:

```text
C:\Users\au662213\repos\biocal3d\biocal3d_workflow
```

Keep the Python companion files and the `annotator` subfolder together.
Existing raw data, preparation snapshots, ROI references, and annotations
stay in their existing locations. The default data-root search finds your
project's `data\02-09-2026-Data collected` folder when the scripts are placed
inside the project. `--data-root` or `BIOCAL3D_DATA_ROOT` overrides it.

## Environment and launchers

All batch launchers automatically activate the **existing `biocal3d` Conda
environment**. They look for your `%LOCALAPPDATA%\anaconda3` installation,
other common locations, `CONDA_BAT`, and Conda on PATH. A nonstandard install
can be supplied with `BIOCAL3D_CONDA_BAT` pointing to `condabin\conda.bat`.
They verify the Python prefix before running. They do not install packages
or create environments silently.

If the new UI dependencies are missing, install them once in Anaconda Prompt:

```bat
conda activate biocal3d
python -m pip install "PySide6>=6.6,<7" "plyfile>=1.0,<2"
```

The existing preprocessing requirements remain in `requirements.txt`, now
including the annotator dependencies. Use that file for a full setup only
when required. PyVista/VTK pins from the existing workflow were retained.
The launchers cannot be execution-tested on Windows in this development
environment; the Python workflow and UI checks were run on Linux.

| Launcher | Action |
| --- | --- |
| `start_preprocessing.bat` | Run raw loading, registration, ROI extraction, PLY and image/map exports |
| `start_labeling.bat` | Open the last prepared snapshot's approved 3D annotation inventory |
| `check_environment.bat` | Print interpreter/versions and check PyVista/VTK imports |
| `create_labeling_plan.bat` | Build grouped calibration/agreement/production queues from an existing snapshot |
| `analyse_agreement.bat` | Compare completed independent overlap ratings |
| `seed_consensus.bat` | Create separate consensus drafts after both ratings are complete |
| `annotator\start_annotator.bat` | Open the standalone 3D annotator with optional CLI arguments |

## Preprocess raw data

Double-click `start_preprocessing.bat` from the package in your repository,
or run it with an explicit data root:

```bat
start_preprocessing.bat --data-root "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected"
```

Equivalent Python command:

```bat
python biocal3d_prepare_all.py prepare --data-root "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected"
```

The sequence is the existing scanner loader → saved reference circle (define
once if missing) → registration on the holder → circular specimen extraction
→ canonical coordinates → ROI PLY and geometry/maps → QC and inventories.
New nested sample folders, G6, and the five configured scanners remain supported.
Source paths must still contain the scanner and BCG group/time/specimen identity.

Each run creates a **new `_development_<timestamp>` snapshot**. A supplied
`--output` must be new/empty. This avoids changing geometry after labeling.
Original raw PLYs, older snapshots, and existing annotations are not overwritten.

Saved references are reused from the legacy reference directory and previous
`_development_*\references` folders. For a custom reference location:

```bat
start_preprocessing.bat --data-root "C:\path\to\raw" --references "C:\path\to\references"
```

The reference raw scan must still exist. The original reference interaction
and registration thresholds are unchanged. If context screenshots fail,
add `--no-context-qc`; review the other QC outputs before labeling.

## Output per sample

Under `samples\SCANNER\Gx\BCGxTy-n`:

| File | Purpose |
| --- | --- |
| `roi.ply` | Triangle ROI in canonical coordinates, normals, original/sampled RGB, and original texture UVs when used |
| `roi.texture.jpg` / `.png` etc. | Matching texture copied beside the PLY, referenced in its header |
| `roi.vtp` | Retained VTK ROI for existing scripts |
| `geometry.npz` | Exact ROI connectivity, raw point/triangle IDs, raw coordinates, and forward/inverse transforms |
| `rgb.png`, `valid_mask.png` | Unlit 2D view and actual mesh coverage |
| `maps.npz` | Pixel → triangle/barycentric → canonical/raw 3D correspondence |
| `metadata.json` | Scanner/specimen/stage, registration QC, physical scale, hashes |
| `qc.png`, optional `context_qc.png` | Color/coverage/height QC and holder/crop-location QC |

PLY face order is identical to the ROI geometry/maps. `source_face_id` stores
the raw triangulated source face IDs when available; `geometry.npz` preserves
both raw vertex and face mappings. Geometry is not decimated or smoothed for
annotation. Texture UVs retain image detail; vertex RGB is included for
convenient color display. In the existing loader, embedded RGB takes precedence
when present; texture + UV are used when embedded RGB is absent. The existing
texture-V convention is respected, including the explicit `--no-flip-v` option.

The run also writes `scan_manifest.csv`, `failures.csv`, `dataset_index.json`,
`annotation_inventory.csv`, reference templates, and QC overview sheets.
**Review registration and appearance for every scanner.** A successful export
is not proof that the circle landed on the correct specimen. Weak-holder
`CHECK` scans are retained for QC but omitted from the normal labeling queue.
`--include-check` explicitly includes them when you intend to review them.

## Label in 3D

After QC, double-click **`start_labeling.bat`**. It opens the last snapshot
created by this package. You can select an existing snapshot explicitly:

```bat
start_labeling.bat --output "C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261007_100936_877172" --annotator johan
```

This also supports older inventories: if an inventory has no `annotation_mesh`
column, its sample folder's `roi.ply` or `roi.vtp` is used. Existing VTP labels
remain usable with their original inputs. Do not move them onto newly prepared
PLYs without an explicit correspondence/migration step.

Equivalent command:

```bat
python biocal3d_prepare_all.py label --output "C:\path\to\snapshot" --annotator johan
```

The label command loads **only the approved inventory**, in its priority order
(partial samples first within split), and verifies newly prepared PLY hashes.
Opening a folder directly in the standalone annotator uses ordinary discovery
and bypasses the workflow's QC queue; use the workflow launcher for this study.
A second rater uses `--annotator rater2` to get independent annotation files.

Paint plaque, inspect the whole ROI, mark reviewed remainder clean, then
**Save & Next**. Number keys: **1 plaque, 2 clean, 3 uncertain, 4 unreviewed**.
Space + left drag rotates. The UI also has surface/points/wireframe toggles,
vertex/face/no shading, adjustable brush size, undo/redo, and autosave.

Working labels save under `SNAPSHOT\annotations\RATER\...` as `.labels.npz`.
**Auto-save PLY too** writes a matched labeled PLY on every NPZ save, including
drafts. Labels are face properties (`face_label`, plus MeshLab-compatible
`quality`); scanner RGB is preserved. NPZ retains draft/completion status,
notes, rater and geometry fingerprint. An exported PLY is a snapshot; editing
a completed annotation updates its NPZ and automatic PLY draft. NPZ completion
status remains authoritative.
For training, use completed NPZs and faces labeled 0/1. Exclude -1/-2.
See `annotator\README.md` for complete controls and data conventions.

## Existing 2D workflows and model inputs

RGB, maps and geometry are still generated for your 2D/3D model branches.
An X-AnyLabeling image queue is now optional:

```bat
start_preprocessing.bat --export-x-anylabeling
```

The legacy `project-labels` subcommand is retained. Instructions for that
2D-labeling path are in `README_2D_legacy.md`. The new direct-3D workflow does
not require X-AnyLabeling or a 2D-to-3D projection step. Rendering 2D masks
from the new face annotations can be added to the training pipeline later.

## Verification

`python -m unittest -v test_workflow.py` tests synthetic colored and textured
ROI export, vertex/triangle correspondence, portable texture copying,
processed-snapshot preparation, CHECK exclusion, and annotator launch arguments.
`annotator\test_core.py`, `test_brush.py`, and `test_ui.py` cover label storage,
PLY round trips, visible-face brushing, numeric keys and desktop operations.
The raw registration module was kept unchanged; a full real-scanner raw-data
run and the Windows batch launchers still need validation on your workstation.

## Dentist study and new annotation controls

See **LABELING_STUDY.md** for calibration, independent dentist overlap, consensus,
production allocation, agreement analysis and batch exports. Start with
`create_labeling_plan.bat` on an existing prepared snapshot; no preprocessing
rerun is required. The updated app includes display-only brightness/contrast/
gamma/saturation, an annotator dropdown with New, top/bottom orientation cues,
and a top-only brushing guard.
