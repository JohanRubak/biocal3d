# Approved ROIs example browser

Use actual specimen scans to select images for the manuscript. Assemble the final multipanel figure in PowerPoint after choosing a specimen. All available groups G1–G6 are treated equally.

## Start on Windows

1. Extract this entire ZIP into a folder **outside Approved ROIs**. Keep the Python files together.
2. Double-click `Start_Browser.bat`. It uses your active `biocal3d` environment first; otherwise it tries to activate it, then falls back to the current Python. It installs nothing.
3. Click **Open Approved ROIs folder…** and select your dataset folder, for example:

   `C:\Users\YOUR_USER\repos\biocal3d\data\Approved ROIs`

If the launcher cannot find your environment, open Anaconda Prompt:

```bat
conda activate biocal3d
cd /d "C:\path\to\Approved_ROI_Example_Browser"
python browse_approved_rois.py --root "C:\path\to\data\Approved ROIs"
```

The browser uses Python 3.10+, NumPy, Pillow, PySide6 and VTK. Your annotation environment likely already has them. Install only missing dependencies if necessary:

```bat
python -m pip install -r requirements.txt
```

## Find a good example

- Start with a **G3–G6 specimen** if you want to illustrate baseline, partial cleaning and final cleaning in the same figure. G1–G2 contain baseline and final cleaning only.
- Select a specimen in the left list. **A / D** browse previous / next. The four scanner panels show the **same physical specimen**, paired by group and specimen number. Each scanner panel shows its cleaning stages side by side: baseline and final cleaning for G1–G2, or baseline, partial cleaning and final cleaning for G3–G6. All stages are visible simultaneously.
- Read the cleaning sequence from left to right within each scanner panel. A useful illustration has visible deposits at baseline, a clear mix of covered and exposed surface after partial cleaning, and little remaining material after final cleaning. Check that this is supported by the actual images; do not infer it solely from the stage name.
- Prefer examples with a complete ROI in all four scanners, comparable orientation, limited acquisition artefacts and a pattern that reflects the broader dataset. If one is an unusual case, explain this in the caption.
- **Scanner colour (unshaded)** shows saved scanner RGB without colour correction or lighting. **Geometry** and **Geometry + triangle edges** help inspect the surface and mesh; use colour mode for the primary illustration.
- **Use original texture if available** uses stored UV coordinates and the original texture image. If unavailable, the panel explicitly reports an RGB fallback. Saved vertex RGB can look coarser than a full texture. Do not describe that coarseness as scanner texture resolution. For the fairest comparison, use the same representation across scanners and describe it in the caption.
- Mouse left drag rotates, middle drag pans, wheel/right drag zooms. **Link cameras** applies the same camera to all panels. **R** restores the top view; **Oblique view** reveals surface relief.
- Canonical axes are used as saved. Center translation aligns the display frames, but **no spatial registration is performed**. Check visible landmarks; the **Rotation** control above each panel can adjust its in-plane angle. This is a display adjustment recorded in the export metadata. Stages are not automatically registered either. Each stage has its own rotation control, saved and exported separately. Existing shortlists with one rotation per scanner remain supported.
- Write a short note and click **Save / update shortlist**. Notes and rotation settings are saved outside the dataset. The shortlist is associated with the current dataset path and export folder. Camera orientation is not saved in the shortlist; restore it before exporting.

## Export for PowerPoint

Choose a stage in the **Export stage** selector and click **Export stage…** for the four-scanner comparison, or click **Export all stages…** for the complete specimen. The selector controls only export; every acquired stage remains visible in the browser. The default is 1600 × 1600 pixels per scan, with 300 dpi metadata; you can raise the pixel count before export.

Exports go to `Pictures\BioCal3D_Figure_Examples` by default. Use **Export folder…** to change this. The export location must be outside Approved ROIs.

Each export creates a new folder containing:

- Individual, unlabelled PNG images with a white background, suitable for insertion into PowerPoint.
- `labelled_overview.png`: a 2 × 2 layout of four scanner panels, each containing stage images from left to right. Missing or failed scans are explicitly marked.
- `iTERO_stages.png`, `TRIOS3_stages.png`, `TRIOS5_stages.png` and `LABscanner_stages.png`: separate labelled scanner panels for flexible layout in PowerPoint.
- `export_metadata.json`: source files and hashes, specimen/stage pairing, colours or texture used, camera, display rotations, notes and any loading errors.
- `selected_scans.csv`: the scan list for this export.

**Fit full ROI on export** is enabled by default. It keeps your view direction and fits every exported ROI with a shared scale. This can change the zoom from the preview, especially for all-stage exports. Turn it off for an intentional close-up using the current linked camera. Close-ups can crop the surface; inspect the resulting PNGs. The same source coordinate units and parallel camera scale are retained across panels; no scale bars are invented. Confirm the acquisition units before adding a scale bar in the final figure.

In PowerPoint use Insert → Pictures, crop consistently, preserve image aspect ratio, and add panel letters and short labels. Use the individual PNGs for the final layout; the labelled overview is primarily for selecting an example. Keep scanner colours unchanged when describing scanner differences. Avoid claiming that a selected illustration demonstrates quantitative differences by itself.

## Dataset assumptions

Expected layout:

```text
Approved ROIs/
    samples/
        iTERO/       (also TRIOS3, TRIOS5 and LABscanner)
            G3/
                BCG3T0-1/
                    geometry.npz
                    roi.vtp
```

`geometry.npz` is preferred; `roi.vtp` is used as a fallback. The loader accepts saved vertices/points, triangular faces and vertex or face RGB. Available ROI masks are respected. Review-version folders and annotation files are ignored. Duplicate scan identities stop discovery so two versions cannot be silently mixed. The scanner folder names identify each scanner. The root can also be `samples` itself or a scanner subfolder, although a full Approved ROIs root is recommended for the comparison.

Stage interpretation follows the current study design:

| Groups | T0 | T1 | T2 |
| --- | --- | --- | --- |
| G1–G2, 12 h | Baseline | Final cleaning | Not acquired |
| G3–G6, 24 h | Baseline | Partial cleaning | Final cleaning |

The app discovers the files actually present; it does not hard-code 48 specimens or 512 scans and does not exclude any group. Missing scans are shown, not replaced by another specimen. There is no smoothing, decimation, colour normalisation or edit to the input files. Texture paths from another PC are rebased onto the current project's `data` folder where possible; otherwise the saved RGB is used and identified. Specimen selection is manual; the app does not estimate plaque or rank examples by biological appearance.

## Optional command line

Write an inventory without opening the GUI:

```bat
python browse_approved_rois.py --root "C:\path\to\Approved ROIs" --inventory "C:\path\to\exports\inventory.csv"
```

Export one specimen from the top with a common scale:

```bat
python browse_approved_rois.py --root "C:\path\to\Approved ROIs" --output "C:\path\to\exports" --export-specimen BCG3-1 --pixels 2000
```

VTK needs a working graphics context for rendering, including command-line exports. Normal Windows desktop use provides this. Inventory generation needs NumPy but no graphics context or Qt.

## Troubleshooting

- **No meshes found:** select the actual Approved ROIs root containing `samples`, not the raw scans, review versions or a results directory.
- **Grey surface:** the panel has no usable RGB or you selected geometry mode. Check the message below it. No artificial colours are supplied.
- **Texture unavailable:** select saved RGB mode, or make the original texture file accessible under the same project's `data` folder. The viewer does not search the entire drive for substitute images.
- **Wrong orientation:** review the top face and landmarks; adjust the rotation above the individual stage image. Do not treat framing as registration.
- **Loading on a network drive:** discovery reads sample folder names first and loads the selected specimen’s 8 or 12 scans. Up to 24 meshes are cached in memory; no cache is written to the dataset.
- **Export failure:** retain the full terminal error and any `export_metadata.json`. Missing mesh loads are recorded, but a graphics or image-writing error stops the export rather than claiming completion. A partially written export folder may remain.

Run the included functional checks with `python -m unittest discover -s tests -v`. Rendering checks additionally need a desktop display/graphics context.

## Windows Qt startup fix (version 1.0.1)

Replace the extracted package files with this version, then run `Start_Browser.bat` from your activated `biocal3d` prompt. The launcher uses that environment directly rather than nesting `conda run`. Before creating a window, the app replaces inherited Qt plugin paths with those belonging to the selected PySide6 installation and adds its DLL directories for this process only. No system settings or installed packages are changed.

The launcher first checks QApplication startup in a separate process. If startup fails, it retries with `QT_DEBUG_PLUGINS=1` and saves the full output to `qt_startup.log` beside the scripts (or `BioCal3D_qt_startup.log` in your user folder if that location is not writable). Run `Diagnose_Qt.bat` for the detailed check without opening the browser. Share the log if the problem remains; it can distinguish an incompatible plugin, missing DLL or package-version problem. A successful Qt probe checks window initialization, not the VTK graphics driver.

## Simultaneous stage layout (version 1.1.0)

Four scanner panels are arranged in a 2 × 2 grid. Each panel contains its own two or three stage viewports, ordered left to right. G1–G2 have no empty partial-cleaning column. Missing scans within an acquired stage retain a labelled placeholder, so another specimen cannot fill the gap. Cameras can be linked across all visible stages and scanners; each stage also has its own in-plane rotation. The previous Windows Qt startup fix and startup diagnostics remain included.
