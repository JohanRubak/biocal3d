# BioCal3D Studio 1.0.0

A dedicated desktop workspace for inspecting, orienting, reviewing and labeling the BioCal3D dataset.

## Install on a Windows PC or laptop

1. Right-click `BioCal3D_Studio_1.0.0.zip` and choose **Extract All**. Do not run setup from inside the ZIP.
2. Open the extracted `BioCal3D_Studio` folder.
3. Double-click **Install_BioCal3D_Studio.cmd**.
4. After setup finishes, open **BioCal3D Studio** using its desktop or Start menu shortcut.

First installation needs internet and downloads a private Python runtime plus the scientific/graphics dependencies. No existing Python, Conda, CUDA, or administrator account is required. The runtime is separate from your existing BioCal3D environment. Allow several minutes and approximately 2 GB free space for the runtime, packages and installation files, plus space for your datasets. Ordinary integrated graphics can work; the 3D views require functioning desktop OpenGL drivers.

Target: Windows 10/11 x64. Windows ARM, 32-bit Windows and restricted institutional installations are not validated. The included setup is an **online installer**, not a precompiled EXE. The user confirmed that installation and the three workspaces function on their Windows PC after the installer hotfix; other target machines still need checking.

To update, close Studio and run the new package's installer. The previous application code is kept as `application-previous`; datasets and labels remain wherever you selected them. Setup logs are under `%LOCALAPPDATA%\BioCal3DStudio\logs`. To remove this source-based installation, delete its desktop/Start menu shortcuts and `%LOCALAPPDATA%\BioCal3DStudio`; that does not delete datasets stored elsewhere.

## macOS and Linux

These source launchers are included, but have not been validated on native macOS or a full Linux desktop. Install Python 3.10–3.12 first (3.11 recommended), then run:

```sh
sh Install_and_Start.command
```

The launcher creates a private virtual environment and starts the same app. To select Python explicitly:

```sh
BIOCAL3D_PYTHON=python3.11 sh Install_and_Start.command
```

Linux may need its distribution's `python3-venv`, desktop X11/Wayland, OpenGL, and Qt xcb runtime libraries. Binary dependency availability depends on the OS and processor architecture. macOS security policy may require allowing the downloaded launcher. Keep the extracted source folder for subsequent launches.

## Appearance update

All four workspaces share the dark Label theme, including buttons,
input fields, menus, plot labels, toolbars and viewport backgrounds. The
inspection figure exports keep their existing publication background.

The labeling sample list uses two small status icons instead of text prefixes:
the left hexagon is ROI approval; the right symbol is label status. A check
means approved, a pencil means draft, an empty circle means not started, and
an exclamation mark means attention is needed. Sample IDs appear first, with
scanner/folder context underneath. Hover a row for the full path and status
text; the same details are available to accessibility tools.

Close Studio and rerun the installer from this updated package to apply it.
The corrected PowerShell dependency check is included.

## Choose a workspace

The left navigation has four workspaces: **Inspect**, **Orient**, **Review**, and **Label**. **Open dataset…** selects the active dataset; the selected path is shown below the buttons. Each workspace opens that dataset when you visit it. **Original models…** points to the raw-data parent, such as `02-09-2026-Data collected`, and remembers that path per dataset. It resolves matching files rather than requiring an exact model selection for every scan.

### Inspect

The previous Approved ROI Example Browser is integrated here, including all four scanner panels, stage comparisons, group/specimen filters, original-model viewing, colour/geometry/wireframe modes, shortlists, notes and figure exports.

- **Link rotation, pan & zoom** is on by default. Wheel zoom and right-drag zoom synchronize across all panels, as do rotation and pan. Uncheck it for independent camera views.
- **One stage** reduces the number of visible panels for laptop screens. Use the stage selector at the bottom to choose the displayed stage. It starts enabled on displays narrower than 1600 pixels.
- **Refresh surfaces** reloads geometry and current ROI reviews. Returning from Review or Label refreshes inspection automatically.
- The inspector reads the current versioned ROI surface, including recent ROI corrections; the status indicates when that surface is still pending. It never displays plaque labels as scanner colour.
- Inspection reads scan data. Shortlists, notes and figure exports are written to the selected export folder outside the dataset.

The 1.0 inspector supports the four scanners in the supplied browser: iTERO, LABscanner, TRIOS3 and TRIOS5. Orientation retains the original engine's optional PRIMESCAN support.

### Orient

The scanner-orientation tool is now a Qt workspace in the same app; no separate Tk window is needed.

1. Select the **full original, unoriented approved snapshot**, with `scan_manifest.csv` and TRIOS5 present.
2. Select the matching original full-model data root.
3. Choose a TRIOS5 representative and a representative from the scanner you are rotating.
4. Use the angle slider/spin box or **Match landmarks**. For landmarks, click A then B in TRIOS5, then the same ordered pair in the other scanner. Use holder/teeth features rather than plaque.
5. **Accept scanner** for every nonreference scanner. TRIOS5 stays at 0°.
6. **Save rotations…** writes the reusable JSON and a preview PNG.
7. **Apply to new copy…** selects that saved JSON, an original dataset (or a calibration/agreement parent), and a new output folder name. **Open oriented copy** opens the result for inspection and labeling.

Orient now uses the same native VTK viewer interaction as Inspect: **middle drag to pan, scroll to zoom, and left drag to rotate the camera**. All three cameras are linked. These camera interactions change the view; the angle control applies the scanner correction. Zoom and pan survive angle changes. **Top view** restores the viewing direction; **Fit all views** frames the full models. **Match landmarks** restores a top view and temporarily uses left clicks to pick the two corresponding landmark pairs. **Cancel landmarks** returns to ordinary view interaction. Full original geometry is displayed in its verified canonical ROI frame; the Matplotlib canvas and toolbar are no longer used in Orient.

The original orientation engine is retained: rigid rotations preserve label/face identity, rotate approved-surface histories and camera/mapping metadata, and regenerate affected raster data. Input datasets are not modified. An already oriented snapshot cannot be oriented a second time. Output must be a separate new folder. Existing checks reject ambiguous or inconsistent geometry/history/frame mappings.

### Review

The complete Verify / correct ROI window is embedded in this workspace, with
its own sample list, scanner/sample search, needs-review filter, and previous,
next and next-unapproved navigation. It retains original-model loading,
extraction parameters, live preview, restore/exclude brush, guides, undo/redo,
top/side views, notes, pending/approved/flagged saves and shared geometry history.
Saving keeps the editor open. **Approve & Next** / Ctrl+Enter advances through
review, independently of plaque-label completion. Leaving with unsaved ROI
corrections offers Save pending, Discard, or Cancel. ROI changes refresh an open
Label workspace; opening Label later reads and migrates labels to the saved
surface, preserving known source-face correspondence. Review alone creates no
plaque annotation files.

### Label

The surface annotator from ROI Review v3.13 is retained. Its Verify / correct
ROI button has been removed; use Review for those tools. Existing controls include independent rater IDs, brushes, uncertainty labels, approval controls, undo/redo, original-model resolution, geometry versions, local read cache and loading feedback on dataset opening.

- A / D: previous / next sample.
- Ctrl+S: save labels.
- Ctrl+Enter: approve ROI and advance.
- Ctrl+Z / Ctrl+Y: undo / redo.
- Hold Space: rotate instead of painting.
- S / W: surface / wireframe.
- F5: reload shared ROI.

Select the correct rater before labeling. Annotations autosave to the dataset's annotation directory (or your explicit `--output`). Switching away saves the current annotation; a save failure prevents the switch. Closing also saves and releases annotation locks. Workspace shortcuts are scoped so a hidden annotator does not capture the inspection/orientation controls.

Before copying, Studio saves and closes its labeling workspace to release source locks; reopening Label loads fresh state. While the copy runs, Review and Label are unavailable. For orientation, do not apply a copy while another application is editing that source dataset. Close other annotators first so the copied source is stable.

## Existing Python environment / study options

For your current environment, install the shared requirements and run:

```sh
python -m pip install -r requirements.txt
python launch_studio.py --root "PATH/TO/Approved ROIs" --page label --annotator johan
```

Options retained for study workflows:

```text
--page inspect|orient|review|label
--root DATASET
--annotator RATER
--output ANNOTATION_OUTPUT
--manifest ANNOTATION_INVENTORY_CSV
--lock-annotator
--review-mode
--no-local-cache
--cache-dir LOCAL_CACHE
```

Use `--annotator INITIAL_CONSENSUS --review-mode` for the prior reviewer workflow. Rater locking and review mode are optional; the default is the ordinary editable annotation workflow. Never mix independent study annotations and consensus editing unintentionally.

## Build an offline Windows distribution

This package includes source and build recipes. On a Windows x64 build PC with Python 3.11, run `setup\Build_Windows_App.cmd`. PyInstaller produces `dist\BioCal3D Studio`, containing its own runtime. Copy the entire folder to the recipient, or compile `setup\BioCal3D_Studio.iss` with Inno Setup 6 to create a single installer. Neither that Windows binary nor the Inno installer is built or code-signed by this delivery.

## Validation

See `VALIDATION.md` for the executed checks and platform limits. Source provenance: ROI Review v3.13, Scanner Orientation (22 October 2026), and Approved ROI Example Browser v1.2.0 (latest retrieved revision). Original archives were not replaced.
