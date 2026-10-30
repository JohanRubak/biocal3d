# Validation — BioCal3D Studio 1.0.0

## Dedicated Review workspace and native Orient viewer

The navigation is now Inspect / Orient / Review / Label. The full ROI editor is
embedded as a reusable Review workspace; it no longer opens from a Verify
button in Label. Review checks cover embedded saving without closing, guarded
workspace switching, original-model lookup, source-face label migration, and
preserving independent label files when only geometry is reviewed. Label
migration is marked for persistence when labels are opened after a correction.

Orient now uses full VTK models in the verified canonical frame. Its native
trackball wheel and middle-button pan paths were exercised with render drawing
disabled. All cameras remained linked, including after camera rotation and
scanner angle changes. The landmark direction calculation was checked in the
new viewer. A full-original loading fixture verified the coordinate transform
and unchanged triangle count. The Matplotlib orientation canvas is removed.

Current full-suite result: **102 passed, 2 skipped** with the pinned dependencies.
The isolated dependency check and source compilation passed as well. The earlier
results document the previous release. Native OpenGL drawing and mouse picking
on a target Windows PC remain a target-PC check for this update.

## Appearance update

Before this update, the user confirmed that installation and all existing
functionality worked on their Windows PC after the PowerShell hotfix.

The shared dark theme and compact sample rows were checked with the four
existing Studio integration tests: **4 passed** on the pinned dependencies.
Offscreen screenshots were visually inspected for labeling and orientation;
inspection controls and renderer background settings were also checked with
render-window drawing disabled. The checks verified that all three workspaces
use the same stylesheet and that rows have status icons, sample-first text,
full-path tooltips and accessible status descriptions. No new hardware-rendering
validation was performed for this cosmetic update. The previous full-suite
results below describe the earlier functional build.

## Windows installer hotfix

The user's Windows PowerShell 5.1 setup log confirmed that all pinned packages
installed successfully, then the inline Python import check lost its embedded
quotation marks. The installer now executes `setup/check_dependencies.py` as a
file, avoiding nested command-line quoting. This check passed with isolated
Python (`-I`) and the pinned dependencies on Linux. The user subsequently confirmed successful Windows installation and existing
functionality; application code was unchanged by this hotfix.

Validated on 30 October 2026 in Linux with Python 3.12, Qt's offscreen platform, and VTK 9.3.1. This is source/runtime validation, not certification on every laptop or operating system.

## Executed results

- Full included regression suite: **98 passed, 2 skipped**. This passed both with the host scientific stack and with the exact package versions in `requirements.txt`.
- After the final workspace/copy-lock changes: the **4 integration tests passed again** using the pinned package versions. These include the combined workflow that opens labeling, saves edits, switches workspaces, releases its annotation locks, and creates an oriented copy.
- Python source compilation: passed.
- Unix launcher shell syntax checks: passed.
- Qt orientation workspace startup and raster UI screenshot inspection: passed.
- Layout check: the combined labeling workspace fits a **1366 × 768** application window. Smaller content areas can scroll; inspection also has a one-stage option.

Pinned stack checked: NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.4, Pillow 11.3.0, PySide6 6.8.3, VTK 9.3.1 and plyfile 1.1.3. Matplotlib emitted dependency deprecation warnings; the checks passed.

## What the tests cover

- Wheel zoom in orientation changes all three axis limits together.
- Toolbar-style zoom/pan limit changes synchronize all three views.
- Changing the orientation angle retains the zoom; Fit restores the full view.
- Inspection camera changes and the VTK trackball wheel path synchronize all 12 cameras without a feedback loop.
- Current versioned ROI corrections appear in inspection, and inspection leaves dataset files unchanged.
- Workspace switching saves edited labels, reuses existing workspaces, and prevents hidden labeling controls from capturing Space.
- A failed save prevents switching away from labeling.
- Orientation can run after labeling was opened: its source locks are released before copying.
- Original orientation-engine regressions verify rigid transforms, annotation identity, reviewed-surface histories, transforms/maps, textures, output-copy rules, frame checks and source preservation.
- Existing geometry, label transfer, brush-state, original-model relocation, rater/review controls, cache, and dataset-discovery tests are included.

## Limits

The test datasets are synthetic. One optional real-data ROI review test was skipped because its external fixture is unavailable. The hardware-visible-surface picking test was skipped because the environment has no working X11/OpenGL display.

Qt/VTK interface integration tests disable actual render-window drawing. They exercise camera/interactor state and the real dataset/annotation loaders, but do not establish native GPU rendering or mouse-picking behavior on a target PC. The orientation raster canvas was checked independently.

The Windows installer and existing functionality were subsequently confirmed by the user on their PC. The agent has not independently executed native Windows rendering, Windows PyInstaller/Inno build recipes, or the macOS launcher on those platforms. A native Windows EXE or offline installer has not been built or signed. The delivered Windows installer downloads the runtime and dependencies during setup. Test first installation and ordinary painting/orientation/export on a Windows x64 PC before wider distribution.

Original source archives were retained unchanged. No user research dataset was modified during development.
