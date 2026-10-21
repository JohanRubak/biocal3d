# Validation and scope

Validated on 21 October 2026.

- All nine included functional tests pass using Python 3.12, NumPy and VTK 9.3.1. The generated 48-specimen fixture contains all six groups and 512 scans. Tests check stage interpretation, identity-based pairing, missing pairs, duplicate rejection, read-only output protection, unchanged RGB, ROI masks, invalid geometry, texture/RGB separation and CSV inventory.
- The previously shared iTERO `BCG1T0-1` approved surface was loaded and exported successfully (2,590 vertices and 4,540 triangles); the resulting image was visually inspected. This validates that sample's array representation. The complete local Approved ROIs folder has not been available here for end-to-end validation.
- A saved VTP version of that sample was read back with unchanged RGB and triangle count.
- A synthetic four-scanner, three-stage fixture was exported to 12 individual PNGs plus a labelled montage. Pairing, filenames, panel dimensions, shared scale metadata, in-plane rotation metadata, stage rows and unchanged source-file hashes were checked. Missing panels were also exercised with the single-scan sample. Rendering used VTK 9.5.2 with an offscreen EGL context, because the validation server has no desktop display.
- Qt application logic was exercised using PySide6 6.8.3 with an offscreen interactor substitute: dataset loading, group filtering, two-/three-stage controls, stage switching, shortlist persistence, rotations, geometry mode and camera controls. The GUI layout and export montage were inspected.

The native embedded VTK viewer and batch launcher still need their first run on a Windows desktop. They use the same PySide6/VTK integration as the existing annotation workflow, but that desktop integration was not claimed as tested by the offscreen checks. No sample meshes, private paths or synthetic demonstration scans are included in this package.

## Version 1.0.1 startup patch

All 13 functional tests pass. Four new tests cover overriding foreign Qt paths with the chosen binding's plugins, retaining DLL-directory handles, handling the conda plugin layout, failing clearly when qwindows.dll is missing, and preserving a failed child-process probe plus its detailed retry. A real QApplication subprocess probe also passed on the Linux validation host using PySide6 6.8.3 with the offscreen platform. The Windows paths are tested with directory fixtures; the user's Windows plugin/DLL problem has not been reproduced or confirmed resolved here.

## Version 1.1.0 simultaneous stages

The Qt smoke test exercised four scanner containers with 8 visible meshes for G1 and 12 for G3, correct left-to-right stage order and T1/T2 final-cleaning labels, switching between two- and three-stage specimens without stale hidden meshes, per-stage rotation shortlist persistence, and an export-stage selector that does not change the visible images. Screenshots of both layouts were inspected. As before, this used an offscreen interactor substitute on Linux, so native Windows graphics remains unverified here.

Two- and three-stage export fixtures generated 8 and 12 individual images respectively, plus a four-scanner overview and four separate scanner panels. Stage-specific rotation metadata was checked alongside legacy scanner-level rotation fallback. All 13 existing core/startup tests still pass.
