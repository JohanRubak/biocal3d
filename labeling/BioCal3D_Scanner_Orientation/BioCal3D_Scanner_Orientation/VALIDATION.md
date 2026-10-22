# Validation

The package was checked in Python 3.12 with NumPy, SciPy, VTK, plyfile, Pillow
and Matplotlib. Your Windows launchers target the existing `biocal3d` environment.

Six integration tests cover:

1. A four-scanner snapshot with approved shared review payloads, geometry NPZs,
   PLY/VTP meshes, UV texture, raster maps and two raters per scan. After rotation,
   the existing BioCal3D loader reopens all eight label files with completion flags
   and notes unchanged. Source files remain byte-for-byte unchanged.
2. Raw-to-canonical mapping composition, unchanged raw coordinates, exact mesh
   connectivity, IDs, labels and masks. Pixel/barycentric mappings are checked
   against the reprojected 3D triangles.
3. Rejection of mismatched coordinate frames, stale label fingerprints and attempts
   to apply orientation twice. A failed single-dataset conversion leaves no output.
4. Circular crop center and tilted plane-normal rotation.
5. Preservation of PLY face properties, including labels and per-face UVs; recovery
   of the saved texture V convention when it differs from the default.
6. Applying one configuration to separate calibration/agreement datasets under a
   common parent, verifying eight rater files per output subset.

An additional end-to-end adapter check used the actual vertices/faces/RGB from a
saved reviewed ROI in `BCG1T0-1.zip`: 2,590 vertices and 4,540 faces. Four scanner
fixtures and synthetic rater labels were created around that real geometry; the
conversion verified eight label files. This does not constitute validation on the
dentists' actual annotations or on all scanner originals.

The full-model lookup/transform/render path was exercised on generated original
PLY fixtures. The three review panels were rendered and visually inspected using
the same drawing code as the GUI. Interactive Tk desktop controls and Windows BAT
execution could not be exercised in this headless Linux environment.

Actual scanner angles must be established on your PC using the full original scans,
then checked on multiple representative specimens/stages. No angles inferred from
the screenshot are embedded in the package.

Run all checks with:

```bat
python -m unittest discover -s tests -v
```
