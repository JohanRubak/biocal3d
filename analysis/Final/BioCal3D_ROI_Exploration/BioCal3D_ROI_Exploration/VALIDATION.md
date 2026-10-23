# Validation of version 1.1.0

Validated on Python 3.12; source grammar also checked for Python 3.11.

- All 16 automated tests pass with `python -m unittest discover -s tests -v`.
- Existing scientific tests cover plane removal, exact triangle-area-integrated
  RMS, texture-seam duplicates, colour area weighting/background exclusion,
  saved texture-map precedence, the CIEDE2000 reference pair, common physical
  scales and hole preservation, noninteger pixel/cell ratios, paired comparisons,
  specimen splits, stale map rejection, resume behaviour and unchanged inputs.
- New checks verify that default results follow the script location, the two
  chosen examples are not replaced with other specimens, and the focused RMS
  panel requires the specified 100 µm calculation scale.
- An end-to-end synthetic dataset with 64 scans, all four scanners, fresh/mature
  protocols, and specimens BCG6-4 and BCG3-1 completed with zero failures. The
  exact requested set of nine PNG/PDF figure pairs and the HTML report were
  generated. Both example titles identify their partial-cleaning scan IDs.
- The end-to-end run omitted --output and verified that the output was created
  beside the script, separate from the input dataset. Input-file SHA256 hashes
  were unchanged. Figure layouts were visually inspected, PNG images verified,
  and PDF header/end markers checked.

Synthetic figures are software checks, not biological findings. The supplied
512-scan results were produced by version 1.0.0; version 1.1.0 has not rerun the
full original meshes here. Run this update on the Approved ROIs on your PC to
generate full-resolution colour maps and native-mesh example panels. Measurement
definitions are unchanged; figure selection, layouts, default examples and the
default output location have been updated.
