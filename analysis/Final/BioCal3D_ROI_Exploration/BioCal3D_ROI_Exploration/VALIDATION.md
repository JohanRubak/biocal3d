# Validation of version 1.0.0

Validated on Python 3.12; source grammar also checked for Python 3.11.

- 13 automated tests pass with `python -m unittest discover -s tests -v`.
- Tests cover plane removal, exact area-integrated RMS, UV-seam duplicates,
  uniform-area sampling, colour weighting/background exclusion, saved texture
  map precedence, CIEDE2000 reference values, common physical scales and hole
  preservation, noninteger pixel/cell ratios, paired stage/scanner comparisons,
  specimen splits, source-map mismatch detection, resume behaviour and source
  data remaining unchanged.
- End-to-end synthetic dataset: 40 scans across four scanners, fresh/mature
  protocols and two specimens per cohort. A training-only run selected 20 scans
  and completed with zero failures, generating 19 PNG/PDF figure pairs and the
  HTML report. Figure layouts were visually inspected and all export files
  checked for integrity. Synthetic patterns are not scientific results.
- Compatibility check: one previously saved reviewed iTERO mesh was adapted to
  the prepared-map schema using the existing rasterizer. The program recovered
  2,590 encoded vertices, 4,540 faces and 110.50191159816526 mm² surface area,
  matching its saved review record. This validates mesh/schema compatibility,
  not colour fidelity of a newly generated adapter projection.

The entire 512-scan Approved ROIs dataset was not available here. Its full
analysis and results are generated when the program runs on the user's PC.
Review processing failures and paired specimen counts before using results.
