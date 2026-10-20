BioCal3D ROI review update v3.9

NEW: create_labeling_plan.bat (or .cmd) now creates portable G5 calibration
and agreement folders from a successfully rebuilt approved snapshot. Defaults:
2 specimens / 24 scans for calibration and 6 specimens / 72 scans for agreement
when all four scanners and three timepoints are present. Actual counts are printed.
Approved geometry and shared ROI approvals are copied; plaque labels start empty
under annotations/bruna and annotations/katrin. No dentist-specific launchers.
Existing studies are never overwritten. Use --output with a NEW folder name.
See biocal3d_workflow/LABELING_STUDY.md for the exact command and study sequence.

Start with ROI_REVIEW_GUIDE.md.

To update your current app: close it, run apply_update.cmd, and select the EXISTING app root containing start_labelling.bat and biocal3d_workflow. Code is backed up; the current Python environment is reused. Then start your usual start_labelling.bat and choose Verify / correct specimen ROI.

Do not replace your prepared dataset or rerun full preprocessing just to inspect the problem cases. Review corrections are saved separately. Use rebuild_reviewed.cmd after review to create a corrected training snapshot.

For a fresh installation, see README_previous_installation.txt.

UI matches the approved compact preview. See ROI_REVIEW_GUIDE.md for renamed controls and shortcuts.
The previously reported Windows/VTK shutdown popup has not been fixed in this UI update.

Fixes the VTP-versus-PLY approval lookup mismatch. Existing reviews are reused after geometry verification; see ROI_REVIEW_GUIDE.md.

Fixes rebuilding T0/T1/T2 timepoint labels. After an incomplete rebuild, choose a NEW empty output folder; keep the same reviewed source snapshot.
