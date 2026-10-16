BioCal3D ROI review update v3.7

Start with ROI_REVIEW_GUIDE.md.

To update your current app: close it, run apply_update.cmd, and select the EXISTING app root containing start_labelling.bat and biocal3d_workflow. Code is backed up; the current Python environment is reused. Then start your usual start_labelling.bat and choose Verify / correct specimen ROI.

Do not replace your prepared dataset or rerun full preprocessing just to inspect the problem cases. Review corrections are saved separately. Use rebuild_reviewed.cmd after review to create a corrected training snapshot.

For a fresh installation, see README_previous_installation.txt.

UI matches the approved compact preview. See ROI_REVIEW_GUIDE.md for renamed controls and shortcuts.
The previously reported Windows/VTK shutdown popup has not been fixed in this UI update.

Fixes the VTP-versus-PLY approval lookup mismatch. Existing reviews are reused after geometry verification; see ROI_REVIEW_GUIDE.md.
