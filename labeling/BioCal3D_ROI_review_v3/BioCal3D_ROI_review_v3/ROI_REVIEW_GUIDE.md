# BioCal3D specimen surface verification — v3

This update adds shared ROI review to the existing annotator and uses the same face-selection algorithm in new preprocessing runs. The existing scanner-colour display, label classes, rater selection, brush controls, reset-label action and coloured PLY exports remain available.

## Install into your current app

1. Close the annotator. Extract this ZIP into a separate folder.
2. Run `apply_update.cmd`. Paste the path to your **existing app root**, containing `start_labelling.bat` and `biocal3d_workflow`. You can also drag that existing app folder onto the CMD file.
3. The installer backs up replaced code under `code_backups` and leaves your existing Python environment, data and labels in place. It rolls code back if copying fails. No new packages are required by the update in the existing supported labeling environment.
4. Start your usual `start_labelling.bat`, open the prepared snapshot you have been using, and select a sample.

Do not run the setup script again merely to apply this update. The full bundle is also included for a fresh installation; use its existing environment setup workflow if you need a separate installation.

## First use: review the two problem cases

1. Click **Verify / correct specimen ROI…**. This saves your current plaque draft first. The review window edits surface selection, not plaque labels.
2. Enable **Show excluded / original model** and **Show circle, plane and height limits**. Green indicates retained faces; red indicates excluded geometry. The yellow circle and translucent plane define the ROI; the blue circles mark the lower/upper height limits. Use **Side view** to inspect walls and holder remnants.
3. If `geometry.npz` beside the ROI identifies the original scanner mesh and transform, the program loads the complete original model. It verifies source triangles against the existing ROI before permitting restoration. If the raw path moved, use **Locate original model…**. The original metadata checksum is checked when available.
4. If the full source or exact mapping is unavailable, the window explicitly says **Local ROI only**. You can remove artifacts within that ROI; geometry outside that crop cannot be restored. Keep `geometry.npz` and `metadata.json` beside prepared `roi.ply`/`roi.vtp` to retain provenance.
5. Adjust circle centre/radius, plane height/tilt, lower/upper height limits and maximum slope. Click **Apply automatic selection**, or enable **Live extraction preview**. Changing parameters with live preview off changes the guides until you apply the selection. Applying automatic selection replaces manual edits; Undo restores them.
6. Rotate and inspect from several directions. Use **Exclude surface** to brush away side walls or holder remnants; use **Restore surface** for valid specimen surface. With the original model hidden, excluded faces cannot be picked for restoration; show it again first. Brushes affect visible faces only. Disable **Brush enabled** to navigate normally, or hold Space + left drag to rotate. Undo/redo/reset operate on geometry selections.
7. Save as **pending**, **approved**, or **flagged**. A flagged case is retained for investigation and blocks final snapshot rebuilding. Cancel discards the current review edits. No smoothing, flattening, remeshing or hole filling is performed.
8. The annotator reloads the selected surface. Existing plaque labels transfer only through verified source-face identity. Newly restored faces are unreviewed, and completion resets for review. Removed faces are excluded from the analysed surface; they are not counted as clean.
9. Complete plaque annotation normally. **Save & Next** now requires an approved specimen surface. Draft saves remain available before approval.

If someone changes a shared surface while you have it open, saving is blocked. Click **Reload shared ROI** to transfer your in-memory labels to the current surface. Old label files are archived when a different geometry revision is saved.

## Shared geometry and independent raters

All raters must open the same dataset root and use the same annotation output root. Shared geometry revisions are stored once under `annotations/_roi_review`; plaque labels remain under each rater's folder. Different output roots create independent review records.

Have one designated geometry reviewer approve/freeze surfaces before the dentists start the independent agreement exercise. Any later geometry correction resets completion for affected annotations; each rater must reopen, inspect and save their labels. Never count agreement on mismatched geometry versions. The agreement/export utilities in this update read shared reviewed surfaces.

If labeling folders are copies of your prepared data, review/rebuild the central snapshot first and distribute prepared surfaces from that corrected snapshot. This update does not automatically synchronize independently copied datasets.

## Inspect the whole dataset

Run `audit_geometry.cmd` from the installed app, paste the prepared snapshot path, and open the resulting `geometry_audit.html`. The CSV reports review status, source-context availability, steep-face area and differences between the existing crop and the automatic proposal. Top and side images highlight suggested exclusions in red. The audit does not modify the dataset or approve a surface.

For a small trial from an activated environment:

```bat
python biocal3d_workflow\audit_roi_geometry.py --root "C:\path\to\prepared_snapshot" --output "C:\path\to\geometry_audit_trial" --limit 20
```

## Make the corrections available to training

The original prepared snapshot remains immutable. The annotator uses reviewed overrides; existing model scripts still reading the old `roi.vtp`/`geometry.npz` do not automatically see these changes.

After approving the surfaces, run `rebuild_reviewed.cmd`. Supply the old snapshot and a separate, new sibling output folder. The command:

- regenerates `roi.vtp`, `roi.ply`, `geometry.npz`, RGB images, valid masks, projection maps and QC figures;
- preserves existing sample identities and split fields, and copies `specimen_splits.csv` unchanged;
- transfers matching saved plaque labels and records geometry review versions;
- produces new manifests and a complete-rebuild record, without copying old model caches or training runs.

Default rebuilding requires every scan in the scan manifest to be approved. Flagged scans always block it. An explicit `--allow-pending` command-line option is available for exploratory rebuilding, not for declaring a verified final dataset. If a saved rater file is stale, reopen and save it on the reviewed surface before rebuilding.

Point training to the corrected snapshot and regenerate all feature/surface caches and raster-derived inputs. Keep the previous model results as exploratory results. Do not mix old and corrected geometry in a single comparison. The existing labeling plan may reference the old dataset fingerprint; regenerate the plan against the corrected snapshot before distributing a new agreement study.

## Future preprocessing runs

The updated `biocal3d_prepare_all.py` uses a shared face-selection function:

1. Keep triangles whose vertices lie inside the reference circle and configured height interval.
2. Reject near-vertical faces using a geometric slope threshold; triangle winding does not affect the test.
3. Keep the broad central sheet connected through shared edges. Merely touching at a vertex does not join sheets.
4. Save extraction settings/metrics and mark surface review pending.

The initial maximum slope is 75 degrees; the existing +/-1.5 mm height interval is retained. These are starting settings to inspect across scanners, not experimentally validated thresholds. Connectivity can remove legitimate disconnected islands, and an attached sloping holder can remain. Review is therefore still required. Use geometric evidence consistently across stages/scanners; do not tune cropping to improve model predictions or plaque coverage.

## Validation and current limits

See `VALIDATION.txt`. Tests cover synthetic side walls, detached holders, real surface relief, full-model identity checks, restoration, label transfer, texture seams, concurrent versions, preprocessing integration and snapshot rebuilding. One previously uploaded experimental ROI was also loaded successfully.

The two full original meshes behind the screenshots were not available in this workspace. Their actual extraction corrections and Windows OpenGL interaction must be checked on your PC. The update does not claim those particular scans have already been repaired.
