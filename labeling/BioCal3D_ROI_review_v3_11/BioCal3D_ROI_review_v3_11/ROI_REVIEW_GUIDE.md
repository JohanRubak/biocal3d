# BioCal3D specimen surface verification — v3.11

## Original data folder lookup in v3.11

In Verify / correct ROI, click **Locate original data folder…** and select the
raw collection root such as **02-09-2026-Data collected**. The app finds and
verifies this sample's original model, remembers that root locally for the
opened dataset, and automatically uses it for subsequent samples and after
restarting on this PC. Use the corresponding local path on each PC. No ROI
reapproval or rebuild is required. If multiple candidates remain, use a narrower
root or the optional **Locate original file…** button. A missing/invalid root
falls back to local ROI viewing with an explanation. See README.txt.

## Performance changes in v3.10

See README.txt for installation and timing details. Automatic drafts save NPZ only;
Save draft, approval, navigation and close update PLY files. The geometry review
and portable G5 study workflow below remain the v3.9 workflow.

## Portable G5 study folders in v3.9

After a successful rebuild, use `create_labeling_plan.bat` in the app root.
It asks for the rebuilt snapshot and makes `G5_labeling_study` beside it.
For an existing older study, choose a new output using `--output`.
The same launcher name in `biocal3d_workflow` and the compatibility launcher
`prepare_pilot_study.bat` run the same folder builder. The dedicated labeling
environment is reused. The complete instructions are in
`biocal3d_workflow/LABELING_STUDY.md`.

Both subsets retain approved, frozen ROI geometry and local scanner textures.
Plaque ratings start empty for bruna and katrin. Labels and approval histories
in the source snapshot remain untouched. Original full scanner models are
not included in the study folders; the ROI inputs are sufficient for labeling.

This update adds shared ROI review to the existing annotator and uses the same face-selection algorithm in new preprocessing runs. The existing scanner-colour display, label classes, rater selection, brush controls, reset-label action and coloured PLY exports remain available.

## Changes in v3.8

- Fixes the rebuild error **invalid literal for int() with base 10: 'T0'**. Prepare-all uses a string timepoint label (**T0/T1/T2**) and a separate numeric **time_idx**. The rebuild preserves those fields in their proper types instead of converting the label to an integer.
- Group, time index and sample number are normalized from CSV text; legacy numeric timepoints are accepted and normalized to T-labels. Conflicting label/index fields are rejected before creating output.
- Adds an integration check using the actual prepare-all metadata schema for both vertex-RGB and UV-textured samples at T0, T1 and T2, followed by VTP approvals/manual correction, a verified rebuild, and another rebuild of that verified snapshot. Geometry corrections, saved labels and approvals survive both exports.

After this failure, install the update and run rebuild_reviewed.cmd with the
same source snapshot and a **new empty output folder**, for example
**Approved ROIs_v2**. The previous failed output contains INCOMPLETE_REBUILD.json
and must not be used for training. Do not repeat preprocessing or ROI approval.

## Changes in v3.7

- Fixes existing approvals stored for **roi.vtp** being missed when the rebuild manifest names **roi.ply**. The app and rebuild locate the existing sibling-format review only after checking the original vertices, triangles and available source-face IDs; the approved review supplies the retained-face mask. Existing review records, corrections and labels remain in place and are used directly; no reapproval is required for this mismatch.
- New folder discovery prefers **roi.ply**, matching preparation manifests. If a valid existing VTP review exists, the app opens that source instead and uses its original labels/history.
- Different geometry, different source IDs or multiple saved sibling review histories block automatic reuse. Nothing is silently selected or overwritten when the review history is ambiguous.
- Completing a queue checks the full scan manifest before claiming the entire dataset is approved. Filtered/inventory-only queues report any exported samples still needing review.
- Rebuild preflight lists all pending/flagged exported ROIs and prints the annotation output folder being read. It reports when an existing VTP review is used for a PLY manifest entry. The rebuild audit records that source.
- Both green approval buttons now brighten on hover.

### Recover the failed rebuild

1. Close the app and apply this code update to the existing installation. The existing folder name (for example v3_4) can remain unchanged after an in-place update.
2. Run **rebuild_reviewed.cmd** again with the same prepared snapshot and an empty/new corrected output folder. A preflight-only failure created no output, so that proposed folder can be reused if still empty.
3. Existing compatible approvals/corrections under **annotations/_roi_review/.../roi.vtp** are used automatically. Do not rename those folders or use --allow-pending to work around this filename mismatch.
4. If the app used a custom annotation output directory, pass it as the third argument to rebuild_reviewed.cmd, or use the Python script's **--annotations** option.

For a read-only full-manifest status check from the app folder:

```bat
.labeling-env\python.exe biocal3d_workflow\rebuild_reviewed_snapshot.py --root "C:\path\to\prepared_snapshot" --check-only
```

## Changes in v3.6

- Compact review toolbar: **Verify / correct ROI…**, **Next unapproved**, **Approve ROI & Next [Ctrl+Enter]**, followed by separate **ROI** and **Label** status badges.
- **Label: pending** becomes **Label: approved** after approving plaque labels, and returns to pending when labels change. This status is specific to the selected annotator and independent of ROI approval.
- **Remaining → clean** replaces the longer remainder button name. It still changes only unreviewed included faces. **Approve Label & Next [N]** replaces Save & Next, with the same completion checks.
- The face-count/area summary, orientation control row, Choose texture, Export mesh and Export all buttons have been removed. Normal label/PLY saves continue. Missing scanner colors/textures must be fixed in the prepared data.
- Main plaque painting always uses the top-only guard. **R** or **Reset view [R]** returns to the top. The ROI correction window retains its own surface-correction interactions.
- The Reload ROI button has been removed. **F5** reloads the latest shared geometry while preserving labels on verified matching source faces; a geometry revision resets label completion.
- Undo/Redo/Reset and the two bulk-label actions share one row. The native Qt preview at 1380×880 gains 180 logical pixels of viewer height compared with v3.5; controls were also checked at 1280×800 and 1600×1000.
- These are UI changes. You do not need to preprocess again or reapprove saved ROI reviews. Use the existing prepared data and annotation output paths.

Known limitation: the previously reported intermittent Windows/VTK shutdown
popup has not been fixed by this UI release. Windows native desktop rendering
was not available for verification. This release does not claim a shutdown fix.

## Changes in v3.5

- **Approve ROI & Next [Ctrl+Enter]** approves the exact currently displayed surface directly in the main window. It does not rerun extraction, change the crop, or complete plaque labeling. Existing plaque labels, notes, annotation completion, camera and label Undo/Redo history are preserved; advancing loads the next sample normally.
- **Next ROI needing review** skips approved surfaces independently of plaque-label completion. The sample list shows ROI status, and progress counts ROI approvals separately from completed plaque annotations.
- Quick approval saves a normal shared review version, using the same review directory and concurrency checks as the correction window. Existing corrected/pending surfaces can be approved as displayed. Repeated approval of an already approved surface does not create another version.
- Review-window defaults: **Show excluded / original model** on, **Live extraction preview** on, and **Restore surface** selected. Opening retains the saved surface; editing extraction parameters triggers the live proposal. Disable live preview before adjusting parameters if you want to keep manual changes while moving the guides.
- Rebuilding still requires explicit approval of every exported scan. Geometry review never approves unseen samples in bulk.

## Fast verification workflow

1. Open the newly preprocessed snapshot with the main annotator's **Open dataset folder**. Do not restrict the queue to the plaque-labeling inventory when checking all exported scans.
2. Inspect the displayed ROI. If it is correct, click **Approve ROI & Next**, or press **Ctrl+Enter** while the viewport/buttons have focus. The next unapproved ROI opens automatically, including pending/flagged samples. Already approved ROIs are skipped.
3. For a sample needing correction, use **Verify / correct specimen ROI…**. Full original context, live preview and Restore surface are selected by default. Review/correct the crop and click **Approve shared surface**, then use **Next ROI needing review** in the main window.
4. Confirm all exported scans are approved. A filtered queue's progress refers only to that queue; the rebuild checks the complete scan manifest.
5. Run **rebuild_reviewed.cmd**, choose the reviewed snapshot and a separate new verified output folder, then use that verified output for labeling and investigations. There is no need to rerun raw preprocessing after installing v3.5.

## Changes in v3.4

- Automatic selection now recognizes shared geometric edges across texture seams. Scanner exports may duplicate the same physical vertex for different UV coordinates; these duplicates no longer split the specimen into artificial components. Exact coordinate matching is used only for analysis, preserving original topology, source IDs, colors and UV seams.
- Steep detail fully enclosed by the selected surface is retained. Exposed steep walls, detached surfaces and triangles outside the circle or height limits remain excluded. This preserves actual surface detail without creating or flattening triangles.
- The opening message compares the saved/imported surface with the proposal at the displayed settings, before you apply. Older preprocessing used wider height limits, while new reviews default to −0.5/+0.5 mm; manual corrections can also differ from the automatic proposal. Such a difference is now explicit, and the imported selection remains visible until Apply.
- Preview counts distinguish height exclusions, exposed steep faces and disconnected faces, and report preserved enclosed detail. Parameter displays use six decimals to avoid unnecessary rounding.
- Repeated Apply at unchanged settings is deterministic. Undo restores the opened crop or manual corrections.

## Changes in v3.3

Showing/hiding the original model preserves camera position, direction and zoom. Use **Fit visible model** (F) explicitly to reframe it.

ROI review shortcuts:

- **Ctrl+Z:** undo the last correction or applied extraction change.
- **Ctrl+Y** or **Ctrl+Shift+Z:** redo.
- **F:** fit the visible model.
- **R:** return to the top view.

Undo/redo restores both the surface selection and its extraction parameters. A pending live preview is resolved before Undo so it cannot overwrite the restored state. Shortcuts work from the viewport and numeric controls. Notes retain their own text Undo/Redo; F/R remain normal text while typing in editors.

## Changes in v3.2

**Entire ROI → plaque** labels every included ROI face as Plaque, replacing any Clean, Uncertain or Unreviewed labels in that ROI. Excluded geometry remains unchanged. It saves a draft, supports Undo/Redo, and is disabled for read-only annotations. The existing **Remaining → clean** button still changes only unreviewed faces.

## Changes in v3.1

- New preprocessing and new ROI reviews start with a -0.5 to +0.5 mm height interval. Existing saved review settings remain unchanged. Imported crop geometry is preserved until you apply a new selection.

- The reviewer opens maximized, with minimize/maximize controls and a wider scrollable controls panel.
- Context visibility uses explicit displayed faces, rather than relying on VTK ghost-cell flags. The reviewer starts with the retained surface only. Turn on the context checkbox to reveal excluded geometry; the camera remains fixed; use Fit visible model explicitly when needed.
- If the original source is unavailable and the current crop has no excluded faces, the context toggle is disabled with a clear local-only message. The program cannot display geometry absent from the crop without the original source and verified mapping.
- Depth and other live-preview changes show face counts after the height and slope filters. The lower/upper boundary planes move with their limits. If no faces remain, the preview becomes empty and saving is blocked, rather than showing the old selection. An unchanged count is reported explicitly; slope/connectivity filtering or missing source geometry can leave the retained surface unchanged.
- Space + left drag works after sliders, buttons, drop-downs and numeric input. Free-text notes keep normal typing spaces; with the pointer over the viewport, Space can switch directly to rotation.
- Original-model loading accepts equivalent triangle geometry despite vertex reindexing, UV seam splitting, corner permutations or verified face reordering. RGB/UV data is aligned to the preparation's source-face numbering; different or ambiguous geometry is still rejected.

**Do I need to run prepare-all first?** No. The package includes the updated `biocal3d_prepare_all.py` for future raw-data preparation. Open the existing prepared snapshot, review its surfaces, and use `rebuild_reviewed.cmd` afterwards to regenerate corrected training inputs.

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
5. Adjust circle centre/radius, plane height/tilt, lower/upper height limits and boundary slope limit. Click **Apply automatic selection**, or enable **Live extraction preview**. Changing parameters with live preview off changes the guides until you apply the selection. Applying automatic selection replaces manual edits; Undo restores them.
6. Rotate and inspect from several directions. Use **Exclude surface** to brush away side walls or holder remnants; use **Restore surface** for valid specimen surface. With the original model hidden, excluded faces cannot be picked for restoration; show it again first. Brushes affect visible faces only. Disable **Brush enabled** to navigate normally, or hold Space + left drag to rotate. Undo/redo/reset operate on geometry selections.
7. Save as **pending**, **approved**, or **flagged**. A flagged case is retained for investigation and blocks final snapshot rebuilding. Cancel discards the current review edits. No smoothing, flattening, remeshing or hole filling is performed.
8. The annotator reloads the selected surface. Existing plaque labels transfer only through verified source-face identity. Newly restored faces are unreviewed, and completion resets for review. Removed faces are excluded from the analysed surface; they are not counted as clean.
9. Complete plaque annotation normally. **Approve Label & Next** now requires an approved specimen surface. Draft saves remain available before approval.

If someone changes a shared surface while you have it open, saving is blocked. Press **F5** to transfer your in-memory labels to the current surface. Old label files are archived when a different geometry revision is saved.

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
2. Use geometric slope to seed the surface and reject exposed steep walls; triangle winding does not affect the test.
3. Keep the broad central sheet connected through geometric shared edges, including exact duplicate texture-seam vertices. Merely touching at a vertex does not join sheets.
4. Restore steep regions only when every edge is manifold and surrounded by the retained sheet or the same steep region, with a connection to the retained sheet. Circle and height limits still apply to every corner. No new geometry is generated.
5. Save extraction settings/metrics and mark surface review pending.

The initial maximum slope is 75 degrees; the initial height interval is -0.5 to +0.5 mm. These are starting settings to inspect across scanners, not experimentally validated thresholds. Connectivity can remove legitimate disconnected islands, and an attached sloping holder can remain. Review is therefore still required. Use geometric evidence consistently across stages/scanners; do not tune cropping to improve model predictions or plaque coverage.

## Validation and current limits

See `VALIDATION.txt`. Tests cover synthetic side walls, detached holders, real surface relief, full-model identity checks, restoration, label transfer, texture seams, concurrent versions, preprocessing integration and snapshot rebuilding. One previously uploaded experimental ROI was also loaded successfully.

The two full original meshes behind the screenshots were not available in this workspace. Their actual extraction corrections and Windows OpenGL interaction must be checked on your PC. The update does not claim those particular scans have already been repaired.
