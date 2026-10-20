BioCal3D ROI review v3.10 — loading, brush and navigation performance update

INSTALL INTO THE EXISTING APPLICATION
1. Close the annotator normally so its current annotation is saved.
2. Extract this ZIP into a separate folder.
3. Run apply_update.cmd and select the EXISTING app root containing
   start_labelling.bat and biocal3d_workflow.
4. Start your usual start_labelling.bat. The title now includes v3.10.
5. Open the same approved ROI snapshot or labeling-study subset and select
   your existing annotator name.

The installer backs up replaced code under code_backups and preserves the
existing environment, prepared data, shared ROI reviews and rater labels.
No preprocessing, rebuilding, ROI reapproval or study recreation is needed.
No extra packages are required in the existing supported labeling environment.

WHAT CAUSED THE PAUSES
The earlier version writes both full PLY files on brush autosaves, even though
NPZ is the resumable annotation. Its PLY library converts triangle lists row by
row in Python on reading/writing. Each save also rereads every annotation's
status. Approved ROIs additionally validate the immutable input and load a
versioned surface; the old program repeats those loads on every revisit.
These operations run on the UI thread. Large meshes and shared/network storage
increase the effect. The actual contribution on your PC still depends on its
meshes, storage and graphics performance.

CHANGES
- Vectorized standard binary PLY records and fixed-length triangle/UV reads.
  Geometry, face labels, source IDs, colors, UVs and ASCII export are preserved.
- Brush autosaves save the working labels NPZ only. They are debounced until
  two seconds after the latest stroke and also covered by the periodic timer.
  A new stroke cancels its pending idle save; there is no pile of callbacks.
- Save draft (Ctrl+S), label approval, leaving a sample, changing dataset/profile
  and normal closing flush the label-colored PLY and, when enabled, original-
  color PLY. The checkbox is now named Save PLY too. If the app crashes, the
  latest automatically saved NPZ remains the source for recovery; a PLY can
  lag until the next explicit save. Open the sample and use Ctrl+S to refresh it.
- Unchanged A/D navigation no longer rewrites annotations or PLY exports.
  Camera-only changes update metadata without rebuilding the viewing meshes.
- Up to three recent mesh inputs are cached, capped at 192 MiB of mesh data.
  File signatures and shared review versions are checked before reuse. Labels
  and completion are reloaded independently for each rater. New/changed geometry
  invalidates reuse and retains the existing correspondence/completion checks.
- Folder discovery defers sibling-format geometry comparisons until a sample
  opens. Conflicting histories still block use, and equivalence is verified
  before editing that sample.
- A local save refreshes its queue row instead of rereading the full dataset.
  Use the existing refresh arrow beside Sort by for a full status refresh after
  another user changes files. Current sample opens/saves always check its ROI.
- Brush overlays update only changed face colors, with one render per paint
  event instead of repeated full overlay allocations and renders.

TESTING AND MEASUREMENTS
See VALIDATION.txt and BENCHMARK_RESULTS.json. Results are synthetic local IO
measurements, not a measurement of your Windows PC or actual dataset. The
release keeps exact source geometry; it does not decimate or simplify meshes.
First opening still does validation, decompression, texture loading, normals
and picking setup. A large scan can therefore still take time on first open.
Large explicit exports can still pause briefly, particularly on network drives.

OPTIONAL LOCAL BENCHMARK
Run benchmark_performance.cmd. It creates temporary synthetic meshes, writes
performance_report.json beside the script, and never touches experimental data.
An optional comparison against old code can be run in Windows CMD:
  benchmark_performance.cmd --baseline "C:\path\to\old_app"
The old-app path must contain the old code, not the app already updated in place.

ROI verification, approved-snapshot rebuilding and G5 study-folder generation
remain available. See ROI_REVIEW_GUIDE.md and biocal3d_workflow/LABELING_STUDY.md.
