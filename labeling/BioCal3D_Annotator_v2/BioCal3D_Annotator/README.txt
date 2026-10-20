BioCal3D ROI review v3.13 — loading visualization only when opening a dataset

INSTALL
Close the annotator, extract this ZIP separately, run apply_update.cmd and select
your existing app root containing start_labelling.bat and biocal3d_workflow.
Start your usual start_labelling.bat; its title includes v3.13. Your existing
labels, approved ROIs, original-data-folder setting and environment are retained.
No preprocessing, rebuild, ROI reapproval or new packages are needed.

WHAT YOU WILL SEE
Opening a dataset shows one loading window until the sample list and first
specimen are ready. It shows the current step and elapsed seconds. Dataset label
checks show actual counts; input-cache copying shows real transferred bytes.
Other stages use an activity bar. Dataset input reads run in a worker; VTK
preparation and drawing run on the interface thread.

Switching samples (A/D, sample list, next unfinished or next ROI), opening ROI
verification and locating original models show NO loading window. Input caching,
batched rendering and the earlier speed improvements remain enabled. The status
bar still reports measured sample load duration.

WHY O: IS SLOWER AND WHAT CHANGED
A shared drive has network and server latency for directory listings, file opens,
metadata checks and reading mesh/texture data. Even small files can be expensive
when opened repeatedly. Large meshes also need CPU/GPU preparation on either
storage location. A local SSD will generally remain faster. This update reduces
shared-drive traffic and repeated preparation without moving the authoritative
annotations or modifying the original inputs.

- Folder discovery prunes annotation histories, exports and caches BEFORE walking
  them. Metadata is taken from the scan manifest where available rather than
  opening hundreds of per-sample JSON files. The initial queue/status pass is
  performed once instead of being repeated for display setup.
- Meshes and scanner textures use a persistent local read-through cache, roughly
  capped at 2 GiB. Recently used input meshes, frame/centroid data, decoded textures
  and requested normals also reuse the bounded in-memory cache.
- A sample in None shading no longer calculates shading normals unnecessarily.
  Initial texture, camera and label setup are combined into one requested render.
- Saved labels and camera metadata are read together from their original NPZ.
  Shared annotation files, review pointers and locks are NOT copied into a local
  working dataset. Saves and approvals still use the dataset you opened.

CACHE BEHAVIOR
The first visit to an uncached sample still reads its files from O:. Later visits,
including after restarting the app, can reuse unchanged local input copies.
Original source file signatures are checked before reuse. Changed sources refresh
the cache; damaged local copies are recopied. A disconnected source is not silently
replaced by a stale local copy. Missing/invalid data is reported through the normal
load-error handling. If the local cache cannot be written, loading falls back to
the selected source. The local cache is a convenience, not a backup or offline mode.

On Windows, inputs are cached under:
  %LOCALAPPDATA%\BioCal3D\AnnotatorCache\v1
You may delete that cache with the app closed; no labels or approvals are stored
there. It will rebuild on demand. For troubleshooting, launch the annotator with
--no-local-cache, or choose a different local location using --cache-dir.
These are Python annotator arguments; the normal BAT launcher needs no changes.

TIMING DIAGNOSTICS
The app writes a small local loading_times.csv at:
  %LOCALAPPDATA%\BioCal3D\AnnotatorCache\loading_times.csv
It records dataset/sample paths, total sample-loading seconds and timed input
steps, without label values or notes. It rotates at about 1 MiB. If loading still
feels slow, this file distinguishes surface/annotation input time from remaining
view/render overhead. It does not measure or identify the precise O: server cause.

ORIGINAL DATA FOLDER LOOKUP FROM v3.11
In Verify / correct ROI, use Locate original data folder… and select the local or
shared 02-09-2026-Data collected root once on that PC. The app remembers that root
for the opened dataset and automatically locates subsequent original models,
including after restarting. Exact scanner/sample matching, available checksums
and source-triangle verification remain in place. Use Locate original file… for
an explicit individual override. See ROI_REVIEW_GUIDE.md for details.

v3.10 PERFORMANCE IMPROVEMENTS ARE INCLUDED
The earlier loading/painting/navigation changes remain in this version.

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
