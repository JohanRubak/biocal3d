# Final classification run archive

The final run supplied on 24 November 2026 is recorded in [final-run-2026-11-24.json](final-run-2026-11-24.json), with a per-file byte count and SHA-256 in [final-run-files.csv](final-run-files.csv). The 438,142,506-byte `run.zip` is kept outside Git; its archive SHA-256 is `f0a36cbe62d6efd2d6ab1c93a1604340f1c13f03eb99308970b633b42129eac1`.

## Supplied content

- Six trained attention/classification heads in `models/`: DINOv3 grayscale, native RGB and harmonized RGB; Concerto geometry, native RGB and harmonized RGB.
- `split.csv` and cohort records covering 512 scans and 48 specimens: 30 training, 6 validation and 12 test specimens. BCG3-1 is in test.
- `results_harmonized/index.html`, figures, confusion counts, metrics, predictions, loss history and selected qualitative examples.
- Saved spatial details and rendered views for selected examples.

The archive contains 106 files. Archive CRCs and all saved NumPy arrays were checked without enabling pickle; the cohort linkage, saved specimen partition, model metadata and prediction coverage were inspected. Each head records a 1,000-epoch maximum, patience 30 and a constant learning-rate schedule. The archive was not retrained or re-evaluated during handover.

## Open the supplied results

Extract `run.zip` into a separate archive folder, preserving its relative paths. Open `results_harmonized/index.html` to inspect the saved report. Read `test_metrics.csv` and `test_predictions.csv` for the stored quantitative results. No GPU or encoder download is needed to read these outputs.

Historical absolute workstation paths in the cohort records are preserved as provenance. Do not overwrite them merely to make an old run resume on another computer. The archive records the source ROI geometry hashes, but their correspondence to a freshly downloaded cleaned data snapshot has not been rechecked here.

## Reproduction limits

`features/<mode>/settings.json` and the frozen feature bags are absent, as are the frozen encoder checkpoints/checkouts and package exports. Obtain the pretrained encoders from the authors' [official sources](../../../docs/frozen-models.md); they do not need to be supplied as project-owned artifacts. The current report-generation code reads those settings and caches, so `--figures-only` is not a standalone reconstruction command for this archive. The original execution command and Git revision are also not established by the ZIP.

Use the saved HTML/CSV/PNG outputs directly. For new training, follow the [reproduction guide](../../../docs/reproduce.md) in a fresh run directory. Recover the original feature settings, encoder revisions/checkpoint hashes and environment records if exact replay of the supplied heads is required.

## Controlled archive location

The intended private dataset location is `artifacts/classification/final-run-2026-11-24/run.zip` under `JohanRubak/BioCal3D-Data`, with a matching `manifest.json`. The supplied ZIP is locally inspected; that external artifact upload is not yet confirmed. An authorized maintainer can publish it from the signed-in computer using the project upload helper and the supplied archive. Record the printed artifact revision here once the upload succeeds. This artifact is separate from the four 512-scan data variants and does not replace their pinned reference snapshot.

To publish the original supplied ZIP from an authorized, signed-in computer, run from the project root:

```bat
python tools\upload_dataset.py --repo-id JohanRubak/BioCal3D-Data --study-run-archive "C:\path\to\run.zip"
```

The helper requires the recorded archive size/SHA-256, verifies the remote ZIP and manifest, and checks that the existing scan files are unchanged. It does not insert run files into the four scan variants.

## Small records to retain instead of large feature caches

The computed feature bags are optional archive material and can be regenerated. Keep the six small `features/<mode>/settings.json` files from the original run when available: these record checkpoint SHA-256, upstream Git commit, package versions, sampling/render settings and source-code hashes. Package/environment exports are also small. Keep the supplied trained heads, split and final outputs. Recomputing features costs time/GPU resources and exact replay depends on matching these original settings; storing every cached feature array is not a handover requirement.
