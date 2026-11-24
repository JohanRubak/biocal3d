# BioCal3D data dictionary

This document specifies the handover release used by the current code. It describes supported file schemas and expected content, rather than claiming every optional field is present in every historical export. Use `dataset_inventory.csv` and the supplied dataset indexes to identify the actual files. The cleaned reference snapshot `92fdb7e462b426ccd3ed738e5ca5c1063ea5c093` contains 8,648 selected data files, approximately 19.69 GiB. File counts describe this revision; metadata-only updates can change the repository revision without changing the data.

## Study identities and counts

| Item | Expected content |
| --- | --- |
| Scanners | `iTERO`, `LABscanner`, `TRIOS3`, `TRIOS5` |
| Groups | `G1`–`G6`, eight physical specimens per group: 48 specimens |
| Fresh biofilm, G1–G2 | T0 baseline and T1 clean, 12-hour biofilm |
| Mature biofilm, G3–G6 | T0 baseline, T1 partial, T2 clean, 24-hour biofilm |
| Unique acquisition | Scanner plus scan ID, e.g. `TRIOS5/BCG3T1-1` |
| Physical specimen | Group plus specimen number, e.g. `BCG3-1`; all its scanners/timepoints stay in the same experimental partition |
| Complete release | 512 scanner/scan identities in each of four variants: 128 per scanner; 192 baseline, 128 partial, 192 clean |

Folders identify scanner, group and acquisition, not independent train/test examples. An approved ROI is an approved geometry crop; it does not imply a dentist-approved plaque segmentation label. Preserve original limitations recorded in metadata and `layout_manifest.json` even when a group directory is renamed to a simple `G#`.

## Processing stages and folder roles

| Stage and location | Expected data | Purpose and relationship |
| --- | --- | --- |
| `raw_native/SCANNER/G#/` | Native full-model `.ply` meshes and referenced image textures; scanner acquisition subfolders may remain | Immutable source meshes in scanner coordinates. Saved native SHA-256 values verify the exact source used for an ROI. Full-model PLY basenames use the canonical scan ID; original basenames are recorded in the layout manifest. |
| `initial_rois/SCANNER/G#/BCG#T#-#/` when separately archived | First unedited `roi.vtp`, `geometry.npz`, QC previews, preprocessing manifest and scanner reference fits | Intermediate output of `preprocessing/preprocessing.py`, before manual geometry approval, cross-scanner reorientation and colour harmonization. Not one of the four currently uploaded variants. |
| Pre-reorientation approved snapshot, when separately archived | Approved ROI mesh/array packages and approval metadata | Input to reproducing the cross-scanner orientation step. This is a distinct historical input; the current reoriented exports do not substitute for it. |
| `approved_rois_reoriented/SCANNER/G#/BCG#T#-#/` | Approved ROI mesh/array packages with native colour and saved transforms | Main approved geometry snapshot. Native full models are aligned in memory using the ROI transform; a separate full-reoriented dataset is not currently required. |
| `full_harmonized/SCANNER/G#/` | Exact harmonized full-model exports, normally `.ply` with vertex RGB | Source of harmonized colour. Coordinate frame can depend on scanner/export batch; retain the exact exports used to generate the matched ROIs. |
| `approved_rois_reoriented_harmonized/SCANNER/G#/BCG#T#-#/` | Matched approved ROI geometry, harmonized vertex RGB, refreshed maps/previews and harmonized source provenance | Geometry and source correspondence match the native-colour approved ROIs. Colour changes; the ROI geometry and transform must stay identical. |

The historical initial-crop `geometry.npz` can contain registration/crop parameters alongside the `roi.vtp` surface; it should not be assumed to have the later approved-snapshot array schema described below.

The four released variants all start with `SCANNER/G#`. ROI packages then use the canonical scan ID as a directory. Full-model PLY files use names such as `BCG3T1-1.ply`, with scanner identity supplied by the parent folder. Existing acquisition subdirectories and texture-image basenames are retained. Original mesh names are recorded in `layout_manifest.json` for source and texture lookup. The inventory is the authoritative mapping between these different representations. Legacy `samples/SCANNER/G#/...` ROI paths are supported by readers; normalize them before publishing the release.

## Full-model file types

| File type | Information held | Requirement |
| --- | --- | --- |
| `.ply` | 3D vertex coordinates and triangle connectivity; colour may be vertex RGB or UV coordinates plus a `TextureFile` image reference | Native and harmonized full-model paths recorded in the current inventory. Preserve exact bytes and topology. |
| `.vtp` | VTK XML polygonal data: points, triangle connectivity, point/cell/field arrays, potentially RGB, UV and texture references | Supported mesh alternative when actually referenced by an export/inventory. |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` | Scanner surface texture pixels, accessed through UV coordinates or a recorded texture path | Required whenever the mesh or ROI metadata references that image. Companion image format varies by acquisition. |
| `.stl`, `.obj`, `.mtl` or other acquisition files | May exist in the original workstation archive; STL holds geometry without scanner RGB, OBJ/MTL may define another geometry/material package | Not automatically part of this release. The current selector is built around audited PLY/VTP models and supported textures; archive additional original files separately or extend and validate selection before adding them. |

A texture image is not a photograph providing independent biofilm ground truth. Renaming a file leaves its byte checksum and triangle numbering unchanged, but requires updated path references. Editing the mesh header or re-exporting geometry can change its checksum or triangle numbering. The supplied filename helper updates the inventory and preserves aliases, texture companions and mesh bytes.

## Files within each ROI package

| File | Information held | Expected status |
| --- | --- | --- |
| `geometry.npz` | Typed NumPy arrays for vertices, faces, colour, coordinate transforms and source correspondence/provenance | Required by the current full-release audit and classification pipeline for all 512 ROIs in each ROI variant. |
| `roi.ply` and/or `roi.vtp` | Interoperable approved surface mesh, colour/UV arrays and any exported mesh properties | Preserve available exports; not every historical package needs both formats. Their geometry and face order must match the package arrays. |
| `metadata.json` or historical `meta.json` | Per-scan provenance, approval/review status, colour source, source filenames/checksums, orientation information and companion-file hashes when recorded | Preserve available files. Required provenance must be available in metadata or the geometry archive, as expected by the audit. |
| `maps.npz` | Raster-to-surface mapping, rendered RGB, valid pixels, triangle IDs, barycentric weights and render geometry | Preserve when supplied. Required for workflows using saved pixel-to-triangle maps and for regenerating their harmonized previews. |
| `rgb.png` | Rendered ROI colour preview corresponding to the saved raster maps | Optional companion; preserve when present. This is a rendering, not an independent photo. |
| `valid_mask.png` | Raster footprint of valid ROI pixels, usually 255 within the ROI and 0 outside | Optional companion paired with raster maps. It is an ROI footprint, **not a plaque segmentation mask**. |
| Referenced texture images | Portable UV texture pixels or scanner colour source | Required if referenced by the ROI representation. The release selector includes declared textures and matching texture sidecars, rather than every image in the source folder. |

Geometry review history under `annotations/`, `versions/` and other working directories is excluded from the compact release. Effective approved geometry must already be exported into the supplied ROI packages. Keep any historically important annotations separately; do not assume these directories or dentist plaque labels are included.

## Arrays and provenance fields

`.npz` files are ZIP containers of named NumPy arrays; open them with `np.load(path, allow_pickle=False)`. Field availability varies by preparation/export version. The rows below document fields consumed or produced by the included tools.

| Geometry field | Information and interpretation |
| --- | --- |
| `vertices` | N×3 vertex coordinates, in millimetres for these prepared specimen meshes. Reoriented ROI coordinates are in their canonical frame. |
| `faces` | M×3 integer vertex indices defining triangles; indexing and ordering are part of source correspondence. |
| `vertex_rgb`, or legacy `rgb`/colour aliases | Per-vertex colour. Harmonized export writes floating RGB in [0,1]; native arrays can also use integer [0,255]. Check dtype/range before conversion. |
| `uv` or a supported texture-coordinate alias | Per-vertex image coordinates for textured meshes, when present. Harmonized vertex-colour export may remove UV. |
| `original_to_canonical` | 4×4 homogeneous transform mapping native scanner coordinates into the approved ROI frame. Required by the paired full-release geometry audit. |
| `canonical_to_original` | Inverse transform when stored; used to move between ROI and native coordinates. |
| `raw_vertex_id`, `raw_triangle_id` | Correspondence to native source mesh indices, when supplied. These connect the crop to full-model geometry and support geometry review/restoration; preserve their values and ordering. |
| `source_ply`, `source`, `source_sha256` | Native full-model provenance. The audit requires the recorded native SHA-256 and resolves the corresponding model under `raw_native`; it can read provenance from package metadata as well. |
| `harmonized_source_ply`, `harmonized_source_sha256` | Exact harmonized full-model source and checksum. Required provenance for the harmonized ROI release. |
| `source_texture`, `texture_path`, `texture_file` | Recorded colour image paths when a texture representation is used. Saved absolute paths can refer to the original workstation; the included readers rebase legacy paths onto the current release. |
| Additional preparation fields | Scanner, group, scan ID, crop radius, fitted/reference axes, registration transforms, fitness/error values and related quality measurements may be present. Retain them; do not infer fields absent from an older export. |

| Raster field in `maps.npz` | Information and interpretation |
| --- | --- |
| `rgb` | H×W×3 rendered RGB array, normalized to [0,1] in the included rasterizer. |
| `valid` | H×W Boolean mask identifying pixels with valid ROI surface coverage. |
| `triangle_id` | H×W map from a valid pixel to a triangle in the ROI `faces` array. Use only pixels marked `valid`. |
| `barycentric` | H×W×3 interpolation weights within each pixel's triangle; used to transfer vertex colour onto pixels. |
| `height_mm` | Per-pixel surface height/depth in the render frame; invalid pixels can be NaN. |
| `pixel_surface_area_mm2` | Surface area represented by valid pixels, used for surface-aware measurements. |
| `origin`, `basis`, `fov_mm` | Render-frame origin/axes and physical field of view. Preserve recorded raster settings; do not infer scale from PNG dimensions alone. |
| `vertices`, `faces`, recorded transforms | Geometry and coordinate relationships used by the map. Face order must agree with `geometry.npz`. |
| `overlap_fraction` | Rasterization overlap diagnostic when stored. |

## Dataset-level metadata

These files sit at the top of an ROI variant unless otherwise stated. Their presence depends on which preparation stages were archived; preserve the ones supplied rather than fabricating missing records.

| File | Information held and role |
| --- | --- |
| `Data/dataset_inventory.csv` | Required release-level table: `scanner`, `sample`, the four variant-relative paths, `raw_sha256`, `harmonized_sha256`, and `issues`. A complete clean audit contains 512 unique rows with empty issue fields. Normalization replaces old paths with the new ones. |
| `dataset_index.json` | Snapshot index with a `scans` list, per-scan `sample_dir`, identity, status, saved hashes and run configuration as recorded. Required by the current classification snapshot loader. |
| `scan_manifest.csv` | Tabular preparation/approval inventory, relative sample paths and recorded quality/status fields; also used by orientation and review workflows. |
| `annotation_inventory.csv` | Approved annotation mesh paths and associated identifiers/hashes, when exported. Its name alone does not establish that independent plaque ground truth is included. |
| `scanner_orientation.json`, `orientation_corrections_used.json` | Saved scanner rotations/transforms and related audit/configuration records, when supplied. |
| `geometry_rebuild.json`, `run_config.json` | Geometry preparation/rebuild settings and run provenance. |
| `harmonized_roi_audit.csv` | Export matching method and per-scan harmonized source correspondence, when produced by the harmonized ROI exporter. |
| `splits.json`, `split_assignments.csv`, `specimen_splits.csv`, `splits.csv` | Saved partition assignments when present. Use the matching archived split to reproduce an existing experiment; avoid new random splits for its reproduction. |
| `classes.txt`, `class_map.json`, `annotation_classes.json` | Saved class names/mappings when present; inspect their scope because acquisition-stage classes and annotation classes serve different tasks. |
| `layout_manifest.json` in each of the four variants | Added by normalization/renaming: original-to-canonical directory and full-model file mappings, including original group labels and limitations. Data provenance, not executable code. |
| Sibling `Data-layout-backup-<timestamp>/` | Previous indexes, the successful pre-migration audit and `rename_plan.json`. Filename-only changes use a sibling `Data-filenames-backup-<timestamp>` with the same rollback roles. Local rollback/archive material outside the upload root; not part of the dataset upload. |

## Additional material to reconcile later

| Material | Why retain it | Current release status |
| --- | --- | --- |
| First unedited ROI exports, `_references/`, `manifest.csv`, `qc.png` and `_contact_sheets/` | Reuse the original selected scanner reference circles and reproduce/check the first crop stage | The four scanner reference fits are supplied under `preprocessing/references` with hashes. Initial crop outputs, QC and manifests remain separate from the four verified variants. |
| Approved ROI snapshot before reorientation | Reproduce the historical orientation operation from its actual input | Separate archive input, if available. |
| Original colour-harmonization training code, fitted weights and configuration | Refit or apply the original learned colour mapping | The current project includes the supplied characterization and ROI colour-transfer code; this does not by itself establish that the original mapping training materials are complete. |
| Classification checkpoints, frozen feature caches, model weights and result tables/figures | Reproduce exact predictions and compare against the saved experiment | The supplied final run is inventoried under `analysis/classification-study/archives`: six trained heads, split, predictions and saved reports. The large ZIP is external to Git; feature settings/caches and complete original environment records are not included. |
| Approval/review history and independent dentist assessments | Document geometry decisions and any later verification of plaque masks | Separate controlled archive unless explicitly included and inventoried. |

## Excluded content and verification

GitHub holds source, installation guidance and documentation. The private Hugging Face dataset holds selected meshes, arrays, images and metadata. Credentials, access tokens, code, installers, Python environments and transfer caches do not belong in the dataset release. Working results and histories are not silently included.

Use [the dataset guide](datasets.md) to locate and audit the folders, then open `BCG3T1-1` from both ROI variants in Studio and check native and harmonized full-model views. The audit checks study identities, recorded source checksums, matching ROI geometry/correspondence and transforms. It does not replace visual review of scanner colours, texture orientation or scientific limitations. A relocated-path check reuses existing source hashes only because normalization leaves model/array bytes unchanged; a fresh audit of a downloaded release reads and checks the actual transferred sources again.
