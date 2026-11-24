---
pretty_name: BioCal3D
language:
- en
tags:
- dentistry
- biofilm
- intraoral-scanning
- 3d
- color-harmonization
---

# BioCal3D dataset

Prepared dental biofilm scans for scanner colour/geometry characterization, cleaning-stage classification and development of plaque segmentation methods.

Use [BioCal3D Project on GitHub](https://github.com/JohanRubak/BioCal3D-Project) for software, installation, analysis commands and the detailed [data dictionary](https://github.com/JohanRubak/BioCal3D-Project/blob/main/docs/data-dictionary.md). This repository contains data and provenance metadata. Access to both private repositories is currently restricted to the BioCal3D project team; contact [Katrin](https://www.au.dk/kkresse@dent.au.dk).

## Download and get started

The cleaned reference data snapshot is **`92fdb7e462b426ccd3ed738e5ca5c1063ea5c093`**. After following the [software setup guide](https://github.com/JohanRubak/BioCal3D-Project/blob/main/docs/setup.md), sign in on the computer you will use and download into a fresh directory:

```bat
hf auth login
hf download JohanRubak/BioCal3D-Data --repo-type dataset --revision 92fdb7e462b426ccd3ed738e5ca5c1063ea5c093 --local-dir "D:\BioCal3D\Data"
set "BIOCAL3D_DATA_ROOT=D:\BioCal3D\Data"
```

Change the destination to suit your computer. Keep all four variants under the same parent directory. From the GitHub code repository root, with the `biocal3d` environment active, validate the download:

```bat
python tools\audit_datasets.py --output "%USERPROFILE%\Downloads\biocal3d_download_audit.csv"
```

Use a new output CSV filename. Expect **512 scan identities and zero audit issues**. Then inspect `BCG3T1-1` across the four scanners in Studio, checking native and harmonized colours and full-model correspondence. See the [dataset guide](https://github.com/JohanRubak/BioCal3D-Project/blob/main/docs/datasets.md) for folder selection and validation.

## Study design

| Item | Description |
| --- | --- |
| Physical specimens | 48, eight in each of six groups |
| Scanners | iTERO, LABscanner, TRIOS3 and TRIOS5 |
| G1–G2 | 12-hour biofilm; T0 baseline, T1 clean |
| G3–G6 | 24-hour biofilm; T0 baseline, T1 partial, T2 clean |
| Total | 512 scanner acquisitions; 128 per scanner |
| Acquisition identity | Scanner plus scan ID, for example `TRIOS5/BCG3T1-1` |

`BCG3T1-1` denotes group 3, timepoint 1 and specimen 1. The physical specimen identifier is `BCG3-1`; all its scanner and timepoint views must stay together when assigning training, validation and test partitions.

## Data folders and file types

The reference snapshot contains approximately **19.69 GiB** of selected data. Different companion-file counts reflect scanner colour representations and export formats.

| Folder | Content | Selected files |
| --- | --- | --- |
| `raw_native` | Native full-model PLY meshes and required scanner texture images | 769 |
| `full_harmonized` | Exact full-model exports with harmonized colour | 513 |
| `approved_rois_reoriented` | Approved ROI geometry, native colour, transforms and available raster maps/previews | 3,770 |
| `approved_rois_reoriented_harmonized` | Matching approved ROI geometry with harmonized colour and updated raster companions | 3,596 |

The four variants contain **8,648 data files**. `dataset_inventory.csv` at the root records all 512 scanner/scan identities, paths to the four variants, native/harmonized full-model source SHA-256 values and audit issues. Storage attributes at the root support binary data transfer.

All variants begin with `SCANNER/G#`. Full-model PLY filenames use the canonical scan ID, such as `BCG3T1-1.ply`; some scanners retain an acquisition subdirectory. ROI packages use `SCANNER/G#/BCG#T#-#/`, without a `samples` wrapper.

| File within an ROI package | Information held |
| --- | --- |
| `geometry.npz` | Vertices, triangles, colour, source correspondence and coordinate transforms |
| `roi.ply` / `roi.vtp`, when present | Interoperable approved surface exports |
| `metadata.json` or historical `meta.json` | Provenance, geometry-review status, colour source and recorded hashes |
| `maps.npz`, when present | Rendered RGB, valid pixels, pixel-to-triangle IDs, barycentric weights and render geometry |
| `rgb.png`, when present | Rendered ROI colour preview |
| `valid_mask.png`, when present | Valid ROI surface footprint; **not a plaque segmentation mask** |
| Referenced texture images | Scanner colour pixels used by UV-textured representations |

Each variant's `layout_manifest.json` records original folder labels and renamed full-model paths, including historical limitation labels. Preserve these manifests and texture companions. A simplified group directory name does not remove the original acquisition limitations. The [data dictionary](https://github.com/JohanRubak/BioCal3D-Project/blob/main/docs/data-dictionary.md) documents required versus optional files and individual array fields.

## Coordinate frames and Studio use

Native full models remain in scanner coordinates. Studio applies each ROI's saved `original_to_canonical` transform when displaying them. A separate reoriented full-model export is not required.

Open `approved_rois_reoriented` with `raw_native` for native-colour inspection, or `approved_rois_reoriented_harmonized` with `full_harmonized` for harmonized inspection. Keep `raw_native` available for source-topology verification during harmonized geometry review. Native and harmonized source checksums identify the exact full meshes used to prepare the ROI.

## Validation and interpretation

Release tooling verified all selected paths and byte sizes, native/harmonized full-model source hashes, and retention of the current release after removing obsolete filenames. The cleaned data snapshot was recorded on 23 November 2026. Later documentation-only commits can exist on the main branch; use the pinned reference revision when reproducing an experiment.

Run the dataset audit and visual checks after downloading. Upload metadata checks do not establish independent byte-for-byte validation of every downloaded ROI/texture or exact reproduction of historical numerical results.

Baseline, partial and clean are acquisition-stage labels. Geometry approval does not imply dentist-approved plaque ground truth. Model attention maps and candidate masks require independent spatial assessment before use as segmentation annotations or quantitative biofilm measurements.

The existing colour harmonization was fitted using the study data. Performance on unseen acquisitions has not been established. See the software [release record](https://github.com/JohanRubak/BioCal3D-Project/blob/main/docs/handover-status.md) for historical preparation inputs, fitted models and run artifacts not fully archived with this release.

## Access and reuse

Access and use are currently restricted to the BioCal3D project team. Contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) for access requests. Any sharing outside the project team requires Katrin's approval. Use your own authorized account. Keep the released data as reference input and save new annotations, derivatives and experiment outputs separately.
