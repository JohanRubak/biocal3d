# Use the BioCal3D datasets

The private [BioCal3D-Data repository](https://huggingface.co/datasets/JohanRubak/BioCal3D-Data) contains the prepared data for the current study. Use the released data for existing analyses; rebuilding crops or harmonization is a separate research workflow.

## Download a reproducible copy

Dataset access is currently restricted to the BioCal3D project team. Contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) for access requests or approval of any external sharing. After [setting up the research environment](setup.md), run these commands in Anaconda Prompt from the code repository root:

```bat
conda activate biocal3d
hf auth login
hf auth whoami
hf download JohanRubak/BioCal3D-Data --repo-type dataset --revision 92fdb7e462b426ccd3ed738e5ca5c1063ea5c093 --local-dir "D:\BioCal3D\Data"
set "BIOCAL3D_DATA_ROOT=D:\BioCal3D\Data"
python tools\audit_datasets.py --output "%USERPROFILE%\Downloads\biocal3d_download_audit.csv"
```

Replace the download directory with a suitable local drive. Use a fresh directory so files from other revisions are not mixed into the release. The cleaned handover snapshot contains approximately 19.69 GiB of selected data. Leave additional free space for download caches, derived outputs and model weights.

Hugging Face sign-in is specific to each computer. Download access needs read permission; ordinary project use does not require a write token. In PowerShell, configure the same data location with `$env:BIOCAL3D_DATA_ROOT = 'D:\BioCal3D\Data'`.

Researchers with network-drive access may instead use `O:\HE_BioCal-3D-2026\Data`. The drive letter is local to the institution; it is not required on another computer. Always point the data-root variable at the parent of the four variants, not at an individual variant.

## Expected release layout

| Folder | Contents | Selected files in the handover snapshot |
| --- | --- | --- |
| `raw_native` | Native full meshes and their required texture companions | 769 |
| `full_harmonized` | Exact harmonized full meshes used for ROI colour export | 513 |
| `approved_rois_reoriented` | Approved, reoriented ROI meshes, arrays, maps and metadata with native colour | 3,770 |
| `approved_rois_reoriented_harmonized` | Matched approved ROI geometry and harmonized colour | 3,596 |

These counts total 8,648 data files. The root also contains `dataset_inventory.csv`, the dataset README and storage attributes. Companion-file counts differ across variants because scanner colour representations and export formats differ; this does not imply missing scans.

| Representation | Example relative path |
| --- | --- |
| Native full model | `raw_native/TRIOS5/G3/BCG3T1-1.ply` |
| Native acquisition package | `raw_native/iTERO/G3/BCG3T1-1/BCG3T1-1.ply` |
| Harmonized full model | `full_harmonized/TRIOS5/G3/BCG3T1-1.ply` |
| Approved ROI | `approved_rois_reoriented/TRIOS5/G3/BCG3T1-1/geometry.npz` |
| Harmonized approved ROI | `approved_rois_reoriented_harmonized/TRIOS5/G3/BCG3T1-1/geometry.npz` |

Canonical scanner names are `iTERO`, `LABscanner`, `TRIOS3` and `TRIOS5`; groups are `G1`–`G6`. Full-model PLY filenames use the scan ID. ROI mesh exports retain `roi.ply`/`roi.vtp` within the scan-ID package. Texture image names and acquisition subdirectories are retained to preserve references. The inventory gives the exact path for each scanner/scan pair.

Each variant's `layout_manifest.json` records original directory labels and any renamed full-model paths. A canonical group name does not remove limitations recorded in the original labels. Saved absolute provenance paths can refer to the preparation workstation; the included readers resolve those references within the current release. See the [data dictionary](data-dictionary.md) for file schemas and field meanings.

## Validate a new download

The audit should find 512 scan identities and zero issues. It checks missing/ambiguous scans, recorded native and harmonized source hashes, ROI correspondence, paired geometry and coordinate transforms. Choose a new output CSV filename for each audit.

Then open `BCG3T1-1` in Studio from both ROI variants and inspect all four scanners. Check both native and harmonized full models, top-view alignment, and texture/colour appearance. File verification does not replace these visual checks.

Release tooling verified the upload and removed obsolete paths at data revision `92fdb7e462b426ccd3ed738e5ca5c1063ea5c093` on 23 November 2026. The [release record](handover-status.md) states what this validation does and does not establish. Later README updates can exist on the main branch without changing this pinned data snapshot.

## Select folders in Studio

| Task | ROI dataset | Full-model folder |
| --- | --- | --- |
| Inspect original scanner colour | `approved_rois_reoriented` | `raw_native` |
| Inspect harmonized colour | `approved_rois_reoriented_harmonized` | `full_harmonized` |
| Review/restore triangles with native colour | The approved ROI variant being reviewed | `raw_native` |
| Review/restore triangles with harmonized colour | `approved_rois_reoriented_harmonized` | `full_harmonized`, with `raw_native` alongside it for topology verification |
| Reproduce scanner orientation | A separately archived approved snapshot before reorientation | `raw_native` |

Use **Open dataset…** for an ROI variant and **Original models…** in Inspect or **Locate original data folder…** in Review for its full models. Keep all four variants under the shared data root.

Native full models remain in scanner coordinates. Studio applies the ROI's saved `original_to_canonical` transform in memory, so a separate reoriented full-model export is not required. Harmonized full models are checked against the recorded harmonized source checksum; their coordinate frame may depend on the original export batch. Review verifies native source topology before mapping harmonized colour onto it.

## Work without changing the reference release

Keep the downloaded release as the reference input. Write new geometry, annotations, analyses and model outputs to a working copy or a separate output directory. Existing filenames, hashes, face/vertex correspondence and transforms are part of provenance. Do not rerun folder migration or filename-normalization tools on the published release for ordinary analysis.

The [preprocessing guide](../preprocessing/README.md) explains initial unedited crops. The [reproduction guide](reproduce.md) explains the current analyses and harmonized ROI export. Additional historical inputs needed to recreate earlier preparation stages are listed in the [release record](handover-status.md).
