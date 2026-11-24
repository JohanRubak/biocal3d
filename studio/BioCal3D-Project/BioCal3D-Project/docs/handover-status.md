# Release record and available materials

Use this record to distinguish the supplied, validated data from historical inputs still required for exact reproduction. The [setup guide](setup.md) and [dataset guide](datasets.md) are the starting points for a new workstation.

## Reference release

| Item | Reference |
| --- | --- |
| Code | [JohanRubak/BioCal3D-Project](https://github.com/JohanRubak/BioCal3D-Project), private |
| Data | [JohanRubak/BioCal3D-Data](https://huggingface.co/datasets/JohanRubak/BioCal3D-Data), private |
| Cleaned data revision | `92fdb7e462b426ccd3ed738e5ca5c1063ea5c093` |
| Data release verification date | 23 November 2026 |
| Data scope | 512 acquisitions, 48 physical specimens, four scanners, four matched dataset variants |
| Selected data files | 8,648 files, approximately 19.69 GiB; dataset README and inventory are additional root files |
| Network-drive copy | `O:\HE_BioCal-3D-2026\Data`, available to authorized institutional users |
| Access contact | [Katrin](https://www.au.dk/kkresse@dent.au.dk) |
| Sharing rule | Restricted to the BioCal3D project team for now |
| Approval contact | Katrin; approval is required before sharing outside the project team |

The upload verifier checked all selected paths and byte sizes and the recorded native/harmonized full-model source hashes. The cleanup verifier removed 366 obsolete remote paths and retained all 8,650 managed current files, including README and inventory. These checks were run on the project workstation and reported by the release tooling. Storage attributes remain at the repository root.

This verification establishes remote file-list/size consistency and full-model provenance. It does not establish an independent byte-for-byte check of every downloaded ROI/texture, numerical reproduction of all historical experiments, or platform-independent rendering. Run the audit and inspect representative scans when setting up a new computer. Studio's portable installer has been used successfully on Windows x64; native macOS/Linux launchers and offline binary packaging require platform-specific validation.

The pinned data revision remains the reproducibility reference even if the dataset's main branch receives later documentation-only updates. Record the actual GitHub commit and data revision used with each experiment; do not substitute an unspecified "latest" version for an archived run.

## Supplied research materials

| Material | Status and location | What it enables |
| --- | --- | --- |
| Saved initial ROI reference fits | Supplied on 24 November 2026; four original files in [preprocessing/references](../preprocessing/references/README.md), with SHA-256 manifest | Reuse each scanner's fitted `BCG1T0-1` circle, basis, crop radius and height offset during initial extraction |
| Final trained classification run | Supplied as `run.zip`; inspected file inventory and model metadata in [classification archives](../analysis/classification-study/archives/README.md) | Read the final saved reports/predictions and preserve all six trained task-specific heads, loss histories, split and example outputs |
| Original frozen DINOv3 / Concerto encoders | Obtain from the authors' [papers, official repositories and checkpoint sources](frozen-models.md) | Install the pretrained feature extractors; these are external models, not missing project-owned source files |

The reference files are saved fit parameters, not the original sequence of perimeter clicks. They are distinct from the later cross-scanner orientation correction. The preprocessing command can copy them into a fresh run using `--reference-root preprocessing/references`.

The final-run archive contains 106 files, six trained heads and the complete 512-scan partition (30/6/12 specimens). Archive CRCs, NumPy readability, split separation, model metadata and prediction coverage were checked. The original ZIP bytes and per-file hashes are recorded in its archive manifest; the large ZIP stays outside Git. Its publication to the private dataset's separate `artifacts/classification/` area is prepared but not yet confirmed.

## ROI preparation and approval workflow

After initial crop extraction, each ROI was inspected manually in BioCal3D Studio and corrected when necessary. The approved ROI packages supplied with the release are the saved outcomes of that manual curation stage.

There is no separate missing "builder/rebuild script" to recover for this step. To repeat the workflow for new data, generate initial crops, inspect and correct each ROI in Studio, save the approved packages, and then perform the required orientation/export steps. Manual decisions are represented by the approved data and available review metadata; rerunning initial extraction alone does not reproduce those decisions. Preserve the released approved packages as the reference input for existing analyses.

## Materials pending or needed for exact historical replay

| Material | Current state | Reproduction purpose |
| --- | --- | --- |
| Colour-harmonization training code, configuration and fitted model | Will be provided later | Refit or apply the original learned mapping; the supplied ROI exporter transfers colour from existing full models |
| Final manuscript and final figure/table archive | Will be provided later | Associate reported conclusions with authoritative documents and run versions |
| Original feature settings and environment/checkout records | Small metadata files not included in the supplied final-run ZIP | Retain `features/<mode>/settings.json`, checkpoint hashes, upstream commits and package versions to regenerate matching features |
| Computed frozen-feature caches | Optional; large arrays do not need to be archived | Regenerate from the reference data and upstream encoders using the saved settings; keep the supplied heads and final outputs |
| Saved cross-scanner orientation selections/corrections | Check `orientation_corrections_used.json` and `scanner_orientation.json` at each reoriented ROI dataset root; the separately saved reusable config defaults to `scanner_rotations_TRIOS5.json` | Preserve the exact later orientation choices; these may already be included in the data release |

## Project access and sharing

Access to the code, data and research artifacts is currently restricted to the BioCal3D project team. Contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) for access requests. Katrin is the approval contact for any sharing outside the project team. Use individual authorized accounts and keep credentials outside the repository.

Historical aggregate benchmark tables/figures included under `analysis/colour-harmonization` are contextual reference outputs. They are separate from new full-cohort measurements and are identified in that analysis guide. They do not replace final run artifacts or the manuscript archive.

## Add recovered materials

Add recovered source/configuration with a documented entry point and validation command. Store large data, fitted weights and complete run outputs in the controlled project archive or a suitable private data/model repository. Update this record with their location, immutable revision/hash, intended workflow and validation status. Materials can be added independently as they become available.
