# BioCal3D Project

Software and research workflows for studying dental biofilm colour and geometry in intraoral scanner data. The study contains 512 acquisitions from 48 physical specimens scanned with iTERO, LABscanner, TRIOS3 and TRIOS5.

For a new project member, start with [installation and environment setup](docs/setup.md), then [download and validate the data](docs/datasets.md). Use [the continuation guide](docs/continuing-the-project.md) to choose a workflow and [the reproduction guide](docs/reproduce.md) to run the analyses.

## Code and data

| Location | What it contains |
| --- | --- |
| [GitHub: BioCal3D-Project](https://github.com/JohanRubak/BioCal3D-Project) | Studio source and installer, preprocessing, analysis code, environment definitions and documentation |
| [Private Hugging Face dataset: BioCal3D-Data](https://huggingface.co/datasets/JohanRubak/BioCal3D-Data) | Full models, approved ROI packages, textures and provenance metadata |
| `O:\HE_BioCal-3D-2026\Data` | Project network-drive copy for researchers with access to that drive |

Access to the code, data and research artifacts is currently restricted to the BioCal3D project team. Contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) for access requests. Katrin must approve any sharing outside the project team. Sign in with your own authorized accounts; credentials are not included in the project.

## Start on a new computer

For the Windows desktop application, run `studio/Install_BioCal3D_Studio.cmd` and open BioCal3D Studio from the Start menu. The installer supplies its own Python runtime. See the [Studio guide](studio/README.md) for controls and platform requirements.

For research commands, install Anaconda or Miniconda, open Anaconda Prompt in this repository's root, and run:

```bat
conda env create -f environment.yml
conda activate biocal3d
hf auth login
hf download JohanRubak/BioCal3D-Data --repo-type dataset --revision 92fdb7e462b426ccd3ed738e5ca5c1063ea5c093 --local-dir "D:\BioCal3D\Data"
set "BIOCAL3D_DATA_ROOT=D:\BioCal3D\Data"
python tools\audit_datasets.py --output "%USERPROFILE%\Downloads\biocal3d_download_audit.csv"
```

Choose a fresh download folder and output CSV filename. The audit should report 512 scans and zero issues. The pinned revision is the cleaned handover data snapshot; later documentation-only commits can exist on the dataset's main branch. Setup details, GPU encoder environments and troubleshooting are in [docs/setup.md](docs/setup.md).

## Configure the data folder

`BIOCAL3D_DATA_ROOT` points to the directory containing all four variants. Set it in each terminal used for research commands; explicit command-line paths override it. Without the variable, commands default to the repository's `data/` directory. Studio selects datasets through its file dialog.

| Folder | Content |
| --- | --- |
| `raw_native` | Native full-model meshes and required textures |
| `full_harmonized` | Exact full-model exports with harmonized colour |
| `approved_rois_reoriented` | Approved reoriented ROI geometry with native scanner colour |
| `approved_rois_reoriented_harmonized` | Matching approved ROI geometry with harmonized colour |

All variants begin with `SCANNER/G#`. Full-model PLY names use the scan ID, for example `BCG3T1-1.ply`. ROI packages use `SCANNER/G#/BCG#T#-#/`, with `geometry.npz`, `roi.ply` and other available companions inside. Acquisition subdirectories and required texture names are retained. Keep the four variants together for reliable source and texture lookup.

The [dataset guide](docs/datasets.md) explains Studio folder selection and validation. The [data dictionary](docs/data-dictionary.md) lists expected files, array fields and provenance.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Inspect scans; orient, review or label geometry | [BioCal3D Studio](studio/README.md) |
| Generate initial unedited ROI crops from native full models | [Preprocessing](preprocessing/README.md) |
| Characterize colour/geometry and evaluate harmonization | [Colour and geometry analysis](analysis/colour-harmonization/README.md) |
| Run DINOv3/Concerto classification, attention maps and candidate masks | [Classification study](analysis/classification-study/README.md) |
| Reproduce the current analysis workflows | [Reproduction guide](docs/reproduce.md) |
| Identify missing historical inputs or run artifacts | [Release record and remaining materials](docs/handover-status.md) |

## Study design and interpretation

Groups 1–2 contain 12-hour biofilm, scanned at baseline (T0) and after cleaning (T1). Groups 3–6 contain 24-hour biofilm, scanned at baseline (T0), after partial cleaning (T1) and after final cleaning (T2). Each group has eight physical specimens. `BCG3T1-1` means group 3, timepoint 1, specimen 1; the scanner completes the acquisition identity.

See the authors' [official pretrained-model papers and repositories](docs/frozen-models.md) for the frozen encoders; the [archived run guide](analysis/classification-study/archives/README.md) describes the supplied BioCal3D heads and outputs.

The included classification protocol assigns 30 specimens to training, 6 to validation and 12 to test. Every scanner and timepoint from one specimen stays in the same partition. BCG3-1 is fixed in test. The DINOv3 and Concerto encoders are frozen; the full-study defaults are 1,000 epochs and early-stopping patience 30.

Approved ROIs describe approved crop geometry, not validated plaque segmentation ground truth. Attention-derived masks remain candidate annotations. Colour harmonization was fitted using the study dataset; its performance on unseen acquisitions has not been established.

## Release and reproducibility

The cleaned data snapshot is `92fdb7e462b426ccd3ed738e5ca5c1063ea5c093`. Release tooling verified 8,648 selected data files, their byte sizes, and native/harmonized full-model source hashes; obsolete remote paths were removed. A fresh downloaded-copy audit and visual check are still required when setting up a new workstation.

Saved scanner crop-reference fits and the final six-head classification run have been supplied. ROI approval was performed by manually inspecting and, when needed, correcting each crop in Studio. Exact numerical replay still depends on matching feature/environment records. The harmonization training code and final manuscript archive will be added later. Their availability is documented in the [release record](docs/handover-status.md). Record the GitHub commit, data revision, environment, specimen split and command with every new experiment.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing geometry, source correspondence, data splits or analysis settings. Store data, weights and complete experiment runs outside Git; keep only the source, configuration and documentation needed to use them here.
