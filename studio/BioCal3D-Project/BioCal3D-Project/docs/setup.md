# Setup on a new computer

## Get the code

Download the GitHub repository ZIP and extract it, or clone its URL. Run the commands below from the project root containing environment.yml. The private repository is https://github.com/JohanRubak/BioCal3D-Project. Access is currently restricted to the project team; contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) to request it.

## Desktop application

On Windows 10/11 x64, double-click studio/Install_BioCal3D_Studio.cmd and open BioCal3D Studio from the Start menu. Setup installs its own portable Python and dependencies under the user's local application folder. Internet access is needed during installation.

See the [Studio guide](../studio/README.md) for full controls, logs, updating, and macOS/Linux launcher limitations.

## Research environment

Install Anaconda or Miniconda and open Anaconda Prompt in the project root.

```bat
conda env create -f environment.yml
conda activate biocal3d
python -m pip check
```

For an existing biocal3d environment, use:

```bat
conda env update -f environment.yml
conda activate biocal3d
python -m pip check
```

This environment supports data auditing, colour/geometry characterization and research utilities. Install preprocessing dependencies when rerunning the original extraction:

```bat
python -m pip install -r preprocessing\requirements.txt
```

DINOv3 and Concerto use separate environments described below.

## Download and validate the private data

The private dataset repository is [JohanRubak/BioCal3D-Data](https://huggingface.co/datasets/JohanRubak/BioCal3D-Data). Use the cleaned reference snapshot below. Obtain read access through [Katrin](https://www.au.dk/kkresse@dent.au.dk), sign in on this computer and download into a fresh directory.

```bat
hf auth login
hf auth whoami
hf download JohanRubak/BioCal3D-Data --repo-type dataset --revision 92fdb7e462b426ccd3ed738e5ca5c1063ea5c093 --local-dir "D:\BioCal3D\Data"
set "BIOCAL3D_DATA_ROOT=D:\BioCal3D\Data"
python tools\audit_datasets.py --output "%USERPROFILE%\Downloads\biocal3d_download_audit.csv"
```

These commands use Anaconda Prompt syntax. For PowerShell, set the same variable with `$env:BIOCAL3D_DATA_ROOT = 'D:\BioCal3D\Data'`. On the project workstation the shared release location is `O:\HE_BioCal-3D-2026\Data`.

Expect 512 scans and zero audit issues. Use a new output CSV filename on each audit. The command pins the cleaned handover data snapshot. For another archived experiment, replace that revision with the one recorded by the experiment. See the [dataset guide](datasets.md) for release contents and Studio folder selection.

Open approved_rois_reoriented_harmonized in Studio. Inspect BCG3T1-1 across all scanners and check its full model in Review. Select full_harmonized for harmonized colour. raw_native must be alongside it for source-triangle verification.

## Encoder environments and checkpoints

Obtain the frozen pretrained encoders from the authors' [official papers, GitHub repositories and checkpoint instructions](frozen-models.md). The BioCal3D heads in the final-run archive are separate task-specific models.

Use the commands from analysis/classification-study:

```bat
cd analysis\classification-study
conda create -n biocal3d-dinov3 python=3.11 -y
conda run -n biocal3d-dinov3 python setup_model_env.py dinov3 --checkpoint "D:\BioCal3D\models\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
conda create -n biocal3d-concerto python=3.10 -y
conda run -n biocal3d-concerto python setup_model_env.py concerto
```

Obtain the official DINOv3 checkpoint through Meta's official DINOv3 access process before running its setup command. Concerto setup downloads its official checkpoint and verifies the pinned checksum. The setup scripts install the encoder-specific PyTorch/CUDA combinations. GPU extraction needs a supported NVIDIA GPU and driver. Desktop Studio and quantitative characterization use their own environments and can be set up separately.

See [classification instructions](../analysis/classification-study/README.md) for checkpoint paths, smoke runs, and feature caching.
