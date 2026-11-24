"""Resume the BioCal3D raw-colour study on Johan's Windows research PC."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DINO_NAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
CONCERTO_NAME = "concerto_small.pth"
MODELS = (
    ("dinov3", DINO_NAME, "hubconf.py"),
    ("concerto", CONCERTO_NAME, "concerto"),
)
IGNORED_DIRS = {".git", "__pycache__", ".conda", "data", "runs", "node_modules"}


def paths_for_pc():
    home = Path.home()
    repo = home / "repos" / "biocal3d"
    classification = repo / "classification"
    installed = Path(__file__).resolve().parent
    if not (installed / "matched_study.py").is_file():
        installed = classification / "BioCal3D_Matched_Study" / "BioCal3D_Matched_Study"
    previous = classification / "BioCal3D_Stage_Ablations" / "BioCal3D_Stage_Ablations"
    roi = repo / "data" / "Approved ROIs_reoriented"
    return installed, previous, roi, (classification, home / "Downloads")


def find_conda():
    candidates = (os.environ.get("CONDA_EXE"), shutil.which("conda.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise ValueError("Open Anaconda Prompt and run this launcher there so it can find conda.exe.")


def search_checkpoint(name, folders):
    for folder in folders:
        for directory, subdirs, files in os.walk(folder):
            subdirs[:] = sorted(d for d in subdirs if d.casefold() not in IGNORED_DIRS)
            for filename in files:
                if filename.casefold() == name.casefold():
                    candidate = Path(directory) / filename
                    if candidate.is_file() and candidate.stat().st_size:
                        return candidate
    return None


def model_assets(study, previous, folders, encoder, filename, entry, explicit=None):
    roots = (previous, study)
    repo = next((root / "vendor" / encoder for root in roots
                 if (root / "vendor" / encoder / entry).exists()), None)
    if explicit is not None:
        checkpoint = explicit.expanduser().resolve()
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise FileNotFoundError(f"The supplied checkpoint does not exist or is empty: {checkpoint}")
    else:
        checkpoint = next((root / "weights" / filename for root in roots
                           if (root / "weights" / filename).is_file()
                           and (root / "weights" / filename).stat().st_size), None)
        if checkpoint is None:
            checkpoint = search_checkpoint(filename, folders)
    return repo, checkpoint


def run_in(conda, study, encoder, *args):
    command = [conda, "run", "--no-capture-output", "-n", f"biocal3d-{encoder}",
               "python", *map(str, args)]
    print(f"> {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=study, check=True)


def finish_assets(conda, study, encoder, filename, entry, repo, checkpoint):
    if repo is None:
        args = ["setup_model_env.py", encoder]
        if checkpoint is not None:
            args.extend(("--checkpoint", checkpoint))
        run_in(conda, study, encoder, *args)
        repo = study / "vendor" / encoder
        if not (repo / entry).exists():
            raise FileNotFoundError(f"Encoder setup did not produce the expected checkout: {repo}")
    elif checkpoint is None:
        # Only Concerto can reach this branch. Reuse the existing
        # hash-checked downloader without reinstalling the environment.
        run_in(conda, study, encoder, "-c",
               "from pathlib import Path; from setup_model_env import concerto_checkpoint; "
               "import sys; concerto_checkpoint(Path(sys.argv[1]))", study / "weights" / filename)
    if checkpoint is None:
        checkpoint = study / "weights" / filename
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise FileNotFoundError(f"Missing or empty {encoder} checkpoint after setup: {checkpoint}")
    return repo, checkpoint


def launch(args):
    study, previous, roi, folders = paths_for_pc()
    for filename in ("stage_ablation.py", "setup_model_env.py", "matched_study.py", "matched_figures.py"):
        if not (study / filename).is_file():
            raise FileNotFoundError(f"Expected {filename} in the extracted study folder: {study}")

    run = study / "runs" / "matched_raw"
    cohort = run / "cohort.json"
    results = run / "matched_results" / "development"
    performance = results / "figures" / "performance.png"
    if (results / "settings.json").is_file():
        print(f"Development scoring is already complete: {results}", flush=True)
        if not performance.is_file() and not args.check_only:
            run_in(find_conda(), study, "dinov3", "matched_figures.py", "--run", run,
                   "--partition", "development")
        print(f"Performance figure: {performance}", flush=True)
        return

    if not (roi / "dataset_index.json").is_file():
        raise FileNotFoundError(f"Expected dataset_index.json in the approved ROI export: {roi}")
    if cohort.is_file():
        with cohort.open(encoding="utf-8-sig") as stream:
            rows = json.load(stream)["rows"]
        lead = {row["partition"] for row in rows if row["physical_id"] == "BCG3-1"}
        if lead != {"test"}:
            raise ValueError(f"The saved split at {cohort} must hold out all scans of BCG3-1.")
        print(f"Reusing the prepared specimen split: {cohort}", flush=True)
    else:
        print(f"The split will be prepared from {roi}.", flush=True)

    assets = {}
    for encoder, filename, entry in MODELS:
        repo, checkpoint = model_assets(study, previous, folders, encoder, filename, entry,
                                        getattr(args, f"{encoder}_checkpoint"))
        assets[encoder] = (repo, checkpoint)
        print(f"{encoder} checkout: {repo or 'setup will create it'}", flush=True)
        print(f"{encoder} checkpoint: {checkpoint or 'file missing'}", flush=True)

    if assets["dinov3"][1] is None:
        raise FileNotFoundError(
            f"Could not locate {DINO_NAME}. Put the official file in {study / 'weights'} "
            "or rerun with --dinov3-checkpoint followed by its full path."
        )
    if args.check_only:
        if assets["concerto"][1] is None:
            print("The launcher will download Concerto Small during the run.", flush=True)
        return

    conda = find_conda()
    if not cohort.is_file():
        run_in(conda, study, "dinov3", "stage_ablation.py", "prepare", "--approved-root", roi,
               "--run", run, "--require-test-specimen", "BCG3-1")

    for encoder, filename, entry in MODELS:
        repo, checkpoint = finish_assets(conda, study, encoder, filename, entry, *assets[encoder])
        run_in(conda, study, encoder, "stage_ablation.py", "extract", "--run", run,
               "--encoder", encoder, "--repo", repo, "--checkpoint", checkpoint)

    run_in(conda, study, "dinov3", "matched_study.py", "--run", run, "--partition", "development")
    run_in(conda, study, "dinov3", "matched_figures.py", "--run", run, "--partition", "development")
    print(f"Performance figure: {performance}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print the model paths without running extraction.")
    parser.add_argument("--dinov3-checkpoint", type=Path)
    parser.add_argument("--concerto-checkpoint", type=Path)
    try:
        launch(parser.parse_args())
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, f"Stopped after a command failed (exit code {exc.returncode}). See the output above.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
