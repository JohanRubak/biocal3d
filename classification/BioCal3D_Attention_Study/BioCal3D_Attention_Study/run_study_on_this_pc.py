"""Launch the BioCal3D attention study with the PC's existing Conda environments."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / "repos" / "biocal3d"
CLASSIFICATION = PROJECT / "classification"
APPROVED_ROI = PROJECT / "data" / "Approved ROIs_reoriented"
HARMONIZED_ROI = APPROVED_ROI.with_name(APPROVED_ROI.name + "_harmonized")
ENCODERS = {
    "dinov3": ("hubconf.py", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "concerto": ("concerto", "concerto_small.pth"),
}
PRIOR = ("BioCal3D_Train_Test_Study", "BioCal3D_Stage_Ablations",
         "BioCal3D_Matched_Study", "BioCal3D_Fixed_Split_Study")
SKIP = {".git", "__pycache__", ".conda", "data", "runs", "node_modules"}


def find_conda():
    for candidate in (os.environ.get("CONDA_EXE"), shutil.which("conda.exe"), shutil.which("conda")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise ValueError("Open Anaconda Prompt so the launcher can find conda.exe.")


def search(folder, check):
    for parent, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d.casefold() not in SKIP)
        match = check(Path(parent), files)
        if match is not None:
            return match
    return None


def assets(encoder, repo, checkpoint):
    entry, filename = ENCODERS[encoder]
    roots = [ROOT, CLASSIFICATION, *(CLASSIFICATION / name / name for name in PRIOR),
             *(CLASSIFICATION / name for name in PRIOR)]
    if repo is None:
        repo = next((root / "vendor" / encoder for root in roots
                     if (root / "vendor" / encoder / entry).exists()), None)
        if repo is None:
            repo = search(CLASSIFICATION, lambda path, _: path if path.name == encoder
                          and path.parent.name == "vendor" and (path / entry).exists() else None)
    if repo is None or not (repo / entry).exists():
        raise FileNotFoundError(f"{encoder} checkout not found. Supply --{encoder}-repo or run setup_model_env.py.")
    if checkpoint is None:
        checkpoint = next((root / "weights" / filename for root in roots
                           if (root / "weights" / filename).is_file()), None)
        if checkpoint is None:
            for folder in (CLASSIFICATION, Path.home() / "Downloads"):
                checkpoint = search(folder, lambda path, files: path / filename if filename in files else None)
                if checkpoint is not None:
                    break
    if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"{encoder} checkpoint not found. Supply --{encoder}-checkpoint {filename}.")
    return repo.resolve(), checkpoint.resolve()


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".launcher_running"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError(f"{run} is locked by another launch. If it was force-closed, check the PID in {lock}.") from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_in(conda, encoder, script, *args):
    command = [conda, "run", "--no-capture-output", "-n", f"biocal3d-{encoder}",
               "python", script, *map(str, args)]
    print(f"> {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def launch(args):
    conda = find_conda()
    run = (args.run or ROOT / "runs" / ("smoke" if args.smoke else "full")).resolve()
    epochs = args.epochs if args.epochs is not None else (3 if args.smoke else None)
    if epochs is not None and epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    artifact = args.harmonization_artifact
    harmonized_roi = args.harmonized_roi.resolve() if args.harmonized_roi is not None else None
    if artifact is not None:
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"Harmonisation artefact missing: {artifact}")
    if not args.prepare_only:
        encoders = {encoder: assets(encoder, getattr(args, f"{encoder}_repo"),
                                    getattr(args, f"{encoder}_checkpoint")) for encoder in ENCODERS}
    with run_lock(run):
        roi_args = ("--harmonized-roi", harmonized_roi) if harmonized_roi is not None else ()
        if not args.figures_only or roi_args:
            run_in(conda, "dinov3", "study_protocol.py", "--run", run, "--roi", args.roi.resolve(),
                   *(("--smoke",) if args.smoke else ()), *roi_args)
        if args.prepare_only:
            return
        colour_args = ("--harmonization-artifact", artifact) if artifact is not None else roi_args
        options = ("--with-harmonized",) if colour_args else ()
        if not args.figures_only:
            for encoder, (repo, checkpoint) in encoders.items():
                run_in(conda, encoder, "feature_cache.py", "--run", run, "--encoder", encoder,
                       "--repo", repo, "--checkpoint", checkpoint, *colour_args)
            training_args = []
            for option, value in (("--epochs", epochs), ("--patience", args.patience)):
                if value is not None:
                    training_args.extend((option, value))
            if args.retrain_heads:
                training_args.append("--retrain-heads")
            run_in(conda, "dinov3", "attention_head.py", "--run", run, *training_args, *options)
        run_in(conda, "dinov3", "study_report.py", "--run", run, "--select-examples", *options)
        report_folder = "results_harmonized" if colour_args else "results"
        examples = run / report_folder / "confidence_examples.json"
        if json.loads(examples.read_text(encoding="utf-8"))["examples"]:
            for encoder, (repo, checkpoint) in encoders.items():
                run_in(conda, encoder, "feature_cache.py", "--run", run, "--encoder", encoder,
                       "--repo", repo, "--checkpoint", checkpoint, "--examples-file", examples, *colour_args)
        run_in(conda, "dinov3", "study_report.py", "--run", run, *options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, default=APPROVED_ROI, help="Approved reoriented ROI export folder.")
    parser.add_argument("--run", type=Path, help="Output folder. Defaults to runs/full or runs/smoke.")
    parser.add_argument("--smoke", action="store_true", help="Check the complete pipeline on three specimens.")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--prepare-only", action="store_true", help="Validate the dataset and save the specimen split.")
    operation.add_argument("--figures-only", action="store_true", help="Regenerate figures from a completed run.")
    operation.add_argument("--retrain-heads", action="store_true", help="Refit heads whose settings changed.")
    parser.add_argument("--epochs", type=int, help="Override the maximum number of training epochs.")
    parser.add_argument("--patience", type=int, help="Epochs without a new best validation loss before stopping.")
    colour = parser.add_mutually_exclusive_group()
    colour.add_argument("--harmonization-artifact", type=Path, help="Enable harmonised RGB with a fitted artefact.")
    colour.add_argument("--harmonized-roi", type=Path, nargs="?", const=HARMONIZED_ROI,
                        help="Use exported harmonized ROIs. Defaults to Approved ROIs_reoriented_harmonized.")
    for encoder in ENCODERS:
        parser.add_argument(f"--{encoder}-repo", type=Path)
        parser.add_argument(f"--{encoder}-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        launch(args)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, f"Study stopped at exit code {exc.returncode}. See the error above.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
