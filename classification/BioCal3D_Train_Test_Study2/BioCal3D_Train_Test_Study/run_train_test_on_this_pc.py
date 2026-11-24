"""Run the raw-colour BioCal3D train/test study on the Windows research PC."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile


DINO_NAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
CONCERTO_NAME = "concerto_small.pth"
MODELS = (
    ("dinov3", DINO_NAME, "hubconf.py"),
    ("concerto", CONCERTO_NAME, "concerto"),
)
ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / "repos" / "biocal3d"
CLASSIFICATION = PROJECT / "classification"
PREVIOUS_STAGE = CLASSIFICATION / "BioCal3D_Stage_Ablations" / "BioCal3D_Stage_Ablations"
PREVIOUS_MATCHED = CLASSIFICATION / "BioCal3D_Matched_Study" / "BioCal3D_Matched_Study"
PREVIOUS_FIXED = CLASSIFICATION / "BioCal3D_Fixed_Split_Study" / "BioCal3D_Fixed_Split_Study"
APPROVED_ROI = PROJECT / "data" / "Approved ROIs_reoriented"
IGNORED_DIRS = {".git", "__pycache__", ".conda", "data", "runs", "node_modules"}



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


def model_assets(encoder, filename, entry, explicit=None):
    roots = (PREVIOUS_STAGE, PREVIOUS_STAGE.parent, PREVIOUS_MATCHED, PREVIOUS_MATCHED.parent, ROOT)
    folders = (CLASSIFICATION, Path.home() / "Downloads")
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



@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".launcher_running"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError(f"A launcher already owns {run}. If a previous process was force-closed, "
                         f"confirm it has stopped before removing {lock}.") from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def prepare_cohort(conda, run):
    cohort = run / "cohort.json"
    if cohort.is_file():
        print(f"Reusing the fixed cohort: {cohort}", flush=True)
        return
    index = APPROVED_ROI / "dataset_index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Expected the approved dataset index at {index}.")
    previous = next((root / "runs" / name / "cohort.json"
                     for root, name in ((PREVIOUS_MATCHED, "matched_raw"), (PREVIOUS_MATCHED.parent, "matched_raw"),
                                        (PREVIOUS_FIXED, "matched_fixed_split"),
                                        (PREVIOUS_FIXED.parent, "matched_fixed_split"))
                     if (root / "runs" / name / "cohort.json").is_file()), None)
    if previous is not None:
        manifest = json.loads(previous.read_text(encoding="utf-8-sig"))
        if Path(manifest["snapshot_audit"]["index"]).resolve() != index.resolve():
            raise ValueError(f"The previous cohort points to a different ROI export: {previous}")
        with NamedTemporaryFile(dir=run, suffix=".json", delete=False) as temporary:
            copy = Path(temporary.name)
        try:
            shutil.copyfile(previous, copy)
            copy.replace(cohort)
        finally:
            copy.unlink(missing_ok=True)
        print(f"Reusing the specimen assignments from {previous}", flush=True)
    else:
        run_in(conda, ROOT, "dinov3", "stage_ablation.py", "prepare",
               "--approved-root", APPROVED_ROI, "--run", run, "--require-test-specimen", "BCG3-1")


def extract(conda, run, assets, partition, scans=None, spatial=False):
    options = []
    if spatial:
        options.append("--spatial")
    if scans:
        options.extend(("--scan", *scans))
    for encoder, (repo, checkpoint) in assets.items():
        run_in(conda, ROOT, encoder, "stage_ablation.py", "extract", "--run", run,
               "--encoder", encoder, "--repo", repo, "--checkpoint", checkpoint,
               "--partition", partition, *options)


def launch(args):
    run = args.run.resolve()
    outputs = run / "matched_results"
    result = outputs / "test"
    protocol_file = run / "protocol.json"
    if protocol_file.is_file() and json.loads(protocol_file.read_text(encoding="utf-8"))["smoke"] != args.smoke:
        raise ValueError("This folder already holds a different run type. Use a fresh --run folder.")
    if args.gallery_only:
        if not (result / "settings.json").is_file():
            raise ValueError("Complete this run before refreshing its gallery.")
        run_in(find_conda(), ROOT, "dinov3", "refresh_gallery.py", "--run", run)
        return
    if (result / "settings.json").is_file():
        print(f"Completed example panels: {result / 'examples'}", flush=True)
        return

    found = {}
    for encoder, filename, entry in MODELS:
        explicit = getattr(args, f"{encoder}_checkpoint")
        repo, checkpoint = model_assets(encoder, filename, entry, explicit)
        found[encoder] = (repo, checkpoint)
        print(f"{encoder} checkpoint: {checkpoint or 'file missing'}", flush=True)
    if found["dinov3"][1] is None:
        raise FileNotFoundError(f"Could not locate {DINO_NAME}. Put it in {ROOT / 'weights'} "
                                "or pass its full path with --dinov3-checkpoint.")

    conda = find_conda()
    prepare_cohort(conda, run)
    options = ("--smoke",) if args.smoke else ()
    run_in(conda, ROOT, "dinov3", "fixed_study.py", "prepare", "--run", run, *options)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    assets = {encoder: finish_assets(conda, ROOT, encoder, filename, entry, *found[encoder])
              for encoder, filename, entry in MODELS}

    if not (outputs / "models" / "settings.json").is_file():
        scans = protocol["training_scan_ids"] if args.smoke else None
        extract(conda, run, assets, "development", scans)
    run_in(conda, ROOT, "dinov3", "fixed_study.py", "fit", "--run", run)

    plan_path = result / "gallery_selection.json"
    if not plan_path.is_file() or not (result / "_preview_predictions.csv").is_file():
        scans = protocol["test_scan_ids"] if args.smoke else None
        extract(conda, run, assets, "test", scans)
    run_in(conda, ROOT, "dinov3", "fixed_study.py", "shortlist", "--run", run)
    chosen = json.loads(plan_path.read_text(encoding="utf-8"))
    scans = [row["scan_id"] for row in chosen["examples"]]
    if not scans:
        raise ValueError("The test gallery selection is empty.")
    extract(conda, run, assets, "test", scans, spatial=True)
    run_in(conda, ROOT, "dinov3", "fixed_study.py", "finish", "--run", run)
    if args.smoke:
        print("Smoke check passed. Open the four BCG3T1-1 panels before running the full study.", flush=True)
    print(f"Example panels: {result / 'examples'}", flush=True)
    print(f"Performance figures: {result / 'figures'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Override the default output folder.")
    parser.add_argument("--smoke", action="store_true", help="Fit and make example panels on a small, isolated split.")
    parser.add_argument("--gallery-only", action="store_true", help="Refresh a completed run's panels and viewers.")
    parser.add_argument("--dinov3-checkpoint", type=Path)
    parser.add_argument("--concerto-checkpoint", type=Path)
    args = parser.parse_args()
    if args.run is None:
        args.run = ROOT / "runs" / ("matched_raw_smoke" if args.smoke else "matched_raw_train_test")
    try:
        with run_lock(args.run.resolve()):
            launch(args)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, f"Stopped at exit code {exc.returncode}. See the command output above.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
