"""Train and compare the paper-style pseudo-label pilot in the BioCal3D environments."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from run_train_test_on_this_pc import MODELS, ROOT, find_conda, model_assets, run_in, run_lock


OUTPUTS = {"binary": "paper_pseudo_label_pilot", "three-stage": "paper_pseudo_label_3stage_pilot"}


def launch(args):
    run = args.run or ROOT / "runs" / ("matched_raw_smoke" if args.smoke else "matched_raw_train_test")
    run = run.resolve()
    if not (run / "matched_results" / "test" / "settings.json").is_file():
        raise ValueError(f"Complete the baseline study first: {run}")
    heads = OUTPUTS if args.classifier == "both" else {args.classifier: OUTPUTS[args.classifier]}
    if args.out is not None and len(heads) != 1:
        raise ValueError("Specify a single --classifier when using --out.")
    folders = {head: (args.out or run / folder).resolve() for head, folder in heads.items()}
    performance = run / "paper_pilot_performance"
    if performance in folders.values():
        raise ValueError("Use a different --out folder from the performance report.")
    conda = find_conda()
    with ExitStack() as stack:
        for out in folders.values():
            stack.enter_context(run_lock(out))
        if not args.compare_only:
            stack.enter_context(run_lock(performance))
            assets = {}
            for family, filename, entry in MODELS:
                explicit = getattr(args, f"{family}_checkpoint")
                repo, checkpoint = model_assets(family, filename, entry, explicit)
                assets[family] = ("--repo", repo, "--checkpoint", checkpoint) if repo and checkpoint else ()
                if not args.metrics_only:
                    for head, out in folders.items():
                        run_in(conda, ROOT, family, "paper_pilot.py", "train", "--run", run, "--out", out,
                               "--classifier", head, "--family", family, "--epochs", args.epochs,
                               "--max-locations", args.max_locations, "--hidden-dim", args.hidden_dim, *assets[family])
            with TemporaryDirectory(dir=performance) as work:
                options = tuple(arg for head, path in folders.items() for arg in (f"--{head}", path))
                for family in assets:
                    run_in(conda, ROOT, family, "paper_pilot_metrics.py", "score", "--run", run, "--work", work,
                           "--family", family, *options, *assets[family])
                run_in(conda, ROOT, "dinov3", "paper_pilot_metrics.py", "report", "--run", run, "--work", work,
                       "--out", performance, *options)
        if not args.metrics_only:
            for head, out in folders.items():
                run_in(conda, ROOT, "dinov3", "paper_pilot.py", "compare", "--run", run, "--out", out,
                       "--classifier", head, "--mean-threshold", args.mean_threshold, "--paper-threshold",
                       args.paper_threshold)
                print(f"Open the {head} examples: {out / 'examples' / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed BioCal3D train/test run.")
    parser.add_argument("--out", type=Path, help="Optional separate pilot folder for a single classifier.")
    parser.add_argument("--smoke", action="store_true", help="Use the completed baseline smoke run and three epochs.")
    parser.add_argument("--classifier", choices=("both", *OUTPUTS), default="both")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--compare-only", action="store_true", help="Regenerate only the mask galleries.")
    operation.add_argument("--metrics-only", action="store_true", help="Score saved heads without fitting or rebuilding galleries.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-locations", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--mean-threshold", type=float, default=0.)
    parser.add_argument("--paper-threshold", type=float, default=.5)
    parser.add_argument("--dinov3-checkpoint", type=Path)
    parser.add_argument("--concerto-checkpoint", type=Path)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 3 if args.smoke else 20
    if args.max_locations is None:
        args.max_locations = 128 if args.smoke else 256
    if args.hidden_dim is None:
        args.hidden_dim = 16 if args.smoke else 32
    try:
        launch(args)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, f"Stopped at exit code {exc.returncode}. See the command output above.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
