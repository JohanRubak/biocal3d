"""Train and compare the paper-style pseudo-label pilot in the BioCal3D environments."""

import argparse
from pathlib import Path
import subprocess

from run_train_test_on_this_pc import MODELS, ROOT, find_conda, model_assets, run_in, run_lock


def launch(args):
    run = args.run or ROOT / "runs" / ("matched_raw_smoke" if args.smoke else "matched_raw_train_test")
    run = run.resolve()
    out = (args.out or run / "paper_pseudo_label_pilot").resolve()
    if not (run / "matched_results" / "test" / "settings.json").is_file():
        raise ValueError(f"Complete the baseline study first: {run}")
    conda = find_conda()
    with run_lock(out):
        if not args.compare_only:
            for family, filename, entry in MODELS:
                explicit = getattr(args, f"{family}_checkpoint")
                repo, checkpoint = model_assets(family, filename, entry, explicit)
                assets = ("--repo", repo, "--checkpoint", checkpoint) if repo and checkpoint else ()
                run_in(conda, ROOT, family, "paper_pilot.py", "train", "--run", run, "--out", out,
                       "--family", family, "--epochs", args.epochs, "--max-locations", args.max_locations,
                       "--hidden-dim", args.hidden_dim, *assets)
        run_in(conda, ROOT, "dinov3", "paper_pilot.py", "compare", "--run", run, "--out", out,
               "--mean-threshold", args.mean_threshold, "--paper-threshold", args.paper_threshold)
    print(f"Open the paired examples: {out / 'examples' / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed BioCal3D train/test run.")
    parser.add_argument("--out", type=Path, help="Optional separate pilot folder.")
    parser.add_argument("--smoke", action="store_true", help="Use the completed baseline smoke run and three epochs.")
    parser.add_argument("--compare-only", action="store_true", help="Regenerate the comparison using saved attention heads.")
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
