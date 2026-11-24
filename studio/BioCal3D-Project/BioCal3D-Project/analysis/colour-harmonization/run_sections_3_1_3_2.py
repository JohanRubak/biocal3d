"""Run the full raw characterization and paired ROI color analyses."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "studio" / "app"))
from dataset_paths import APPROVED_HARMONIZED, APPROVED_REORIENTED, data_root

import pandas as pd
from investigate_approved_rois import Figures, STAGES, plt


def scanner_agreement_plots(folder, args):
    summary = pd.read_csv(folder / "scanner_agreement_summary.csv")
    figs = Figures(folder / "figures", args)
    for cohort, cohort_data in summary.groupby("cohort"):
        stages = [stage for stage in STAGES if stage in set(cohort_data.stage)]
        fig, axes = plt.subplots(1, len(stages), figsize=(4.5*len(stages), 4),
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        for ax, stage in zip(axes.flat, stages):
            data = cohort_data[(cohort_data.stage == stage) &
                               (cohort_data.metric == "deltaE00_mean_Lab_corrected_minus_raw")]
            data = data.sort_values(["scanner_a", "scanner_b"])
            positions = list(range(len(data)))[::-1]
            ax.axvline(0, color=".5", ls="--", lw=1)
            ax.hlines(positions, data.ci_low, data.ci_high, color="#008477")
            ax.scatter(data["median"], positions, color="#008477")
            ax.set_yticks(positions, data.scanner_a + " / " + data.scanner_b)
            ax.set_title(stage.capitalize())
        low, high = axes[0, 0].get_xlim()
        if high-low < .2:
            axes[0, 0].set_xlim(-.1, .1)
        fig.suptitle(f"Change in between-scanner color difference · {cohort}")
        fig.supxlabel("Corrected minus raw ΔE₀₀ · median and 95% confidence interval")
        figs.save(fig, f"scanner_agreement_change_{cohort}", "", "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=data_root() / APPROVED_REORIENTED)
    p.add_argument("--corrected-root", type=Path, default=data_root() / APPROVED_HARMONIZED)
    p.add_argument("--output", type=Path, help="Output folder. Default: results beside this script.")
    p.add_argument("--raw-model-root", type=Path)
    p.add_argument("--corrected-model-root", type=Path)
    p.add_argument("--models-manifest", type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--samples", type=int, default=1024, help="Sample count for RGB SWD.")
    p.add_argument("--projections", type=int, default=32, help="Projection count for RGB SWD.")
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    args.raw_root = args.raw_root.resolve()
    args.corrected_root = args.corrected_root.resolve()
    args.output = args.output.resolve() if args.output else base / "results"
    common = ["--bootstrap", str(args.bootstrap), "--seed", str(args.seed), "--dpi", str(args.dpi)]
    paired = ["--raw-root", str(args.raw_root), "--corrected-root", str(args.corrected_root)]
    raw_output = args.output / "section_3_1_raw"
    lab_output = args.output / "section_3_2_lab"
    rgb_output = args.output / "section_3_2_rgb"
    commands = [
        ["investigate_approved_rois.py", "--data-root", str(args.raw_root),
         "--output", str(raw_output), "--scope", "all", "--resume", *common],
        ["analyse_harmonization.py", *paired, "--output", str(lab_output), *common],
        ["analyse_roi_rgb.py", *paired, "--output", str(rgb_output),
         "--samples", str(args.samples), "--projections", str(args.projections), *common],
    ]
    for script, *arguments in commands:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(base / script), *arguments], check=True)
    scanner_agreement_plots(lab_output, args)
    example_command = [sys.executable, str(base / "specimen_comparison.py"), *paired,
                       "--output", str(args.output / "specimen_comparison"), "--dpi", str(args.dpi)]
    for option in ["raw_model_root", "corrected_model_root", "models_manifest"]:
        value = getattr(args, option)
        if value is not None:
            example_command.extend(["--"+option.replace("_", "-"), str(value)])
    example_status = subprocess.run(example_command, check=False).returncode
    if example_status:
        print("Quantitative analyses completed. Specimen example needs attention; see its console output.")
        return example_status
    print(f"Completed Sections 3.1 and 3.2 analysis outputs in {args.output}")
    print("Figures saved as PNG and PDF in:")
    for folder in [raw_output / "figures", lab_output / "figures",
                   lab_output / "corrected_only_figures", args.output / "specimen_comparison",
                   lab_output / "manuscript_figures", rgb_output / "manuscript_figures"]:
        print(f"  {folder}")


if __name__ == "__main__":
    raise SystemExit(main())
