"""Plot saved BioCal3D results and optionally replay development head fits."""
from pathlib import Path
import argparse
import csv
import json
import re
import struct
import warnings
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss

MODES = ("dinov3_gray", "dinov3_rgb", "concerto_geometry", "concerto_rgb")
LABELS = {
    "dinov3_gray": "DINOv3 grayscale",
    "dinov3_rgb": "DINOv3 RGB",
    "concerto_geometry": "Concerto geometry",
    "concerto_rgb": "Concerto RGB",
}
COLORS = dict(zip(MODES, ("#6483a1", "#15466f", "#9b7865", "#9a4d2c")))
STAGE_LABELS = {"baseline": "Baseline", "partial": "Partial", "clean": "Final clean"}
ITERATION_ARRAY = re.compile(
    rb"\x8c\x07n_iter_\x94(?P<wrapper>.{0,130}?)ub\x0f\xff{15}"
    rb"(?P<value>.{4})\x95.{8}\x8c\x05coef_", re.S
)


def read_results(source):
    """Read CSV/JSON and a known scalar layout without deserializing joblib."""
    if source.is_file():
        with ZipFile(source) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "metrics.json"]
            if len(matches) != 1:
                raise ValueError("Provide an archive containing exactly one metrics.json.")
            parent = Path(matches[0]).parent
            contents = {Path(name).name: archive.read(name) for name in archive.namelist()
                        if Path(name).parent == parent and Path(name).suffix in (".csv", ".json", ".joblib")}
    else:
        contents = {path.name: path.read_bytes() for path in source.iterdir()
                    if path.is_file() and path.suffix in (".csv", ".json", ".joblib")}

    def rows(name):
        return list(csv.DictReader(contents[name].decode("utf-8-sig").splitlines()))

    metrics = json.loads(contents["metrics.json"])
    iterations = {}
    for mode in MODES:
        head = contents.get(f"{mode}_all_development_head.joblib", b"")
        match = ITERATION_ARRAY.search(head)
        # This byte layout is verified for this project's uncompressed protocol-4
        # joblib files. For any other layout leave the count unavailable.
        if (head.startswith(b"\x80\x04") and match and b"\x8c\x02i4" in match["wrapper"]
                and b"\x8c\x01<\x94NNN" in head[:400]
                and b"(K\x03h\x1bNNN" in match["wrapper"]):
            iterations[mode] = struct.unpack("<i", match["value"])[0]
        else:
            iterations[mode] = None
    return metrics, rows("summary.csv"), rows("predictions.csv"), iterations


def save(fig, destination):
    fig.savefig(destination, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {destination}", flush=True)


def plot_performance(metrics, summary, predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), layout="constrained")
    pooled = {row["mode"]: row for row in summary if row["scanner"] == "all"}
    assessment = metrics["assessment"]
    first = metrics["metrics"][MODES[0]]["pooled"]
    fig.suptitle(f"BioCal3D three-stage classification\n{assessment.capitalize()} · "
                 f"{first['specimens']} specimens · {first['scans']} scans per condition",
                 fontsize=16, fontweight="bold")
    ax = axes[0, 0]
    x = np.arange(len(MODES))
    bars = ax.bar(x, [100 * float(pooled[mode]["balanced_accuracy"]) for mode in MODES],
                  color=[COLORS[mode] for mode in MODES], width=.56)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.scatter(x, [100 * float(pooled[mode]["partial_recall"]) for mode in MODES],
               facecolors="white", edgecolors="#212a35", zorder=4, label="Partial-cleaning recall")
    ax.set(xticks=x, xticklabels=[LABELS[m].replace(" ", "\n", 1) for m in MODES],
           ylim=(0, 110), ylabel="Score (%)", title="A   Pooled balanced accuracy")
    ax.legend(loc="lower right", facecolor="white", framealpha=.95)

    ax = axes[0, 1]
    scanners = list(dict.fromkeys(row["scanner"] for row in summary if row["scanner"] != "all"))
    by_scanner = {(r["mode"], r["scanner"]): 100 * float(r["balanced_accuracy"])
                  for r in summary if r["scanner"] != "all"}
    matrix = np.array([[by_scanner.get((mode, scanner), np.nan) for scanner in scanners] for mode in MODES])
    im = ax.imshow(matrix, cmap="Blues", vmin=33.33, vmax=100, aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 77 else "#172b3a")
    ax.set(xticks=range(len(scanners)), xticklabels=scanners, yticks=x,
           yticklabels=[LABELS[m] for m in MODES], title="B   Balanced accuracy by scanner")
    fig.colorbar(im, ax=ax, label="Balanced accuracy (%)", fraction=.05, pad=.04)

    ax = axes[1, 0]
    contrasts = [d for d in metrics["paired_deltas"] if d["metric"] in ("balanced_accuracy", "partial_recall")]
    for i, row in enumerate(contrasts):
        value = 100 * row["paired_delta"]
        color = COLORS[row["higher_input"]]
        if row["ci95"]:
            lower, upper = [100 * v for v in row["ci95"]]
            ax.hlines(i, lower, upper, color=color, linewidth=2)
            ax.scatter([lower, upper], [i, i], marker="|", color=color)
        ax.scatter(value, i, color=color, zorder=3)
    ax.axvline(0, color="#59646c", linestyle="--", linewidth=1)
    ax.set(yticks=range(len(contrasts)),
           yticklabels=[f"{LABELS[d['higher_input']].split()[0]} · "
                        f"{'balanced accuracy' if d['metric']=='balanced_accuracy' else 'partial recall'}"
                        for d in contrasts],
           xlabel="RGB minus comparison condition (percentage points)",
           title="C   Paired colour effect with 95% intervals")
    ax.invert_yaxis()

    ax = axes[1, 1]
    folds = sorted({int(row["fold"]) for row in predictions if row.get("fold")})
    if folds:
        for mode in MODES:
            values = []
            for fold in folds:
                rows = [r for r in predictions if r["mode"] == mode and r["fold"] == str(fold)]
                values.append(100 * balanced_accuracy_score([r["truth"] for r in rows],
                                                            [r["prediction"] for r in rows]))
            ax.plot([f + 1 for f in folds], values, "-o", color=COLORS[mode], label=LABELS[mode])
        ax.set(xticks=[f + 1 for f in folds], xlabel="Development validation fold",
               ylabel="Balanced accuracy (%)", ylim=(25, 105),
               title="D   Validation results by fold")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    else:
        fields = ["baseline_recall", "partial_recall", "clean_recall"]
        positions = np.arange(3)
        for mode in MODES:
            ax.plot(positions, [100 * float(pooled[mode][k]) for k in fields],
                    "-o", color=COLORS[mode], label=LABELS[mode])
        ax.set(xticks=positions, xticklabels=list(STAGE_LABELS.values()), ylabel="Recall (%)",
               ylim=(0, 105), title="D   Recall by stage on the locked test")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.17, axis="y")
        ax.set_axisbelow(True)
    save(fig, output / "performance.png")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8), layout="constrained")
    fig.suptitle(f"Cleaning-stage confusion matrices\n{assessment.capitalize()}",
                 fontsize=15, fontweight="bold")
    stages = [STAGE_LABELS.get(s, s) for s in metrics["class_order"]]
    for ax, mode in zip(axes.flat, MODES):
        counts = np.array(metrics["metrics"][mode]["pooled"]["confusion_matrix"])
        percentages = np.divide(counts, counts.sum(axis=1, keepdims=True),
                                out=np.zeros_like(counts, dtype=float), where=counts.sum(axis=1, keepdims=True)>0)
        im = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(counts):
            ax.text(j, i, f"{count}\n({percentages[i, j]:.0%})", ha="center", va="center",
                    color="white" if percentages[i, j] > .6 else "#172b3a")
        ax.set(xticks=range(len(stages)), yticks=range(len(stages)),
               xticklabels=stages, yticklabels=stages, xlabel="Predicted", ylabel="Actual", title=LABELS[mode])
    fig.colorbar(im, ax=axes, label="Fraction of actual stage", shrink=.75)
    save(fig, output / "confusion_matrices.png")


def replay(run, output, budgets, saved_metrics, saved_predictions):
    from sklearn.exceptions import ConvergenceWarning
    from stage_ablation import STAGES, digest, head, load_embeddings, read_json, write_csv

    cohort = run / "cohort.json"
    if saved_metrics["cohort_sha256"] != digest(cohort):
        raise ValueError("The results and --trace-run refer to different specimen manifests.")
    if saved_metrics["assessment"] != "specimen out-of-fold development":
        raise ValueError("Use development results as the input when replaying optimization.")
    manifest = read_json(cohort)
    rows = [row for row in manifest["rows"] if row["partition"] == "development"]
    folds = np.array([row["fold"] for row in rows])
    labels = np.array([STAGES.index(row["stage"]) for row in rows])
    original = {(r["mode"], r["scan_id"]): np.array([float(r[f"p_{stage}"]) for stage in STAGES])
                for r in saved_predictions}
    cap = head().get_params()["logisticregression__max_iter"]
    if any(b < 1 or b > cap for b in budgets):
        raise ValueError(f"Iteration budgets must be between 1 and {cap}.")
    budgets = sorted(set(budgets) | {cap})
    history = []
    largest_difference = 0.
    for mode in MODES:
        features = load_embeddings(run, rows, mode)
        for fold in sorted(set(folds)):
            validation = folds == fold
            training = ~validation
            final = head().fit(features[training], labels[training])
            final_steps = int(final.named_steps["logisticregression"].n_iter_[0])
            for budget in budgets:
                if budget >= final_steps:
                    model = final
                else:
                    # Independent deterministic refits at increasing caps. Warnings
                    # are expected at deliberately small caps.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        model = head().set_params(logisticregression__max_iter=budget).fit(
                            features[training], labels[training])
                train_prob = model.predict_proba(features[training])
                val_prob = model.predict_proba(features[validation])
                if budget == cap:
                    expected = np.stack([original[(mode, row["scan_id"])] for i, row in enumerate(rows) if validation[i]])
                    largest_difference = max(largest_difference, float(np.max(np.abs(expected - val_prob))))
                history.append({
                    "mode": mode, "fold": int(fold), "iteration_cap": budget,
                    "actual_iterations": int(model.named_steps["logisticregression"].n_iter_[0]),
                    "training_log_loss": float(log_loss(labels[training], train_prob)),
                    "validation_log_loss": float(log_loss(labels[validation], val_prob)),
                    "validation_balanced_accuracy": float(balanced_accuracy_score(labels[validation], val_prob.argmax(1))),
                })
            print(f"{LABELS[mode]} · fold {fold + 1} · {final_steps} optimizer iterations", flush=True)
    write_csv(output / "optimization_replay.csv", history)
    print(f"Largest replay vs saved validation probability difference: {largest_difference:.6g}", flush=True)
    if largest_difference > .001:
        print("Replay differs from the saved fit. Check the scikit-learn/SciPy environment before comparing curves.",
              flush=True)
    plot_replay(history, output, largest_difference)


def plot_replay(history, output, largest_difference):
    last = max(1, max(row["actual_iterations"] for row in history))
    caps = sorted({row["iteration_cap"] for row in history})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    fig.suptitle(f"Development classifier optimization replay\nIndependent refits at increasing iteration caps · "
                 f"largest probability difference from original {largest_difference:.3g}",
                 fontsize=14, fontweight="bold")
    plots = [("training_log_loss", "Training log loss", 1),
             ("validation_log_loss", "Validation log loss", 1),
             ("validation_balanced_accuracy", "Validation balanced accuracy (%)", 100)]
    for ax, (field, title, scale) in zip(axes, plots):
        for mode in MODES:
            selected = [r for r in history if r["mode"] == mode]
            means = [scale * np.mean([r[field] for r in selected if r["iteration_cap"] == budget]) for budget in caps]
            keep = [i for i, budget in enumerate(caps) if budget < last] + [len(caps)-1]
            ax.plot([min(last, caps[i]) for i in keep], [means[i] for i in keep], "-o",
                    color=COLORS[mode], label=LABELS[mode], markersize=4)
        ax.set(xscale="log", xlabel="Maximum optimizer iterations allowed", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[-1].legend(frameon=False, fontsize=8)
    save(fig, output / "optimization_replay.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Result folder or ZIP with one metrics.json.")
    parser.add_argument("--out", type=Path, default=Path("ablation_plots"))
    parser.add_argument("--trace-run", type=Path, help="Existing run containing cohort.json and development feature caches.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 5, 10, 20, 40, 80, 160, 320])
    args = parser.parse_args()
    try:
        metrics, summary, predictions, iterations = read_results(args.results)
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"Assessment: {metrics['assessment']}")
        for mode in MODES:
            count = iterations[mode]
            if count is not None:
                print(f"{LABELS[mode]}: saved all-development head used {count} optimizer iterations.")
        plot_performance(metrics, summary, predictions, args.out)
        if args.trace_run:
            replay(args.trace_run, args.out, args.budgets, metrics, predictions)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
