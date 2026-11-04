"""Compare cleaning-stage classification from GMM and boosted candidate masks."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import segment_color_boosting as boosting
from segment_color_gmm import STAGES, segment

METHODS = ("GMM", "Boosting")
FEATURES = ("candidate_fraction", "largest_region_fraction", "log_region_count")


def read_cohort(run):
    manifest = json.loads((run / "cohort.json").read_text(encoding="utf-8-sig"))
    rows = manifest["rows"]
    development = [row for row in rows if row["partition"] == "development"]
    test = [row for row in rows if row["partition"] == "test"]
    for partition, part in (("development", development), ("test", test)):
        if {row["stage"] for row in part} != set(STAGES):
            raise ValueError(f"The {partition} partition must contain all three cleaning stages.")
    if len({row["scan_id"] for row in development + test}) != len(development) + len(test):
        raise ValueError("Duplicate scan IDs in the cohort.")
    if {r["physical_id"] for r in development} & {r["physical_id"] for r in test}:
        raise ValueError("Development and test must have disjoint physical specimens.")

    folds = {}
    for row in development:
        fold = row.get("fold")
        if not isinstance(fold, int) or fold < 0:
            raise ValueError("Development scans must have a saved integer fold in cohort.json.")
        previous = folds.setdefault(row["physical_id"], fold)
        if previous != fold:
            raise ValueError("All scans of a specimen must belong to the same development fold.")
    if len(set(folds.values())) < 2:
        raise ValueError("At least two development folds are required.")
    return development, test


def fit_boosting(args, rows, destination):
    boosting.fit(SimpleNamespace(run=args.run, scanner=None, stage=None,
                                 scan=[row["scan_id"] for row in rows], model=destination,
                                 trees=args.trees, pixels_per_scan=args.pixels_per_scan))
    return joblib.load(destination)


def boosted_mask(image, roi, model):
    # Match the standalone boosting predictor's gate, cutoff, and cleanup.
    features, eligible, scale = boosting.pixel_features(image, roi)
    scores = np.zeros(len(features))
    for step, tree in model["trees"]:
        scores += step * tree.predict(features)
    candidate = np.zeros_like(roi)
    candidate[eligible] = (scores > model["threshold"]) & boosting.color_gate(features)
    components, _ = ndi.label(candidate)
    sizes = np.bincount(components.ravel())
    return candidate & (sizes[components] >= max(1, round(10 * scale ** 2)))


def describe_mask(mask, roi):
    components, count = ndi.label(mask)
    sizes = np.bincount(components.ravel())[1:]
    area = roi.sum()
    return (float(mask.sum() / area), float(sizes.max(initial=0) / area), float(np.log1p(count)))


def extract_features(run, rows, model, features, progress_label):
    if {r["physical_id"] for r in rows} & set(model["specimens"]):
        raise ValueError("A boosted mask would be generated for a training specimen.")
    for index, row in enumerate(rows, 1):
        image, roi = boosting.read_render(run, row)
        gmm, _ = segment(image, roi, membership_threshold=.95)
        masks = (gmm, boosted_mask(image, roi, model))
        for method, mask in zip(METHODS, masks):
            features[method][row["scan_id"]] = describe_mask(mask, roi)
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(f"{progress_label}: masks {index}/{len(rows)}", flush=True)


def labels(rows):
    return np.array([STAGES.index(row["stage"]) for row in rows])


def matrix(rows, features):
    return np.array([features[row["scan_id"]] for row in rows])


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(method, test, truth, predicted, metrics, stages, matrices):
    scanners = np.array([r["scanner"] for r in test])
    for scanner in ("All scanners", *sorted(set(scanners))):
        selected = np.ones(len(test), dtype=bool) if scanner == "All scanners" else scanners == scanner
        actual, guess = truth[selected], predicted[selected]
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, guess, labels=np.arange(len(STAGES)), zero_division=0)
        present = support > 0
        metrics.append({
            "method": method, "scanner": scanner, "scans": len(actual),
            "specimens": len({row["physical_id"] for row, keep in zip(test, selected) if keep}),
            "accuracy": accuracy_score(actual, guess),
            "balanced_accuracy": balanced_accuracy_score(actual, guess),
            "macro_precision": float(precision[present].mean()),
            "macro_f1": float(f1[present].mean()),
        })
        for index, stage in enumerate(STAGES):
            stages.append({
                "method": method, "scanner": scanner, "stage": stage, "support": int(support[index]),
                "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]),
            })
    matrices[method] = confusion_matrix(truth, predicted, labels=np.arange(len(STAGES))).tolist()


def plot_confusion(out, matrices):
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib unavailable. Confusion counts are in confusion_matrices.json.")
        return
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, len(METHODS), figsize=(8.4, 4), layout="constrained")
    fig.suptitle("Cleaning-stage classification from candidate masks | Locked test", fontweight="bold")
    for ax, method in zip(axes, METHODS):
        counts = np.asarray(matrices[method])
        fraction = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(fraction, cmap="Blues", vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(counts):
            ax.text(j, i, f"{count}\n({fraction[i, j]:.0%})", ha="center", va="center",
                    color="white" if fraction[i, j] > .58 else "#172b3a")
        ax.set(title=method, xlabel="Predicted stage", ylabel="Recorded stage",
               xticks=np.arange(3), yticks=np.arange(3),
               xticklabels=("Baseline", "Partial", "Final clean"),
               yticklabels=("Baseline", "Partial", "Final clean"))
    fig.savefig(out / "confusion_matrices.png", dpi=180, facecolor="white")
    plt.close(fig)


def compare(args):
    development, test = read_cohort(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    models = args.out / "models"
    features = {method: {} for method in METHODS}
    folds = sorted({row["fold"] for row in development})
    print(f"{len(development)} development scans, {len(test)} test scans, {len(folds)} saved folds.",
          flush=True)

    for fold in folds:
        training = [row for row in development if row["fold"] != fold]
        validation = [row for row in development if row["fold"] == fold]
        print(f"Fitting mask booster for held-out development fold {fold}.", flush=True)
        model = fit_boosting(args, training, models / f"boost_fold_{fold}.joblib")
        extract_features(args.run, validation, model, features, f"Development fold {fold}")

    # Fit both classifiers before reading the held-out test renders.
    classifiers = {}
    with threadpool_limits(limits=2):
        for method in METHODS:
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=3000))
            classifier.fit(matrix(development, features[method]), labels(development))
            joblib.dump(classifier, models / f"{method.lower()}_stage_classifier.joblib")
            classifiers[method] = classifier

    print("Fitting final mask booster on all development specimens.", flush=True)
    model = fit_boosting(args, development, models / "boost_all_development.joblib")
    extract_features(args.run, test, model, features, "Locked test")
    metrics, stages, matrices, predictions, feature_rows = [], [], {}, [], []
    truth = labels(test)
    for method in METHODS:
        classifier = classifiers[method]
        test_features = matrix(test, features[method])
        probabilities = classifier.predict_proba(test_features)
        predicted = classifier.predict(test_features)
        report(method, test, truth, predicted, metrics, stages, matrices)
        for row, p, guess in zip(test, probabilities, predicted):
            predictions.append({
                "method": method, "scan_id": row["scan_id"], "specimen": row["physical_id"],
                "scanner": row["scanner"], "truth": row["stage"], "prediction": STAGES[guess],
                **{f"p_{stage}": float(p[i]) for i, stage in enumerate(STAGES)},
            })
        for row in development + test:
            feature_rows.append({
                "method": method, "scan_id": row["scan_id"], "partition": row["partition"],
                "scanner": row["scanner"], "stage": row["stage"], "fold": row.get("fold"),
                **dict(zip(FEATURES, features[method][row["scan_id"]])),
            })
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "per_stage_metrics.csv", stages)
    write_csv(args.out / "test_predictions.csv", predictions)
    write_csv(args.out / "mask_features.csv", feature_rows)
    (args.out / "confusion_matrices.json").write_text(
        json.dumps({"class_order": STAGES, "matrices": matrices}, indent=2), encoding="utf-8")
    plot_confusion(args.out, matrices)
    for record in metrics:
        if record["scanner"] == "All scanners":
            print(f"{record['method']}: balanced accuracy {record['balanced_accuracy']:.1%}, "
                  f"macro precision {record['macro_precision']:.1%}, macro F1 {record['macro_f1']:.1%}")
    print(f"Saved results in {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("mask_stage_comparison"))
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--pixels-per-scan", type=int, default=1024)
    args = parser.parse_args()
    if args.trees < 1 or args.pixels_per_scan < 32:
        parser.error("--trees must be positive and --pixels-per-scan must be at least 32.")
    try:
        compare(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
