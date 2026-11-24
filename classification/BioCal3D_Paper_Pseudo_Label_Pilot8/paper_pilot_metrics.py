"""Score the fitted attention heads and render matched held-out performance reports."""

import argparse
from html import escape
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from fixed_study import context, fitted_settings, test_folder
from matched_figures import COLORS, plt, read_csv, save
from matched_study import MATURE_SCOPE, SCOPES, balanced, verify_renders
from paper_pilot import (CLASSES, FAMILIES, MODES, FrozenTrainingEncoder, load_model,
                         native_bag, predict_native, predicted_label, saved_native)
from stage_ablation import save_npz
from study_data import STAGES, write_csv, write_json

HEAD_NAMES = {"binary": "Binary attention", "three-stage": "Three-class attention",
              "mean-pooling": "Mean-pooling logistic regression"}
INPUT_NAMES = dict(zip(MODES, ("Grayscale", "Raw RGB", "Geometry", "Geometry + RGB")))
TASK_NAMES = {"binary": "Plaque stage vs fully cleaned", "three-stage": "Three cleaning stages"}
CLASS_NAMES = {"baseline": "Baseline", "partial": "Partial", "clean": "Fully cleaned",
               "plaque_stage": "Plaque stage"}


def classifier_folders(args):
    return {name: path for name, path in (("binary", args.binary), ("three-stage", args.three_stage))
            if path is not None}


def score(run, destination, family, folders, repo, checkpoint):
    _, roles, settings = context(run)
    fitted_settings(run, settings)
    rows = roles["test"]
    config = next(config for config in settings["configs"] if config["encoder"] == family)
    if family == "dinov3":
        verify_renders(run, rows, config)
    modes = FAMILIES[family]
    models = {(head, mode): load_model(run, folder, mode, roles, head)
              for head, folder in folders.items() for mode in modes}
    predictions = {f"{head}:{mode}": [] for head, mode in models}
    encoder = None
    for i, row in enumerate(rows, 1):
        for mode in modes:
            native = saved_native(run, row, mode, config)
            if native is None:
                if encoder is None:
                    if repo is None or checkpoint is None:
                        raise ValueError(f"Supply the {family} repo and checkpoint to score test scans without spatial caches.")
                    encoder = FrozenTrainingEncoder(run, family, config, repo, checkpoint)
                native = encoder.native(row, mode)
            x, weights, eligible = native_bag(*native)
            if not eligible.any() or not np.isfinite(x[eligible]).all():
                raise ValueError(f"Invalid spatial test features for {row['scan_id']} {mode}.")
            # Leave the thread limit before the next scan's encoder call.
            with threadpool_limits(limits=1, user_api="blas"):
                for head in folders:
                    _, p = predict_native(x, weights, eligible, models[head, mode])
                    if not np.isfinite(p).all():
                        raise ValueError(f"Invalid {head} probabilities for {row['scan_id']} {mode}.")
                    predictions[f"{head}:{mode}"].append(p)
        if i == 1 or i % 32 == 0 or i == len(rows):
            print(f"{family}: scored {i}/{len(rows)} held-out scans", flush=True)
    save_npz(destination / f"{family}.npz", scan_ids=np.array([row["scan_id"] for row in rows]),
             **{key: np.stack(values) for key, values in predictions.items()})


def mean_pooling(run, rows):
    lookup = {row["scan_id"]: row for row in rows}
    indexed = {}
    for record in read_csv(test_folder(run) / "predictions.csv"):
        mode, scan = record["mode"], record["scan_id"]
        if mode not in MODES:
            continue
        row = lookup[scan]
        if ((mode, scan) in indexed or record["truth"] != row["stage"]
                or record["scanner"] != row["scanner"] or record["specimen"] != row["physical_id"]):
            raise ValueError(f"Mean-pooling test prediction differs from the saved split for {scan} {mode}.")
        indexed[mode, scan] = [float(record[f"p_{stage}"]) for stage in STAGES]
    return {mode: np.array([indexed[mode, row["scan_id"]] for row in rows]) for mode in MODES}


def task_predictions(scores, task):
    result = {}
    for (head, mode), probabilities in scores.items():
        if task == "three-stage" and head == "binary":
            continue
        if task == "binary" and head != "binary":
            positive = probabilities[:, [STAGES.index("baseline"), STAGES.index("partial")]].sum(axis=1)
            probabilities = np.column_stack((probabilities[:, STAGES.index("clean")], positive))
        result[head, mode] = probabilities
    return result


def evaluate(rows, scores, task, seed):
    classes = CLASSES[task]
    truth = [row["stage"] for row in rows]
    if task == "binary":
        truth = ["clean" if stage == "clean" else "plaque_stage" for stage in truth]
    labels = np.array([classes.index(stage) for stage in truth])
    guesses = {key: np.array([classes.index(predicted_label(p, task)) for p in probabilities])
               for key, probabilities in scores.items()}
    specimen = np.array([row["physical_id"] for row in rows])
    group = np.array([row["group"] for row in rows])
    scanner = np.array([row["scanner"] for row in rows])
    summaries, per_class, deltas, matrices = [], [], [], {}
    indices = np.arange(len(classes))
    for scope in (*SCOPES, MATURE_SCOPE):
        selected = group >= 3 if scope == MATURE_SCOPE else (
            np.ones(len(rows), dtype=bool) if scope == SCOPES[0] else scanner == scope)
        ids, first, inverse = np.unique(specimen[selected], return_index=True, return_inverse=True)
        group_by_id = group[selected][first]
        rng = np.random.default_rng(seed)
        strata = [np.flatnonzero(group_by_id == g) for g in np.unique(group_by_id)]
        draws = np.concatenate([rng.choice(s, size=(2000, len(s)), replace=True) for s in strata], axis=1)
        matrices[scope], bootstrap = {}, {}
        for (head, mode), predicted in guesses.items():
            cm = confusion_matrix(labels[selected], predicted[selected], labels=indices)
            actual, assigned, correct = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
            if not actual.all():
                raise ValueError(f"Missing {task} class support in {scope}.")
            recall = correct / actual
            precision = np.divide(correct, assigned, out=np.zeros(len(classes)), where=assigned != 0)
            f1 = 2 * correct / (actual + assigned)
            summaries.append(dict(scope=scope, classifier=head, mode=mode, scans=int(selected.sum()),
                                  specimens=len(ids), balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()),
                                  accuracy=float(correct.sum() / cm.sum()), macro_precision=float(precision.mean())))
            per_class.extend(dict(scope=scope, classifier=head, mode=mode, stage=label, support=int(actual[i]),
                                  recall=float(recall[i]), precision=float(precision[i]), f1=float(f1[i]))
                             for i, label in enumerate(classes))
            matrices[scope].setdefault(head, {})[mode] = cm.tolist()
            counts = np.zeros((len(ids), len(classes), len(classes)), dtype=int)
            np.add.at(counts, (inverse, labels[selected], predicted[selected]), 1)
            bootstrap[head, mode] = balanced(counts[draws].sum(axis=1))
        for head, head_matrices in matrices[scope].items():
            for family, (reference, colour) in FAMILIES.items():
                samples = bootstrap[head, colour] - bootstrap[head, reference]
                low, high = np.quantile(samples, [.025, .975])
                point = balanced(np.array(head_matrices[colour])) - balanced(np.array(head_matrices[reference]))
                deltas.append(dict(scope=scope, classifier=head, family=family, contrast="Raw colour − no colour",
                                   comparison=colour, reference=reference, specimens=len(ids),
                                   delta_balanced_accuracy=float(point), ci95_low=float(low), ci95_high=float(high)))
    return summaries, per_class, deltas, matrices


def figures(out, task, summaries, deltas, matrices, title):
    table = {(r["classifier"], r["mode"], r["scope"]): r for r in summaries}
    heads = list(matrices[SCOPES[0]])
    families = ("DINOv3", "Concerto")
    input_labels = [f"{families[i // 2]} · {INPUT_NAMES[mode]}" for i, mode in enumerate(MODES)]
    names = [f"{HEAD_NAMES[head]} · {label}" for head in heads for label in input_labels]
    fig, axes = plt.subplots(1, 2, figsize=(15, 2 + .43 * len(names)), layout="constrained")
    for axis, (metric, label) in zip(axes, (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1"))):
        values = np.array([[table[head, mode, scope][metric] for scope in SCOPES]
                           for head in heads for mode in MODES])
        axis.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(range(len(SCOPES)), SCOPES, rotation=28, ha="right")
        axis.set_yticks(range(len(names)), names if axis is axes[0] else [""] * len(names))
        for i, j in np.ndindex(values.shape):
            value = values[i, j]
            axis.text(j, i, f"{value:.1%}", color="white" if value > .67 else "#142630",
                      ha="center", va="center", fontsize=9)
        for boundary in range(len(MODES), len(names), len(MODES)):
            axis.axhline(boundary - .5, color="white", linewidth=1.5)
        axis.set_title(label)
    fig.suptitle(title)
    save(fig, out / "performance.png")

    lookup = {(r["classifier"], r["scope"], r["family"]): r for r in deltas}
    bound = max(5, 105 * max(abs(r[key]) for r in deltas for key in ("ci95_low", "ci95_high", "delta_balanced_accuracy")))
    fig, axes = plt.subplots(len(heads), len(SCOPES), figsize=(15, 1.8 * len(heads) + .8),
                             squeeze=False, sharex=True, layout="constrained")
    for i, head in enumerate(heads):
        for j, scope in enumerate(SCOPES):
            axis = axes[i, j]
            for k, family in enumerate(FAMILIES):
                r = lookup[head, scope, family]
                axis.hlines(k, 100 * r["ci95_low"], 100 * r["ci95_high"], color=COLORS[1], linewidth=2)
                axis.plot(100 * r["delta_balanced_accuracy"], k, "o", color=COLORS[1], markersize=5)
            axis.axvline(0, color="0.55", linestyle="--", linewidth=1)
            axis.set(xlim=(-bound, bound), ylim=(1.5, -.5))
            axis.set_yticks(range(2), families if j == 0 else ["", ""])
            axis.set_title(f"{scope}\n{HEAD_NAMES[head]}", fontsize=9)
    fig.supxlabel("Raw colour − no colour in balanced accuracy (percentage points) · paired specimen bootstrap 95% interval")
    fig.suptitle(title)
    save(fig, out / "paired_colour_deltas.png")

    classes = CLASSES[task]
    labels = [CLASS_NAMES[cls] for cls in classes]
    links = []
    for scope, by_head in matrices.items():
        fig, axes = plt.subplots(len(heads), len(MODES), figsize=(12, 2.85 * len(heads)),
                                 squeeze=False, layout="constrained")
        for i, head in enumerate(heads):
            for j, mode in enumerate(MODES):
                axis = axes[i, j]
                raw = np.asarray(by_head[head][mode])
                cm = raw / raw.sum(axis=1, keepdims=True)
                axis.imshow(cm, cmap="Blues", vmin=0, vmax=1)
                for a, b in np.ndindex(cm.shape):
                    axis.text(b, a, f"{cm[a,b]:.0%}\n({raw[a,b]})", ha="center", va="center",
                              color="white" if cm[a,b] > .58 else "#18212b", fontsize=9)
                axis.set_xticks(range(len(labels)), labels if i == len(heads)-1 else [""]*len(labels),
                                rotation=23, ha="right")
                axis.set_yticks(range(len(labels)), labels if j == 0 else [""]*len(labels))
                if i == 0:
                    axis.set_title(input_labels[j], fontsize=10)
                if j == 0:
                    axis.set_ylabel(HEAD_NAMES[head], fontsize=9)
        fig.suptitle(f"{title} · {scope}\nRows are recorded classes. Columns are predicted classes.")
        filename = f"confusion_{scope.split(' (')[0].replace(' ', '_').lower()}.png"
        save(fig, out / filename)
        links.append((filename, f"Confusion matrices · {scope}"))
    return links


def export_task(out, task, rows, scores, seed, smoke):
    out.mkdir(parents=True, exist_ok=True)
    classes = CLASSES[task]
    summaries, per_class, deltas, matrices = evaluate(rows, scores, task, seed)
    records = []
    for (head, mode), probabilities in scores.items():
        for row, p in zip(rows, probabilities):
            truth = row["stage"] if task == "three-stage" or row["stage"] == "clean" else "plaque_stage"
            records.append(dict(classifier=head, mode=mode, scan_id=row["scan_id"], specimen=row["physical_id"],
                                scanner=row["scanner"], group=row["group"], recorded_stage=row["stage"],
                                truth=truth, prediction=predicted_label(p, task),
                                **{f"p_{label}": float(value) for label, value in zip(classes, p)}))
    write_csv(out / "predictions.csv", records)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per_stage_metrics.csv", per_class)
    write_csv(out / "paired_deltas.csv", deltas)
    write_json(out / "confusion_matrices.json", dict(class_order=classes, matrices=matrices))
    title = f"{'Smoke check' if smoke else 'Held-out test'} · {TASK_NAMES[task]}"
    return figures(out, task, summaries, deltas, matrices, title)


def report(run, destination, out, folders):
    manifest, roles, settings = context(run)
    rows = roles["test"]
    scores = {}
    expected_ids = np.array([row["scan_id"] for row in rows])
    for family, modes in FAMILIES.items():
        with np.load(destination / f"{family}.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["scan_ids"], expected_ids):
                raise ValueError(f"{family} predictions do not cover the exact saved test split.")
            for head in folders:
                for mode in modes:
                    scores[head, mode] = arrays[f"{head}:{mode}"]
    scores.update({("mean-pooling", mode): values for mode, values in mean_pooling(run, rows).items()})
    for (head, mode), values in scores.items():
        shape = (len(rows), len(CLASSES["binary" if head == "binary" else "three-stage"]))
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1):
            raise ValueError(f"Invalid held-out probabilities for {head} {mode}.")

    blocks = []
    for task in ("binary", "three-stage"):
        if task == "three-stage" and task not in folders:
            continue
        name = task.replace("-", "_")
        links = export_task(out / name, task, rows, task_predictions(scores, task),
                            manifest["split_seed"], settings["protocol"]["smoke"])
        links = [("performance.png", "Balanced accuracy and macro F1"),
                 ("paired_colour_deltas.png", "Paired colour effects"), *links,
                 ("summary.csv", "Summary metrics"), ("per_stage_metrics.csv", "Per-class metrics"),
                 ("predictions.csv", "Per-scan predictions"), ("paired_deltas.csv", "Paired colour effects CSV"),
                 ("confusion_matrices.json", "Confusion matrix counts")]
        listing = "".join(f'<li><a href="{name}/{path}">{escape(label)}</a></li>' for path, label in links)
        blocks.append(f"<h2>{TASK_NAMES[task]}</h2><ul>{listing}</ul>")

    smoke = "Smoke check" if settings["protocol"]["smoke"] else "Held-out test"
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BioCal3D pilot performance</title>'
        '<style>body{font:16px system-ui;max-width:850px;margin:35px auto;padding:0 16px;'
        'background:#f6f8f9;color:#172c35}li{margin:9px 0}a{color:#075f65}</style>'
        f'<h1>BioCal3D pilot performance</h1><p>{smoke} · {len(rows)} scans</p>'
        '<p>The binary task groups baseline and partial scans into plaque stage. For three-class models, the '
        'binary score is the sum of these two probabilities. Scores at or above 0.5 predict plaque stage.</p>'
        '<p>These metrics evaluate scan classification. Candidate-mask agreement is reported in the example galleries.</p>'
        + "".join(blocks) + '</html>'
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Open the performance report: {out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--three-stage", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    folders = classifier_folders(args)
    if not folders or (args.action == "score" and args.family is None) or (args.action == "report" and args.out is None):
        parser.error("Specify classifier folders and --family for scoring or --out for reporting.")
    try:
        if args.action == "score":
            score(args.run, args.work, args.family, folders, args.repo, args.checkpoint)
        else:
            report(args.run, args.work, args.out, folders)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
