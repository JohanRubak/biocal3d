"""Make common performance figures and a matched gallery of held-out scan predictions."""

import csv
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from compare_mask_stage_classification import boosted_mask, describe_mask
from explain_stage_classifier import decomposition, image_values, load_cache
from make_stage_heatmaps import render_inputs
from matched_study import CONTRASTS, INPUTS, LEAD_SCAN, LEAD_SPECIMEN, MATURE_SCOPE, SCOPES, SOURCE_LABELS, render
from segment_color_gmm import segment
from stage_ablation import read_json
from study_data import STAGES, digest, write_csv

COLORS = ("#526c80", "#007e79", "#ad542e")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def performance(args, results, out):
    records = read_csv(results / "summary.csv")
    summary = {(r["mode"], r["scope"]): r for r in records}
    modes = [(family, col, mode) for family, items in INPUTS.items() for col, mode in enumerate(items)
             if (mode, SCOPES[0]) in summary]
    labels = [f"{family} · {'Geometry' if mode == 'concerto_geometry' else SOURCE_LABELS[col]}"
              for family, col, mode in modes]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    for axis, (metric, title) in zip(axes, (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1"))):
        data = np.array([[float(summary[mode, scope][metric]) for scope in SCOPES] for _, _, mode in modes])
        axis.imshow(data, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(range(len(SCOPES)), SCOPES, rotation=28, ha="right")
        axis.set_yticks(range(len(modes)), labels if axis is axes[0] else [""] * len(modes))
        for i, j in np.ndindex(data.shape):
            axis.text(j, i, f"{data[i,j]:.1%}", color="white" if data[i,j] > .67 else "#142630",
                      ha="center", va="center", fontsize=9)
        axis.set_title(title)
    fig.suptitle(f"BioCal3D cleaning-stage classification · {args.partition}")
    save(fig, out / "performance.png")

    contrasts = read_csv(results / "paired_deltas.csv")
    data = {(r["scope"], r["family"], r["contrast"]): r for r in contrasts}
    available = [contrast for contrast in CONTRASTS if any(row["contrast"] == contrast for row in contrasts)]
    fig, axes = plt.subplots(len(available), len(SCOPES), figsize=(15, 3 * len(available)),
                             squeeze=False, sharex=True, layout="constrained")
    bound = max(5, max(abs(float(r[key])) for r in contrasts for key in ("ci95_low", "ci95_high")) * 105,
                max(abs(float(r["delta_balanced_accuracy"])) for r in contrasts) * 105)
    for i, contrast in enumerate(available):
        for j, scope in enumerate(SCOPES):
            axis = axes[i, j]
            for k, family in enumerate(INPUTS):
                r = data[scope, family, contrast]
                point, lo, hi = [100 * float(r[key]) for key in ("delta_balanced_accuracy", "ci95_low", "ci95_high")]
                axis.hlines(k, lo, hi, color=COLORS[i + 1], linewidth=2)
                axis.plot(point, k, "o", color=COLORS[i + 1], markersize=5)
            axis.axvline(0, color="0.55", linestyle="--", linewidth=1)
            axis.set(xlim=(-bound, bound), ylim=(len(INPUTS) - .5, -.5))
            axis.set_yticks(range(len(INPUTS)), list(INPUTS) if j == 0 else [])
            axis.set_title(f"{scope}\n{contrast}", fontsize=9)
    fig.supxlabel("Change in balanced accuracy (percentage points), paired specimen bootstrap 95% interval")
    save(fig, out / "paired_colour_deltas.png")

    document = read_json(results / "confusion_matrices.json")
    for scope, matrices in document["matrices"].items():
        n_inputs = len(modes) // len(INPUTS)
        fig, axes = plt.subplots(len(INPUTS), n_inputs, figsize=(3.65 * n_inputs, 11.4),
                                 squeeze=False, layout="constrained")
        for i, (family, items) in enumerate(INPUTS.items()):
            for j, mode in enumerate(items[:n_inputs]):
                axis = axes[i, j]
                raw = np.asarray(matrices[mode])
                cm = raw / raw.sum(axis=1, keepdims=True)
                axis.imshow(cm, cmap="Blues", vmin=0, vmax=1)
                for x, y in np.ndindex(cm.shape):
                    axis.text(y, x, f"{cm[x,y]:.0%}\n({raw[x,y]})", ha="center", va="center",
                              color="white" if cm[x,y] > .58 else "#18212b", fontsize=9)
                axis.set_xticks(range(len(STAGES)), STAGES, rotation=24, ha="right")
                axis.set_yticks(range(len(STAGES)), STAGES if j == 0 else [])
                title = "Geometry" if mode == "concerto_geometry" else SOURCE_LABELS[j]
                axis.set_title(f"{family} · {title}", fontsize=10)
        fig.suptitle(f"{scope} · recorded stage by predicted stage")
        save(fig, out / f"confusion_{scope.split(' (')[0].replace(' ', '_').lower()}.png")


def gallery_selection(rows, predicted):
    lead = [row for row in rows if row["sample_name"] == LEAD_SCAN]
    result = [(row, "Manuscript example") for row in sorted(lead, key=lambda x: (x["scanner"] != "TRIOS5", x["scanner"]))]
    candidates = [row for row in rows if row["physical_id"] != LEAD_SPECIMEN]
    counts = {row["scan_id"]: sum(p["prediction"] == row["stage"] for p in predicted[row["scan_id"]].values())
              for row in candidates}
    total = len(next(iter(predicted.values())))
    used = {LEAD_SPECIMEN}
    categories = (
        ("All conditions correct", lambda count: count == total, -1),
        ("Conditions disagree on correctness", lambda count: 0 < count < total, 1),
        ("All conditions incorrect", lambda count: count == 0, 1),
    )
    for label, condition, order in categories:
        choices = [row for row in candidates if row["physical_id"] not in used and condition(counts[row["scan_id"]])]
        if not choices:
            continue
        choice = min(choices, key=lambda row: (order * counts[row["scan_id"]], row["scan_id"]))
        result.append((choice, label))
        used.add(choice["physical_id"])
    # Prefer a successful and a failed example even when neither is unanimous.
    fallbacks = [
        ("Highest agreement with recorded stage", lambda n: n > 0, -1),
        ("Most incorrect conditions", lambda n: n < total, 1),
    ]
    if any(reason == "All conditions correct" for _, reason in result):
        fallbacks.reverse()
    for label, condition, order in fallbacks:
        if len(result) >= len(lead) + 3:
            break
        choices = [row for row in candidates if row["physical_id"] not in used and condition(counts[row["scan_id"]])]
        if choices:
            choice = min(choices, key=lambda row: (order * counts[row["scan_id"]], row["scan_id"]))
            result.append((choice, label))
            used.add(choice["physical_id"])
    return result


def check_prediction(model, features, record):
    probability = model.predict_proba(np.asarray(features)[None])[0]
    expected = np.array([float(record[f"p_{stage}"]) for stage in STAGES])
    if not np.allclose(probability, expected, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Could not replay {record['mode']} prediction for {record['scan_id']}.")


def gallery_data(run, results):
    settings = read_json(results / "settings.json")
    if settings["cohort_sha256"] != digest(run / "cohort.json"):
        raise ValueError("The predictions belong to a different cohort.")
    rows = [row for row in read_json(run / "cohort.json")["rows"] if row["partition"] == "test"]
    predicted = defaultdict(dict)
    for record in read_csv(results / "predictions.csv"):
        bucket = predicted[record["scan_id"]]
        if record["mode"] in bucket:
            raise ValueError(f"Duplicate saved prediction for {record['scan_id']}.")
        bucket[record["mode"]] = record
    modes = [mode for family in INPUTS.values() for mode in family]
    if set(predicted) != {row["scan_id"] for row in rows} or any(
        set(bucket) != set(modes) for bucket in predicted.values()
    ):
        raise ValueError("The gallery requires all 12 predictions for every held-out scan.")
    return modes, predicted, gallery_selection(rows, predicted)


def examples(args, results, out, models):
    modes, predicted, selections = gallery_data(args.run, results)
    heads = {mode: joblib.load(models / f"{mode}_stage.joblib") for mode in modes}
    boosters = {variant: joblib.load(models / f"boost_{variant}_training.joblib")
                for variant in ("gray", "rgb", "harmonized")}
    gallery_rows = []
    for row, reason in selections:
        scan = row["scan_id"]
        raw, roi, center, field = render_inputs(args.run, row)
        inputs = [render(args.run, row, variant) for variant in ("gray", "rgb", "harmonized")]
        if any(not np.array_equal(roi, mask) for _, mask in inputs):
            raise ValueError(f"ROI differs across conditions for {scan}.")
        target = STAGES.index(row["stage"])
        alternatives = [stage for stage in STAGES if stage != row["stage"]]
        # Keep one contrast per scan across all models. Prefer the common wrong
        # prediction when there is one, then the strongest alternate on average.
        against = STAGES.index(max(alternatives, key=lambda stage: (
            sum(record["prediction"] == stage for record in predicted[scan].values()),
            sum(float(record[f"p_{stage}"]) for record in predicted[scan].values()),
        )))
        arrays = {"valid_roi": roi, "recorded_stage": np.array(row["stage"]),
                  "contrast_stage": np.array(STAGES[against])}
        fig, grid = plt.subplots(5, 4, figsize=(10.3, 14), layout="constrained",
                                 gridspec_kw={"width_ratios": (1, 1, 1, .07)})
        axes = grid[:, :3]
        for unused in grid[:3, 3]:
            unused.set_axis_off()
        for j, (image, _) in enumerate(inputs):
            axes[0, j].imshow(image)
            axes[0, j].set_title(("Grayscale render", "Raw RGB render", "Harmonised RGB render")[j])
            axes[0, j].set_axis_off()
        for i, (family, family_modes) in enumerate(INPUTS.items(), 1):
            spatial = []
            for j, (mode, (image, _), variant) in enumerate(zip(family_modes, inputs, ("gray", "rgb", "harmonized"))):
                record, model = predicted[scan][mode], heads[mode]
                axis = axes[i, j]
                axis.imshow(raw)
                axis.set_axis_off()
                correct = record["prediction"] == row["stage"]
                input_label = "Geometry" if mode == "concerto_geometry" else SOURCE_LABELS[j]
                axis.set_title(f"{family} · {input_label}\n{record['prediction']}  "
                               f"({float(record['p_' + record['prediction']]):.0%})  "
                               f"{'correct' if correct else 'incorrect'}", fontsize=9)
                mask_fraction = None
                if family in ("GMM", "Boosting"):
                    mask = (segment(image, roi, .95, luminance_only=variant == "gray")[0] if family == "GMM"
                            else boosted_mask(image, roi, boosters[variant]))
                    features = describe_mask(mask, roi)
                    mask_fraction = features[0]
                    arrays[f"{mode}_candidate_mask"] = mask
                    check_prediction(model, features, record)
                    axis.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap(["#ffe263"]),
                                vmin=0, vmax=1, alpha=.74)
                else:
                    cache = load_cache(args.run, row, mode)
                    check_prediction(model, cache["pooled"], record)
                    values = decomposition(model, cache, target, against)["combined"]
                    overlay = image_values(cache, values, roi, center, field)
                    projected = np.where(roi, overlay, np.nan)
                    arrays[f"{mode}_logit_contribution"] = projected
                    spatial.append((axis, projected))
                gallery_rows.append(dict(scan_id=scan, reason=reason, family=family, correct=int(correct),
                                         mask_fraction=mask_fraction, contrast_target=row["stage"],
                                         contrast_against=STAGES[against], **{
                    key: record[key] for key in ("mode", "truth", "prediction", *("p_" + s for s in STAGES))
                }))
            if spatial:
                finite = [np.abs(array[np.isfinite(array)]) for _, array in spatial]
                if any(len(array) == 0 for array in finite):
                    raise ValueError(f"No spatial support for {family}, {scan}.")
                limit = max(1e-8, max(float(np.quantile(array, .99)) for array in finite))
                for axis, array in spatial:
                    overlay = axis.imshow(np.ma.masked_invalid(array), vmin=-limit, vmax=limit,
                                          cmap="coolwarm", alpha=.75)
                bar = fig.colorbar(overlay, cax=grid[i, 3], extend="both")
                bar.ax.set_title("Logits", fontsize=8, pad=6)
                bar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{row['scanner']} · {row['sample_name']} · recorded stage {row['stage']}\n"
                     f"{reason} · neural contrast "
                     f"{row['stage']} vs {STAGES[against]}\n"
                     "Yellow = candidate mask. Red/blue = evidence for/against the named stage contrast.",
                     fontsize=12)
        save(fig, out / f"{scan}.png")
        np.savez_compressed(out / f"{scan}_maps.npz", **arrays)
    write_csv(out / "gallery_predictions.csv", gallery_rows)
    print(f"Saved {len(selections)} matched scan panels at {out}.")

