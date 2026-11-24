"""Score locked test specimens and export performance, learned attention and PCA figures."""
import argparse
from contextlib import ExitStack
from html import escape
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.special import softmax
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from threadpoolctl import threadpool_limits

from attention_head import forward, load_bags, read_model, score_bags
from feature_cache import bag_path, details_path, key_for, mode_settings, view_path
from plaque_mask_viewer import write_mask_viewer
from spatial import match_scan, patch_overlay, point_overlay, plt, save_figure_png
from study_data import SCANNERS, STAGES, write_csv, write_json
from study_io import read_json, save_npz
from study_protocol import INPUTS, LEAD_SCAN, LEAD_SPECIMEN, SEED, input_modes, is_lead
from supplemental_figures import balanced_bars, scan_panel

FAMILIES = ("dinov3", "concerto")
TITLES = ("DINOv3", "Concerto")
STAGE_LABELS = ("Baseline", "Partial", "Fully cleaned")
SCOPES = ("All scanners", *SCANNERS)
EXAMPLE_LABELS = {
    "highest": "Highest confidence",
    "lowest": "Lowest confidence",
    "most_correct": "Most models correct",
    "least_correct": "Fewest models correct",
    "highest_combined": "Highest combined score",
    "lowest_combined": "Lowest combined score",
}


def row_label(mode):
    return INPUTS[mode][2]


def balanced(cm):
    return (np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(axis=-1)).mean(axis=-1)


def metrics(rows, probabilities, modes, smoke):
    truth = np.array([STAGES.index(row["stage"]) for row in rows])
    scanner = np.array([row["scanner"] for row in rows])
    physical = np.array([row["physical_id"] for row in rows])
    group = np.array([row["group"] for row in rows])
    matrices, summaries, classes, deltas = {}, [], [], []
    for scope in SCOPES:
        keep = np.ones(len(rows), dtype=bool) if scope == SCOPES[0] else scanner == scope
        ids, first, inverse = np.unique(physical[keep], return_index=True, return_inverse=True)
        groups = group[keep][first]
        rng = np.random.default_rng(SEED)
        strata = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        draws = np.concatenate([rng.choice(s, size=(2000, len(s)), replace=True) for s in strata], axis=1)
        matrices[scope], bootstrap = {}, {}
        for mode in modes:
            predicted = probabilities[mode].argmax(axis=1)
            counts = np.zeros((len(ids), len(STAGES), len(STAGES)), dtype=np.int64)
            np.add.at(counts, (inverse, truth[keep], predicted[keep]), 1)
            matrix = counts.sum(axis=0)
            support, predicted_n = matrix.sum(axis=1), matrix.sum(axis=0)
            if not support.all():
                raise ValueError(f"Missing test class in {scope}.")
            correct = matrix.diagonal()
            recall = correct / support
            precision = np.divide(correct, predicted_n, out=np.zeros(len(STAGES)), where=predicted_n > 0)
            f1 = 2 * correct / (support + predicted_n)
            matrices[scope][mode] = matrix.tolist()
            bootstrap[mode] = balanced(counts[draws].sum(axis=1))
            low, high = np.quantile(bootstrap[mode], [.025, .975]) if not smoke else ("", "")
            summaries.append(dict(scope=scope, mode=mode, specimens=len(ids), scans=int(keep.sum()),
                                  balanced_accuracy=float(recall.mean()), ci95_low=low, ci95_high=high,
                                  macro_f1=float(f1.mean()), accuracy=float(correct.sum() / matrix.sum())))
            classes.extend(dict(scope=scope, mode=mode, stage=stage, support=int(support[i]),
                                precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                           for i, stage in enumerate(STAGES))
        for family, title in zip(FAMILIES, TITLES):
            selected = [mode for mode in modes if INPUTS[mode][0] == family]
            for left, right in zip(selected, selected[1:]):
                draws_delta = bootstrap[right] - bootstrap[left]
                low, high = np.quantile(draws_delta, [.025, .975]) if not smoke else ("", "")
                delta = balanced(np.array(matrices[scope][right])) - balanced(np.array(matrices[scope][left]))
                deltas.append(dict(scope=scope, family=title, reference=left, comparison=right,
                                   delta_balanced_accuracy=delta, ci95_low=low, ci95_high=high))
    return matrices, summaries, classes, deltas


def performance_figures(out, modes, matrices, summaries, deltas, title, smoke):
    chart = {(r["mode"], r["scope"]): r["balanced_accuracy"] for r in summaries}
    values = np.array([[chart[mode, scope] for scope in SCOPES] for mode in modes])
    fig, ax = plt.subplots(figsize=(10, 2.4 + .48 * len(modes)), layout="constrained")
    image = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SCOPES)), SCOPES, rotation=22, ha="right")
    ax.set_yticks(range(len(modes)), [row_label(mode) for mode in modes])
    for y, x in np.ndindex(values.shape):
        value = values[y, x]
        ax.text(x, y, f"{value:.1%}", ha="center", va="center", color="white" if value > .65 else "#142630")
    fig.colorbar(image, ax=ax, label="Balanced accuracy")
    fig.suptitle(f"{title} · balanced accuracy")
    save_figure_png(fig, out / "balanced_accuracy.png")

    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    comparisons = list(dict.fromkeys((r["reference"], r["comparison"]) for r in deltas))
    for index, (reference, comparison) in enumerate(comparisons):
        items = [r for r in deltas if (r["reference"], r["comparison"]) == (reference, comparison)]
        y = np.arange(len(SCOPES)) + (index - (len(comparisons) - 1) / 2) * .16
        color = f"C{index}"
        if not smoke:
            ax.hlines(y, [r["ci95_low"] * 100 for r in items],
                      [r["ci95_high"] * 100 for r in items], color=color, linewidth=1.8)
        variant = INPUTS[comparison][1]
        label = f"{items[0]['family']} · " + ("harmonised − raw RGB" if variant == "harmonized"
                                              else "raw RGB − no RGB")
        ax.plot([r["delta_balanced_accuracy"] * 100 for r in items], y, "o", color=color, label=label)
    ax.axvline(0, color=".55", linestyle="--", linewidth=1)
    limits = [abs(float(r[key]) * 100) for r in deltas
              for key in ("delta_balanced_accuracy", "ci95_low", "ci95_high") if r[key] != ""]
    extent = max(5, max(limits) * 1.12)
    ax.set_xlim(-extent, extent)
    ax.set_yticks(range(len(SCOPES)), SCOPES)
    ax.invert_yaxis()
    ax.set_xlabel("Change in balanced accuracy (percentage points)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncols=2, fontsize=9)
    interval = "" if smoke else " · paired specimen bootstrap 95% intervals"
    fig.suptitle(f"{title} · effect of colour{interval}", fontsize=11)
    save_figure_png(fig, out / "paired_colour_effects.png")

    files = []
    for scope, per_mode in matrices.items():
        n = len(modes) // 2
        fig, axes = plt.subplots(2, n, figsize=(3.25 * n, 5.8), squeeze=False, layout="constrained")
        for ax, mode in zip(axes.ravel(), modes):
            cm = np.asarray(per_mode[mode])
            fraction = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
            for row, col in np.ndindex(cm.shape):
                ax.text(col, row, f"{cm[row, col]}\n{fraction[row, col]:.0%}", ha="center", va="center",
                        color="white" if fraction[row, col] > .6 else "#142630")
            ax.set_xticks(range(len(STAGES)), STAGE_LABELS, rotation=20, ha="right")
            ax.set_yticks(range(len(STAGES)), STAGE_LABELS)
            ax.set_title(row_label(mode), fontsize=9)
        fig.supxlabel("Predicted stage")
        fig.supylabel("Observed stage")
        fig.suptitle(f"{title} · {scope} · counts and row percentages", fontsize=11)
        file = f"confusion_{scope.lower().replace(' ', '_')}.png"
        save_figure_png(fig, out / file)
        files.append((file, f"Confusion matrices · {scope}"))
    return files


def roc_figure(out, rows, modes, probabilities, title):
    """Plot pooled one-vs-rest ROC curves for every active model/input condition."""
    truth = np.array([STAGES.index(row["stage"]) for row in rows])
    families = (("dinov3", "DINOv3"), ("concerto", "Concerto"))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for family_index, (family, family_title) in enumerate(families):
        selected = [mode for mode in modes if INPUTS[mode][0] == family]
        for config_index, mode in enumerate(selected):
            ax = axes[family_index, config_index]
            scores = probabilities[mode]
            class_aucs = []
            for class_index, stage_title in enumerate(STAGE_LABELS):
                binary_truth = truth == class_index
                fpr, tpr, _ = roc_curve(binary_truth, scores[:, class_index])
                class_auc = auc(fpr, tpr)
                class_aucs.append(class_auc)
                ax.plot(fpr, tpr, linewidth=1.8,
                        label=f"{stage_title} (AUC {class_auc:.3f})")
            ax.plot((0, 1), (0, 1), color=".55", linestyle="--", linewidth=1)
            macro_auc = float(np.mean(class_aucs))
            ax.set_title(f"{family_title} · {INPUTS[mode][2].split(' · ', 1)[1]}\n"
                         f"Macro OvR AUC {macro_auc:.3f}", fontsize=10)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6),
                   yticks=np.linspace(0, 1, 6))
            ax.grid(color="#e2e8ed", linewidth=.7)
            ax.legend(loc="lower right", fontsize=7.5, frameon=True)
        for config_index in range(len(selected), axes.shape[1]):
            axes[family_index, config_index].set_visible(False)
    fig.supxlabel("False-positive rate")
    fig.supylabel("True-positive rate")
    fig.suptitle(f"{title} · one-vs-rest ROC curves by input condition", fontsize=13)
    save_figure_png(fig, out / "roc_curves.png")


def loss_figure(out, modes, histories, title):
    n = len(modes) // 2
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 6.1), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for ax, mode in zip(axes.ravel(), modes):
        history, epoch, max_epochs = histories[mode]
        xs = np.arange(1, len(history) + 1)
        ax.plot(xs, history[:, 0], color="#2a6697", label="Train")
        ax.plot(xs, history[:, 1], color="#d4741d", label="Validation")
        ax.axvline(epoch, color=".4", linestyle="--", label="Lowest validation loss")
        ax.plot(epoch, history[epoch - 1, 1], "o", color="#d4741d", markersize=3)
        ending = "early stop" if len(history) < max_epochs else "epoch limit"
        ax.set_title(f"{row_label(mode)}\nBest epoch {epoch} · {ending} at {len(history)}", fontsize=9)
    observed = np.concatenate([history.ravel() for history, _, _ in histories.values()])
    low, high = observed.min(), observed.max()
    pad = max(.01, .05 * (high - low))
    axes[0, 0].set_ylim(max(0, low - pad), high + pad)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Epoch")
    fig.supylabel("Class-weighted cross-entropy")
    fig.suptitle(f"{title} · training and validation loss")
    save_figure_png(fig, out / "loss_progress.png")


def open_illustration(run, row, modes, settings, models):
    raw_key = key_for(settings["dinov3_rgb"], row)
    with np.load(view_path(run, row), allow_pickle=False) as view:
        if view["key"].item() != raw_key:
            raise ValueError(f"View does not match its scan: {row['scan_id']}")
        rgb, roi, center, field = view["rgb"], view["roi"], view["center"], float(view["field"].item())
    feature_bags, maps, weights, pred = {}, {}, {}, {}
    for mode in modes:
        with ExitStack() as stack:
            bag_cache = stack.enter_context(np.load(bag_path(run, mode, row), allow_pickle=False))
            source = bag_cache if is_lead(row) else stack.enter_context(
                np.load(details_path(run, mode, row), allow_pickle=False))
            key = key_for(settings[mode], row)
            if source["key"].item() != key or bag_cache["key"].item() != key:
                raise ValueError(f"Illustration cache does not match its scan: {row['scan_id']} {mode}")
            bag, native, indices = bag_cache["bag"], source["native"], source["indices"]
            logits, alpha, _, _ = forward(bag[None], np.ones((1, len(bag)), dtype=bool), models[mode])
            alpha = alpha[0]
            relative = alpha * len(alpha)
            pred[mode] = softmax(logits[0])
            if native.ndim == 3:
                coverage = source["coverage"]
                shape = coverage.shape
                x = native.reshape(native.shape[0], -1).T
                keep = coverage.ravel() >= .5
                feature_bags[mode] = (x[keep], keep, None, {"shape": shape})
                scores = np.zeros(coverage.size, dtype=np.float32)
                scores[indices] = relative
                support = np.zeros(coverage.size, dtype=bool)
                support[indices] = True
                maps[mode] = patch_overlay(scores.reshape(shape), support.reshape(shape), roi.shape[1], roi.shape[0])
            else:
                xyz = source["xyz"]
                feature_bags[mode] = (native, np.ones(len(native), dtype=bool), None, {"xyz_mm": xyz})
                maps[mode] = point_overlay(xyz[indices], relative, center, field, roi)
            maps[mode] = np.where(roi, maps[mode], np.nan)
            weights[f"{mode}_attention"] = alpha
            weights[f"{mode}_indices"] = indices
            weights[f"{mode}_map_relative_to_uniform"] = maps[mode]
    return (rgb, roi, center, field), match_scan(feature_bags, center, field, row["scan_id"]), maps, pred, weights


def pca_maps(examples, leads, modes, items, views):
    mapped = {}
    with threadpool_limits(limits=4, user_api="blas"):
        for family in FAMILIES:
            family_modes = [mode for mode in modes if INPUTS[mode][0] == family]
            data = [items[row["scan_id"]][mode][0] for mode in family_modes for row in leads]
            tokens = np.concatenate(data)
            if min(tokens.shape) < 3:
                raise ValueError("The primary specimen has too few matched cells for PCA.")
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(tokens)
            low, high = np.quantile(pca.transform(tokens), [.02, .98], axis=0)
            for mode in family_modes:
                for row in examples:
                    scan = row["scan_id"]
                    x, supported, _, meta = items[scan][mode]
                    _, roi, _, _ = views[scan]
                    colours = np.clip((pca.transform(x) - low) / np.maximum(high - low, 1e-8), 0, 1)
                    shape = meta["shape"]
                    grid = np.zeros((len(supported), 3), dtype=np.float32)
                    grid[supported] = colours
                    support = supported.reshape(shape)
                    w, h = roi.shape[1], roi.shape[0]
                    projected = np.stack([patch_overlay(ch.reshape(shape), support, w, h) for ch in grid.T], axis=-1)
                    nearest = np.asarray(Image.fromarray(support.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST))
                    valid = roi & (nearest > 0) & np.isfinite(projected).all(axis=-1)
                    mapped[mode, scan] = np.where(valid[..., None], projected, 1)
    return mapped


def _show_plaque_mask(ax, background, mask):
    ax.imshow(background)
    if mask.any():
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (0, .78, .9, .62)
        ax.imshow(overlay)


def _finish_image_axis(ax):
    ax.set(xticks=[], yticks=[])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def qualitative_figures(out, leads, modes, views, maps, pca, predictions, masks, limit):
    """Export all active input conditions in a compact four-scanner layout."""
    # The earlier constrained layout compressed each nested scanner grid and
    # left a large empty band between the upper and lower scanner panels. A
    # fixed GridSpec gives each scanner exactly six configuration rows.
    fig = plt.figure(figsize=(15.5, 12.4))
    fig.subplots_adjust(left=.035, right=.93, top=.89, bottom=.025)
    panels = fig.add_gridspec(2, 2, wspace=.12, hspace=.24)
    panel_lookup = {row["scanner"]: row for row in leads}
    for panel_index, scanner in enumerate(SCANNERS):
        row = panel_lookup[scanner]
        scan = row["scan_id"]
        raw, roi, _, _ = views[scan]
        background = np.where(roi[..., None], raw, 255)
        panel_grid = panels[panel_index // 2, panel_index % 2].subgridspec(
            len(modes), 4, width_ratios=[1.35, 1, 1, 1], wspace=.035, hspace=.12)
        axes = np.asarray([[fig.add_subplot(panel_grid[r, c]) for c in range(1, 4)]
                           for r in range(len(modes))])
        for index, mode in enumerate(modes):
            probability = predictions[scan][mode]
            guess = STAGE_LABELS[int(probability.argmax())]
            axes[index, 0].imshow(background)
            axes[index, 0].imshow(
                np.ma.masked_invalid(maps[scan][mode]), cmap="magma", vmin=0, vmax=limit, alpha=.86)
            axes[index, 1].imshow(pca[mode, scan], interpolation="nearest")
            mask = masks[scan][f"{mode}_proposed_plaque"]
            _show_plaque_mask(axes[index, 2], background, mask)
            label_ax = fig.add_subplot(panel_grid[index, 0])
            label_ax.axis("off")
            label = row_label(mode).replace("geometry + ", "geometry +\n")
            label_ax.text(.98, .5, f"{label}\n{guess} · {probability.max():.0%}",
                          ha="right", va="center", fontsize=7)
            if index == 0:
                bounds = panel_grid.get_subplot_params(fig)
                fig.text((bounds.left + bounds.right) / 2, bounds.top + .022,
                         scanner, ha="center", weight="bold", fontsize=12)
                axes[index, 0].set_title("Attention overlay", fontsize=8)
                axes[index, 1].set_title("PCA feature map", fontsize=8)
                axes[index, 2].set_title("Proposed pseudo-label", fontsize=8)
            for ax in axes[index]:
                _finish_image_axis(ax)
    fig.suptitle(
        f"{LEAD_SCAN} · all active model configurations by scanner\n"
        "Attention overlays, PCA feature maps, and proposed pseudo-labels",
        fontsize=13, weight="bold", y=.98)
    save_figure_png(fig, out / f"{LEAD_SCAN}_qualitative_all_configurations.png", dpi=300)
    # Do not leave the redundant selected-config export in a completed report.
    (out / f"{LEAD_SCAN}_qualitative_selected.png").unlink(missing_ok=True)


def partial_examples_figure(out, selection, examples, views, maps, pca, predictions, masks, modes, limit):
    """Show attention, matched PCA, and proposed masks for unique ranked cases."""
    roles = ("least_correct", "lowest_combined", "highest_combined", "most_correct")
    entries = {entry["role"]: entry for entry in selection["examples"]}
    if any(role not in entries for role in roles):
        return None
    row_by_scan = {row["scan_id"]: row for row in examples}
    role_scan_groups = {}
    for role in roles:
        entry = entries[role]
        role_scan_groups.setdefault(entry["scan_id"], []).append(role)
    chosen = [(role_scan_groups[scan_id], entries[role_scan_groups[scan_id][0]], row_by_scan[scan_id])
              for scan_id in role_scan_groups]
    role_titles = {
        "least_correct": "Fewest models correct",
        "lowest_combined": "Lowest combined score",
        "highest_combined": "Highest combined score",
        "most_correct": "Most models correct",
    }
    panel_columns = len(chosen) if len(chosen) <= 3 else 2
    panel_rows = int(np.ceil(len(chosen) / panel_columns))
    panel_width, panel_height = 6.1, 1.65 * len(modes) + 1.25
    fig = plt.figure(figsize=(panel_width * panel_columns, panel_height * panel_rows))
    fig.subplots_adjust(left=.025, right=.985, top=.84, bottom=.025, wspace=.16, hspace=.24)
    outer = fig.add_gridspec(panel_rows, panel_columns, wspace=.14, hspace=.35)
    fig.suptitle("Partial-stage examples · attention, features, and proposed masks",
                 fontsize=14, weight="bold", y=.985)
    for case_index, (case_roles, entry, row) in enumerate(chosen):
        scan = row["scan_id"]
        rgb, roi, _, _ = views[scan]
        background = np.where(roi[..., None], rgb, 255)
        criteria = " + ".join(role_titles[role] for role in case_roles)
        details = []
        if "least_correct" in case_roles or "most_correct" in case_roles:
            details.append(f"{entry['models_correct']}/{len(modes)} models correct")
        if "lowest_combined" in case_roles or "highest_combined" in case_roles:
            details.append(f"combined score {entry['correct_confidence_score']:.1%}")
        panel_grid = outer[case_index // panel_columns, case_index % panel_columns].subgridspec(
            len(modes), 4, width_ratios=[1.35, 1, 1, 1], wspace=.025, hspace=.07)
        axes = np.asarray([[fig.add_subplot(panel_grid[r, c]) for c in range(1, 4)]
                           for r in range(len(modes))])
        for mode_index, mode in enumerate(modes):
            prediction = predictions[scan][mode]
            guess = STAGE_LABELS[int(prediction.argmax())]
            axes[mode_index, 0].imshow(background)
            axes[mode_index, 0].imshow(
                np.ma.masked_invalid(maps[scan][mode]), cmap="magma", vmin=0, vmax=limit, alpha=.86)
            axes[mode_index, 1].imshow(pca[mode, scan], interpolation="nearest")
            mask = masks[scan][f"{mode}_proposed_plaque"]
            _show_plaque_mask(axes[mode_index, 2], background, mask)
            label_ax = fig.add_subplot(panel_grid[mode_index, 0])
            label_ax.axis("off")
            label = row_label(mode).replace("geometry + ", "geometry +\n")
            label_ax.text(.98, .5, f"{label}\n{guess} · {prediction.max():.0%}",
                          ha="right", va="center", fontsize=7)
            if mode_index == 0:
                bounds = panel_grid.get_subplot_params(fig)
                x = (bounds.left + bounds.right) / 2
                fig.text(x, bounds.top + .066 / panel_rows,
                         f"{chr(65 + case_index)} · {row['scanner']} · {row['sample_name']}",
                         ha="center", weight="bold", fontsize=10)
                fig.text(x, bounds.top + .032 / panel_rows,
                         criteria.replace(" + ", "\n") + "\n" + " · ".join(details),
                         ha="center", fontsize=8)
                axes[mode_index, 0].set_title("Attention overlay", fontsize=8)
                axes[mode_index, 1].set_title("PCA feature map", fontsize=8)
                axes[mode_index, 2].set_title("Proposed pseudo-label", fontsize=8)
            for ax in axes[mode_index]:
                _finish_image_axis(ax)
    filename = "partial_stage_selected_model_examples.png"
    save_figure_png(fig, out / filename, dpi=300)
    return filename


def illustrations(run, out, rows, modes, settings, models, probabilities, selection):
    leads = sorted([row for row in rows if is_lead(row)], key=lambda row: SCANNERS.index(row["scanner"]))
    if [row["scanner"] for row in leads] != SCANNERS:
        raise ValueError(f"Expected {LEAD_SCAN} on all scanners in test.")
    positions = {row["scan_id"]: i for i, row in enumerate(rows)}
    extremes = {}
    for entry in selection["examples"]:
        extremes.setdefault(entry["scan_id"], []).append(entry)
    examples = leads + [rows[positions[scan]] for scan in extremes]
    views, items, maps, preds, native, corrected_views = {}, {}, {}, {}, {}, {}
    harmonizer = settings.get("dinov3_harmonized", {}).get("harmonizer", {})
    with threadpool_limits(limits=4, user_api="blas"):
        for row in examples:
            scan = row["scan_id"]
            views[scan], items[scan], maps[scan], preds[scan], native[scan] = open_illustration(
                run, row, modes, settings, models)
            if "roi_index_sha256" in harmonizer:
                with np.load(view_path(run, row, True), allow_pickle=False) as view:
                    if view["key"].item() != key_for(settings["dinov3_harmonized"], row):
                        raise ValueError(f"Harmonized view does not match its scan: {scan}")
                    corrected_views[scan] = view["rgb"]
            for mode in modes:
                if not np.allclose(preds[scan][mode], probabilities[mode][positions[scan]], atol=1e-6):
                    raise ValueError(f"Illustration predictions disagree with test predictions: {mode} {scan}.")
    all_maps = [maps[row["scan_id"]][mode] for row in examples for mode in modes]
    limit = max(2, float(np.quantile(np.concatenate([m[np.isfinite(m)] for m in all_maps]), .99)))
    pca = pca_maps(examples, leads, modes, items, views)
    folder = out / "examples"
    folder.mkdir(exist_ok=True)
    links = []
    masks_by_scan = {}
    with threadpool_limits(limits=4, user_api="blas"):
        for row in examples:
            scan = row["scan_id"]
            selected = extremes.get(scan, [])
            role = selected[0]["role"] if selected else "primary"
            filename = f"{role}_{row['scanner']}_{row['sample_name']}.png"
            label = " / ".join(EXAMPLE_LABELS[entry["role"]] for entry in selected) or "Primary"
            note = "\n".join(example_summary(entry, "\n") for entry in selected) or "Primary example"
            shape = items[scan]["dinov3_rgb"][3]["shape"]
            masks, scores = scan_panel(folder / filename, row, modes, views[scan], maps[scan], pca,
                                       preds[scan], limit, shape, note, harmonized_rgb=corrected_views.get(scan))
            masks_by_scan[scan] = masks
            save_npz(out / f"{row['scanner']}_{row['sample_name']}_attention.npz",
                     **native[scan], **masks, roi=views[scan][1])
            label = f"{label} · {row['scanner']} · {row['sample_name']}"
            links.append((f"examples/{filename}", label))
            viewer = Path(filename).with_suffix(".html").name
            raw, roi, _, _ = views[scan]
            write_mask_viewer(folder / viewer, row, raw, roi, scores, preds[scan],
                              harmonized_rgb=corrected_views.get(scan))
            links.append((f"examples/{viewer}", f"Adjust plaque threshold · {label}"))
    qualitative_figures(out, leads, modes, views, maps, pca, preds, masks_by_scan, limit)
    example_figure = partial_examples_figure(out, selection, examples, views, maps, pca,
                                             preds, masks_by_scan, modes, limit)
    if example_figure:
        links.append((example_figure, "Selected partial-stage model examples"))
    preview_rows = 2 if corrected_views else 1
    nrows = preview_rows + len(modes)
    for kind in ("attention", "features"):
        fig, axes = plt.subplots(nrows, len(leads), figsize=(11.8, 2.65 * nrows),
                                 squeeze=False, layout="constrained")
        for col, row in enumerate(leads):
            scan = row["scan_id"]
            raw, roi, _, _ = views[scan]
            rgb = np.where(roi[..., None], raw, 255)
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(row["scanner"], weight="bold")
            if corrected_views:
                corrected = np.where(roi[..., None], corrected_views[scan], 255)
                axes[1, col].imshow(corrected)
            for index, mode in enumerate(modes, preview_rows):
                ax = axes[index, col]
                if kind == "attention":
                    ax.imshow(corrected if corrected_views and INPUTS[mode][1] == "harmonized" else rgb)
                    overlay = ax.imshow(np.ma.masked_invalid(maps[scan][mode]), cmap="magma",
                                        vmin=0, vmax=limit, alpha=.86)
                    probability = preds[scan][mode]
                    guess = int(probability.argmax())
                    ax.set_title(f"{STAGE_LABELS[guess]} · {probability[guess]:.0%}", fontsize=9)
                else:
                    ax.imshow(pca[mode, scan], interpolation="nearest")
            for index, ax in enumerate(axes[:, col]):
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col == 0:
                    if index < preview_rows:
                        label = ("Original scan", "Harmonised scan")[index]
                    else:
                        label = row_label(modes[index - preview_rows])
                    ax.set_ylabel(label, fontsize=9)
        if kind == "attention":
            fig.colorbar(overlay, ax=axes[preview_rows:, :], shrink=.75, label="Attention relative to uniform weighting")
            subtitle = "Learned attention · all panels on one scale"
        else:
            subtitle = "Matched patch grid · shared PCA colours within each encoder"
        fig.suptitle(f"{LEAD_SCAN} · observed partial cleaning stage\n{subtitle}", fontsize=13)
        save_figure_png(fig, out / f"{LEAD_SCAN}_{'learned_attention' if kind == 'attention' else 'matched_features'}.png")
    current = {out / path for path, _ in links}
    for role in EXAMPLE_LABELS:
        for scanner in SCANNERS:
            for suffix in ("png", "html"):
                for old_panel in folder.glob(f"{role}_{scanner}_*.{suffix}"):
                    if old_panel not in current:
                        old_panel.unlink()
    return links


def example_summary(entry, separator):
    role = entry["role"]
    if role in ("highest", "lowest"):
        detail = f"Median top probability {entry['median_top_probability']:.1%}"
    elif role in ("most_correct", "least_correct"):
        detail = f"{entry['models_correct']}/{len(entry['model_predictions'])} predicted partial"
    else:
        detail = (f"{entry['correct_confidence_score']:.1%} combined"
                  f"{separator}{entry['models_correct']}/{len(entry['model_predictions'])} correct")
    return EXAMPLE_LABELS[role] + separator + detail


def confidence_examples(rows, modes, probabilities, smoke):
    candidates = []
    for i, row in enumerate(rows):
        if row["stage"] != "partial" or is_lead(row) or (not smoke and row["physical_id"] == LEAD_SPECIMEN):
            continue
        top = {mode: float(probabilities[mode][i].max()) for mode in modes}
        guesses = {mode: STAGES[int(probabilities[mode][i].argmax())] for mode in modes}
        correct = [top[mode] for mode in modes if guesses[mode] == row["stage"]]
        candidates.append(dict(scan_id=row["scan_id"], sample_name=row["sample_name"], scanner=row["scanner"],
                               observed_stage=row["stage"], model_predictions=guesses, top_probabilities=top,
                               models_correct=len(correct), correct_confidence_score=sum(correct) / len(modes),
                               median_top_probability=float(np.median(list(top.values()))),
                               mean_top_probability=float(np.mean(list(top.values())))))

    def confidence_key(row):
        return row["median_top_probability"], row["mean_top_probability"], row["scan_id"]

    examples = []
    if len(candidates) >= 2:
        candidates.sort(key=confidence_key)
        examples = [dict(candidates[-1], role="highest"), dict(candidates[0], role="lowest")]
        used = {row["scan_id"] for row in examples}
        for score, high_role, low_role in (
            ("models_correct", "most_correct", "least_correct"),
            ("correct_confidence_score", "highest_combined", "lowest_combined"),
        ):
            high = max(candidates, key=lambda r: (r[score], r["scan_id"] not in used, *confidence_key(r)))
            used.add(high["scan_id"])
            low = min(candidates, key=lambda r: (r[score], r["scan_id"] in used, *confidence_key(r)))
            used.add(low["scan_id"])
            examples.extend([dict(high, role=high_role), dict(low, role=low_role)])
    return dict(rule="Median of the active models' maximum class probabilities, with mean and scan ID as tie-breakers.",
                correctness_rule="Number of models predicting the observed partial stage. Ties prefer unused "
                                 "scans, then use the confidence ranking in the same direction.",
                combined_rule="Sum of the top probabilities of correct models divided by the number of active "
                              "models. Incorrect predictions contribute zero. Ties use the correct-count ranking's "
                              "tie-breakers.",
                observed_stage_filter="partial", excluded_primary=LEAD_SCAN if smoke else LEAD_SPECIMEN,
                candidates=len(candidates), examples=examples)


def test_outputs(run, harmonized):
    cohort = read_json(run / "cohort.json")
    rows = [row for row in cohort["rows"] if row["role"] == "test"]
    smoke = cohort["smoke"]
    modes = [mode for family in FAMILIES for mode in input_modes(harmonized, family)]
    out = run / ("results_harmonized" if harmonized else "results")
    histories, models, settings, probabilities = {}, {}, {}, {}
    predictions, curves = [], []
    plans = set()
    for mode in modes:
        settings[mode] = mode_settings(run, mode)
        model, history, epoch, plan = read_model(run, mode, settings[mode])
        plans.add(plan)
        if len(plans) > 1:
            raise ValueError("The active heads have different training schedules. Complete training with "
                             "--retrain-heads before generating the report.")
        max_epochs = plan[0]
        histories[mode], models[mode] = (history, epoch, max_epochs), model
        curves.extend(dict(mode=mode, epoch=i, train_loss=float(t), validation_loss=float(v),
                           selected=(i == epoch), max_epochs=max_epochs,
                           early_stopped=(len(history) < max_epochs)) for i, (t, v) in enumerate(history, 1))
        p = softmax(score_bags(load_bags(run, mode, rows, settings[mode]), model), axis=1)
        if not np.isfinite(p).all():
            raise ValueError(f"Invalid test probabilities: {mode}.")
        probabilities[mode] = p
        predictions.extend(dict(mode=mode, scan_id=row["scan_id"], specimen=row["physical_id"],
                                group=row["group"], scanner=row["scanner"], truth=row["stage"],
                                prediction=STAGES[int(values.argmax())],
                                **{f"p_{stage}": float(v) for stage, v in zip(STAGES, values)})
                           for row, values in zip(rows, p))
    return out, rows, smoke, modes, histories, models, settings, probabilities, predictions, curves


def report(run, harmonized=False, select_only=False):
    out, rows, smoke, modes, histories, models, settings, probabilities, predictions, curves = test_outputs(
        run, harmonized)
    out.mkdir(parents=True, exist_ok=True)
    selection = confidence_examples(rows, modes, probabilities, smoke)
    write_json(out / "confidence_examples.json", selection)
    no_extra = "Fewer than two additional test scans have an observed partial stage. Showing the four primary views."
    if select_only:
        if not selection["examples"]:
            print(no_extra, flush=True)
        for entry in selection["examples"]:
            print(f"{entry['scan_id']} · {example_summary(entry, ' · ')}", flush=True)
        return
    matrices, summaries, classes, deltas = metrics(rows, probabilities, modes, smoke)
    write_csv(out / "test_predictions.csv", predictions)
    write_csv(out / "loss_history.csv", curves)
    write_csv(out / "test_metrics.csv", summaries)
    write_csv(out / "per_stage_metrics.csv", classes)
    write_csv(out / "paired_colour_effects.csv", deltas)
    write_json(out / "confusion_counts.json", dict(stage_order=STAGES, matrices=matrices))
    title = "Pipeline smoke check" if smoke else "BioCal3D held-out test"
    loss_figure(out, modes, histories, "Pipeline smoke check" if smoke else "BioCal3D training")
    confusion_files = performance_figures(out, modes, matrices, summaries, deltas, title, smoke)
    roc_figure(out, rows, modes, probabilities, title)
    balanced_bars(out, modes, summaries, title, smoke)
    example_links = illustrations(run, out, rows, modes, settings, models, probabilities, selection)
    links = [("loss_progress.png", "Training and validation loss"),
             ("balanced_accuracy.png", "Balanced accuracy · overall and by scanner"),
             ("balanced_accuracy_bars.png", "Balanced accuracy · horizontal bar chart"),
             ("paired_colour_effects.png", "Paired colour effects"),
             ("roc_curves.png", "One-vs-rest ROC curves by input condition"),
             (f"{LEAD_SCAN}_qualitative_all_configurations.png", f"{LEAD_SCAN} · all configurations by scanner"),
             *confusion_files, (f"{LEAD_SCAN}_learned_attention.png", f"{LEAD_SCAN} · learned attention"),
             (f"{LEAD_SCAN}_matched_features.png", f"{LEAD_SCAN} · matched PCA feature maps"),
             *example_links, ("confidence_examples.json", "Example selections, model votes and ranking rules"),
             ("test_predictions.csv", "Test predictions CSV"), ("test_metrics.csv", "Test metrics CSV"),
             ("per_stage_metrics.csv", "Per-stage metrics CSV"), ("loss_history.csv", "Loss history CSV"),
             ("paired_colour_effects.csv", "Paired colour effects CSV"),
             ("confusion_counts.json", "Confusion matrix counts JSON"), ("../split.csv", "Specimen split CSV")]
    html = "".join(f'<li><a href="{path}">{escape(label)}</a></li>' for path, label in links)
    note = "" if selection["examples"] else f"<p>{no_extra}</p>"
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BioCal3D attention study</title>'
        '<style>body{font:16px system-ui;max-width:850px;margin:3rem auto;padding:0 1rem}li{margin:.55rem 0}</style>'
        f'<h1>{title}</h1><p>{len(rows)} scans from {len({row["physical_id"] for row in rows})} test specimens.</p>'
        f'{note}<ul>{html}</ul></html>', encoding="utf-8")
    print(f"Open the report: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--with-harmonized", action="store_true")
    parser.add_argument("--select-examples", action="store_true")
    args = parser.parse_args()
    try:
        report(args.run, args.with_harmonized, args.select_examples)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
