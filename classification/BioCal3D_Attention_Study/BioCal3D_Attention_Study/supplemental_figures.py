"""Horizontal accuracy bars and one attention–PCA–mask panel per scan."""
import numpy as np
from matplotlib.patches import Patch

from plaque_masks import THRESHOLD, plaque_scores
from spatial import plt, save_figure_png
from study_data import STAGES
from study_protocol import INPUTS

PALETTE = {"gray": "#3a867e", "geometry": "#3a867e", "rgb": "#2c659b", "harmonized": "#c9923c"}
STAGE_NAMES = {"baseline": "Baseline", "partial": "Partial", "clean": "Fully cleaned"}


def balanced_bars(out, modes, summaries, title, smoke):
    scopes = list(dict.fromkeys(r["scope"] for r in summaries))
    lookup = {(r["mode"], r["scope"]): r for r in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    fig.subplots_adjust(left=.14, right=.98, bottom=.18, top=.76, wspace=.12)
    ypos = np.arange(len(scopes))
    for ax, (family, label) in zip(axes, (("dinov3", "DINOv3"), ("concerto", "Concerto"))):
        selected = [mode for mode in modes if INPUTS[mode][0] == family]
        width = .24 if len(selected) == 3 else .30
        for i, mode in enumerate(selected):
            records = [lookup[mode, scope] for scope in scopes]
            offsets = ypos + (i - (len(selected) - 1) / 2) * width
            vals = np.asarray([r["balanced_accuracy"] for r in records]) * 100
            ax.barh(offsets, vals, height=width * .86, color=PALETTE[INPUTS[mode][1]], zorder=2)
            if not smoke:
                low = np.asarray([r["ci95_low"] for r in records]) * 100
                high = np.asarray([r["ci95_high"] for r in records]) * 100
                ax.hlines(offsets, low, high, linewidth=1, color="#233d50", zorder=3)
                ax.vlines(np.concatenate((low, high)), np.tile(offsets - .055, 2),
                          np.tile(offsets + .055, 2), linewidth=.9, color="#233d50", zorder=3)
            for y, value in zip(offsets, vals):
                ax.text(103, y, f"{value:.1f}", va="center", ha="left", fontsize=9, color="#183448")
        ax.set(xlim=(0, 114), ylim=(len(scopes) - .5, -.5), xticks=np.arange(0, 101, 20))
        ax.set_yticks(ypos, scopes)
        ax.set_title(label, color="#183448", weight="bold")
        ax.set_xlabel("Balanced accuracy (%)", color="#183448")
        ax.grid(axis="x", color="#e0e8ed", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#b7c7d2")
        ax.tick_params(axis="y", length=0)
    handles = [Patch(color=PALETTE["gray"], label="DINOv3 grayscale / Concerto geometry only"),
               Patch(color=PALETTE["rgb"], label="DINOv3 RGB / Concerto geometry + RGB")]
    if any(INPUTS[mode][1] == "harmonized" for mode in modes):
        handles.append(Patch(color=PALETTE["harmonized"], label="Harmonised RGB"))
    fig.suptitle(f"{title} · balanced accuracy across scanners", x=.04, y=.97, ha="left",
                 weight="bold", fontsize=16, color="#183448")
    fig.legend(handles=handles, ncols=len(handles), loc="upper left", bbox_to_anchor=(.035, .9),
               frameon=False, fontsize=9)
    note = ("Pipeline check on one test specimen." if smoke else
            f"95% intervals use a paired bootstrap over {summaries[0]['specimens']} test specimens, "
            "stratified by group.")
    fig.text(.04, .035, note, fontsize=9, color="#456478")
    save_figure_png(fig, out / "balanced_accuracy_bars.png")


def scan_panel(path, row, modes, view, attention, features, pred, limit, shape, note, *, harmonized_rgb=None):
    raw, roi, _, _ = view
    original = np.where(roi[..., None], raw, 255)
    fig = plt.figure(figsize=(10.2, 2.3 + 2.0 * len(modes)), layout="constrained")
    grid = fig.add_gridspec(len(modes) + 1, 4, height_ratios=[.9] + [1] * len(modes),
                           width_ratios=[.72, 1, 1, 1])
    first = fig.add_subplot(grid[0, 2])
    first.imshow(original)
    first.set_title("Original scan", fontsize=11, weight="bold")
    first.set_axis_off()
    if harmonized_rgb is not None:
        corrected = np.where(roi[..., None], harmonized_rgb, 255)
        first = fig.add_subplot(grid[0, 1])
        first.imshow(corrected)
        first.set_title("Harmonised scan", fontsize=11, weight="bold")
        first.set_axis_off()
    detail = fig.add_subplot(grid[0, 0])
    detail.text(.05, .5, f"Observed stage\n{STAGE_NAMES[row['stage']]}", va="center", fontsize=11, color="#183448")
    detail.set_axis_off()
    note_ax = fig.add_subplot(grid[0, 3])
    note_ax.text(.03, .5, note, va="center", fontsize=9, color="#456478")
    note_ax.set_axis_off()

    masks, scores, attention_axes = {}, {}, []
    for i, mode in enumerate(modes, 1):
        axes = [fig.add_subplot(grid[i, col]) for col in range(1, 4)]
        attention_axes.append(axes[0])
        values = pred[mode]
        guess = STAGES[int(values.argmax())]
        harmonized = harmonized_rgb is not None and INPUTS[mode][1] == "harmonized"
        background = corrected if harmonized else original
        axes[0].imshow(background)
        overlay = axes[0].imshow(np.ma.masked_invalid(attention[mode]), cmap="magma", vmin=0, vmax=limit, alpha=.86)
        label_ax = fig.add_subplot(grid[i, 0])
        name, variant = INPUTS[mode][2].split(" · ", 1)
        label_ax.text(.05, .5, f"{name}\n{variant}\n\n{STAGE_NAMES[guess]} · {max(values):.0%}",
                      va="center", fontsize=9, color="#183448")
        label_ax.set_axis_off()
        axes[1].imshow(features[mode, row["scan_id"]], interpolation="nearest")
        scores[mode] = plaque_scores(attention[mode], harmonized_rgb if harmonized else raw, roi, shape, guess,
                                     INPUTS[mode][1] == "geometry")
        candidate = scores[mode] >= THRESHOLD
        masks[f"{mode}_proposed_plaque"] = candidate
        axes[2].imshow(background)
        if candidate.any():
            tint = np.zeros((*candidate.shape, 4), dtype=np.float32)
            tint[candidate] = (0, .78, .9, .62)
            axes[2].imshow(tint)
        status = "Clean prediction" if guess == "clean" else f"{100 * candidate.sum() / roi.sum():.1f}% of scan ROI"
        axes[2].set_xlabel(status, fontsize=8)
        if i == 1:
            for ax, title in zip(axes, ("Learned attention", "Matched PCA", "Proposed plaque mask")):
                ax.set_title(title, fontsize=10, weight="bold")
        for ax in axes:
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.colorbar(overlay, ax=attention_axes, orientation="horizontal", fraction=.025, pad=.018,
                 label="Attention relative to uniform weighting")
    fig.suptitle(f"{row['sample_name']} · {row['scanner']}", fontsize=15, weight="bold")
    save_figure_png(fig, path)
    return masks, scores
