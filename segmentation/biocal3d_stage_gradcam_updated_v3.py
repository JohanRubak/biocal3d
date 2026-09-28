"""Train a stage-classification head on frozen spatial DINOv2 features.

Place beside biocal3d_prepare_dino.py. Requires torch, numpy, pandas,
scikit-learn, pillow, matplotlib. First regenerate maps and DINO features with
the updated preparation script (including map hashes).

Examples
--------
python biocal3d_stage_gradcam.py --live
python biocal3d_stage_gradcam.py --step

Interpretation
--------------
Image-level labels are baseline / partial / clean. These are known experimental
stages used as WEAK supervision for plaque localization, not pixel labels.
The DINOv2 encoder remains frozen. Only a small spatial classification head is
trained. Grad-CAM is computed at that head's hidden spatial feature layer, not
at DINO attention maps.

No test-set inference is performed. Best epoch is selected by weighted
validation cross entropy. With only a few independent validation specimens,
all validation performance remains exploratory.
"""

from pathlib import Path
import argparse
import hashlib
import json
import random
import copy
import re
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from biocal3d_prepare_dino import DATA_ROOT, LivePreview


CLASSES = ["baseline", "partial", "clean"]
STAGE_ORDER = {name: i for i, name in enumerate(CLASSES)}


# ============================================================
# DATASET + MODEL
# ============================================================

class Features(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)
        self.items = []

        for row in self.rows.to_dict("records"):
            folder = Path(row["map_dir"])

            with np.load(folder / "dino_features.npz") as z:
                digest = hashlib.sha256((folder / "maps.npz").read_bytes()).hexdigest()

                if "map_sha256" not in z or str(z["map_sha256"]) != digest:
                    raise ValueError(
                        f"Stale or unverified DINO features: {folder}. "
                        "Rerun preparation --feature-only."
                    )

                f = z["features"].astype(np.float32).transpose(2, 0, 1)
                mask = z["patch_valid"].astype(np.float32)[None]

            if mask.sum() == 0:
                raise ValueError(f"No valid feature patches: {folder}")

            self.items.append(
                (
                    torch.from_numpy(f),
                    torch.from_numpy(mask),
                    CLASSES.index(row["stage"]),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class StageHead(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(hidden, len(CLASSES), 1)

    def forward(self, x, mask, return_features=False):
        activation = self.spatial(x)
        spatial_logits = self.classifier(activation)
        logits = (spatial_logits * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        return (logits, activation) if return_features else logits


def gradcam(model, x, mask, target):
    """Grad-CAM at StageHead.spatial output for one target class."""
    model.zero_grad(set_to_none=True)
    logits, activation = model(x, mask, return_features=True)

    gradient = torch.autograd.grad(logits[:, target].sum(), activation)[0]
    weights = (gradient * mask).sum((2, 3), keepdim=True) / mask.sum(
        (2, 3), keepdim=True
    ).clamp_min(1)

    cam = torch.relu((weights * activation).sum(1, keepdim=True)) * mask
    raw_max = cam.amax().item()
    cam = cam / cam.amax((2, 3), keepdim=True).clamp_min(1e-12)

    return cam.detach()[0, 0].cpu().numpy(), raw_max


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device, class_weights):
    """Evaluate with the SAME class-weighted CE definition used for training.

    The numerator and denominator are accumulated globally so the returned loss
    is directly comparable between train and validation curves.
    """
    ys, preds, probabilities = [], [], []
    weighted_loss_numerator = 0.0
    weighted_loss_denominator = 0.0

    model.eval()
    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(x, mask)

            weighted_loss_numerator += nn.functional.cross_entropy(
                logits,
                y,
                weight=class_weights,
                reduction="sum",
            ).item()
            weighted_loss_denominator += class_weights[y].sum().item()

            ys.extend(y.cpu().tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())

    weighted_loss = weighted_loss_numerator / max(weighted_loss_denominator, 1e-12)

    return (
        weighted_loss,
        balanced_accuracy_score(ys, preds),
        ys,
        preds,
        probabilities,
    )


# ============================================================
# CORE FIGURES
# ============================================================

def plot_training_progress(history, best_epoch, out_path):
    """Loss curves + validation balanced accuracy."""
    hist = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(
        hist["epoch"],
        hist["weighted_train_loss"],
        marker="o",
        markersize=3,
        label="Weighted train loss",
    )
    axes[0].plot(
        hist["epoch"],
        hist["weighted_val_loss"],
        marker="o",
        markersize=3,
        label="Weighted validation loss",
    )
    axes[0].axvline(best_epoch, linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted cross-entropy")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        hist["epoch"],
        hist["val_balanced_accuracy"],
        marker="o",
        markersize=3,
        label="Validation balanced accuracy",
    )
    axes[1].axvline(best_epoch, linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation performance")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Stage-classification training progress", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_figure(ys, preds, split_name, out_path):
    """Confusion matrix with counts and row-normalized percentages."""
    cm = confusion_matrix(ys, preds, labels=list(range(len(CLASSES))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")

    ax.set_xticks(range(len(CLASSES)), labels=[c.title() for c in CLASSES])
    ax.set_yticks(range(len(CLASSES)), labels=[c.title() for c in CLASSES])
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(f"{split_name.title()} confusion matrix")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            pct = 100.0 * cm_norm[i, j]
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of true class")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def _extract_group_number(row):
    """Best-effort extraction of BioCal group number from manifest fields."""
    for key in ("group", "group_id", "group_name"):
        if key in row and pd.notna(row[key]):
            text = str(row[key])
            match = re.search(r"(?:BC)?G?\s*([1-5])", text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

    for key in ("sample_name", "physical_id"):
        if key in row and pd.notna(row[key]):
            text = str(row[key])
            match = re.search(r"BCG\s*([1-5])", text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
            match = re.search(r"\bG\s*([1-5])\b", text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

    return None


def partial_is_relevant(row):
    """True for groups that contain a partial-cleaning stage (G3-G5)."""
    if str(row.get("stage", "")).lower() == "partial":
        return True

    group_number = _extract_group_number(row)
    return group_number is not None and group_number >= 3


def make_cam_overlay(rgb, full_cam, valid, max_alpha=0.65):
    """Overlay heat only where CAM evidence is present.

    Previous versions used a fixed alpha everywhere inside the ROI. That made a
    zero/near-zero CAM look like a dark purple mask. Here alpha is proportional
    to CAM intensity, so CAM=0 leaves the original RGB unchanged.
    """
    heat = colormaps["inferno"](full_cam)[..., :3]
    alpha = np.clip(max_alpha * full_cam, 0.0, max_alpha)[..., None]

    overlay = rgb.copy()
    overlay[valid] = (
        (1.0 - alpha[valid]) * rgb[valid]
        + alpha[valid] * heat[valid]
    )
    return overlay


def save_comparison_figure(folder, rgb, overlays_by_class, row, predicted_stage, probs):
    """Save RGB + stage-relevant Grad-CAM panels without title overlap."""
    include_partial = partial_is_relevant(row)

    classes_to_show = ["baseline", "clean"]
    if include_partial:
        classes_to_show = ["baseline", "partial", "clean"]

    images = [rgb] + [overlays_by_class[name] for name in classes_to_show]
    titles = ["RGB"] + [
        f"{name.title()}-class Grad-CAM\np={probs[CLASSES.index(name)]:.3f}"
        for name in classes_to_show
    ]

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.75))
    axes = np.atleast_1d(axes)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.axis("off")

    group_number = _extract_group_number(row)
    group_text = f"G{group_number} | " if group_number is not None else ""
    caption = (
        f"{row['scanner']} {row['sample_name']} | {group_text}"
        f"true={row['stage']} | predicted={predicted_stage}"
    )

    # Put scan metadata below the image panels. This removes all competition
    # between the figure-level caption and the two-line panel titles.
    fig.text(
        0.5,
        0.025,
        caption,
        ha="center",
        va="bottom",
        fontsize=11.5,
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        bottom=0.105,
        top=0.88,
        wspace=0.05,
    )
    fig.savefig(folder / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return images, titles, caption


# ============================================================
# ADDITIONAL ANALYSIS FIGURES
# ============================================================

def add_prediction_diagnostics(predictions):
    """Add true-class probability and simple uncertainty diagnostics."""
    df = predictions.copy()

    true_probs = []
    confidences = []
    margins = []
    entropies = []
    correct = []

    for row in df.to_dict("records"):
        probs = np.array([row[f"p_{c}"] for c in CLASSES], dtype=float)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum()

        true_probs.append(float(probs[CLASSES.index(row["stage"])]))
        confidences.append(float(probs.max()))
        sorted_probs = np.sort(probs)[::-1]
        margins.append(float(sorted_probs[0] - sorted_probs[1]))
        entropies.append(float(-(probs * np.log(probs)).sum()))
        correct.append(bool(row["stage"] == row["predicted_stage"]))

    df["true_class_probability"] = true_probs
    df["prediction_confidence"] = confidences
    df["probability_margin"] = margins
    df["prediction_entropy"] = entropies
    df["correct"] = correct

    return df


def plot_validation_probability_heatmap(predictions, out_path, table_path):
    """Cross-scanner heatmap of probability assigned to the correct stage."""
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        return

    val["stage_order"] = val["stage"].map(STAGE_ORDER)
    val["group_number"] = val.apply(lambda r: _extract_group_number(r), axis=1)
    val["group_number"] = val["group_number"].fillna(99)

    # Stable row order: group -> specimen -> biological stage.
    row_order_df = (
        val[["physical_id", "stage", "group_number", "stage_order"]]
        .drop_duplicates()
        .sort_values(["group_number", "physical_id", "stage_order"])
    )
    row_keys = list(zip(row_order_df["physical_id"], row_order_df["stage"]))

    scanners = sorted(val["scanner"].unique())

    prob_lookup = {
        (r.physical_id, r.stage, r.scanner): r.true_class_probability
        for r in val.itertuples()
    }
    correct_lookup = {
        (r.physical_id, r.stage, r.scanner): bool(r.correct)
        for r in val.itertuples()
    }

    matrix = np.full((len(row_keys), len(scanners)), np.nan, dtype=float)
    for i, key in enumerate(row_keys):
        for j, scanner in enumerate(scanners):
            matrix[i, j] = prob_lookup.get((key[0], key[1], scanner), np.nan)

    export = pd.DataFrame(
        matrix,
        index=[f"{pid} | {stage}" for pid, stage in row_keys],
        columns=scanners,
    )
    export.to_csv(table_path)

    fig_height = max(4.6, 0.46 * len(row_keys) + 1.8)
    fig_width = max(7.0, 1.45 * len(scanners) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xticks(range(len(scanners)), labels=scanners, rotation=25, ha="right")
    ax.set_yticks(
        range(len(row_keys)),
        labels=[f"{pid} | {stage}" for pid, stage in row_keys],
    )
    ax.set_xlabel("Scanner")
    ax.set_ylabel("Physical specimen and true stage")
    ax.set_title("Validation: probability assigned to the true stage")

    for i, (pid, stage) in enumerate(row_keys):
        for j, scanner in enumerate(scanners):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            is_correct = correct_lookup.get((pid, stage, scanner), True)
            suffix = "" if is_correct else "\nX"
            text_color = "white" if value < 0.35 or value > 0.78 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}{suffix}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold" if not is_correct else "normal",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(true stage)")
    fig.text(0.5, 0.01, "X = misclassified scan", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_true_probability_distribution(predictions, out_path, summary_path):
    """Distribution of P(true stage), with individual scanner observations."""
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        return

    stage_values = [
        val.loc[val["stage"] == stage, "true_class_probability"].to_numpy()
        for stage in CLASSES
    ]

    summary_rows = []
    for stage, values in zip(CLASSES, stage_values):
        if len(values) == 0:
            continue
        summary_rows.append(
            dict(
                stage=stage,
                n_scans=len(values),
                mean_true_probability=float(np.mean(values)),
                median_true_probability=float(np.median(values)),
                min_true_probability=float(np.min(values)),
                max_true_probability=float(np.max(values)),
            )
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    positions = np.arange(1, len(CLASSES) + 1)

    ax.boxplot(
        stage_values,
        positions=positions,
        widths=0.48,
        showfliers=False,
        labels=[c.title() for c in CLASSES],
    )

    scanners = sorted(val["scanner"].unique())
    offsets = np.linspace(-0.18, 0.18, max(len(scanners), 1))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scanner_color = {
        scanner: cycle[i % len(cycle)] for i, scanner in enumerate(scanners)
    }

    for scanner_idx, scanner in enumerate(scanners):
        sub = val[val["scanner"] == scanner]
        for stage_idx, stage in enumerate(CLASSES):
            points = sub[sub["stage"] == stage]
            if points.empty:
                continue

            x = np.full(len(points), positions[stage_idx] + offsets[scanner_idx])
            correct = points["correct"].to_numpy(dtype=bool)
            y = points["true_class_probability"].to_numpy(dtype=float)

            # Correct observations.
            if np.any(correct):
                ax.scatter(
                    x[correct],
                    y[correct],
                    s=42,
                    alpha=0.85,
                    color=scanner_color[scanner],
                    label=scanner if stage_idx == 0 else None,
                )

            # Misclassifications get an X marker and are annotated below.
            if np.any(~correct):
                ax.scatter(
                    x[~correct],
                    y[~correct],
                    s=70,
                    marker="x",
                    linewidths=2,
                    color=scanner_color[scanner],
                )
                for xi, yi, sample in zip(
                    x[~correct],
                    y[~correct],
                    points.loc[~correct, "sample_name"],
                ):
                    ax.annotate(
                        str(sample),
                        (xi, yi),
                        xytext=(4, -10),
                        textcoords="offset points",
                        fontsize=7.5,
                        rotation=20,
                    )

    ax.axhline(1.0 / len(CLASSES), linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("Probability assigned to the true stage")
    ax.set_xlabel("True experimental stage")
    ax.set_title("Validation confidence by true stage")
    ax.grid(axis="y", alpha=0.2)

    # Build scanner legend explicitly so duplicate point labels never matter.
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            color=scanner_color[scanner], label=scanner
        )
        for scanner in scanners
    ]
    handles.append(Line2D([0], [0], marker="x", linestyle="", label="Misclassified"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_saved_cam_overlay(row, out_root, class_name, max_alpha):
    """Load RGB + saved normalized CAM and recreate the intensity-weighted overlay."""
    with np.load(Path(row["map_dir"]) / "maps.npz") as z:
        rgb = z["rgb"]
        valid = z["valid"]

    cam_path = (
        out_root
        / "gradcam"
        / row["split"]
        / row["scanner"]
        / row["sample_name"]
        / f"gradcam_{class_name}.npz"
    )
    with np.load(cam_path) as z:
        full_cam = z["normalized_cam"]

    return rgb, make_cam_overlay(rgb, full_cam, valid, max_alpha=max_alpha)


def plot_uncertain_error_montage(predictions, out_root, out_path, table_path, n_total, max_alpha):
    """Show all validation errors + lowest-confidence correct scans.

    Each row is one scan. Columns are RGB, baseline CAM, partial CAM, clean CAM.
    For G1-G2 the partial panel is deliberately blank because that stage is not
    part of the biological design for those groups.
    """
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        return

    errors = val[~val["correct"]].sort_values("true_class_probability")
    correct = val[val["correct"]].sort_values("true_class_probability")

    requested = max(int(n_total), len(errors))
    n_needed = max(0, requested - len(errors))
    selected = pd.concat([errors, correct.head(n_needed)], ignore_index=True)
    selected = selected.sort_values(
        ["correct", "true_class_probability"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected.to_csv(table_path, index=False)
    if selected.empty:
        return

    nrows = len(selected)
    ncols = 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.5, max(3.2 * nrows, 4.0)),
        squeeze=False,
    )

    column_titles = ["RGB", "Baseline Grad-CAM", "Partial Grad-CAM", "Clean Grad-CAM"]
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=11, pad=8)

    for i, row in selected.iterrows():
        row_dict = row.to_dict()

        # RGB is identical regardless of class; use baseline load to retrieve it.
        rgb, baseline_overlay = _load_saved_cam_overlay(
            row_dict, out_root, "baseline", max_alpha
        )
        _, clean_overlay = _load_saved_cam_overlay(
            row_dict, out_root, "clean", max_alpha
        )

        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 1].imshow(np.clip(baseline_overlay, 0, 1))

        if partial_is_relevant(row_dict):
            _, partial_overlay = _load_saved_cam_overlay(
                row_dict, out_root, "partial", max_alpha
            )
            axes[i, 2].imshow(np.clip(partial_overlay, 0, 1))
        else:
            axes[i, 2].text(
                0.5,
                0.5,
                "Not applicable\n(G1-G2)",
                ha="center",
                va="center",
                transform=axes[i, 2].transAxes,
                fontsize=10,
            )

        axes[i, 3].imshow(np.clip(clean_overlay, 0, 1))

        status = "ERROR" if not row["correct"] else "uncertain correct"
        row_label = (
            f"{row['scanner']} {row['sample_name']}\n"
            f"true={row['stage']} → pred={row['predicted_stage']}\n"
            f"P(true)={row['true_class_probability']:.3f} | {status}"
        )

        for j in range(ncols):
            axes[i, j].axis("off")

        # ax.text remains visible with axis('off'), unlike a conventional ylabel.
        axes[i, 0].text(
            -0.055,
            0.5,
            row_label,
            transform=axes[i, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
        )

    fig.suptitle(
        "Validation errors and lowest-confidence correct classifications",
        fontsize=14,
        y=0.998,
    )
    fig.tight_layout(rect=[0.08, 0.01, 1, 0.98])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dino_pca(train_ds, val_ds, out_path, table_path):
    """PCA of masked-mean frozen DINO features.

    Color = biological stage, marker = scanner, black edge = validation scan.
    This checks whether frozen DINO representations organize primarily by stage,
    scanner, or neither before the trained classification head is applied.
    """
    vectors = []
    meta_rows = []

    for split, ds in [("train", train_ds), ("val", val_ds)]:
        for i, row in enumerate(ds.rows.to_dict("records")):
            feat, mask, _ = ds[i]
            feat_np = feat.numpy().astype(np.float64)
            mask_np = mask.numpy().astype(np.float64)

            denom = mask_np.sum()
            pooled = (feat_np * mask_np).sum(axis=(1, 2)) / max(denom, 1e-12)

            # L2-normalize each pooled scan embedding before PCA so variation in
            # vector magnitude does not dominate the visualization.
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            vectors.append(pooled)
            meta_rows.append({**row, "split": split})

    if len(vectors) < 3:
        return

    x = np.vstack(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)

    scores = pd.DataFrame(meta_rows)
    scores["PC1"] = coords[:, 0]
    scores["PC2"] = coords[:, 1]
    scores.to_csv(table_path, index=False)

    scanners = sorted(scores["scanner"].unique())
    markers = ["o", "s", "^", "D", "P", "v", "X", "<", ">"]
    marker_map = {scanner: markers[i % len(markers)] for i, scanner in enumerate(scanners)}

    # Default matplotlib color cycle, fixed by stage.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stage_color = {stage: cycle[i % len(cycle)] for i, stage in enumerate(CLASSES)}

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    for stage in CLASSES:
        for scanner in scanners:
            sub = scores[(scores["stage"] == stage) & (scores["scanner"] == scanner)]
            if sub.empty:
                continue

            train_sub = sub[sub["split"] == "train"]
            val_sub = sub[sub["split"] == "val"]

            if not train_sub.empty:
                ax.scatter(
                    train_sub["PC1"],
                    train_sub["PC2"],
                    c=stage_color[stage],
                    marker=marker_map[scanner],
                    s=42,
                    alpha=0.68,
                    linewidths=0.3,
                )

            if not val_sub.empty:
                ax.scatter(
                    val_sub["PC1"],
                    val_sub["PC2"],
                    c=stage_color[stage],
                    marker=marker_map[scanner],
                    s=72,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=1.2,
                )

    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("Frozen DINOv2 pooled-feature PCA")
    ax.grid(alpha=0.18)

    stage_handles = [
        Patch(facecolor=stage_color[stage], label=stage.title()) for stage in CLASSES
    ]
    scanner_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[scanner],
            linestyle="",
            color="black",
            label=scanner,
            markersize=7,
        )
        for scanner in scanners
    ]
    split_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="gray", label="Train", markersize=6),
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor="gray",
            markeredgecolor="black", markeredgewidth=1.2, color="gray",
            label="Validation", markersize=7,
        ),
    ]

    legend1 = ax.legend(handles=stage_handles, title="Stage", loc="upper right")
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=scanner_handles, title="Scanner", loc="lower right")
    ax.add_artist(legend2)
    ax.legend(handles=split_handles, title="Split", loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_class_metrics(predictions, out_path):
    """Save per-class validation precision/recall/F1 without overstating certainty."""
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        return

    y_true = val["stage"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    y_pred = val["predicted_stage"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )

    rows = []
    for i, stage in enumerate(CLASSES):
        rows.append(
            dict(
                stage=stage,
                precision=precision[i],
                recall=recall[i],
                f1=f1[i],
                support_scans=int(support[i]),
                note="Scan-level; repeated scanners are not independent specimens",
            )
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DATA_ROOT / "_dino_preparation_80_10_10",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--live", action="store_true")
    p.add_argument("--step", action="store_true")
    p.add_argument(
        "--cam-alpha",
        type=float,
        default=0.65,
        help="Maximum Grad-CAM overlay opacity. Actual alpha is CAM intensity times this value.",
    )
    p.add_argument(
        "--montage-n",
        type=int,
        default=6,
        help="Target number of validation scans in the error/uncertainty montage. All errors are always included.",
    )
    args = p.parse_args()

    if min(args.epochs, args.patience, args.batch_size, args.montage_n) <= 0 or args.lr <= 0:
        p.error("Epochs, patience, batch size, montage-n and learning rate must be positive")
    if not (0.0 <= args.cam_alpha <= 1.0):
        p.error("--cam-alpha must be between 0 and 1")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    manifest = pd.read_csv(args.root / "scan_manifest.csv")

    if manifest.groupby("physical_id").split.nunique().max() != 1:
        raise ValueError("Specimen leakage across splits")

    trainrows = manifest[manifest.split == "train"].copy()
    valrows = manifest[manifest.split == "val"].copy()

    for name, rows in [("train", trainrows), ("val", valrows)]:
        if set(rows.stage) != set(CLASSES):
            raise ValueError(
                f"{name} must contain all three stages; inspect missing scans before proceeding"
            )

    out = args.root / "stage_classification"
    out.mkdir(exist_ok=True)
    analysis_figures = out / "analysis_figures"
    analysis_tables = out / "analysis_tables"
    analysis_figures.mkdir(exist_ok=True)
    analysis_tables.mkdir(exist_ok=True)

    train = Features(trainrows)
    val = Features(valrows)

    trainloader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valloader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    channels = train[0][0].shape[0]
    model = StageHead(channels).to(device)

    counts = np.bincount([item[2] for item in train.items], minlength=len(CLASSES))
    weights = torch.tensor(
        len(train) / (len(CLASSES) * counts),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    best = float("inf")
    best_epoch = 0
    state = None
    history = []

    print(
        f"Train: {len(train)} scans / {trainrows.physical_id.nunique()} specimens. "
        f"Validation: {len(val)} scans / {valrows.physical_id.nunique()} specimens. "
        f"Device: {device}",
        flush=True,
    )
    print(
        "Training head only. Encoder frozen. Stages are not pixel-level plaque labels.",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_num = 0.0
        train_loss_den = 0.0

        for x, mask, y in trainloader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Reconstruct the weighted numerator/denominator for a globally
            # comparable epoch loss (same definition as validation loss).
            batch_weight_sum = weights[y].sum().item()
            train_loss_num += loss.item() * batch_weight_sum
            train_loss_den += batch_weight_sum

        train_loss = train_loss_num / max(train_loss_den, 1e-12)
        val_loss, ba, _, _, _ = evaluate(model, valloader, device, weights)

        history.append(
            dict(
                epoch=epoch,
                weighted_train_loss=train_loss,
                weighted_val_loss=val_loss,
                # Kept for backwards compatibility with older analysis scripts.
                val_loss=val_loss,
                val_balanced_accuracy=ba,
            )
        )
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}: "
            f"weighted train loss={train_loss:.4f}, "
            f"weighted val loss={val_loss:.4f}, "
            f"val balanced accuracy={ba:.3f}",
            flush=True,
        )

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            state = copy.deepcopy(model.state_dict())

            torch.save(
                dict(
                    state_dict=state,
                    channels=channels,
                    classes=CLASSES,
                    best_epoch=epoch,
                ),
                out / "best_stage_head.pt",
            )

        if epoch - best_epoch >= args.patience:
            print("Early stopping.", flush=True)
            break

    if state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    plot_training_progress(
        history,
        best_epoch,
        out / "training_progress.png",
    )
    print(f"Saved training figure: {out / 'training_progress.png'}", flush=True)

    model.load_state_dict(state)
    model.eval()

    viewer = LivePreview(args.live or args.step, args.step)
    allpred = []

    try:
        for split, ds in [("train", train), ("val", val)]:
            loss, ba, ys, preds, probs = evaluate(
                model,
                DataLoader(ds, batch_size=args.batch_size, shuffle=False),
                device,
                weights,
            )

            cm = confusion_matrix(ys, preds, labels=list(range(len(CLASSES))))
            pd.DataFrame(
                cm,
                index=CLASSES,
                columns=CLASSES,
            ).to_csv(out / f"{split}_confusion_matrix.csv")

            plot_confusion_matrix_figure(
                ys,
                preds,
                split,
                out / f"{split}_confusion_matrix.png",
            )
            print(
                f"Saved {split} confusion matrix figure: {out / f'{split}_confusion_matrix.png'}",
                flush=True,
            )

            for i, row in enumerate(ds.rows.to_dict("records")):
                x, mask, y = ds[i]
                x = x[None].to(device)
                mask = mask[None].to(device)

                folder = out / "gradcam" / split / row["scanner"] / row["sample_name"]
                folder.mkdir(parents=True, exist_ok=True)

                with np.load(Path(row["map_dir"]) / "maps.npz") as z:
                    rgb = z["rgb"]
                    valid = z["valid"]

                overlays_by_class = {}
                record = {
                    **row,
                    "predicted_stage": CLASSES[preds[i]],
                }

                for c, name in enumerate(CLASSES):
                    cam, raw_max = gradcam(model, x, mask, c)

                    full = np.array(
                        Image.fromarray(cam).resize(
                            (rgb.shape[1], rgb.shape[0]),
                            Image.Resampling.BILINEAR,
                        ),
                        dtype=np.float32,
                        copy=True,
                    )
                    full[~valid] = 0

                    # CHANGED: CAM-dependent alpha instead of fixed alpha.
                    overlay = make_cam_overlay(
                        rgb,
                        full,
                        valid,
                        max_alpha=args.cam_alpha,
                    )
                    overlays_by_class[name] = overlay

                    np.savez_compressed(
                        folder / f"gradcam_{name}.npz",
                        normalized_cam=full,
                        raw_positive_max=raw_max,
                    )
                    Image.fromarray(
                        np.uint8(np.clip(overlay, 0, 1) * 255)
                    ).save(folder / f"gradcam_{name}.png")

                    record[f"p_{name}"] = probs[i][c]
                    record[f"cam_positive_max_{name}"] = raw_max

                preview_images, preview_titles, caption = save_comparison_figure(
                    folder=folder,
                    rgb=rgb,
                    overlays_by_class=overlays_by_class,
                    row=row,
                    predicted_stage=CLASSES[preds[i]],
                    probs=probs[i],
                )

                print(
                    f"[Grad-CAM {split} {i + 1}/{len(ds)}] {caption}",
                    flush=True,
                )

                viewer.show(preview_images, preview_titles, caption)
                allpred.append(record)

        # --------------------------------------------------------
        # Prediction table + uncertainty diagnostics
        # --------------------------------------------------------
        predictions = add_prediction_diagnostics(pd.DataFrame(allpred))
        predictions.to_csv(out / "stage_predictions.csv", index=False)

        # --------------------------------------------------------
        # Scanner-level summary (scan-level; repeated specimens)
        # --------------------------------------------------------
        metrics = []
        for keys, group in predictions.groupby(["split", "scanner"]):
            metrics.append(
                dict(
                    split=keys[0],
                    scanner=keys[1],
                    n_scans=len(group),
                    n_specimens=group["physical_id"].nunique(),
                    accuracy=accuracy_score(group.stage, group.predicted_stage),
                    balanced_accuracy=balanced_accuracy_score(
                        group.stage,
                        group.predicted_stage,
                    ),
                    mean_true_class_probability=group["true_class_probability"].mean(),
                )
            )

        pd.DataFrame(metrics).to_csv(
            out / "metrics_by_scanner.csv",
            index=False,
        )

        # --------------------------------------------------------
        # NEW ANALYSIS FIGURES/TABLES
        # --------------------------------------------------------
        plot_validation_probability_heatmap(
            predictions,
            analysis_figures / "validation_true_stage_probability_heatmap.png",
            analysis_tables / "validation_true_stage_probability_matrix.csv",
        )

        plot_validation_true_probability_distribution(
            predictions,
            analysis_figures / "validation_true_probability_by_stage.png",
            analysis_tables / "validation_true_probability_summary.csv",
        )

        plot_uncertain_error_montage(
            predictions,
            out,
            analysis_figures / "validation_errors_and_uncertain_examples.png",
            analysis_tables / "validation_errors_and_uncertain_examples.csv",
            n_total=args.montage_n,
            max_alpha=args.cam_alpha,
        )

        plot_dino_pca(
            train,
            val,
            analysis_figures / "dino_pca_stage_scanner.png",
            analysis_tables / "dino_pca_scores.csv",
        )

        save_class_metrics(
            predictions,
            analysis_tables / "validation_class_metrics.csv",
        )

        print(f"Saved additional figures to: {analysis_figures}", flush=True)
        print(f"Saved additional tables to: {analysis_tables}", flush=True)

        (out / "run.json").write_text(
            json.dumps(
                dict(
                    classes=CLASSES,
                    best_epoch=best_epoch,
                    best_val_loss=best,
                    best_weighted_val_loss=best,
                    loss_definition=(
                        "Class-weighted cross entropy for both training and validation; "
                        "global weighted numerator divided by global target-weight sum."
                    ),
                    encoder="frozen cached dinov2_vits14_reg",
                    target_layer="StageHead.spatial ReLU output",
                    device=device,
                    interpretation=(
                        "Stage attribution, not plaque segmentation. CAM normalized per "
                        "image/class; intensity is not classifier confidence."
                    ),
                    cam_overlay=(
                        f"Inferno heat overlay with alpha = normalized_CAM * {args.cam_alpha:.3f}; "
                        "zero CAM leaves RGB unchanged."
                    ),
                    comparison_figures=(
                        "G1-G2: RGB + baseline + clean Grad-CAM. "
                        "G3-G5: RGB + baseline + partial + clean Grad-CAM. "
                        "Scan metadata is placed below panels to prevent title overlap."
                    ),
                    additional_figures=[
                        "validation_true_stage_probability_heatmap.png",
                        "validation_true_probability_by_stage.png",
                        "validation_errors_and_uncertain_examples.png",
                        "dino_pca_stage_scanner.png",
                    ],
                    dino_pca=(
                        "Masked spatial mean of frozen DINO features, L2-normalized per scan, "
                        "then PCA. Color encodes stage, marker encodes scanner, black edge "
                        "marks validation observations."
                    ),
                    validation_warning=(
                        "Metrics are scan-level and repeated scanners of the same physical "
                        "specimen are not independent observations."
                    ),
                    test_used=False,
                ),
                indent=2,
            )
        )

    finally:
        viewer.close()

    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
