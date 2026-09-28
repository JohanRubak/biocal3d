"""Train a stage-classification head on frozen spatial DINOv2 features.

Place beside biocal3d_prepare_dino.py. Requires torch, numpy, pandas,
scikit-learn, pillow, matplotlib. First regenerate maps and DINO features with
updated preparation script (including map hashes).

python biocal3d_stage_gradcam.py --live
python biocal3d_stage_gradcam.py --step

Image-level labels: baseline / partial / clean. These are known experimental
stages used as WEAK supervision for plaque localization, not pixel labels.
Encoder remains frozen. Only a small spatial classification head is trained.
Grad-CAM is computed at that head's hidden spatial feature layer, not attention.
No test-set inference. Best epoch selected by validation cross entropy.
With 4 validation specimens, results are exploratory and uncertain.
"""

from pathlib import Path
import argparse
import hashlib
import json
import random
import copy
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from biocal3d_prepare_dino import DATA_ROOT, LivePreview


CLASSES = ["baseline", "partial", "clean"]


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
    # Frozen encoder features are constants; head parameters create autograd graph.
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


def evaluate(model, loader, device):
    ys, preds, probabilities = [], [], []
    loss_sum = 0.0

    model.eval()
    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(x, mask)
            loss_sum += nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()

            ys.extend(y.cpu().tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())

    return (
        loss_sum / len(ys),
        balanced_accuracy_score(ys, preds),
        ys,
        preds,
        probabilities,
    )


def plot_training_progress(history, best_epoch, out_path):
    """Save one figure containing loss curves and validation balanced accuracy."""
    hist = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Loss panel
    axes[0].plot(
        hist["epoch"],
        hist["weighted_train_loss"],
        marker="o",
        markersize=3,
        label="Weighted train loss",
    )
    axes[0].plot(
        hist["epoch"],
        hist["val_loss"],
        marker="o",
        markersize=3,
        label="Validation loss",
    )
    axes[0].axvline(best_epoch, linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Balanced-accuracy panel
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

    fig.suptitle("Stage-classification training progress")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_figure(ys, preds, split_name, out_path):
    """Save confusion matrix with counts and row-normalized percentages."""
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


def _extract_group_number(row):
    """Best-effort extraction of BioCal group number from manifest fields."""
    # First try explicit group-like columns if present.
    for key in ("group", "group_id", "group_name"):
        if key in row and pd.notna(row[key]):
            text = str(row[key])
            match = re.search(r"(?:BC)?G?\s*([1-5])", text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

    # Then infer from names such as BCG3T1-4.
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
    """Return True for groups that contain a partial-cleaning stage (G3-G5)."""
    # A partial-labelled scan is always from a partial-capable group.
    if str(row.get("stage", "")).lower() == "partial":
        return True

    group_number = _extract_group_number(row)
    return group_number is not None and group_number >= 3


def save_comparison_figure(folder, rgb, overlays_by_class, row, predicted_stage, probs):
    """Save RGB + relevant Grad-CAM class overlays for one scan."""
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
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.8))
    axes = np.atleast_1d(axes)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    group_number = _extract_group_number(row)
    group_text = f"G{group_number} | " if group_number is not None else ""
    caption = (
        f"{row['scanner']} {row['sample_name']} | {group_text}"
        f"true={row['stage']} | predicted={predicted_stage}"
    )

    fig.suptitle(caption, fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return images, titles, caption


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
    args = p.parse_args()

    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.lr <= 0:
        p.error("Epochs, patience, batch size and learning rate must be positive")

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
                f"{name} must contain all three stages; "
                "inspect missing scans before proceeding"
            )

    out = args.root / "stage_classification"
    out.mkdir(exist_ok=True)

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

    counts = np.bincount([item[2] for item in train.items], minlength=3)
    weights = torch.tensor(
        len(train) / (3 * counts),
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
        "Training head only. Encoder frozen. "
        "Stages are not pixel-level plaque labels.",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, mask, y in trainloader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, mask), y)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(y)

        vl, ba, _, _, _ = evaluate(model, valloader, device)
        train_loss = running / len(train)

        history.append(
            dict(
                epoch=epoch,
                weighted_train_loss=train_loss,
                val_loss=vl,
                val_balanced_accuracy=ba,
            )
        )
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}: "
            f"train loss={train_loss:.4f}, "
            f"val loss={vl:.4f}, "
            f"val balanced accuracy={ba:.3f}",
            flush=True,
        )

        if vl < best:
            best = vl
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

    # ------------------------------------------------------------
    # NEW: training progress figure
    # ------------------------------------------------------------
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
            )

            # Keep the original CSV confusion matrix.
            cm = confusion_matrix(ys, preds, labels=[0, 1, 2])
            pd.DataFrame(
                cm,
                index=CLASSES,
                columns=CLASSES,
            ).to_csv(out / f"{split}_confusion_matrix.csv")

            # ----------------------------------------------------
            # NEW: confusion matrix figure
            # ----------------------------------------------------
            plot_confusion_matrix_figure(
                ys,
                preds,
                split,
                out / f"{split}_confusion_matrix.png",
            )
            print(
                f"Saved {split} confusion matrix figure: "
                f"{out / f'{split}_confusion_matrix.png'}",
                flush=True,
            )

            for i, row in enumerate(ds.rows.to_dict("records")):
                x, mask, y = ds[i]
                x = x[None].to(device)
                mask = mask[None].to(device)

                folder = (
                    out
                    / "gradcam"
                    / split
                    / row["scanner"]
                    / row["sample_name"]
                )
                folder.mkdir(parents=True, exist_ok=True)

                with np.load(Path(row["map_dir"]) / "maps.npz") as z:
                    rgb = z["rgb"]
                    valid = z["valid"]

                overlays_by_class = {}
                record = {
                    **row,
                    "predicted_stage": CLASSES[preds[i]],
                }

                # Still compute and save ALL three class Grad-CAMs for every scan.
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

                    heat = colormaps["inferno"](full)[..., :3]
                    overlay = rgb.copy()
                    overlay[valid] = 0.6 * rgb[valid] + 0.4 * heat[valid]
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

                # ------------------------------------------------
                # CHANGED: comparison now includes partial Grad-CAM
                # for groups where partial cleaning exists (G3-G5).
                # ------------------------------------------------
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

                # Live preview follows the same relevant panels as comparison.png.
                viewer.show(preview_images, preview_titles, caption)

                allpred.append(record)

        predictions = pd.DataFrame(allpred)
        predictions.to_csv(out / "stage_predictions.csv", index=False)

        metrics = []
        for keys, group in predictions.groupby(["split", "scanner"]):
            metrics.append(
                dict(
                    split=keys[0],
                    scanner=keys[1],
                    n_scans=len(group),
                    balanced_accuracy=balanced_accuracy_score(
                        group.stage,
                        group.predicted_stage,
                    ),
                )
            )

        pd.DataFrame(metrics).to_csv(
            out / "metrics_by_scanner.csv",
            index=False,
        )

        (out / "run.json").write_text(
            json.dumps(
                dict(
                    classes=CLASSES,
                    best_epoch=best_epoch,
                    best_val_loss=best,
                    encoder="frozen cached dinov2_vits14_reg",
                    target_layer="StageHead.spatial ReLU output",
                    device=device,
                    interpretation=(
                        "Stage attribution, not plaque segmentation. "
                        "CAM normalized per image/class; intensity is not confidence."
                    ),
                    comparison_figures=(
                        "G1-G2: RGB + baseline + clean Grad-CAM. "
                        "G3-G5: RGB + baseline + partial + clean Grad-CAM."
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
