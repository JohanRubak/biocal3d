"""BioCal3D stage-classification robustness analyses.

This script extends the frozen-DINOv2 stage classifier with three analyses:

1) Specimen-grouped cross-validation
   - Outer folds are split by physical specimen, never by scan.
   - Stratification is performed on experimental group (G1-G4/G5 as present).
   - The untouched manifest test split is EXCLUDED by default.
   - Each outer fold uses an inner specimen-level validation split for epoch
     selection, then retrains on the full outer-training set for the selected
     number of epochs before evaluating the outer fold.

2) Strict leave-one-scanner-out (LOSO) generalization
   - For each held-out scanner, training uses ONLY the manifest 'train' split
     from the other scanners.
   - Evaluation uses ONLY the manifest 'val' split from the held-out scanner.
   - Therefore both scanner identity and validation specimens are unseen by the
     fitted final model. The manifest 'test' split remains untouched.

5) Color ablation: RGB-DINO vs grayscale-DINO
   - RGB uses the existing cached dino_features.npz.
   - Grayscale converts each prepared RGB map to luminance, repeats it to three
     channels, runs the same frozen DINOv2 encoder, and caches features as
     dino_features_gray.npz.
   - The SAME specimen folds and model procedure are used for RGB and grayscale.

Important interpretation
------------------------
The labels baseline / partial / clean are experimental stage labels, not pixel
plaque annotations. These analyses test stage-classification robustness and
scanner/color dependence; they do not validate plaque segmentation.

Typical use
-----------
python biocal3d_stage_robustness_analysis.py

If grayscale DINO features are already cached:
python biocal3d_stage_robustness_analysis.py --skip-gray-preparation

Run RGB only:
python biocal3d_stage_robustness_analysis.py --modalities rgb

Outputs are written to:
<root>/stage_robustness_analysis/
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import random
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

from biocal3d_prepare_dino import DATA_ROOT


CLASSES = ["baseline", "partial", "clean"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
RGB_FEATURE_FILE = "dino_features.npz"
GRAY_FEATURE_FILE = "dino_features_gray.npz"


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_savefig(fig, out_path, **kwargs):
    """Robust figure saving for Windows/Pillow Errno 22 path failures."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(out_path), **kwargs)
        return out_path
    except OSError as exc:
        if getattr(exc, "errno", None) != 22:
            raise
        old_cwd = Path.cwd()
        try:
            os.chdir(out_path.parent)
            fig.savefig(out_path.name, **kwargs)
        finally:
            os.chdir(old_cwd)
        return out_path


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def normalize_group_value(v):
    """Return a stable string stratum such as 'G1', 'G2', ..."""
    s = str(v)
    if s.upper().startswith("G"):
        return s.upper()
    try:
        return f"G{int(float(v))}"
    except Exception:
        return s


def specimen_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per physical specimen with its experimental-group stratum."""
    tmp = rows[["physical_id", "group"]].copy()
    tmp["stratum"] = tmp["group"].map(normalize_group_value)
    n_group = tmp.groupby("physical_id")["stratum"].nunique()
    if n_group.max() != 1:
        bad = n_group[n_group > 1].index.tolist()
        raise ValueError(f"Some physical specimens map to multiple groups: {bad[:10]}")
    return tmp.drop_duplicates("physical_id").reset_index(drop=True)


def verify_no_specimen_overlap(a: pd.DataFrame, b: pd.DataFrame, label_a="A", label_b="B"):
    overlap = set(a["physical_id"]) & set(b["physical_id"])
    if overlap:
        raise ValueError(
            f"Specimen leakage between {label_a} and {label_b}: "
            f"{sorted(overlap)[:10]}"
        )


def feature_filename(modality: str) -> str:
    if modality == "rgb":
        return RGB_FEATURE_FILE
    if modality == "gray":
        return GRAY_FEATURE_FILE
    raise ValueError(modality)


# ============================================================
# MODEL + FEATURE DATASET
# ============================================================

class StageHead(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(hidden, len(CLASSES), 1)

    def forward(self, x, mask):
        activation = self.spatial(x)
        spatial_logits = self.classifier(activation)
        logits = (spatial_logits * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        return logits


class CachedFeatures(Dataset):
    """Load cached DINO features into memory for one subset."""

    def __init__(self, rows: pd.DataFrame, feature_file: str):
        self.rows = rows.reset_index(drop=True).copy()
        self.items = []

        for row in self.rows.to_dict("records"):
            folder = Path(row["map_dir"])
            path = folder / feature_file
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing feature file: {path}\n"
                    f"For grayscale, rerun without --skip-gray-preparation."
                )

            with np.load(path) as z:
                if "features" not in z or "patch_valid" not in z:
                    raise ValueError(f"Malformed feature cache: {path}")

                # Verify against the exact prepared map when hash metadata exists.
                maps_path = folder / "maps.npz"
                digest = hashlib.sha256(maps_path.read_bytes()).hexdigest()
                if "map_sha256" in z and str(z["map_sha256"]) != digest:
                    raise ValueError(
                        f"Stale feature cache: {path}. The corresponding maps.npz changed."
                    )

                f = z["features"].astype(np.float32).transpose(2, 0, 1)
                mask = z["patch_valid"].astype(np.float32)[None]

            if mask.sum() == 0:
                raise ValueError(f"No valid feature patches: {path}")

            y = CLASS_TO_IDX[row["stage"]]
            self.items.append((torch.from_numpy(f), torch.from_numpy(mask), y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def class_weights_from_dataset(ds: CachedFeatures, device: str):
    y = np.array([item[2] for item in ds.items], dtype=int)
    counts = np.bincount(y, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError(
            f"Training subset is missing one or more classes. Counts={counts.tolist()}"
        )
    return torch.tensor(
        len(y) / (len(CLASSES) * counts),
        dtype=torch.float32,
        device=device,
    )


def evaluate(model, loader, device, weights):
    model.eval()
    ys, preds, probs = [], [], []
    loss_num = 0.0
    loss_den = 0.0

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            logits = model(x, mask)

            loss_num += nn.functional.cross_entropy(
                logits, y, weight=weights, reduction="sum"
            ).item()
            loss_den += weights[y].sum().item()

            p = logits.softmax(1)
            ys.extend(y.cpu().tolist())
            preds.extend(p.argmax(1).cpu().tolist())
            probs.extend(p.cpu().tolist())

    loss = loss_num / max(loss_den, 1e-12)
    ba = balanced_accuracy_score(ys, preds)
    return loss, ba, ys, preds, probs


def train_with_inner_validation(
    fit_rows,
    inner_val_rows,
    feature_file,
    device,
    epochs,
    patience,
    batch_size,
    lr,
    seed,
):
    """Select epoch using only inner validation specimens."""
    set_seed(seed)
    fit_ds = CachedFeatures(fit_rows, feature_file)
    val_ds = CachedFeatures(inner_val_rows, feature_file)

    channels = fit_ds[0][0].shape[0]
    model = StageHead(channels).to(device)
    weights = class_weights_from_dataset(fit_ds, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_num = 0.0
        train_den = 0.0

        for x, mask, y in fit_loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_weight_sum = weights[y].sum().item()
            train_num += loss.item() * batch_weight_sum
            train_den += batch_weight_sum

        train_loss = train_num / max(train_den, 1e-12)
        val_loss, val_ba, _, _, _ = evaluate(model, val_loader, device, weights)
        history.append(
            {
                "epoch": epoch,
                "weighted_train_loss": train_loss,
                "weighted_inner_val_loss": val_loss,
                "inner_val_balanced_accuracy": val_ba,
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break

    if best_epoch <= 0:
        raise RuntimeError("Inner validation did not select a valid epoch")

    del model, fit_ds, val_ds, fit_loader, val_loader
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    return best_epoch, best_loss, pd.DataFrame(history)


def train_fixed_epochs(
    train_rows,
    feature_file,
    device,
    n_epochs,
    batch_size,
    lr,
    seed,
):
    """Retrain from scratch on the full outer-training set for selected epochs."""
    set_seed(seed)
    ds = CachedFeatures(train_rows, feature_file)
    channels = ds[0][0].shape[0]
    model = StageHead(channels).to(device)
    weights = class_weights_from_dataset(ds, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(n_epochs):
        model.train()
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, mask), y)
            loss.backward()
            optimizer.step()

    return model, weights, ds


def predict_rows(model, weights, test_rows, feature_file, device, batch_size):
    ds = CachedFeatures(test_rows, feature_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    loss, ba, ys, preds, probs = evaluate(model, loader, device, weights)

    out = ds.rows.copy()
    out["true_index"] = ys
    out["predicted_index"] = preds
    out["predicted_stage"] = [CLASSES[i] for i in preds]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = [p[i] for p in probs]
    out["true_class_probability"] = [probs[i][ys[i]] for i in range(len(ys))]
    out["prediction_confidence"] = [max(p) for p in probs]
    out["correct"] = np.array(ys) == np.array(preds)
    out["evaluation_weighted_loss"] = loss
    out["evaluation_balanced_accuracy"] = ba
    return out


# ============================================================
# SPLITTING
# ============================================================

def make_outer_specimen_folds(dev_rows, n_splits, seed):
    specs = specimen_table(dev_rows)
    if n_splits > specs["physical_id"].nunique():
        raise ValueError("More outer folds than physical specimens")

    counts = specs["stratum"].value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            "Cannot stratify specimen folds because at least one experimental group "
            f"has fewer than {n_splits} specimens: {counts.to_dict()}"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(specs["physical_id"], specs["stratum"]), start=1
    ):
        train_ids = set(specs.iloc[train_idx]["physical_id"])
        test_ids = set(specs.iloc[test_idx]["physical_id"])
        folds.append((fold, train_ids, test_ids))
    return folds


def split_inner_specimens(outer_train_rows, seed, fraction=0.20):
    specs = specimen_table(outer_train_rows)
    n_specs = len(specs)
    n_strata = specs["stratum"].nunique()
    test_size = max(n_strata, int(math.ceil(n_specs * fraction)))
    if test_size >= n_specs - n_strata + 1:
        test_size = n_strata

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    fit_idx, val_idx = next(splitter.split(specs["physical_id"], specs["stratum"]))
    fit_ids = set(specs.iloc[fit_idx]["physical_id"])
    val_ids = set(specs.iloc[val_idx]["physical_id"])

    fit_rows = outer_train_rows[outer_train_rows["physical_id"].isin(fit_ids)].copy()
    val_rows = outer_train_rows[outer_train_rows["physical_id"].isin(val_ids)].copy()
    verify_no_specimen_overlap(fit_rows, val_rows, "inner-fit", "inner-val")
    return fit_rows, val_rows


# ============================================================
# GRAYSCALE DINO FEATURE PREPARATION
# ============================================================

def load_dinov2(model_name: str, device: str):
    print(f"Loading frozen DINOv2 model '{model_name}'...", flush=True)
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2 through torch.hub. If your machine is offline, "
            "make sure the facebookresearch/dinov2 repository/model is already in "
            "the torch hub cache, or temporarily run with internet access."
        ) from exc
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prepare_grayscale_features(rows, device, model_name, force=False):
    """Create grayscale DINO features while preserving the prepared map geometry."""
    unique = rows.drop_duplicates("map_dir").reset_index(drop=True)
    missing = []
    for row in unique.to_dict("records"):
        folder = Path(row["map_dir"])
        target = folder / GRAY_FEATURE_FILE
        if force or not target.exists():
            missing.append(row)

    if not missing:
        print("All grayscale DINO features are already cached.", flush=True)
        return

    print(
        f"Preparing grayscale DINO features for {len(missing)} scans "
        f"(cache file: {GRAY_FEATURE_FILE})...",
        flush=True,
    )
    model = load_dinov2(model_name, device)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)[None, :, None, None]

    for i, row in enumerate(missing, start=1):
        folder = Path(row["map_dir"])
        maps_path = folder / "maps.npz"
        rgb_feature_path = folder / RGB_FEATURE_FILE
        target = folder / GRAY_FEATURE_FILE

        with np.load(maps_path) as z:
            rgb = z["rgb"].astype(np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 rgb map in {maps_path}; got {rgb.shape}")

        h, w, _ = rgb.shape
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                f"Prepared map size must be divisible by DINO patch size 14; got {h}x{w} in {maps_path}"
            )

        # Preserve the exact patch-valid mask from the already verified RGB cache.
        if not rgb_feature_path.exists():
            raise FileNotFoundError(
                f"RGB feature cache is required to copy patch_valid: {rgb_feature_path}"
            )
        with np.load(rgb_feature_path) as zrgb:
            patch_valid = zrgb["patch_valid"].astype(np.float32)

        # ITU-R / sRGB luminance weights; repeated to 3 channels for DINOv2.
        gray = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        gray3 = np.repeat(gray[..., None], 3, axis=2)
        x = torch.from_numpy(gray3.transpose(2, 0, 1)).unsqueeze(0).to(device)
        x = (x - mean) / std

        with torch.inference_mode():
            output = model.forward_features(x)
            if "x_norm_patchtokens" not in output:
                raise KeyError(
                    "DINOv2 forward_features output lacks 'x_norm_patchtokens'. "
                    f"Available keys: {list(output.keys())}"
                )
            tokens = output["x_norm_patchtokens"][0]

        ph, pw = h // 14, w // 14
        if tokens.shape[0] != ph * pw:
            raise ValueError(
                f"Unexpected DINO patch-token count {tokens.shape[0]} for {h}x{w}; expected {ph*pw}."
            )
        features = tokens.reshape(ph, pw, -1).detach().cpu().numpy().astype(np.float16)

        if patch_valid.shape != (ph, pw):
            raise ValueError(
                f"RGB patch_valid shape {patch_valid.shape} does not match grayscale tokens {(ph, pw)}"
            )

        digest = hashlib.sha256(maps_path.read_bytes()).hexdigest()
        np.savez_compressed(
            target,
            features=features,
            patch_valid=patch_valid.astype(np.float32),
            map_sha256=np.array(digest),
            modality=np.array("grayscale_repeated_rgb"),
            dino_model=np.array(model_name),
        )

        if i == 1 or i % 10 == 0 or i == len(missing):
            print(f"[gray DINO {i:>3}/{len(missing)}] {row['scanner']} {row['sample_name']}", flush=True)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# METRICS + FIGURES
# ============================================================

def classification_summary(pred: pd.DataFrame):
    y = pred["stage"].map(CLASS_TO_IDX).to_numpy()
    p = pred["predicted_stage"].map(CLASS_TO_IDX).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=list(range(len(CLASSES))), zero_division=0
    )
    summary = {
        "n_scans": len(pred),
        "n_specimens": pred["physical_id"].nunique(),
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "mean_true_class_probability": pred["true_class_probability"].mean(),
        "mean_specimen_accuracy": pred.groupby("physical_id")["correct"].mean().mean(),
    }
    class_rows = []
    for i, c in enumerate(CLASSES):
        class_rows.append(
            {
                "stage": c,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
        )
    return summary, pd.DataFrame(class_rows)


def plot_confusion(pred, title, out_path):
    y = pred["stage"].map(CLASS_TO_IDX).to_numpy()
    p = pred["predicted_stage"].map(CLASS_TO_IDX).to_numpy()
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    row_sum = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
    for r in range(len(CLASSES)):
        for c in range(len(CLASSES)):
            text_color = "white" if frac[r, c] >= 0.5 else "black"
            ax.text(
                c, r, f"{cm[r,c]}\n({100*frac[r,c]:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=11,
            )
    ax.set_xticks(range(len(CLASSES)), [x.title() for x in CLASSES])
    ax.set_yticks(range(len(CLASSES)), [x.title() for x in CLASSES])
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_cv_modality_comparison(fold_metrics, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    modalities = list(dict.fromkeys(fold_metrics["modality"].tolist()))
    x = np.arange(1, fold_metrics["fold"].nunique() + 1)
    for modality in modalities:
        g = fold_metrics[fold_metrics["modality"] == modality].sort_values("fold")
        ax.plot(g["fold"], g["balanced_accuracy"], marker="o", label=modality.upper())
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xlabel("Outer specimen fold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Specimen-grouped cross-validation")
    ax.grid(alpha=0.25)
    ax.legend(title="Input")
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loso_comparison(metrics, out_path):
    scanners = sorted(metrics["held_out_scanner"].unique())
    modalities = list(dict.fromkeys(metrics["modality"].tolist()))
    x = np.arange(len(scanners))
    width = 0.8 / max(1, len(modalities))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for j, modality in enumerate(modalities):
        g = metrics[metrics["modality"] == modality].set_index("held_out_scanner")
        vals = [g.loc[s, "balanced_accuracy"] if s in g.index else np.nan for s in scanners]
        offset = (j - (len(modalities)-1)/2) * width
        ax.bar(x + offset, vals, width=width, label=modality.upper())
    ax.set_xticks(x, scanners, rotation=20)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Completely unseen scanner")
    ax.set_title("Strict leave-one-scanner-out validation\n(train specimens also disjoint from validation specimens)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Input")
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ANALYSIS 1: SPECIMEN-GROUPED CV
# ============================================================

def run_grouped_cv(dev_rows, modalities, args, device, out_root):
    out_dir = out_root / "01_grouped_specimen_cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = make_outer_specimen_folds(dev_rows, args.outer_folds, args.seed)
    all_predictions = []
    fold_metrics = []
    class_metrics = []

    for modality in modalities:
        ffile = feature_filename(modality)
        modality_dir = out_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== GROUPED CV: {modality.upper()} ===", flush=True)

        for fold, train_ids, test_ids in folds:
            outer_train = dev_rows[dev_rows["physical_id"].isin(train_ids)].copy()
            outer_test = dev_rows[dev_rows["physical_id"].isin(test_ids)].copy()
            verify_no_specimen_overlap(outer_train, outer_test, "outer-train", "outer-test")

            fit_rows, inner_val_rows = split_inner_specimens(
                outer_train,
                seed=args.seed + 1000 * fold,
                fraction=args.inner_val_fraction,
            )

            print(
                f"[{modality} fold {fold}/{args.outer_folds}] "
                f"outer train={outer_train.physical_id.nunique()} specimens, "
                f"outer test={outer_test.physical_id.nunique()}, "
                f"inner fit={fit_rows.physical_id.nunique()}, "
                f"inner val={inner_val_rows.physical_id.nunique()}",
                flush=True,
            )

            best_epoch, best_inner_loss, history = train_with_inner_validation(
                fit_rows,
                inner_val_rows,
                ffile,
                device,
                args.epochs,
                args.patience,
                args.batch_size,
                args.lr,
                seed=args.seed + fold,
            )
            history.to_csv(modality_dir / f"fold_{fold}_selection_history.csv", index=False)

            model, weights, train_ds = train_fixed_epochs(
                outer_train,
                ffile,
                device,
                best_epoch,
                args.batch_size,
                args.lr,
                seed=args.seed + 100 + fold,
            )
            pred = predict_rows(
                model, weights, outer_test, ffile, device, args.batch_size
            )
            pred["analysis"] = "grouped_specimen_cv"
            pred["modality"] = modality
            pred["fold"] = fold
            pred["selected_epoch"] = best_epoch
            pred["best_inner_val_loss"] = best_inner_loss
            all_predictions.append(pred)

            summary, per_class = classification_summary(pred)
            summary.update(
                {
                    "analysis": "grouped_specimen_cv",
                    "modality": modality,
                    "fold": fold,
                    "selected_epoch": best_epoch,
                    "best_inner_val_loss": best_inner_loss,
                }
            )
            fold_metrics.append(summary)
            per_class["analysis"] = "grouped_specimen_cv"
            per_class["modality"] = modality
            per_class["fold"] = fold
            class_metrics.append(per_class)

            print(
                f"    selected epoch={best_epoch}; outer BA={summary['balanced_accuracy']:.3f}; "
                f"accuracy={summary['accuracy']:.3f}",
                flush=True,
            )

            del model, weights, train_ds
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    pred_all = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)
    class_df = pd.concat(class_metrics, ignore_index=True)

    pred_all.to_csv(out_dir / "grouped_cv_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "grouped_cv_fold_metrics.csv", index=False)
    class_df.to_csv(out_dir / "grouped_cv_class_metrics.csv", index=False)

    pooled_rows = []
    pooled_class_rows = []
    for modality in modalities:
        g = pred_all[pred_all["modality"] == modality]
        summary, pc = classification_summary(g)
        summary.update({"analysis": "grouped_specimen_cv_pooled", "modality": modality})
        pooled_rows.append(summary)
        pc["analysis"] = "grouped_specimen_cv_pooled"
        pc["modality"] = modality
        pooled_class_rows.append(pc)
        plot_confusion(
            g,
            f"Grouped specimen CV – {modality.upper()}",
            out_dir / f"grouped_cv_confusion_{modality}.png",
        )

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_class_df = pd.concat(pooled_class_rows, ignore_index=True)
    pooled_df.to_csv(out_dir / "grouped_cv_pooled_summary.csv", index=False)
    pooled_class_df.to_csv(out_dir / "grouped_cv_pooled_class_metrics.csv", index=False)

    # Scanner-specific pooled outer-fold metrics.
    scanner_rows = []
    for (modality, scanner), g in pred_all.groupby(["modality", "scanner"]):
        summary, _ = classification_summary(g)
        summary.update({"modality": modality, "scanner": scanner})
        scanner_rows.append(summary)
    pd.DataFrame(scanner_rows).to_csv(
        out_dir / "grouped_cv_scanner_metrics.csv", index=False
    )

    plot_grouped_cv_modality_comparison(
        metrics_df,
        out_dir / "grouped_cv_rgb_vs_gray.png",
    )

    return pred_all, metrics_df, pooled_df


# ============================================================
# ANALYSIS 2: STRICT LEAVE-ONE-SCANNER-OUT
# ============================================================

def run_loso(manifest, modalities, args, device, out_root):
    out_dir = out_root / "02_leave_one_scanner_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "train" not in set(manifest["split"]) or "val" not in set(manifest["split"]):
        raise ValueError("LOSO requires manifest split labels 'train' and 'val'.")

    train_pool = manifest[manifest["split"] == "train"].copy()
    val_pool = manifest[manifest["split"] == "val"].copy()
    verify_no_specimen_overlap(train_pool, val_pool, "manifest train", "manifest val")

    scanners = sorted(set(train_pool["scanner"]) & set(val_pool["scanner"]))
    all_predictions = []
    metric_rows = []
    class_rows = []

    for modality in modalities:
        ffile = feature_filename(modality)
        modality_dir = out_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== STRICT LOSO: {modality.upper()} ===", flush=True)

        for s_idx, held_out in enumerate(scanners, start=1):
            outer_train = train_pool[train_pool["scanner"] != held_out].copy()
            outer_test = val_pool[val_pool["scanner"] == held_out].copy()

            if len(outer_test) == 0:
                continue

            # Strong checks: no held-out scanner in training and no validation specimens in training.
            if held_out in set(outer_train["scanner"]):
                raise AssertionError("Held-out scanner leaked into LOSO training")
            verify_no_specimen_overlap(outer_train, outer_test, "LOSO train", "LOSO test")

            fit_rows, inner_val_rows = split_inner_specimens(
                outer_train,
                seed=args.seed + 5000 + s_idx,
                fraction=args.inner_val_fraction,
            )

            print(
                f"[{modality} LOSO {held_out}] "
                f"train scanners={sorted(outer_train.scanner.unique().tolist())}; "
                f"test scans={len(outer_test)} / {outer_test.physical_id.nunique()} specimens",
                flush=True,
            )

            best_epoch, best_inner_loss, history = train_with_inner_validation(
                fit_rows,
                inner_val_rows,
                ffile,
                device,
                args.epochs,
                args.patience,
                args.batch_size,
                args.lr,
                seed=args.seed + 6000 + s_idx,
            )
            history.to_csv(
                modality_dir / f"loso_{held_out}_selection_history.csv", index=False
            )

            model, weights, train_ds = train_fixed_epochs(
                outer_train,
                ffile,
                device,
                best_epoch,
                args.batch_size,
                args.lr,
                seed=args.seed + 7000 + s_idx,
            )
            pred = predict_rows(
                model, weights, outer_test, ffile, device, args.batch_size
            )
            pred["analysis"] = "strict_leave_one_scanner_out"
            pred["modality"] = modality
            pred["held_out_scanner"] = held_out
            pred["selected_epoch"] = best_epoch
            pred["best_inner_val_loss"] = best_inner_loss
            all_predictions.append(pred)

            summary, pc = classification_summary(pred)
            summary.update(
                {
                    "analysis": "strict_leave_one_scanner_out",
                    "modality": modality,
                    "held_out_scanner": held_out,
                    "selected_epoch": best_epoch,
                    "best_inner_val_loss": best_inner_loss,
                    "training_scanners": ";".join(sorted(outer_train.scanner.unique())),
                }
            )
            metric_rows.append(summary)
            pc["analysis"] = "strict_leave_one_scanner_out"
            pc["modality"] = modality
            pc["held_out_scanner"] = held_out
            class_rows.append(pc)

            plot_confusion(
                pred,
                f"LOSO {held_out} – {modality.upper()}",
                modality_dir / f"loso_{held_out}_confusion.png",
            )

            print(
                f"    selected epoch={best_epoch}; BA={summary['balanced_accuracy']:.3f}; "
                f"accuracy={summary['accuracy']:.3f}",
                flush=True,
            )

            del model, weights, train_ds
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    pred_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    class_df = pd.concat(class_rows, ignore_index=True)

    pred_df.to_csv(out_dir / "loso_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "loso_metrics.csv", index=False)
    class_df.to_csv(out_dir / "loso_class_metrics.csv", index=False)

    plot_loso_comparison(metrics_df, out_dir / "loso_rgb_vs_gray.png")
    return pred_df, metrics_df


# ============================================================
# ANALYSIS 5: COLOR ABLATION SUMMARY
# ============================================================

def write_color_ablation_summary(grouped_metrics, grouped_pooled, loso_metrics, out_root):
    out_dir = out_root / "05_color_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Paired grouped-CV fold comparison: same held-out specimens for both modalities.
    wide = grouped_metrics.pivot(
        index="fold", columns="modality", values=["balanced_accuracy", "accuracy", "macro_f1"]
    )
    wide.columns = [f"{metric}_{modality}" for metric, modality in wide.columns]
    wide = wide.reset_index()
    if {"balanced_accuracy_rgb", "balanced_accuracy_gray"}.issubset(wide.columns):
        wide["balanced_accuracy_gray_minus_rgb"] = (
            wide["balanced_accuracy_gray"] - wide["balanced_accuracy_rgb"]
        )
    if {"macro_f1_rgb", "macro_f1_gray"}.issubset(wide.columns):
        wide["macro_f1_gray_minus_rgb"] = wide["macro_f1_gray"] - wide["macro_f1_rgb"]
    wide.to_csv(out_dir / "grouped_cv_paired_rgb_vs_gray.csv", index=False)

    # LOSO scanner-wise paired comparison.
    lwide = loso_metrics.pivot(
        index="held_out_scanner",
        columns="modality",
        values=["balanced_accuracy", "accuracy", "macro_f1"],
    )
    lwide.columns = [f"{metric}_{modality}" for metric, modality in lwide.columns]
    lwide = lwide.reset_index()
    if {"balanced_accuracy_rgb", "balanced_accuracy_gray"}.issubset(lwide.columns):
        lwide["balanced_accuracy_gray_minus_rgb"] = (
            lwide["balanced_accuracy_gray"] - lwide["balanced_accuracy_rgb"]
        )
    lwide.to_csv(out_dir / "loso_paired_rgb_vs_gray.csv", index=False)

    # Compact overall table.
    rows = []
    for _, r in grouped_pooled.iterrows():
        rows.append(
            {
                "analysis": "grouped_specimen_cv_pooled",
                "modality": r["modality"],
                "balanced_accuracy": r["balanced_accuracy"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "mean_true_class_probability": r["mean_true_class_probability"],
                "n_scans": r["n_scans"],
                "n_specimens": r["n_specimens"],
            }
        )

    for modality, g in loso_metrics.groupby("modality"):
        rows.append(
            {
                "analysis": "strict_loso_mean_across_scanners",
                "modality": modality,
                "balanced_accuracy": g["balanced_accuracy"].mean(),
                "accuracy": g["accuracy"].mean(),
                "macro_f1": g["macro_f1"].mean(),
                "mean_true_class_probability": g["mean_true_class_probability"].mean(),
                "n_scans": g["n_scans"].sum(),
                "n_specimens": np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "color_ablation_summary.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DATA_ROOT / "_dino_preparation_80_10_10",
        help="DINO preparation root containing scan_manifest.csv.",
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        choices=["rgb", "gray"],
        default=["rgb", "gray"],
        help="Input representations to compare.",
    )
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-val-fraction", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--dino-model",
        default="dinov2_vits14_reg",
        help="torch.hub DINOv2 model used for grayscale feature extraction.",
    )
    p.add_argument(
        "--skip-gray-preparation",
        action="store_true",
        help="Assume dino_features_gray.npz already exists for all required scans.",
    )
    p.add_argument(
        "--force-gray-features",
        action="store_true",
        help="Regenerate grayscale DINO feature caches even if they already exist.",
    )
    args = p.parse_args()

    if args.outer_folds < 2:
        p.error("--outer-folds must be at least 2")
    if not (0 < args.inner_val_fraction < 0.5):
        p.error("--inner-val-fraction must be between 0 and 0.5")
    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.lr <= 0:
        p.error("epochs, patience, batch-size and lr must be positive")

    device = choose_device(args.device)
    set_seed(args.seed)

    manifest_path = args.root / "scan_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    required = {"scanner", "sample_name", "physical_id", "group", "stage", "split", "map_dir"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    unexpected_stages = set(manifest["stage"]) - set(CLASSES)
    if unexpected_stages:
        raise ValueError(f"Unexpected stages in manifest: {sorted(unexpected_stages)}")

    # Keep the final manifest test set untouched. Grouped CV is run only on train+val.
    dev_rows = manifest[manifest["split"].isin(["train", "val"])].copy()
    if len(dev_rows) == 0:
        raise ValueError("No train/val rows found in scan_manifest.csv")

    out_root = args.root / "stage_robustness_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    print("BioCal3D stage robustness analysis", flush=True)
    print(f"Root: {args.root}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Modalities: {args.modalities}", flush=True)
    print(
        f"Development data: {len(dev_rows)} scans / {dev_rows.physical_id.nunique()} specimens. "
        f"Manifest test split excluded: {(manifest.split == 'test').sum()} scans.",
        flush=True,
    )

    # Verify original split is specimen-safe when used for strict LOSO.
    train_rows = manifest[manifest["split"] == "train"]
    val_rows = manifest[manifest["split"] == "val"]
    if len(train_rows) and len(val_rows):
        verify_no_specimen_overlap(train_rows, val_rows, "manifest train", "manifest val")

    # Verify existing RGB caches early.
    if "rgb" in args.modalities or "gray" in args.modalities:
        first_rgb = Path(dev_rows.iloc[0]["map_dir"]) / RGB_FEATURE_FILE
        if not first_rgb.exists():
            raise FileNotFoundError(
                f"Existing RGB DINO features were not found: {first_rgb}\n"
                "Run biocal3d_prepare_dino.py first."
            )

    if "gray" in args.modalities and not args.skip_gray_preparation:
        # Need every train/val row plus original train/val used by LOSO; dev_rows already covers both.
        prepare_grayscale_features(
            dev_rows,
            device=device,
            model_name=args.dino_model,
            force=args.force_gray_features,
        )

    # Save exact configuration.
    (out_root / "run_config.json").write_text(
        json.dumps(
            {
                "classes": CLASSES,
                "modalities": args.modalities,
                "outer_folds": args.outer_folds,
                "inner_val_fraction": args.inner_val_fraction,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "device": device,
                "dino_model": args.dino_model,
                "rgb_feature_file": RGB_FEATURE_FILE,
                "gray_feature_file": GRAY_FEATURE_FILE,
                "grouped_cv_data": "manifest train + val; manifest test excluded",
                "loso_design": (
                    "train on manifest train specimens from the other 3 scanners; "
                    "evaluate on manifest val specimens from held-out scanner"
                ),
                "interpretation": (
                    "Experimental stage classification, not plaque segmentation. "
                    "Grayscale ablation tests dependence on chromatic information."
                ),
            },
            indent=2,
        )
    )

    grouped_pred, grouped_metrics, grouped_pooled = run_grouped_cv(
        dev_rows, args.modalities, args, device, out_root
    )

    loso_pred, loso_metrics = run_loso(
        manifest, args.modalities, args, device, out_root
    )

    write_color_ablation_summary(
        grouped_metrics, grouped_pooled, loso_metrics, out_root
    )

    print("\nDone.", flush=True)
    print(f"Results: {out_root}", flush=True)
    print("Key files:", flush=True)
    print("  01_grouped_specimen_cv/grouped_cv_pooled_summary.csv", flush=True)
    print("  01_grouped_specimen_cv/grouped_cv_rgb_vs_gray.png", flush=True)
    print("  02_leave_one_scanner_out/loso_metrics.csv", flush=True)
    print("  02_leave_one_scanner_out/loso_rgb_vs_gray.png", flush=True)
    print("  05_color_ablation/color_ablation_summary.csv", flush=True)


if __name__ == "__main__":
    main()
