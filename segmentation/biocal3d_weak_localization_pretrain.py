"""BioCal3D weak-localization pretraining with frozen multi-layer DINOv2 features.

Goal
----
Turn the current stage-classification experiment into a more segmentation-oriented
pretraining stage without pretending that stage labels are pixel plaque labels.

Architecture
------------
Prepared ROI -> frozen DINOv2 ViT-S/14 intermediate blocks (default 3,6,9,12)
             -> per-block 1x1 projection 384 -> 64
             -> concatenate + 1x1 fusion -> 128
             -> residual 3x3 spatial adapter blocks
             -> shared 37x37x128 spatial representation
                  |-> attention-pooled 3-class stage head
                  |-> 1-channel candidate plaque/burden map

Weak supervision
----------------
1) Stage cross-entropy: baseline / partial / clean.
2) Temporal burden ranking within the SAME physical specimen + scanner:
       baseline > partial > clean
   (or baseline > clean for two-stage groups).
3) Cross-scanner burden consistency for the SAME physical specimen + stage.
4) Optional cross-scanner feature consistency for the pooled shared representation.

Important
---------
The 1-channel map is NOT a validated plaque segmentation. Before using it as a
segmentation target, validate spatial correspondence (manual partial-sample
annotations / perturbation testing) and then add true/pseudo spatial supervision.

The script can initialize the final-block 384->64 projection from your existing
stage-classification checkpoint (StageHead.spatial.0), so the learned 64-channel
projection is not discarded.

Typical use
-----------
python biocal3d_weak_localization_pretrain.py

If intermediate DINO caches already exist:
python biocal3d_weak_localization_pretrain.py --skip-feature-preparation

Do not reuse the old stage projection:
python biocal3d_weak_localization_pretrain.py --no-init-stage-head
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from biocal3d_prepare_dino import DATA_ROOT


CLASSES = ["baseline", "partial", "clean"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
STAGE_RANK = {"baseline": 2, "partial": 1, "clean": 0}

RGB_SINGLE_FEATURE_FILE = "dino_features.npz"


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def safe_savefig(fig, out_path: Path, **kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(out_path), **kwargs)
    except OSError as exc:
        if getattr(exc, "errno", None) != 22:
            raise
        old = Path.cwd()
        try:
            os.chdir(out_path.parent)
            fig.savefig(out_path.name, **kwargs)
        finally:
            os.chdir(old)


def normalize_group_value(v):
    s = str(v)
    if s.upper().startswith("G"):
        return s.upper()
    try:
        return f"G{int(float(v))}"
    except Exception:
        return s


def verify_specimen_safe_split(manifest: pd.DataFrame):
    n = manifest.groupby("physical_id")["split"].nunique()
    if len(n) and n.max() > 1:
        bad = n[n > 1].index.tolist()
        raise ValueError(f"Specimens occur in multiple manifest splits: {bad[:10]}")


def infer_group(row) -> str:
    if "group" in row and pd.notna(row["group"]):
        return normalize_group_value(row["group"])
    m = re.search(r"BCG(\d+)", str(row.get("sample_name", "")), re.I)
    return f"G{m.group(1)}" if m else "unknown"


# ============================================================================
# MULTI-LAYER DINO FEATURE PREPARATION
# ============================================================================

def multilayer_cache_name(modality: str, blocks: list[int]) -> str:
    b = "-".join(str(x) for x in blocks)
    return f"dino_multilayer_{modality}_blocks_{b}.npz"


def load_dino(model_name: str, device: str):
    print(f"Loading frozen DINOv2 '{model_name}'...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prepare_multilayer_features(
    rows: pd.DataFrame,
    device: str,
    model_name: str,
    blocks: list[int],
    modality: str,
    force: bool = False,
):
    """Cache normalized patch tokens from selected DINO blocks.

    Human block numbers are 1-based (3,6,9,12). DINO's API receives 0-based
    indices, hence [b-1 for b in blocks].
    """
    cache_file = multilayer_cache_name(modality, blocks)
    unique = rows.drop_duplicates("map_dir").reset_index(drop=True)

    todo = []
    for r in unique.to_dict("records"):
        target = Path(r["map_dir"]) / cache_file
        if force or not target.exists():
            todo.append(r)

    if not todo:
        print(f"All multi-layer {modality} DINO caches already exist: {cache_file}", flush=True)
        return

    model = load_dino(model_name, device)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)[None, :, None, None]
    block_indices = [b - 1 for b in blocks]

    if min(block_indices) < 0:
        raise ValueError("Block numbers must be >= 1")

    print(
        f"Preparing multi-layer DINO features for {len(todo)} scans; "
        f"blocks={blocks}; modality={modality}",
        flush=True,
    )

    for i, row in enumerate(todo, 1):
        folder = Path(row["map_dir"])
        maps_path = folder / "maps.npz"
        base_feature_path = folder / RGB_SINGLE_FEATURE_FILE
        target = folder / cache_file

        with np.load(maps_path) as z:
            rgb = z["rgb"].astype(np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 rgb map in {maps_path}, got {rgb.shape}")

        # Reuse the verified valid-patch geometry from the original DINO cache.
        if not base_feature_path.exists():
            raise FileNotFoundError(
                f"Missing original DINO feature cache needed for patch_valid: {base_feature_path}"
            )
        with np.load(base_feature_path) as z:
            patch_valid = z["patch_valid"].astype(bool)

        if modality == "rgb":
            arr = rgb
        elif modality == "gray":
            gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            arr = np.repeat(gray[..., None], 3, axis=2)
        else:
            raise ValueError(modality)

        h, w, _ = arr.shape
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        x = (x - mean) / std

        with torch.inference_mode():
            if not hasattr(model, "get_intermediate_layers"):
                raise AttributeError("Loaded DINO model has no get_intermediate_layers()")
            outputs = model.get_intermediate_layers(
                x,
                n=block_indices,
                reshape=True,
                return_class_token=False,
                norm=True,
            )

        if len(outputs) != len(blocks):
            raise RuntimeError(
                f"Expected {len(blocks)} intermediate outputs, got {len(outputs)}"
            )

        payload = {
            "patch_valid": patch_valid.astype(np.uint8),
            "map_sha256": np.array(hashlib.sha256(maps_path.read_bytes()).hexdigest()),
            "modality": np.array(modality),
            "dino_model": np.array(model_name),
            "blocks": np.array(blocks, dtype=np.int16),
        }
        for b, feat in zip(blocks, outputs):
            # get_intermediate_layers(... reshape=True) -> B,C,Hpatch,Wpatch
            f = feat[0].detach().cpu().numpy().astype(np.float16)
            if f.shape[1:] != patch_valid.shape:
                raise ValueError(
                    f"Block {b} feature grid {f.shape[1:]} != patch_valid {patch_valid.shape}"
                )
            payload[f"block_{b}"] = f

        np.savez_compressed(target, **payload)

        if i == 1 or i % 10 == 0 or i == len(todo):
            print(f"[multi-DINO {i:>3}/{len(todo)}] {row['scanner']} {row['sample_name']}", flush=True)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# DATASET: ONE ITEM = ONE PHYSICAL SPECIMEN, ALL SCANNERS/STAGES
# ============================================================================

class SpecimenDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache_file: str, blocks: list[int]):
        self.rows = rows.reset_index(drop=True).copy()
        self.cache_file = cache_file
        self.blocks = list(blocks)
        self.specimen_ids = sorted(self.rows["physical_id"].astype(str).unique())

    def __len__(self):
        return len(self.specimen_ids)

    def __getitem__(self, idx):
        pid = self.specimen_ids[idx]
        g = self.rows[self.rows["physical_id"].astype(str) == pid].reset_index(drop=True)

        per_block = {b: [] for b in self.blocks}
        masks = []
        labels = []
        meta = []

        for row in g.to_dict("records"):
            folder = Path(row["map_dir"])
            path = folder / self.cache_file
            if not path.exists():
                raise FileNotFoundError(f"Missing multi-layer feature cache: {path}")

            with np.load(path) as z:
                maps_path = folder / "maps.npz"
                digest = hashlib.sha256(maps_path.read_bytes()).hexdigest()
                if "map_sha256" in z and str(z["map_sha256"]) != digest:
                    raise ValueError(f"Stale multi-layer cache: {path}")

                mask = z["patch_valid"].astype(np.float32)[None]
                for b in self.blocks:
                    key = f"block_{b}"
                    if key not in z:
                        raise KeyError(f"{path} lacks {key}")
                    per_block[b].append(torch.from_numpy(z[key].astype(np.float32)))

            masks.append(torch.from_numpy(mask))
            labels.append(CLASS_TO_IDX[row["stage"]])
            meta.append(
                {
                    "physical_id": str(row["physical_id"]),
                    "scanner": str(row["scanner"]),
                    "stage": str(row["stage"]),
                    "sample_name": str(row["sample_name"]),
                    "map_dir": str(row["map_dir"]),
                    "group": infer_group(row),
                }
            )

        return {
            "features": {b: torch.stack(per_block[b], dim=0) for b in self.blocks},
            "mask": torch.stack(masks, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "meta": meta,
        }


def specimen_collate(items):
    blocks = list(items[0]["features"].keys())
    return {
        "features": {b: torch.cat([it["features"][b] for it in items], dim=0) for b in blocks},
        "mask": torch.cat([it["mask"] for it in items], dim=0),
        "labels": torch.cat([it["labels"] for it in items], dim=0),
        "meta": sum([it["meta"] for it in items], []),
    }


# ============================================================================
# MODEL
# ============================================================================

class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.10):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.GroupNorm(groups, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class WeakLocalizationModel(nn.Module):
    def __init__(
        self,
        blocks: list[int],
        in_channels: int = 384,
        projection_channels: int = 64,
        fused_channels: int = 128,
        n_res_blocks: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.blocks = list(blocks)

        self.projections = nn.ModuleDict(
            {
                str(b): nn.Sequential(
                    nn.Conv2d(in_channels, projection_channels, 1),
                    nn.GroupNorm(8 if projection_channels % 8 == 0 else 1, projection_channels),
                    nn.GELU(),
                )
                for b in self.blocks
            }
        )

        fused_in = len(self.blocks) * projection_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, fused_channels, 1),
            nn.GroupNorm(8 if fused_channels % 8 == 0 else 1, fused_channels),
            nn.GELU(),
        )
        self.spatial_blocks = nn.Sequential(
            *[
                ResidualSpatialBlock(fused_channels, groups=8, dropout=dropout)
                for _ in range(n_res_blocks)
            ]
        )

        # Attention pooling makes the stage classifier spatially selective.
        self.stage_attention = nn.Conv2d(fused_channels, 1, 1)
        self.stage_classifier = nn.Sequential(
            nn.Linear(fused_channels, max(64, fused_channels // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, fused_channels // 2), len(CLASSES)),
        )

        # Candidate plaque/burden map. Not a validated segmentation yet.
        self.plaque_head = nn.Conv2d(fused_channels, 1, 1)

    @staticmethod
    def masked_softmax(logits, mask):
        # logits/mask: N,1,H,W
        valid = mask > 0.5
        very_neg = torch.finfo(logits.dtype).min
        x = torch.where(valid, logits, torch.tensor(very_neg, device=logits.device, dtype=logits.dtype))
        n = x.shape[0]
        attn = torch.softmax(x.view(n, 1, -1), dim=-1).view_as(x)
        attn = attn * mask
        attn = attn / attn.sum((2, 3), keepdim=True).clamp_min(1e-12)
        return attn

    def forward(self, features: dict[int, torch.Tensor], mask: torch.Tensor):
        projected = [self.projections[str(b)](features[b]) for b in self.blocks]
        x = self.fuse(torch.cat(projected, dim=1))
        shared = self.spatial_blocks(x)

        attn_logits = self.stage_attention(shared)
        attention = self.masked_softmax(attn_logits, mask)
        pooled = (shared * attention).sum((2, 3))
        stage_logits = self.stage_classifier(pooled)

        plaque_logits = self.plaque_head(shared)
        plaque_prob = torch.sigmoid(plaque_logits) * mask
        burden = plaque_prob.sum((2, 3)).squeeze(1) / mask.sum((2, 3)).squeeze(1).clamp_min(1)

        # Masked mean shared feature for cross-scanner representation consistency.
        mean_feat = (shared * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        mean_feat = nn.functional.normalize(mean_feat, dim=1)

        return {
            "stage_logits": stage_logits,
            "shared": shared,
            "attention": attention,
            "plaque_logits": plaque_logits,
            "plaque_prob": plaque_prob,
            "burden": burden,
            "mean_feature": mean_feat,
        }


def initialize_from_old_stage_head(model: WeakLocalizationModel, checkpoint: Path, block: int):
    """Copy old StageHead 384->64 conv into the selected DINO-block projection."""
    if not checkpoint.exists():
        print(f"No old stage checkpoint found at {checkpoint}; using random projection init.", flush=True)
        return False

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    w = state.get("spatial.0.weight")
    b = state.get("spatial.0.bias")
    target_conv = model.projections[str(block)][0]

    if w is None:
        print("Old checkpoint has no 'spatial.0.weight'; skipping reuse.", flush=True)
        return False
    if tuple(w.shape) != tuple(target_conv.weight.shape):
        print(
            f"Old projection shape {tuple(w.shape)} != new {tuple(target_conv.weight.shape)}; skipping reuse.",
            flush=True,
        )
        return False

    with torch.no_grad():
        target_conv.weight.copy_(w)
        if b is not None and target_conv.bias is not None and tuple(b.shape) == tuple(target_conv.bias.shape):
            target_conv.bias.copy_(b)

    print(
        f"Initialized DINO block {block} 384->64 projection from old stage-classification head.",
        flush=True,
    )
    return True


# ============================================================================
# WEAK LOSSES
# ============================================================================

def temporal_ranking_loss(burden: torch.Tensor, meta: list[dict], margin: float):
    """Enforce baseline > partial > clean within same physical_id + scanner."""
    groups = defaultdict(dict)
    for i, m in enumerate(meta):
        groups[(m["physical_id"], m["scanner"])][m["stage"]] = i

    losses = []
    for stage_idx in groups.values():
        pairs = []
        if "baseline" in stage_idx and "partial" in stage_idx:
            pairs.append((stage_idx["baseline"], stage_idx["partial"]))
        if "partial" in stage_idx and "clean" in stage_idx:
            pairs.append((stage_idx["partial"], stage_idx["clean"]))
        # Two-stage groups, or backup direct ordering.
        if "partial" not in stage_idx and "baseline" in stage_idx and "clean" in stage_idx:
            pairs.append((stage_idx["baseline"], stage_idx["clean"]))

        for high, low in pairs:
            losses.append(torch.relu(margin - (burden[high] - burden[low])))

    if not losses:
        return burden.sum() * 0.0
    return torch.stack(losses).mean()


def scanner_burden_consistency_loss(burden: torch.Tensor, meta: list[dict]):
    """Same physical specimen + stage should have similar burden across scanners."""
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["physical_id"], m["stage"])].append(i)

    losses = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        vals = burden[idxs]
        losses.append(((vals - vals.mean()) ** 2).mean())

    if not losses:
        return burden.sum() * 0.0
    return torch.stack(losses).mean()


def scanner_feature_consistency_loss(mean_feature: torch.Tensor, meta: list[dict]):
    """Reduce scanner identity in the shared representation for paired scans."""
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["physical_id"], m["stage"])].append(i)

    losses = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        f = mean_feature[idxs]
        centroid = nn.functional.normalize(f.mean(0, keepdim=True), dim=1)
        # 1 - cosine similarity to paired-scan centroid.
        losses.append((1.0 - (f * centroid).sum(1)).mean())

    if not losses:
        return mean_feature.sum() * 0.0
    return torch.stack(losses).mean()


def class_weights_from_rows(rows: pd.DataFrame, device: str):
    y = rows["stage"].map(CLASS_TO_IDX).to_numpy()
    counts = np.bincount(y, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError(f"Training split missing stage(s), counts={counts.tolist()}")
    return torch.tensor(
        len(y) / (len(CLASSES) * counts), dtype=torch.float32, device=device
    )


def compute_loss(outputs, labels, meta, class_weights, args):
    stage = nn.functional.cross_entropy(
        outputs["stage_logits"], labels, weight=class_weights
    )
    rank = temporal_ranking_loss(outputs["burden"], meta, args.rank_margin)
    burden_cons = scanner_burden_consistency_loss(outputs["burden"], meta)
    feat_cons = scanner_feature_consistency_loss(outputs["mean_feature"], meta)

    total = (
        stage
        + args.lambda_rank * rank
        + args.lambda_burden_consistency * burden_cons
        + args.lambda_feature_consistency * feat_cons
    )
    return total, {
        "stage": stage,
        "rank": rank,
        "burden_consistency": burden_cons,
        "feature_consistency": feat_cons,
    }


# ============================================================================
# TRAIN / EVALUATE
# ============================================================================

def move_batch(batch, device):
    features = {b: x.to(device, non_blocking=True) for b, x in batch["features"].items()}
    mask = batch["mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return features, mask, labels, batch["meta"]


def run_epoch(model, loader, device, class_weights, args, optimizer=None):
    training = optimizer is not None
    model.train(training)

    sums = defaultdict(float)
    n_scans = 0
    ys, preds = [], []

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        features, mask, labels, meta = move_batch(batch, device)

        with torch.set_grad_enabled(training):
            outputs = model(features, mask)
            total, pieces = compute_loss(outputs, labels, meta, class_weights, args)

            if training:
                (total / args.grad_accum).backward()
                if step % args.grad_accum == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        bs = len(labels)
        n_scans += bs
        sums["total"] += float(total.detach().cpu()) * bs
        for k, v in pieces.items():
            sums[k] += float(v.detach().cpu()) * bs

        p = outputs["stage_logits"].argmax(1)
        ys.extend(labels.detach().cpu().tolist())
        preds.extend(p.detach().cpu().tolist())

    out = {k: v / max(n_scans, 1) for k, v in sums.items()}
    out["accuracy"] = accuracy_score(ys, preds)
    out["balanced_accuracy"] = balanced_accuracy_score(ys, preds)
    return out


@torch.no_grad()
def evaluate_detailed(model, loader, device, class_weights, args):
    model.eval()
    rows = []
    metric_accum = defaultdict(float)
    n_scans = 0

    for batch in loader:
        features, mask, labels, meta = move_batch(batch, device)
        outputs = model(features, mask)
        total, pieces = compute_loss(outputs, labels, meta, class_weights, args)
        probs = outputs["stage_logits"].softmax(1)
        pred = probs.argmax(1)

        bs = len(labels)
        n_scans += bs
        metric_accum["total"] += float(total.cpu()) * bs
        for k, v in pieces.items():
            metric_accum[k] += float(v.cpu()) * bs

        for i, m in enumerate(meta):
            rec = dict(m)
            rec["true_stage"] = CLASSES[int(labels[i])]
            rec["predicted_stage"] = CLASSES[int(pred[i])]
            rec["correct"] = int(pred[i]) == int(labels[i])
            rec["burden"] = float(outputs["burden"][i].cpu())
            for c, name in enumerate(CLASSES):
                rec[f"p_{name}"] = float(probs[i, c].cpu())
            rec["true_class_probability"] = float(probs[i, int(labels[i])].cpu())
            rows.append(rec)

    pred_df = pd.DataFrame(rows)
    y = pred_df["true_stage"].map(CLASS_TO_IDX).to_numpy()
    p = pred_df["predicted_stage"].map(CLASS_TO_IDX).to_numpy()
    metrics = {k: v / max(n_scans, 1) for k, v in metric_accum.items()}
    metrics["accuracy"] = accuracy_score(y, p)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y, p)
    return metrics, pred_df


def plot_training(history: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(history["epoch"], history["train_total"], label="Train total")
    ax.plot(history["epoch"], history["val_total"], label="Validation total")
    ax.plot(history["epoch"], history["val_stage"], label="Validation stage CE", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Weak-localization pretraining")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(pred_df: pd.DataFrame, out_path: Path):
    y = pred_df["true_stage"].map(CLASS_TO_IDX).to_numpy()
    p = pred_df["predicted_stage"].map(CLASS_TO_IDX).to_numpy()
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    rs = cm.sum(1, keepdims=True)
    frac = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    im = ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
    for r in range(3):
        for c in range(3):
            ax.text(
                c, r, f"{cm[r,c]}\n({100*frac[r,c]:.1f}%)",
                ha="center", va="center",
                color="white" if frac[r,c] >= 0.5 else "black",
            )
    ax.set_xticks(range(3), [x.title() for x in CLASSES])
    ax.set_yticks(range(3), [x.title() for x in CLASSES])
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title("Validation stage classification")
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_validation_maps(model, dataset, device, blocks, out_root: Path, max_examples: int = 12):
    """Save candidate burden map + stage attention for selected validation scans."""
    out_root.mkdir(parents=True, exist_ok=True)
    saved = 0

    # Iterate whole specimens; save up to max_examples scans total.
    for sidx in range(len(dataset)):
        item = dataset[sidx]
        features = {b: item["features"][b].to(device) for b in blocks}
        mask = item["mask"].to(device)
        outputs = model(features, mask)
        probs = outputs["stage_logits"].softmax(1)

        for i, meta in enumerate(item["meta"]):
            if saved >= max_examples:
                return
            folder = out_root / meta["scanner"] / meta["sample_name"]
            folder.mkdir(parents=True, exist_ok=True)

            with np.load(Path(meta["map_dir"]) / "maps.npz") as z:
                rgb = z["rgb"].astype(np.float32)
                valid = z["valid"].astype(bool)

            plaque = outputs["plaque_prob"][i, 0].cpu().numpy()
            attn = outputs["attention"][i, 0].cpu().numpy()
            # Normalize attention only for display; preserve raw attention in NPZ.
            attn_disp = attn / max(float(attn.max()), 1e-12)

            def upsample(a):
                return np.array(
                    Image.fromarray(a.astype(np.float32), mode="F").resize(
                        (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )

            plaque_full = upsample(plaque)
            attn_full = upsample(attn_disp)
            plaque_full[~valid] = 0
            attn_full[~valid] = 0

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(rgb)
            axes[1].imshow(attn_full, cmap="inferno", alpha=np.clip(attn_full * 0.65, 0, 0.65))
            axes[1].set_title("Stage attention")
            axes[2].imshow(rgb)
            axes[2].imshow(plaque_full, cmap="inferno", alpha=np.clip(plaque_full * 0.65, 0, 0.65))
            axes[2].set_title("Candidate plaque/burden map")
            for ax in axes:
                ax.axis("off")

            true_idx = CLASS_TO_IDX[meta["stage"]]
            pred_idx = int(probs[i].argmax().cpu())
            fig.suptitle(
                f"{meta['scanner']} {meta['sample_name']} | true={meta['stage']} | "
                f"pred={CLASSES[pred_idx]} | P(true)={float(probs[i,true_idx]):.3f} | "
                f"burden={float(outputs['burden'][i]):.3f}",
                fontsize=10,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            safe_savefig(fig, folder / "weak_localization_preview.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            np.savez_compressed(
                folder / "weak_localization_maps.npz",
                plaque_probability_patch=plaque,
                stage_attention_patch=attn,
                burden=np.array(float(outputs["burden"][i].cpu())),
                stage_probabilities=probs[i].cpu().numpy(),
                classes=np.array(CLASSES),
            )
            saved += 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DATA_ROOT / "_dino_preparation_80_10_10")
    p.add_argument("--modality", choices=["rgb", "gray"], default="rgb")
    p.add_argument("--blocks", nargs="+", type=int, default=[3, 6, 9, 12])
    p.add_argument("--dino-model", default="dinov2_vits14_reg")
    p.add_argument("--skip-feature-preparation", action="store_true")
    p.add_argument("--force-features", action="store_true")

    p.add_argument("--projection-channels", type=int, default=64)
    p.add_argument("--fused-channels", type=int, default=128)
    p.add_argument("--res-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--specimen-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--rank-margin", type=float, default=0.10)
    p.add_argument("--lambda-rank", type=float, default=1.0)
    p.add_argument("--lambda-burden-consistency", type=float, default=0.5)
    p.add_argument("--lambda-feature-consistency", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-preview-examples", type=int, default=12)

    p.add_argument(
        "--init-stage-head",
        type=Path,
        default=None,
        help="Existing best_stage_head.pt. Default: <root>/stage_classification/best_stage_head.pt",
    )
    p.add_argument("--no-init-stage-head", action="store_true")
    args = p.parse_args()

    if min(args.epochs, args.patience, args.specimen_batch_size, args.grad_accum) <= 0:
        p.error("epochs/patience/specimen-batch-size/grad-accum must be positive")
    if args.projection_channels != 64 and not args.no_init_stage_head:
        print(
            "Note: projection_channels != 64, so old 384->64 stage projection cannot be reused unless shapes happen to match.",
            flush=True,
        )
    if not args.blocks:
        p.error("At least one DINO block is required")

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    manifest_path = args.root / "scan_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    required = {"scanner", "sample_name", "physical_id", "stage", "split", "map_dir"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    bad_stage = set(manifest["stage"]) - set(CLASSES)
    if bad_stage:
        raise ValueError(f"Unexpected stages: {sorted(bad_stage)}")
    verify_specimen_safe_split(manifest)

    train_rows = manifest[manifest["split"] == "train"].copy()
    val_rows = manifest[manifest["split"] == "val"].copy()
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("This script expects manifest train and val splits")

    # Keep test split untouched.
    dev_rows = manifest[manifest["split"].isin(["train", "val"])].copy()
    cache_file = multilayer_cache_name(args.modality, args.blocks)

    if not args.skip_feature_preparation:
        prepare_multilayer_features(
            dev_rows,
            device=device,
            model_name=args.dino_model,
            blocks=args.blocks,
            modality=args.modality,
            force=args.force_features,
        )

    # Verify at least one cache before training.
    probe = Path(dev_rows.iloc[0]["map_dir"]) / cache_file
    if not probe.exists():
        raise FileNotFoundError(
            f"Missing multi-layer cache {probe}. Rerun without --skip-feature-preparation."
        )

    out = args.root / f"weak_localization_pretrain_{args.modality}"
    out.mkdir(parents=True, exist_ok=True)

    train_ds = SpecimenDataset(train_rows, cache_file, args.blocks)
    val_ds = SpecimenDataset(val_rows, cache_file, args.blocks)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.specimen_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=specimen_collate,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.specimen_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=specimen_collate,
        pin_memory=device.startswith("cuda"),
    )

    # Infer DINO channel width from cache instead of hard-coding 384.
    with np.load(probe) as z:
        in_channels = int(z[f"block_{args.blocks[-1]}"].shape[0])

    model = WeakLocalizationModel(
        blocks=args.blocks,
        in_channels=in_channels,
        projection_channels=args.projection_channels,
        fused_channels=args.fused_channels,
        n_res_blocks=args.res_blocks,
        dropout=args.dropout,
    )

    init_ckpt = args.init_stage_head or (args.root / "stage_classification" / "best_stage_head.pt")
    reused_stage_projection = False
    if not args.no_init_stage_head:
        reused_stage_projection = initialize_from_old_stage_head(model, init_ckpt, args.blocks[-1])

    model = model.to(device)
    class_weights = class_weights_from_rows(train_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    print(
        f"Train: {len(train_rows)} scans / {train_rows.physical_id.nunique()} specimens | "
        f"Val: {len(val_rows)} scans / {val_rows.physical_id.nunique()} specimens",
        flush=True,
    )
    print(
        f"Multi-layer DINO blocks={args.blocks}; projection={args.projection_channels}; "
        f"fused={args.fused_channels}; residual blocks={args.res_blocks}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, class_weights, args, optimizer=optimizer)
        va = run_epoch(model, val_loader, device, class_weights, args, optimizer=None)

        rec = {"epoch": epoch}
        for k, v in tr.items():
            rec[f"train_{k}"] = v
        for k, v in va.items():
            rec[f"val_{k}"] = v
        history.append(rec)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train total={tr['total']:.4f} stage={tr['stage']:.4f} "
            f"rank={tr['rank']:.4f} | val total={va['total']:.4f} "
            f"BA={va['balanced_accuracy']:.3f}",
            flush=True,
        )

        if va["total"] < best_val:
            best_val = va["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "blocks": args.blocks,
                    "in_channels": in_channels,
                    "projection_channels": args.projection_channels,
                    "fused_channels": args.fused_channels,
                    "res_blocks": args.res_blocks,
                    "classes": CLASSES,
                    "best_epoch": best_epoch,
                    "best_val_total_loss": best_val,
                    "modality": args.modality,
                    "cache_file": cache_file,
                    "reused_stage_projection": reused_stage_projection,
                },
                out / "best_weak_localization_model.pt",
            )

        if epoch - best_epoch >= args.patience:
            print("Early stopping.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model")

    model.load_state_dict(best_state)
    model.eval()

    history = pd.DataFrame(history)
    plot_training(history, out / "training_progress.png")

    val_metrics, val_pred = evaluate_detailed(
        model, val_loader, device, class_weights, args
    )
    val_pred.to_csv(out / "validation_predictions.csv", index=False)
    pd.DataFrame([val_metrics]).to_csv(out / "validation_metrics.csv", index=False)
    plot_confusion(val_pred, out / "validation_confusion_matrix.png")

    save_validation_maps(
        model,
        val_ds,
        device,
        args.blocks,
        out / "validation_maps",
        max_examples=args.max_preview_examples,
    )

    config = {
        "purpose": "weak localization pretraining before validated plaque segmentation",
        "dino_model": args.dino_model,
        "dino_frozen": True,
        "dino_blocks": args.blocks,
        "modality": args.modality,
        "cache_file": cache_file,
        "projection_channels": args.projection_channels,
        "fused_channels": args.fused_channels,
        "res_blocks": args.res_blocks,
        "loss_weights": {
            "stage_ce": 1.0,
            "temporal_ranking": args.lambda_rank,
            "cross_scanner_burden_consistency": args.lambda_burden_consistency,
            "cross_scanner_feature_consistency": args.lambda_feature_consistency,
        },
        "rank_margin": args.rank_margin,
        "old_stage_projection_checkpoint": str(init_ckpt),
        "reused_old_384_to_64_projection": reused_stage_projection,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "test_split_used": False,
        "warning": (
            "Candidate plaque/burden map is weakly supervised by stage ordering and scanner consistency; "
            "it is not yet a validated plaque segmentation."
        ),
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    print("\nDone.", flush=True)
    print(f"Output: {out}", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Validation balanced accuracy: {val_metrics['balanced_accuracy']:.3f}", flush=True)
    print("Next: validate partial-sample spatial maps / Grad-CAM before using pseudo-labels.", flush=True)


if __name__ == "__main__":
    main()
