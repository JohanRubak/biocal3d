"""Interactive patch-level pseudo-label verification/correction for BioCal3D.

Purpose
-------
Review the candidate plaque maps produced by biocal3d_weak_localization_pretrain_v3.py
and convert them into human-verified patch labels.

Labels
------
  1  = plaque
  0  = background / no plaque
 -1  = ignore / uncertain

The editor works on the native DINO patch grid (typically 37x37), while displaying
labels over the full-resolution RGB ROI. This avoids pretending the weak map has
pixel-level precision.

Recommended workflow
--------------------
1) First review VALIDATION partial scans to test spatial correspondence:

   python biocal3d_pseudolabel_editor.py --split val

   These validation annotations are for evaluation/validation, NOT model fitting.

2) If the weak maps are useful, export training maps:

   python biocal3d_weak_localization_pretrain_v3.py \
       --skip-feature-preparation --validation-only --export-train-maps

3) Correct TRAIN partial scans:

   python biocal3d_pseudolabel_editor.py --split train

4) Retrain with verified training pseudo-labels:

   python biocal3d_weak_localization_pretrain_v3.py \
       --skip-feature-preparation --use-pseudo-labels

Controls
--------
Mouse on "Corrected pseudo-labels" panel:
  left drag   = current selected label mode

Radio buttons:
  Plaque      = label 1
  Background  = label 0
  Ignore      = label -1

Keyboard:
  1           plaque mode
  0           background mode
  x           ignore mode
  [ / ]       decrease / increase brush radius
  s           save as VERIFIED and go to next
  left/right  previous / next scan
  r           reset to automatic threshold initialization
  c           clear all to ignore
  q           quit

The initial pseudo-label is conservative:
  probability >= high threshold -> plaque
  probability <= low threshold  -> background
  middle values                 -> ignore

Only valid DINO patches can be labeled.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.widgets import Button, RadioButtons, Slider
from PIL import Image

from biocal3d_prepare_dino import DATA_ROOT


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_PLAQUE = 1


def pseudo_label_path(root: Path, row: dict) -> Path:
    return (
        Path(root)
        / str(row["split"])
        / str(row["scanner"])
        / str(row["sample_name"])
        / "pseudo_labels.npz"
    )


def output_preview_path(root: Path, row: dict) -> Path:
    return pseudo_label_path(root, row).with_name("pseudo_labels_preview.png")


def model_map_path(model_output: Path, split: str, row: dict) -> Path:
    folder_name = "validation_maps" if split == "val" else "train_maps"
    return (
        Path(model_output)
        / folder_name
        / str(row["scanner"])
        / str(row["sample_name"])
        / "weak_localization_maps.npz"
    )


def conservative_initial_labels(prob, patch_valid, low, high):
    labels = np.full(prob.shape, LABEL_IGNORE, dtype=np.int8)
    labels[(prob <= low) & patch_valid] = LABEL_BACKGROUND
    labels[(prob >= high) & patch_valid] = LABEL_PLAQUE
    return labels


def label_rgba(labels, patch_valid, alpha=0.50):
    """Create patch-grid RGBA overlay: plaque=orange, background=blue, ignore=transparent."""
    h, w = labels.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    plaque = (labels == LABEL_PLAQUE) & patch_valid
    bg = (labels == LABEL_BACKGROUND) & patch_valid
    # Orange/red for plaque.
    out[plaque, 0] = 1.0
    out[plaque, 1] = 0.25
    out[plaque, 2] = 0.05
    out[plaque, 3] = alpha
    # Cyan/blue for confirmed background.
    out[bg, 0] = 0.05
    out[bg, 1] = 0.45
    out[bg, 2] = 1.0
    out[bg, 3] = alpha * 0.75
    return out


def upsample_float(arr, shape_hw):
    h, w = shape_hw
    return np.array(
        Image.fromarray(arr.astype(np.float32), mode="F").resize(
            (w, h), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    )


def upsample_rgba(arr, shape_hw):
    h, w = shape_hw
    im = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="RGBA")
    return np.asarray(im.resize((w, h), Image.Resampling.NEAREST)).astype(np.float32) / 255.0


def heat_overlay(rgb, prob, valid, max_alpha=0.65):
    prob_full = np.array(
        Image.fromarray(prob.astype(np.float32), mode="F").resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    prob_full[~valid] = 0
    heat = colormaps["inferno"](np.clip(prob_full, 0, 1))[..., :3].astype(np.float32)
    a = (max_alpha * np.clip(prob_full, 0, 1))[..., None]
    out = rgb.copy()
    out[valid] = (1 - a[valid]) * out[valid] + a[valid] * heat[valid]
    return np.clip(out, 0, 1)


def save_status(root: Path, row: dict, labels, verified=True):
    path = Path(root) / "annotation_status.csv"
    rec = {
        "split": row["split"],
        "scanner": row["scanner"],
        "sample_name": row["sample_name"],
        "physical_id": row["physical_id"],
        "stage": row["stage"],
        "verified": bool(verified),
        "n_plaque": int((labels == LABEL_PLAQUE).sum()),
        "n_background": int((labels == LABEL_BACKGROUND).sum()),
        "n_ignore": int((labels == LABEL_IGNORE).sum()),
        "edited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        df = pd.read_csv(path)
        key = (
            (df["split"].astype(str) == str(rec["split"]))
            & (df["scanner"].astype(str) == str(rec["scanner"]))
            & (df["sample_name"].astype(str) == str(rec["sample_name"]))
        )
        df = df.loc[~key].copy()
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    else:
        df = pd.DataFrame([rec])
    df = df.sort_values(["split", "scanner", "physical_id", "stage", "sample_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class Editor:
    def __init__(self, rows, model_output, pseudo_root, low, high, start_index=0):
        self.rows = rows.reset_index(drop=True)
        self.model_output = Path(model_output)
        self.pseudo_root = Path(pseudo_root)
        self.low = float(low)
        self.high = float(high)
        self.index = int(np.clip(start_index, 0, max(len(self.rows) - 1, 0)))
        self.mode = LABEL_PLAQUE
        self.radius = 1
        self.mouse_down = False
        self.labels = None
        self.initial = None
        self.prob = None
        self.patch_valid = None
        self.rgb = None
        self.valid_full = None
        self.row = None

        self.fig = plt.figure(figsize=(15.5, 7.8))
        self.ax_rgb = self.fig.add_axes([0.03, 0.16, 0.26, 0.74])
        self.ax_prob = self.fig.add_axes([0.31, 0.16, 0.26, 0.74])
        self.ax_edit = self.fig.add_axes([0.59, 0.16, 0.26, 0.74])
        self.ax_radio = self.fig.add_axes([0.87, 0.55, 0.11, 0.22])
        self.ax_radius = self.fig.add_axes([0.87, 0.47, 0.11, 0.035])
        self.ax_prev = self.fig.add_axes([0.87, 0.36, 0.05, 0.05])
        self.ax_next = self.fig.add_axes([0.93, 0.36, 0.05, 0.05])
        self.ax_save = self.fig.add_axes([0.87, 0.28, 0.11, 0.055])
        self.ax_reset = self.fig.add_axes([0.87, 0.20, 0.05, 0.05])
        self.ax_clear = self.fig.add_axes([0.93, 0.20, 0.05, 0.05])

        self.radio = RadioButtons(self.ax_radio, ("Plaque", "Background", "Ignore"), active=0)
        self.slider = Slider(self.ax_radius, "Brush", 0, 5, valinit=1, valstep=1)
        self.b_prev = Button(self.ax_prev, "Prev")
        self.b_next = Button(self.ax_next, "Next")
        self.b_save = Button(self.ax_save, "Save + next")
        self.b_reset = Button(self.ax_reset, "Reset")
        self.b_clear = Button(self.ax_clear, "Clear")

        self.radio.on_clicked(self.on_radio)
        self.slider.on_changed(self.on_radius)
        self.b_prev.on_clicked(lambda evt: self.goto(self.index - 1))
        self.b_next.on_clicked(lambda evt: self.goto(self.index + 1))
        self.b_save.on_clicked(lambda evt: self.save_and_next())
        self.b_reset.on_clicked(lambda evt: self.reset())
        self.b_clear.on_clicked(lambda evt: self.clear())

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()

    def on_radio(self, label):
        self.mode = {"Plaque": 1, "Background": 0, "Ignore": -1}[label]

    def on_radius(self, val):
        self.radius = int(val)

    def load_current(self):
        self.row = self.rows.iloc[self.index].to_dict()
        mpath = model_map_path(self.model_output, self.row["split"], self.row)
        if not mpath.exists():
            split_dir = "validation_maps" if self.row["split"] == "val" else "train_maps"
            raise FileNotFoundError(
                f"Missing candidate map: {mpath}\n"
                f"For train annotations, first export train maps with:\n"
                f"  python biocal3d_weak_localization_pretrain_v3.py "
                f"--skip-feature-preparation --validation-only --export-train-maps\n"
                f"Expected under model output/{split_dir}/..."
            )

        with np.load(mpath) as z:
            self.prob = z["plaque_probability_patch"].astype(np.float32)

        with np.load(Path(self.row["map_dir"]) / "maps.npz") as z:
            self.rgb = z["rgb"].astype(np.float32)
            self.valid_full = z["valid"].astype(bool)

        # Original DINO cache provides the exact valid-patch geometry.
        with np.load(Path(self.row["map_dir"]) / "dino_features.npz") as z:
            self.patch_valid = z["patch_valid"].astype(bool)

        if self.patch_valid.shape != self.prob.shape:
            raise ValueError(
                f"patch_valid {self.patch_valid.shape} != candidate map {self.prob.shape}"
            )

        self.initial = conservative_initial_labels(
            self.prob, self.patch_valid, self.low, self.high
        )

        ppath = pseudo_label_path(self.pseudo_root, self.row)
        if ppath.exists():
            with np.load(ppath) as z:
                lab = z["labels_patch"].astype(np.int8)
            self.labels = lab.copy()
        else:
            self.labels = self.initial.copy()

        self.redraw()

    def redraw(self):
        for ax in (self.ax_rgb, self.ax_prob, self.ax_edit):
            ax.clear()
            ax.axis("off")

        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title("RGB")

        cand = heat_overlay(self.rgb, self.prob, self.valid_full)
        self.ax_prob.imshow(cand)
        self.ax_prob.set_title("Candidate plaque map")

        self.ax_edit.imshow(self.rgb)
        rgba = label_rgba(self.labels, self.patch_valid)
        self.ax_edit.imshow(
            upsample_rgba(rgba, self.rgb.shape[:2]),
            interpolation="nearest",
        )
        self.ax_edit.set_title("Corrected pseudo-labels")

        n_p = int((self.labels == LABEL_PLAQUE).sum())
        n_b = int((self.labels == LABEL_BACKGROUND).sum())
        n_i = int((self.labels == LABEL_IGNORE).sum())
        saved = pseudo_label_path(self.pseudo_root, self.row).exists()
        self.fig.suptitle(
            f"{self.index+1}/{len(self.rows)} | {self.row['scanner']} {self.row['sample_name']} | "
            f"{self.row['physical_id']} | stage={self.row['stage']} | "
            f"plaque={n_p}, background={n_b}, ignore={n_i} | "
            f"{'SAVED' if saved else 'not saved'}",
            fontsize=12,
        )
        self.fig.text(
            0.87, 0.10,
            "1 plaque | 0 background | x ignore\n"
            "drag to paint | [ ] brush\n"
            "s save+next | arrows navigate\n"
            "r reset | c clear | q quit",
            fontsize=9,
            va="top",
        )
        self.fig.canvas.draw_idle()

    def event_to_patch(self, event):
        if event.inaxes is not self.ax_edit or event.xdata is None or event.ydata is None:
            return None
        h, w = self.rgb.shape[:2]
        ph, pw = self.labels.shape
        x = int(np.floor(event.xdata / max(w, 1) * pw))
        y = int(np.floor(event.ydata / max(h, 1) * ph))
        x = int(np.clip(x, 0, pw - 1))
        y = int(np.clip(y, 0, ph - 1))
        return y, x

    def paint(self, event):
        rc = self.event_to_patch(event)
        if rc is None:
            return
        cy, cx = rc
        rr = self.radius
        yy, xx = np.ogrid[: self.labels.shape[0], : self.labels.shape[1]]
        brush = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr ** 2
        brush &= self.patch_valid
        self.labels[brush] = self.mode
        self.redraw()

    def on_press(self, event):
        if event.inaxes is self.ax_edit and event.button == 1:
            self.mouse_down = True
            self.paint(event)

    def on_release(self, event):
        self.mouse_down = False

    def on_motion(self, event):
        if self.mouse_down:
            self.paint(event)

    def reset(self):
        self.labels = self.initial.copy()
        self.redraw()

    def clear(self):
        self.labels[:] = LABEL_IGNORE
        self.labels[~self.patch_valid] = LABEL_IGNORE
        self.redraw()

    def save(self):
        ppath = pseudo_label_path(self.pseudo_root, self.row)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ppath,
            labels_patch=self.labels.astype(np.int8),
            verified=np.array(True),
            source_probability_patch=self.prob.astype(np.float32),
            patch_valid=self.patch_valid.astype(np.uint8),
            low_threshold=np.array(self.low, dtype=np.float32),
            high_threshold=np.array(self.high, dtype=np.float32),
            split=np.array(str(self.row["split"])),
            scanner=np.array(str(self.row["scanner"])),
            sample_name=np.array(str(self.row["sample_name"])),
            physical_id=np.array(str(self.row["physical_id"])),
            stage=np.array(str(self.row["stage"])),
            edited_at_utc=np.array(datetime.now(timezone.utc).isoformat()),
        )
        save_status(self.pseudo_root, self.row, self.labels, verified=True)
        self.save_preview()
        print(f"Saved VERIFIED pseudo-label: {ppath}", flush=True)

    def save_preview(self):
        path = output_preview_path(self.pseudo_root, self.row)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(self.rgb); axes[0].set_title("RGB")
        axes[1].imshow(heat_overlay(self.rgb, self.prob, self.valid_full)); axes[1].set_title("Candidate")
        axes[2].imshow(self.rgb)
        axes[2].imshow(upsample_rgba(label_rgba(self.labels, self.patch_valid), self.rgb.shape[:2]))
        axes[2].set_title("Verified pseudo-label")
        for ax in axes: ax.axis("off")
        fig.suptitle(
            f"{self.row['scanner']} {self.row['sample_name']} | {self.row['stage']}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def save_and_next(self):
        self.save()
        self.goto(self.index + 1)

    def goto(self, index):
        if not len(self.rows):
            return
        self.index = int(np.clip(index, 0, len(self.rows) - 1))
        self.load_current()

    def on_key(self, event):
        key = str(event.key).lower()
        if key == "1":
            self.radio.set_active(0)
        elif key == "0":
            self.radio.set_active(1)
        elif key == "x":
            self.radio.set_active(2)
        elif key == "[":
            self.slider.set_val(max(0, self.radius - 1))
        elif key == "]":
            self.slider.set_val(min(5, self.radius + 1))
        elif key == "s":
            self.save_and_next()
        elif key == "left":
            self.goto(self.index - 1)
        elif key == "right":
            self.goto(self.index + 1)
        elif key == "r":
            self.reset()
        elif key == "c":
            self.clear()
        elif key == "q":
            plt.close(self.fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DATA_ROOT / "_dino_preparation_80_10_10")
    p.add_argument("--model-output", type=Path, default=None)
    p.add_argument("--pseudo-root", type=Path, default=None)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["baseline", "partial", "clean", "all"],
        default=["partial"],
        help="Default: partial only. Use --stages all to review every stage.",
    )
    p.add_argument("--scanners", nargs="+", default=None)
    p.add_argument("--low-threshold", type=float, default=0.10)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--skip-verified", action="store_true")
    args = p.parse_args()

    if not (0 <= args.low_threshold < args.high_threshold <= 1):
        p.error("Require 0 <= low-threshold < high-threshold <= 1")

    if args.model_output is None:
        args.model_output = args.root / "weak_localization_pretrain_rgb"
    if args.pseudo_root is None:
        args.pseudo_root = args.root / "weak_pseudo_labels"

    manifest = pd.read_csv(args.root / "scan_manifest.csv")
    rows = manifest[manifest["split"] == args.split].copy()
    if "all" not in args.stages:
        rows = rows[rows["stage"].isin(args.stages)].copy()
    if args.scanners:
        rows = rows[rows["scanner"].isin(args.scanners)].copy()

    rows = rows.sort_values(["scanner", "physical_id", "stage", "sample_name"]).reset_index(drop=True)

    if args.skip_verified:
        keep = []
        for row in rows.to_dict("records"):
            path = pseudo_label_path(args.pseudo_root, row)
            verified = False
            if path.exists():
                try:
                    with np.load(path) as z:
                        verified = bool(z["verified"].item()) if "verified" in z else False
                except Exception:
                    verified = False
            keep.append(not verified)
        rows = rows[np.asarray(keep, dtype=bool)].reset_index(drop=True)

    if len(rows) == 0:
        raise ValueError("No scans match the requested split/stage/scanner filters")

    print(f"Review set: {len(rows)} scans | split={args.split} | stages={args.stages}", flush=True)
    if args.split == "val":
        print(
            "NOTE: validation annotations should be used for spatial validation, not fitted into the model. ",
            "Create TRAIN annotations before using --use-pseudo-labels for direct supervision.",
            flush=True,
        )

    editor = Editor(
        rows=rows,
        model_output=args.model_output,
        pseudo_root=args.pseudo_root,
        low=args.low_threshold,
        high=args.high_threshold,
        start_index=args.start_index,
    )
    plt.show()


if __name__ == "__main__":
    main()
