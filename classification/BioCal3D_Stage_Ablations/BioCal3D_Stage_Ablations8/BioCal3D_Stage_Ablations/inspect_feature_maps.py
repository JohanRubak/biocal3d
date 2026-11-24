"""Inspect native DINOv3/Concerto features with PCA and clicked cosine references.

No stage labels, stage heads or pretrained scene-segmentation heads are used.
Run beside make_feature_pseudomasks.py and the original stage-ablation modules.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from make_feature_pseudomasks import load_local, render, project
from stage_ablation import MODES, feature_path, read_json
from study_data import digest, write_json

VERSION = "feature-inspector-1.0"


def positions(meta, eligible, image_shape, center, field):
    h, w = image_shape
    if "shape" in meta:
        gh, gw = meta["shape"]
        yy, xx = np.mgrid[:gh, :gw]
        return np.column_stack(((xx.ravel() + .5) * w / gw - .5,
                                (yy.ravel() + .5) * h / gh - .5))[eligible]
    xyz = meta["xyz_mm"]
    return np.column_stack(((xyz[:, 0] - center[0] + field / 2) * w / field - .5,
                            (center[1] + field / 2 - xyz[:, 1]) * h / field - .5))


def per_scan_pca(x, seed=123):
    # Like the repository PCA demos: fit the axes within the displayed specimen.
    # These colours are not shared across scans or models.
    n = min(9, len(x) - 1, x.shape[1])
    if n < 6:
        raise ValueError("Need at least seven native locations and six feature channels for PCA")
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
    coords = pca.fit_transform(x).astype(np.float32)
    return coords, pca.explained_variance_ratio_


def pca_rgb(coords, start=0, style="components"):
    if style == "concerto":
        # The Concerto example mixes PCs 1-3 and 4-6 into RGB, then rescales.
        values = .6 * coords[:, :3] + .4 * coords[:, 3:6]
    else:
        start = min(start, coords.shape[1] - 3)
        values = coords[:, start:start + 3]
    low, high = values.min(axis=0), values.max(axis=0)
    return np.clip((values - low) / np.maximum(high - low, 1e-9), 0, 1)


def choose_region(xy, click, radius_mm, image_shape, field):
    # Convert both axes to physical mm before selecting a circular reference region.
    h, w = image_shape
    delta = (xy - click) * np.array([field / w, field / h])
    distance = np.linalg.norm(delta, axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] > .8:
        raise ValueError("No native feature within 0.8 mm of this click; choose a covered region")
    ids = np.flatnonzero(distance <= radius_mm) if radius_mm else np.array([], dtype=int)
    return ids if len(ids) else np.array([nearest])


def reference_similarities(target_x, reference_x, positive_groups, negative_groups):
    target = normalize(target_x)
    source = normalize(reference_x)
    result = {}
    for name, groups in (("positive", positive_groups), ("negative", negative_groups)):
        if not groups:
            continue
        prototypes = np.stack([source[ids].mean(axis=0) for ids in groups])
        if np.any(np.linalg.norm(prototypes, axis=1) < 1e-10):
            raise ValueError("Reference feature average is zero; choose another region")
        # Max over clicked region prototypes preserves multiple appearances.
        result[name] = np.clip((target @ normalize(prototypes).T).max(axis=1), -1, 1)
    if "positive" in result and "negative" in result:
        result["margin"] = result["positive"] - result["negative"]
    return result


def prompt_labels(margin, uncertainty):
    labels = np.full(len(margin), -1, np.int8)
    labels[margin > uncertainty] = 1
    labels[margin < -uncertainty] = 0
    return labels


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


class Inspector:
    def __init__(self, args, target_row, reference_row):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        self.plt, self.args = plt, args
        self.target_row, self.reference_row = target_row, reference_row
        self.rgb, self.valid, self.center, self.field = render(args.run, target_row)
        self.ref_rgb, self.ref_valid, self.ref_center, self.ref_field = render(args.run, reference_row)
        self.clicks = []
        self.kind = "positive"
        self.radius = args.radius_mm
        self.uncertainty = args.margin
        self.pc_start = args.pc_start - 1
        self.robust = not args.fixed_scale
        self.data = {}
        for mode in args.modes:
            x, eligible, weights, meta = load_local(args.run, target_row, mode, args.min_coverage)
            if target_row["row_id"] == reference_row["row_id"]:
                rx, re, rm = x, eligible, meta
            else:
                rx, re, _, rm = load_local(args.run, reference_row, mode, args.min_coverage)
            if x.shape[1] != rx.shape[1]:
                raise ValueError(f"Reference and target channel counts differ for {mode}")
            coords, variance = per_scan_pca(x)
            xyz = positions(rm, re, self.ref_valid.shape, self.ref_center, self.ref_field)
            clusters = MiniBatchKMeans(n_clusters=min(args.clusters, len(x)), n_init=3,
                                      random_state=123, batch_size=2048).fit_predict(normalize(x))
            self.data[mode] = dict(x=x, reference_x=rx, eligible=eligible, weights=weights,
                meta=meta, reference_meta=rm, reference_xy=xyz, coords=coords,
                pca_variance=variance, clusters=clusters)
        self.fig, self.axes = plt.subplots(len(args.modes), 5, figsize=(16, 3.25 * len(args.modes) + 1.5), squeeze=False)
        self.fig.subplots_adjust(left=.025, right=.98, top=.76, bottom=.08, wspace=.18, hspace=.35)
        self.fig.suptitle(f"{target_row['scan_id']} | PCA, cosine similarity and prompted masks", fontsize=13)
        self.status = self.fig.text(.03, .025, "Select Plaque or Clean, then click an original image. S saves. Arrows browse.", fontsize=9)
        self.reference_fig = None
        if target_row["row_id"] != reference_row["row_id"]:
            self.reference_fig, self.reference_ax = plt.subplots(figsize=(6, 6))
            self.reference_ax.imshow(self.ref_rgb)
            self.reference_ax.set_title("Click reference: " + reference_row["scan_id"])
            self.reference_ax.set_axis_off()
            self.reference_fig.canvas.mpl_connect("button_press_event", self.on_reference_click)
        self.buttons = []
        actions = [("Plaque", lambda _: self.set_kind("positive")), ("Clean", lambda _: self.set_kind("negative")),
                   ("Undo", self.undo), ("Clear", self.clear), ("Save", lambda _: self.save()),
                   ("Scale", self.toggle_scale), ("Prev", lambda _: self.navigate(-1)),
                   ("Next", lambda _: self.navigate(1))]
        for i, (label, callback) in enumerate(actions):
            button = Button(self.fig.add_axes([.03 + .09 * i, .9, .08, .04]), label)
            button.on_clicked(callback); self.buttons.append(button)
        self.radius_slider = Slider(self.fig.add_axes([.09, .855, .2, .02]), "Radius mm", 0, 1,
                                    valinit=self.radius, valfmt="%.2f")
        self.margin_slider = Slider(self.fig.add_axes([.42, .855, .2, .02]), "Uncertain +/-", 0, .2,
                                    valinit=self.uncertainty, valfmt="%.3f")
        max_start = min(d["coords"].shape[1] - 3 for d in self.data.values())
        self.pc_slider = Slider(self.fig.add_axes([.77, .855, .18, .02]), "First PC", 1, max_start + 1,
                                valinit=min(self.pc_start + 1, max_start + 1), valstep=1)
        self.pc_start = int(self.pc_slider.val) - 1
        self.radius_slider.on_changed(self.change_radius)
        self.margin_slider.on_changed(self.change_margin)
        self.pc_slider.on_changed(self.change_pc)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.navigation = None
        self.colorbars = []
        self.refresh()

    def notify(self, message):
        self.status.set_text(message)
        self.fig.canvas.draw_idle()
        print(message, flush=True)

    def set_kind(self, kind):
        self.kind = kind
        self.notify("Click a " + ("plaque-like" if kind == "positive" else "clean-like") + " reference region.")

    def add_click(self, x, y, kind=None, radius=None):
        col, row = int(round(x)), int(round(y))
        if not (0 <= row < self.ref_valid.shape[0] and 0 <= col < self.ref_valid.shape[1] and self.ref_valid[row, col]):
            self.notify("Click inside the reference specimen."); return False
        click = dict(x=float(x), y=float(y), kind=kind or self.kind,
                     radius_mm=self.radius if radius is None else float(radius))
        try:
            for data in self.data.values():
                ids = choose_region(data["reference_xy"], [x, y], click["radius_mm"], self.ref_valid.shape, self.ref_field)
                opposite = [c for c in self.clicks if c["kind"] != click["kind"]]
                other_ids = [choose_region(data["reference_xy"], [c["x"], c["y"]], c["radius_mm"], self.ref_valid.shape, self.ref_field) for c in opposite]
                if other_ids and np.intersect1d(ids, np.concatenate(other_ids)).size:
                    raise ValueError("Plaque/clean prompts overlap in at least one model's native cells; move them farther apart")
        except ValueError as exc:
            self.notify(str(exc)); return False
        self.clicks.append(click)
        self.refresh()
        return True

    def on_click(self, event):
        if event.inaxes not in self.axes[:, 0]:
            return
        if self.reference_fig is not None:
            self.notify("Select reference points in the separate reference window."); return
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            self.add_click(event.xdata, event.ydata)

    def on_reference_click(self, event):
        if event.inaxes == self.reference_ax and event.xdata is not None and event.ydata is not None and event.button == 1:
            self.add_click(event.xdata, event.ydata)

    def undo(self, _=None):
        if self.clicks: self.clicks.pop()
        self.refresh()

    def clear(self, _=None):
        self.clicks.clear(); self.refresh()

    def change_radius(self, value):
        self.radius = float(value)

    def change_margin(self, value):
        self.uncertainty = float(value); self.refresh()

    def change_pc(self, value):
        self.pc_start = int(value) - 1; self.refresh()

    def toggle_scale(self, _=None):
        self.robust = not self.robust; self.refresh()

    def navigate(self, direction):
        if self.navigation:
            self.navigation(direction)

    def on_key(self, event):
        if event.key == "p": self.set_kind("positive")
        elif event.key == "c": self.set_kind("negative")
        elif event.key == "s": self.save()
        elif event.key in ("backspace", "u"): self.undo()
        elif event.key == "right": self.navigate(1)
        elif event.key == "left": self.navigate(-1)

    def compute(self, data):
        groups = {k: [choose_region(data["reference_xy"], [c["x"], c["y"]], c["radius_mm"], self.ref_valid.shape, self.ref_field)
                      for c in self.clicks if c["kind"] == k] for k in ("positive", "negative")}
        return reference_similarities(data["x"], data["reference_x"], groups["positive"], groups["negative"])

    def image(self, values, data):
        return project(values, data["meta"], data["eligible"], self.valid, self.center, self.field)

    def marks(self, ax):
        for i, c in enumerate(self.clicks):
            color = "gold" if c["kind"] == "positive" else "cyan"
            ax.plot(c["x"], c["y"], marker="+", color=color, markersize=10, markeredgewidth=2)
            ax.annotate(str(i + 1), (c["x"] + 4, c["y"] - 4), color=color, fontsize=8)

    def refresh(self):
        for bar in self.colorbars: bar.remove()
        self.colorbars = []
        for axrow, (mode, data) in zip(self.axes, self.data.items()):
            for ax in axrow:
                ax.clear(); ax.set_axis_off()
            axrow[0].imshow(self.rgb); axrow[0].set_title(mode + "\nOriginal / reference clicks", fontsize=10)
            if self.reference_fig is None: self.marks(axrow[0])
            rgb = pca_rgb(data["coords"], self.pc_start, self.args.pca_style)
            axrow[1].imshow(np.nan_to_num(self.image(rgb, data)))
            title = "PCA: 0.6 x PCs 1-3 + 0.4 x 4-6" if self.args.pca_style == "concerto" else f"PCA: PCs {self.pc_start + 1}-{self.pc_start + 3}"
            axrow[1].set_title(title + "\nPer-specimen colour scale", fontsize=10)
            scores = self.compute(data)
            for column, name, title in ((2, "positive", "Plaque-reference similarity"), (3, "negative", "Clean-reference similarity")):
                ax = axrow[column]
                if name not in scores:
                    ax.text(.5, .5, "Add a " + name + " reference", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title, fontsize=10); continue
                values = scores[name]
                low, high = np.quantile(values, [.01, .99]) if self.robust else (-1., 1.)
                if high - low < 1e-6:
                    low, high = max(-1., float(low) - .005), min(1., float(high) + .005)
                im = ax.imshow(np.ma.masked_invalid(self.image(values, data)), cmap="magma", vmin=low, vmax=high)
                bar = self.fig.colorbar(im, ax=ax, fraction=.04, pad=.02, format="%.3f")
                bar.ax.tick_params(labelsize=7); self.colorbars.append(bar)
                ax.set_title(title + ("\nCosine, display clipped p1-p99" if self.robust else "\nCosine, fixed [-1,1]"), fontsize=10)
            axrow[4].imshow(self.rgb)
            if "margin" in scores:
                labels = prompt_labels(scores["margin"], self.uncertainty)
                label_img = self.image(labels, data)
                overlay = np.zeros((*self.valid.shape, 4), np.float32)
                overlay[label_img == 1] = [1, .75, 0, .8]
                overlay[label_img == -1] = [.2, .7, 1, .35]
                axrow[4].imshow(overlay)
                axrow[4].set_title("Prompted candidate mask\nGold: plaque-like; blue: uncertain", fontsize=10)
            else:
                axrow[4].set_title("Prompted segmentation\nNeeds plaque AND clean references", fontsize=10)
        if self.reference_fig is not None:
            self.reference_ax.clear(); self.reference_ax.imshow(self.ref_rgb)
            self.reference_ax.set_title("Click reference: " + self.reference_row["scan_id"])
            self.reference_ax.set_axis_off(); self.marks(self.reference_ax)
            self.reference_fig.canvas.draw_idle()
        self.status.set_text(f"{len(self.clicks)} reference regions | next click: {self.kind} | uncertainty margin +/-{self.uncertainty:.3f} | S: save; arrows: browse")
        self.fig.canvas.draw_idle()

    def save(self):
        from matplotlib import pyplot as plt
        root = self.args.out
        root.mkdir(parents=True, exist_ok=True)
        stem = safe_name(self.target_row["scan_id"])
        i = 1
        while (root / f"{stem}_{i:03d}").exists(): i += 1
        out = root / f"{stem}_{i:03d}"
        out.mkdir()
        metadata = dict(version=VERSION, run=str(self.args.run.resolve()),
            target_scan=self.target_row["scan_id"], reference_scan=self.reference_row["scan_id"],
            target_geometry_sha256=self.target_row["geometry_sha256"], reference_geometry_sha256=self.reference_row["geometry_sha256"],
            cohort_sha256=digest(self.args.run / "cohort.json"), modes=self.args.modes,
            clicks=self.clicks, uncertainty_margin=self.uncertainty, pca_style=self.args.pca_style,
            selected_first_pc=self.pc_start + 1, robust_display_scale=self.robust,
            min_coverage=self.args.min_coverage,
            feature_cache_sha256={mode: digest(feature_path(self.args.run, mode, self.target_row)) for mode in self.args.modes},
            reference_cache_sha256={mode: digest(feature_path(self.args.run, mode, self.reference_row)) for mode in self.args.modes},
            interpretation="Raw full-dimensional feature cosine; clicked plaque/clean prototypes. Prompted candidate masks, not automatic or validated segmentation.")
        write_json(out / "prompts.json", metadata)
        Image.fromarray(self.ref_rgb).save(out / "reference_original.png")
        Image.fromarray(self.rgb).save(out / "target_original.png")
        ref_plot, ref_axis = plt.subplots(figsize=(5, 5), layout="constrained")
        ref_axis.imshow(self.ref_rgb); self.marks(ref_axis)
        ref_axis.set_title("Reference: " + self.reference_row["scan_id"], fontsize=9)
        ref_axis.set_axis_off(); ref_plot.savefig(out / "reference_marked.png", dpi=150); plt.close(ref_plot)
        self.fig.savefig(out / "comparison.png", dpi=150)
        for mode, data in self.data.items():
            folder = out / mode; folder.mkdir()
            scores = self.compute(data)
            arrays = dict(scores)
            arrays.update(eligible=data["eligible"], pca_coordinates=data["coords"], pca_variance=data["pca_variance"],
                          cluster=data["clusters"], feature_cache_sha256=np.array(digest(feature_path(self.args.run, mode, self.target_row))),
                          reference_cache_sha256=np.array(digest(feature_path(self.args.run, mode, self.reference_row))), **data["meta"])
            panel, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
            panel.suptitle(mode + " | " + self.target_row["scan_id"], fontsize=12)
            axes[0, 0].imshow(self.rgb); axes[0, 0].set_title("Original target")
            for axis, start in ((axes[0, 1], 0), (axes[0, 2], 3)):
                image = self.image(pca_rgb(data["coords"], start), data)
                axis.imshow(np.nan_to_num(image)); axis.set_title(f"PCA: PCs {start+1}-{start+3}")
                Image.fromarray(np.uint8(np.nan_to_num(image) * 255)).save(folder / f"pca_pc{start+1}_{start+3}.png")
            blend = self.image(pca_rgb(data["coords"], style="concerto"), data)
            Image.fromarray(np.uint8(np.nan_to_num(blend) * 255)).save(folder / "pca_concerto_blend.png")
            for axis, name in ((axes[1, 0], "positive"), (axes[1, 1], "negative")):
                if name in scores:
                    image = self.image(scores[name], data)
                    arrays[name + "_image"] = image
                    im = axis.imshow(np.ma.masked_invalid(image), cmap="magma", vmin=-1, vmax=1)
                    panel.colorbar(im, ax=axis, shrink=.7)
                    axis.set_title(name + " cosine [-1,1]")
                    Image.fromarray(image.astype(np.float32)).save(folder / f"{name}_cosine.tiff")
                else:
                    axis.text(.5, .5, "No " + name + " reference", ha="center", transform=axis.transAxes)
            axes[1, 2].imshow(self.rgb)
            if "margin" in scores:
                labels = prompt_labels(scores["margin"], self.uncertainty)
                arrays["pseudo_label"] = labels
                label_img = self.image(labels, data)
                covered = np.isfinite(label_img) & self.valid
                mask = np.full(self.valid.shape, 128, np.uint8)
                mask[label_img == 0] = 0; mask[label_img == 1] = 255
                Image.fromarray(mask).save(folder / "prompted_mask.png")
                Image.fromarray(covered.astype(np.uint8) * 255).save(folder / "covered_mask.png")
                Image.fromarray(self.valid.astype(np.uint8) * 255).save(folder / "valid_mask.png")
                arrays["margin_image"] = self.image(scores["margin"], data)
                overlay = np.zeros((*self.valid.shape, 4), np.float32)
                overlay[label_img == 1] = [1, .75, 0, .8]
                overlay[label_img == -1] = [.2, .7, 1, .35]
                axes[1, 2].imshow(overlay)
                axes[1, 2].set_title("Prompted mask\nGold: plaque-like; blue: uncertain", fontsize=10)
            else:
                axes[1, 2].set_title("Both reference classes needed")
            for ax in axes.flat: ax.set_axis_off()
            panel.savefig(folder / "maps.png", dpi=160, bbox_inches="tight", pad_inches=.15); plt.close(panel)
            cluster_img = self.image(data["clusters"], data)
            cluster_rgb = plt.get_cmap("tab20", self.args.clusters)(np.nan_to_num(cluster_img).astype(int))[..., :3]
            cluster_rgb[~np.isfinite(cluster_img)] = 0
            Image.fromarray(np.uint8(cluster_rgb * 255)).save(folder / "unsupervised_clusters.png")
            np.savez_compressed(folder / "native_maps.npz", **arrays)
            if "xyz_mm" in data["meta"]:
                self.save_ply(folder, data, scores)
        self.notify("Saved " + str(out))
        return out

    def save_ply(self, folder, data, scores):
        from make_feature_pseudomasks import point_ply
        n = len(data["x"])
        labels = prompt_labels(scores["margin"], self.uncertainty) if "margin" in scores else np.full(n, -1)
        values = scores.get("margin", np.zeros(n))
        point_ply(folder / "pca_points.ply", data["meta"]["xyz_mm"], values, labels, data["clusters"],
                  pca_rgb(data["coords"], self.pc_start, self.args.pca_style))
        for name in ("positive", "negative"):
            if name in scores:
                rgb = self.plt.get_cmap("magma")((scores[name] + 1) / 2)[:, :3]
                point_ply(folder / f"{name}_similarity_points.ply", data["meta"]["xyz_mm"], scores[name], labels,
                          data["clusters"], rgb)
        if "margin" in scores:
            rgb = np.tile([.6, .6, .6], (n, 1)); rgb[labels == -1] = [.2, .7, 1]; rgb[labels == 1] = [1, .75, 0]
            point_ply(folder / "prompted_segmentation_points.ply", data["meta"]["xyz_mm"], values, labels, data["clusters"], rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scan", help="Exact target scan ID; omit to browse the chosen partition")
    parser.add_argument("--reference", help="Optional other scan ID to click as the reference")
    parser.add_argument("--partition", choices=["validation", "development", "test"], default="validation")
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["dinov3_rgb", "concerto_rgb"])
    parser.add_argument("--out", type=Path, default=Path("feature_inspection"))
    parser.add_argument("--radius-mm", type=float, default=0.)
    parser.add_argument("--margin", type=float, default=.02)
    parser.add_argument("--min-coverage", type=float, default=.5)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--pc-start", type=int, default=1)
    parser.add_argument("--pca-style", choices=["components", "concerto"], default="components")
    parser.add_argument("--fixed-scale", action="store_true", help="Cosine colour range [-1,1] instead of display percentiles")
    parser.add_argument("--positive", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument("--negative", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument("--load-prompts", type=Path, help="Replay exported prompts.json")
    parser.add_argument("--list-scans", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Save maps without opening a desktop window")
    args = parser.parse_args()
    if not 0 <= args.radius_mm <= 1 or not 0 <= args.margin <= .2 or not 0 < args.min_coverage <= 1 or args.clusters < 2 or not 1 <= args.pc_start <= 7:
        parser.error("Require radius 0..1 mm, margin 0..0.2, coverage (0,1], clusters>=2 and first PC 1..7")
    try:
        all_rows = read_json(args.run / "cohort.json")["rows"]
        rows = [r for r in all_rows if r["partition"] == ("development" if args.partition == "validation" else args.partition)]
        if args.partition == "validation": rows = [r for r in rows if r["fold"] == args.validation_fold]
        if args.list_scans:
            for row in rows: print(row["scan_id"], row["stage"], sep="\t")
            return
        replay = read_json(args.load_prompts) if args.load_prompts else None
        if replay:
            if replay["cohort_sha256"] != digest(args.run / "cohort.json"):
                raise ValueError("Replayed prompts belong to a different cohort")
            args.scan, args.reference = replay["target_scan"], replay["reference_scan"]
            args.modes, args.margin = replay["modes"], replay["uncertainty_margin"]
            args.pca_style, args.pc_start = replay["pca_style"], replay["selected_first_pc"]
            args.min_coverage = replay["min_coverage"]
            args.fixed_scale = not replay["robust_display_scale"]
            if set(args.modes) - set(MODES):
                raise ValueError("Unknown mode in saved prompts")
        ids = [r["scan_id"] for r in rows]
        if not rows or (args.scan and args.scan not in ids):
            raise ValueError("Target not in selected partition. Use --list-scans or choose --partition development/test.")
        index = ids.index(args.scan) if args.scan else 0
        lookup = {r["scan_id"]: r for r in all_rows}
        if args.reference and args.reference not in lookup:
            raise ValueError("Unknown reference scan ID")
        if replay:
            for name, row in (("feature_cache_sha256", rows[index]), ("reference_cache_sha256", lookup[args.reference])):
                for mode in args.modes:
                    if replay.get(name, {}).get(mode) != digest(feature_path(args.run, mode, row)):
                        raise ValueError("Spatial features changed since the prompts were saved")
        import matplotlib.pyplot as plt
        if not args.headless:
            try:
                plt.switch_backend("TkAgg")
            except (ImportError, RuntimeError) as exc:
                raise ValueError("Desktop display unavailable. Use a local terminal with Tkinter, or --headless to export maps.") from exc
        while True:
            row = rows[index]
            ref = lookup[args.reference] if args.reference else row
            viewer = Inspector(args, row, ref)
            if replay:
                for c in replay["clicks"]:
                    if not viewer.add_click(c["x"], c["y"], c["kind"], c["radius_mm"]):
                        raise ValueError("A replayed reference could not be applied")
            for kind in ("positive", "negative"):
                for x, y in getattr(args, kind) or []:
                    if not viewer.add_click(x, y, kind):
                        raise ValueError("A command-line reference could not be applied")
            if args.headless:
                viewer.save(); plt.close("all"); break
            state = {"direction": None}
            def navigate(direction):
                state["direction"] = direction
                plt.close(viewer.fig)
                if viewer.reference_fig is not None: plt.close(viewer.reference_fig)
            viewer.navigation = navigate
            plt.show()
            if state["direction"] is None: break
            index = (index + state["direction"]) % len(rows)
            replay = None
            args.positive = args.negative = None
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
