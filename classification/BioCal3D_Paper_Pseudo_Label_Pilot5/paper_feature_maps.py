"""Render Figure 4-style DINOv3 patch feature maps from saved test-scan caches."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from fixed_split import LEAD_SCAN
from fixed_study import context
from make_feature_pseudomasks import load_local
from make_stage_heatmaps import patch_overlay, plt, render_inputs
from paper_pilot_views import save_figure_png
from study_data import SCANNERS

ROOT = Path(__file__).resolve().parent
MODES = ("dinov3_rgb", "dinov3_gray")
ROW_LABELS = ("Original scan", "DINOv3 · RGB input", "DINOv3 · grayscale input")


def feature_image(local, pca, low, high, roi):
    tokens, eligible, _, meta = local
    channels = np.clip((pca.transform(tokens) - low) / np.maximum(high - low, 1e-8), 0, 1)
    shape = meta["shape"]
    grid = np.zeros((eligible.size, 3), dtype=np.float32)
    grid[eligible] = channels
    coverage = np.where(eligible.reshape(shape), meta["coverage"], 0)
    height, width = roi.shape
    projected = np.stack([patch_overlay(channel.reshape(shape), coverage, width, height)
                          for channel in grid.T], axis=-1)
    valid = roi & np.isfinite(projected).all(axis=-1)
    return np.where(valid[..., None], projected, 1)


def export(run, out, sample):
    _, roles, _ = context(run)
    rows = [row for row in roles["test"] if row["sample_name"].casefold() == sample.casefold()]
    rows.sort(key=lambda row: SCANNERS.index(row["scanner"]))
    if not rows:
        raise ValueError(f"No held-out scans match {sample}. Use a specimen-stage ID such as {LEAD_SCAN}.")
    sample = rows[0]["sample_name"]
    originals = [render_inputs(run, row)[:2] for row in rows]
    features = {(mode, row["scan_id"]): load_local(run, row, mode, min_coverage=.5)
                for mode in MODES for row in rows}

    # One basis and one colour range for all displayed inputs and scanners.
    # The fit is for this descriptive figure only.
    tokens = np.concatenate([local[0] for local in features.values()])
    if min(tokens.shape) < 3:
        raise ValueError("The selected scans have too few patch embeddings for three PCA components.")
    with threadpool_limits(limits=2, user_api="blas"):
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(tokens)
        coords = pca.transform(tokens)
        low, high = np.quantile(coords, [.02, .98], axis=0)
        maps = {(mode, row["scan_id"]): feature_image(features[mode, row["scan_id"]], pca, low, high, roi)
                for mode in MODES for row, (_, roi) in zip(rows, originals)}

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample}_dinov3_features.png"
    fig, axes = plt.subplots(3, len(rows), squeeze=False, figsize=(max(3.7, 2.7 * len(rows)), 8.1),
                             layout="constrained")
    for col, (row, (rgb, roi)) in enumerate(zip(rows, originals)):
        images = (np.where(roi[..., None], rgb, 255), *(maps[mode, row["scan_id"]] for mode in MODES))
        for index, image in enumerate(images):
            axis = axes[index, col]
            axis.imshow(image)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if col == 0:
                axis.set_ylabel(ROW_LABELS[index], fontsize=10)
        axes[0, col].set_title(row["scanner"], fontsize=11, weight="semibold")
    fig.suptitle(f"{sample} · final DINOv3 patch embeddings\n"
                 "Shared PCA colour scale across all feature panels", fontsize=13)
    save_figure_png(fig, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Use a saved train/test run at a custom path.")
    parser.add_argument("--smoke", action="store_true", help="Read the default smoke run.")
    parser.add_argument("--sample", default=LEAD_SCAN, help="Held-out specimen-stage ID with spatial caches.")
    parser.add_argument("--out", type=Path, default=ROOT / "dinov3_feature_maps")
    args = parser.parse_args()
    run = args.run or ROOT / "runs" / ("matched_raw_smoke" if args.smoke else "matched_raw_train_test")
    try:
        path = export(run.resolve(), args.out.resolve(), args.sample)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
