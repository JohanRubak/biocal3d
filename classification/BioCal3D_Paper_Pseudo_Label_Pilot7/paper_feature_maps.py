"""Render Figure 4-style DINOv3 and Concerto feature maps from saved test-scan caches."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from fixed_split import LEAD_SCAN
from fixed_study import context
from make_feature_pseudomasks import load_local, project
from make_stage_heatmaps import patch_overlay, plt, render_inputs
from paper_pilot_views import save_figure_png
from study_data import SCANNERS

ROOT = Path(__file__).resolve().parent
INPUTS = {
    "dinov3": (("dinov3_rgb", "DINOv3 · RGB input"), ("dinov3_gray", "DINOv3 · grayscale input")),
    "concerto": (("concerto_rgb", "Concerto · geometry + RGB"), ("concerto_geometry", "Concerto · geometry only")),
}
TITLES = {"dinov3": "final DINOv3 patch embeddings", "concerto": "Concerto point embeddings"}
MAX_FIT_LOCATIONS = 2048


def feature_image(local, pca, low, high, roi, center, field, nearest=False):
    tokens, eligible, _, meta = local
    channels = np.clip((pca.transform(tokens) - low) / np.maximum(high - low, 1e-8), 0, 1)
    if "shape" in meta and not nearest:
        shape = meta["shape"]
        grid = np.zeros((eligible.size, 3), dtype=np.float32)
        grid[eligible] = channels
        coverage = np.where(eligible.reshape(shape), meta["coverage"], 0)
        height, width = roi.shape
        projected = np.stack([patch_overlay(channel.reshape(shape), coverage, width, height)
                              for channel in grid.T], axis=-1)
    else:
        projected = project(channels, meta, eligible, roi, center, field)
    valid = roi & np.isfinite(projected).all(axis=-1)
    return np.where(valid[..., None], projected, 1)


def points_to_cells(tokens, xyz, shape, center, field):
    """Average Concerto embeddings inside each cell of the DINOv3 patch grid."""
    height, width = shape
    cols = np.floor(((xyz[:, 0] - center[0]) / field + .5) * width).astype(np.intp)
    rows = np.floor(((center[1] - xyz[:, 1]) / field + .5) * height).astype(np.intp)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cell = rows[inside] * width + cols[inside]
    counts = np.bincount(cell, minlength=height * width)
    totals = np.zeros((height * width, tokens.shape[1]), dtype=np.float32)
    np.add.at(totals, cell, tokens[inside])
    np.divide(totals, counts[:, None], out=totals, where=counts[:, None] > 0)
    return totals, counts > 0


def match_scan(local, center, field, scan):
    shape = local["dinov3_rgb"][3]["shape"]
    cell_features, supported = {}, []
    for mode, (tokens, eligible, _, meta) in local.items():
        if "shape" in meta:
            if meta["shape"] != shape:
                raise ValueError(f"DINOv3 inputs have different patch grids for {scan}.")
            values = np.zeros((eligible.size, tokens.shape[1]), dtype=np.float32)
            values[eligible] = tokens
            support = eligible
        else:
            values, support = points_to_cells(tokens, meta["xyz_mm"], shape, center, field)
        cell_features[mode] = values
        supported.append(support)
    common = np.logical_and.reduce(supported)
    if not common.any():
        raise ValueError(f"No spatial cells contain features for all four inputs in {scan}.")
    return {mode: (values[common], common, None, {"shape": shape})
            for mode, values in cell_features.items()}


def shared_pca(features, sample_limit):
    # One basis and one colour range for all displayed inputs and scanners.
    # The fit is for this descriptive figure only.
    rng = np.random.default_rng(0)
    samples = []
    for local, _, _, _ in features:
        if sample_limit is not None and len(local) > sample_limit:
            local = local[rng.choice(len(local), size=sample_limit, replace=False)]
        samples.append(local)
    tokens = np.concatenate(samples)
    if min(tokens.shape) < 3:
        raise ValueError("The selected scans have too few embeddings for three PCA components.")
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(tokens)
    low, high = np.quantile(pca.transform(tokens), [.02, .98], axis=0)
    return pca, low, high


def export(run, out, sample, encoder="dinov3"):
    _, roles, _ = context(run)
    rows = [row for row in roles["test"] if row["sample_name"].casefold() == sample.casefold()]
    rows.sort(key=lambda row: SCANNERS.index(row["scanner"]))
    if not rows:
        raise ValueError(f"No held-out scans match {sample}. Use a specimen-stage ID such as {LEAD_SCAN}.")
    sample = rows[0]["sample_name"]
    matched = encoder == "both"
    encoders = tuple(INPUTS) if matched else (encoder,)
    modes, labels = zip(*(item for name in encoders for item in INPUTS[name]))
    row_labels = ("Original scan", *labels)
    originals = [render_inputs(run, row) for row in rows]
    features = {}
    for row, (_, _, center, field) in zip(rows, originals):
        scan = row["scan_id"]
        local = {mode: load_local(run, row, mode, min_coverage=.5) for mode in modes}
        if matched:
            local = match_scan(local, center, field, scan)
        features.update({(mode, scan): entry for mode, entry in local.items()})

    if matched:
        shapes = {local[3]["shape"] for local in features.values()}
        fields = [entry[3] for entry in originals]
        if len(shapes) != 1 or not np.allclose(fields, fields[0]):
            raise ValueError("Matched feature figures require the same patch grid and field of view in all scans.")
        height, width = shapes.pop()
        title = (f"{sample} · DINOv3 and Concerto on a shared grid\n"
                 f"{width} × {height} cells · shared PCA colours within each encoder")
    else:
        title = f"{sample} · {TITLES[encoder]}\nShared PCA colour scale across all feature panels"

    maps = {}
    with threadpool_limits(limits=2, user_api="blas"):
        for name in encoders:
            model_modes = [mode for mode, _ in INPUTS[name]]
            pca, low, high = shared_pca(
                [features[mode, row["scan_id"]] for mode in model_modes for row in rows],
                sample_limit=None if matched else MAX_FIT_LOCATIONS)
            for mode in model_modes:
                for row, (_, roi, center, field) in zip(rows, originals):
                    key = mode, row["scan_id"]
                    maps[key] = feature_image(features[key], pca, low, high, roi, center, field, nearest=matched)

    count = len(row_labels)
    out.mkdir(parents=True, exist_ok=True)
    name = "matched" if matched else encoder
    path = out / f"{sample}_{name}_features.png"
    fig, axes = plt.subplots(count, len(rows), squeeze=False, figsize=(max(3.7, 2.7 * len(rows)), 2.7 * count),
                             layout="constrained")
    for col, (row, (rgb, roi, _, _)) in enumerate(zip(rows, originals)):
        images = (np.where(roi[..., None], rgb, 255), *(maps[mode, row["scan_id"]] for mode in modes))
        for index, image in enumerate(images):
            axis = axes[index, col]
            axis.imshow(image, interpolation="nearest" if matched and index > 0 else None)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if col == 0:
                axis.set_ylabel(row_labels[index], fontsize=10)
        axes[0, col].set_title(row["scanner"], fontsize=11, weight="bold")
    fig.suptitle(title, fontsize=13)
    save_figure_png(fig, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=(*INPUTS, "both"), default="dinov3",
                        help="Use both for one matched-grid comparison.")
    parser.add_argument("--run", type=Path, help="Use a saved train/test run at a custom path.")
    parser.add_argument("--smoke", action="store_true", help="Read the default smoke run.")
    parser.add_argument("--sample", default=LEAD_SCAN, help="Held-out specimen-stage ID with spatial caches.")
    parser.add_argument("--out", type=Path, help="Output folder. Defaults to <encoder>_feature_maps.")
    args = parser.parse_args()
    run = args.run or ROOT / "runs" / ("matched_raw_smoke" if args.smoke else "matched_raw_train_test")
    folder = "matched" if args.encoder == "both" else args.encoder
    out = args.out or ROOT / f"{folder}_feature_maps"
    try:
        path = export(run.resolve(), out.resolve(), args.sample, args.encoder)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
