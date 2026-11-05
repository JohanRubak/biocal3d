"""Make Grad-CAM-style patch and point maps for saved BioCal3D stage classifiers."""

import argparse
import csv
import re
import sys
from tempfile import NamedTemporaryFile
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from images_2d import coverage_mask, pool_numpy, to_grayscale
from stage_ablation import MODES, RESULT_SUBDIR, STAGES, digest, feature_path, read_json
from study_data import signature, write_csv

MODE_LABELS = {
    "dinov3_gray": "DINOv3 grayscale",
    "dinov3_rgb": "DINOv3 RGB",
    "concerto_geometry": "Concerto geometry only",
    "concerto_rgb": "Concerto geometry + RGB",
}
STAGE_LABELS = {"baseline": "Baseline", "partial": "Partially cleaned", "clean": "Final clean"}


def render_inputs(run, row):
    folder = run / "renders" / "test" / "images" / row["scanner"] / row["sample_name"]
    rgb = np.array(Image.open(folder / "rgb.png").convert("RGB"))
    mask = np.array(Image.open(folder / "valid_mask.png").convert("L")) > 0
    with np.load(folder / "maps.npz", allow_pickle=False) as maps:
        if maps["source_geometry_sha256"].item() != row["geometry_sha256"]:
            raise ValueError(f"Render belongs to a different mesh: {row['scan_id']}.")
        center = maps["center_mm"]
        field = float(maps["field_mm"])
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Render and ROI mask have different dimensions: {row['scan_id']}.")
    return rgb, mask, center, field


def spatial_effect(run, row, mode, model, config_key, prediction, target, against):
    path = feature_path(run, mode, row)
    expected_key = signature([config_key, row["geometry_sha256"], row["row_id"]])
    xyz = None
    with np.load(path, allow_pickle=False) as cache:
        if cache["key"].item() != expected_key:
            raise ValueError(f"Feature cache identity mismatch: {path}")
        if "local" not in cache:
            encoder = mode.split("_", 1)[0]
            raise ValueError(f"Spatial features missing for {mode}, {row['scan_id']}. "
                             f"Repeat {encoder} extraction with --partition test --spatial.")
        local = cache["local"]
        baseline = cache["pooled"]
        if mode.startswith("dinov3_"):
            coverage = cache["coverage"]
            full = pool_numpy(local, coverage)
            features = local.reshape(local.shape[0], -1).T
            weights = coverage.ravel()
            max_eligible = weights >= .05
        else:
            xyz = ((cache["xyz_scaled"] * cache["scale_mm"]) @ cache["rotation"].T
                   + cache["center_mm"])
            features = local
            full = np.concatenate((local.mean(axis=0), local.max(axis=0)))
            weights = np.ones(len(features))
            max_eligible = np.ones(len(features), dtype=bool)

    expected_probabilities = [float(prediction[f"p_{stage}"]) for stage in STAGES]
    if not np.allclose(model.predict_proba(baseline[None])[0],
                       expected_probabilities, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Saved prediction and head disagree for {mode}, {row['scan_id']}.")
    if not np.allclose(full, baseline, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Spatial features do not reproduce the cached pooling: {path}")

    scaler = model.named_steps["standardscaler"]
    head = model.named_steps["logisticregression"]
    if not np.array_equal(head.classes_, np.arange(len(STAGES))):
        raise ValueError("Unexpected classifier class order.")
    contrast = np.full(len(STAGES), -1 / (len(STAGES) - 1))
    if against is not None:
        contrast[:] = 0
        contrast[against] = -1
    contrast[target] = 1

    # Gradient of the logit contrast with respect to the unscaled pooled vector.
    mean_grad, max_grad = np.split((contrast @ head.coef_) / scaler.scale_, 2)
    mean_ref, max_ref = np.split(scaler.mean_, 2)
    _, maxima = np.split(full, 2)

    # Mean pooling contributes in proportion to coverage. Max pooling sends each
    # channel's contribution to its winner(s), dividing it equally across ties.
    # Center each pooling branch at its own development-set scaler mean.
    mean_scores = (weights / weights.sum(dtype=np.float64)) * (
        features @ mean_grad - mean_ref @ mean_grad
    )
    winners = (features == maxima) & max_eligible[:, None]
    winner_count = winners.sum(axis=0)
    if not winner_count.all():
        raise ValueError(f"No spatial winner for a cached max-pooling channel: {path}")
    max_scores = winners @ ((maxima - max_ref) * max_grad / winner_count)
    scores = mean_scores + max_scores

    logit_margin = float(model.decision_function(full[None])[0] @ contrast)
    intercept = float(head.intercept_ @ contrast)
    if not np.isclose(scores.sum() + intercept, logit_margin, atol=2e-4, rtol=2e-5):
        raise ValueError(f"Spatial contributions do not reconstruct the class score: {path}")

    if xyz is None:
        shape = coverage.shape
        return dict(scores=scores.reshape(shape), mean_scores=mean_scores.reshape(shape),
                    max_scores=max_scores.reshape(shape), coverage=coverage,
                    logit_margin=logit_margin, intercept=intercept)
    return dict(scores=scores, mean_scores=mean_scores, max_scores=max_scores, xyz_mm=xyz,
                logit_margin=logit_margin, intercept=intercept)


def patch_overlay(scores, coverage, width, height):
    support = coverage > 0
    def resize(array):
        image = Image.fromarray(array.astype(np.float32))
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    total = resize(np.where(support, scores, 0))
    weight = resize(support)
    return np.divide(total, weight, out=np.full_like(total, np.nan), where=weight > .05)


def point_overlay(xyz, scores, center, field, mask):
    # Interpolate for the flat preview only, in the same physical projection as
    # the original render. The PLY and NPZ keep all original sampled positions.
    height, width = mask.shape
    rows, cols = np.where(mask)
    xy = np.column_stack(((cols + .5) * field / width + center[0] - field / 2,
                          center[1] + field / 2 - (rows + .5) * field / height))
    distance, ids = cKDTree(xyz[:, :2]).query(xy, k=min(6, len(xyz)))
    if distance.ndim == 1:
        distance, ids = distance[:, None], ids[:, None]
    weight = np.exp(-.5 * (distance / .15) ** 2)
    weight[distance > .45] = 0
    total = weight.sum(axis=1)
    overlay = np.full(mask.shape, np.nan)
    overlay[rows, cols] = np.divide((weight * scores[ids]).sum(axis=1), total,
                                    out=np.full(len(rows), np.nan), where=total > 0)
    return overlay


def write_point_cloud(path, xyz, scores, display, cmap, signed):
    colors = np.rint(cmap(display)[:, :3] * 255).astype(np.uint8)
    if signed:
        colors[display == .5] = 150
    else:
        colors[display == 0] = 150
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float evidence\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write(header + "end_header\n")
        np.savetxt(output, np.column_stack((xyz, colors, scores)),
                   fmt="%.7f %.7f %.7f %d %d %d %.9g")


def evidence_overlay(data, scores, mask, center, field):
    height, width = mask.shape
    if "xyz_mm" in data:
        return point_overlay(data["xyz_mm"], scores, center, field, mask)
    return patch_overlay(scores, data["coverage"], width, height)


def rim_support(data, mask, rim, center, field):
    """Return the eligible native locations and the rim fraction at each."""
    if "coverage" in data:
        coverage = data["coverage"]
        height, width = coverage.shape
        rim_coverage = coverage_mask(rim, height, width)
        fraction = np.divide(rim_coverage, coverage, out=np.zeros_like(coverage),
                             where=coverage > 0)
        return coverage, np.clip(fraction, 0, 1)
    xyz = data["xyz_mm"]
    height, width = mask.shape
    cols = np.floor((xyz[:, 0] - center[0] + field / 2) * width / field).astype(int)
    rows = np.floor((center[1] + field / 2 - xyz[:, 1]) * height / field).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    support = np.zeros(len(xyz), dtype=bool)
    fraction = np.zeros(len(xyz), dtype=bool)
    support[inside] = mask[rows[inside], cols[inside]]
    fraction[inside] = rim[rows[inside], cols[inside]]
    return support.astype(float), fraction


def draw_overlay(ax, base, mask, overlay, scale, cmap, norm):
    shown = np.clip(overlay / scale, norm.vmin, norm.vmax)
    alpha = np.where(mask & np.isfinite(shown), .84 * np.abs(shown) ** .65, 0)
    ax.imshow(base)
    ax.imshow(np.nan_to_num(shown), cmap=cmap, norm=norm, alpha=alpha,
              interpolation="nearest")


def save_png(fig, destination, dpi):
    """Write a complete PNG before replacing its final filename."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", suffix=".png", prefix=".plot-",
                                dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            fig.savefig(stream, format="png", dpi=dpi, facecolor="white")
        temporary.replace(destination)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def diagnose_pooling(output, maps, rgb, mask, center, field, scan, stage, comparison):
    # Distances are measured in the orthographic projection. Padding also
    # makes the image border count as outside the valid surface.
    spacing = (field / mask.shape[0], field / mask.shape[1])
    distance = distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1]
    rim = mask & (distance <= .75)
    components = (("scores", "Combined"), ("mean_scores", "Mean pooling"),
                  ("max_scores", "Max pooling"))
    fig, axes = plt.subplots(3, len(maps) + 1, figsize=(2.8 * (len(maps) + 1), 9),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"{scan}  |  {STAGE_LABELS[stage]} vs {comparison}\n"
                 "Common scale within each model column", fontweight="bold")
    base = to_grayscale(rgb)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[1, 0].text(.5, .5, "Mean pooling\nDistributes each feature\nacross the surface",
                    ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[2, 0].text(.5, .5, "Max pooling\nAssigns each feature to\nits strongest location(s)",
                    ha="center", va="center", transform=axes[2, 0].transAxes)
    cmap, norm = plt.get_cmap("coolwarm"), Normalize(-1, 1)
    records = []
    for column, (mode, data) in enumerate(maps.items(), 1):
        overlays = [evidence_overlay(data, data[key], mask, center, field)
                    for key, _ in components]
        visible = [np.abs(overlay[mask & np.isfinite(overlay)]) for overlay in overlays]
        scale = max(1e-12, *(float(np.quantile(values, .99)) if len(values) else 0
                             for values in visible))
        support, fraction = rim_support(data, mask, rim, center, field)
        total_support = support.sum()
        for index, ((key, label), overlay) in enumerate(zip(components, overlays)):
            ax = axes[index, column]
            draw_overlay(ax, base, mask, overlay, scale, cmap, norm)
            scores = data[key]
            absolute = np.abs(scores)
            visible_absolute = np.where(support > 0, absolute, 0)
            absolute_sum = float(visible_absolute.sum())
            rim_share = float((visible_absolute * fraction).sum() / absolute_sum) if absolute_sum else None
            value = f"{scores.sum():+.2f} logit units"
            title = f"{MODE_LABELS[mode]}\nCombined: {value}" if index == 0 else value
            ax.set_title(title, fontsize=10)
            records.append({
                "scan_id": scan, "mode": mode, "component": label,
                "signed_logit_contribution": float(scores.sum()),
                "absolute_native_evidence": float(absolute.sum()),
                "rim_width_mm": .75,
                "rim_share_absolute_visible_evidence": rim_share,
                "rim_share_visible_native_support":
                    float((support * fraction).sum() / total_support) if total_support else None,
                "visible_share_absolute_native_evidence":
                    absolute_sum / float(absolute.sum()) if absolute.sum() else None,
                "column_display_clip": scale,
            })
    for ax in axes.flat:
        ax.set_axis_off()
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                       ax=axes.ravel().tolist(), shrink=.56)
    bar.set_ticks([-1, 0, 1], labels=["Opposes", "Neutral", "Supports"])
    save_png(fig, output / "pooling_diagnostic.png", dpi=170)
    write_csv(output / "pooling_diagnostic.csv", records)


def save_scan(args, row, models, keys, predictions):
    scan = row["scan_id"]
    stage = row["stage"] if args.target == "recorded" else args.target
    target = STAGES.index(stage)
    against = STAGES.index(args.against) if args.against else None
    rgb, mask, center, field = render_inputs(args.run, row)
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", scan)
    contrast_name = f"{stage}_vs_{args.against or 'others'}"
    style_name = "signed" if args.signed else "positive"
    output = args.out / f"{filename}_{row['row_id']}_{contrast_name}_{style_name}"

    maps = {mode: spatial_effect(args.run, row, mode, model, keys[mode],
                                predictions[mode, scan], target, against)
            for mode, model in models.items()}
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.15 * (len(maps) + 1), 4.25),
                             layout="constrained")
    comparison = STAGE_LABELS[args.against].lower() if args.against else "mean of other stage logits"
    fig.suptitle(f"{scan}  |  {STAGE_LABELS[stage]} vs {comparison}", fontweight="bold")
    axes[0].imshow(rgb)
    axes[0].set_title("Original scan\n" + STAGE_LABELS[row["stage"]])
    base = to_grayscale(rgb)
    cmap = plt.get_cmap("coolwarm" if args.signed else "magma")
    norm = Normalize(-1, 1) if args.signed else Normalize(0, 1)
    summary = []
    for ax, (mode, data) in zip(axes[1:], maps.items()):
        scores = data["scores"]
        xyz = data.get("xyz_mm")
        overlay = evidence_overlay(data, scores, mask, center, field)
        rendered = overlay[mask & np.isfinite(overlay)]
        if not len(rendered):
            raise ValueError(f"No spatial evidence projects into the scan ROI: {mode}, {scan}.")
        display_values = np.abs(rendered) if args.signed else rendered[rendered > 0]
        limit = float(np.quantile(display_values, .99)) if len(display_values) else 0.0
        scale = max(limit, 1e-12)
        draw_overlay(ax, base, mask, overlay, scale, cmap, norm)
        pred = predictions[mode, scan]
        ax.set_title(f"{MODE_LABELS[mode]}\nPredicted {STAGE_LABELS[pred['prediction']].lower()}")
        np.savez_compressed(output / f"{mode}_attribution.npz", **data,
                            target_stage=stage, contrast_stage=args.against or "mean_other_logits")
        if xyz is not None:
            write_point_cloud(output / f"{mode}.ply", xyz, scores,
                              norm(np.clip(scores / scale, norm.vmin, norm.vmax)), cmap, args.signed)
        summary.append({
            "scan_id": scan, "mode": mode, "recorded_stage": row["stage"],
            "target_stage": stage, "contrast_stage": args.against or "mean_other_logits",
            "predicted_stage": pred["prediction"], "target_probability": pred[f"p_{stage}"],
            "logit_margin": data["logit_margin"], "intercept": data["intercept"],
            "sum_spatial_evidence": float(scores.sum()), "display_clip_p99": limit,
        })
    for ax in axes:
        ax.set_axis_off()
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           ax=axes.tolist(), shrink=.67)
    if args.signed:
        colorbar.set_ticks([-1, 0, 1], labels=["Opposes", "Neutral", "Supports"])
    else:
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Relative evidence within each panel")
    save_png(fig, output / "heatmaps.png", dpi=200)
    write_csv(output / "map_summary.csv", summary)
    if args.diagnose_pooling:
        diagnose_pooling(output, maps, rgb, mask, center, field, scan, stage, comparison)
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Defaults to RUN/gradcam/test.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--scan", nargs="+", help="Exact test scan ID(s) from predictions.csv.")
    parser.add_argument("--stage", choices=STAGES, help="Select scans by recorded stage.")
    parser.add_argument("--target", choices=("recorded", *STAGES), default="recorded",
                        help="Stage to explain across all modes. Defaults to each scan's recorded stage.")
    parser.add_argument("--against", choices=STAGES,
                        help="Compare with this stage instead of the mean of the other logits.")
    parser.add_argument("--signed", action="store_true", help="Show evidence for and against the target.")
    parser.add_argument("--diagnose-pooling", action="store_true",
                        help="Save separate mean/max maps and boundary evidence summaries.")
    parser.add_argument("--list-scans", action="store_true")
    args = parser.parse_args()
    args.out = args.out or args.run / "gradcam" / "test"

    try:
        cohort_path = args.run / "cohort.json"
        rows = [r for r in read_json(cohort_path)["rows"] if r["partition"] == "test"]
        if args.scan:
            unknown = set(args.scan) - {r["scan_id"] for r in rows}
            if unknown:
                raise ValueError(f"Unknown locked test scan ID(s): {', '.join(sorted(unknown))}.")
            rows = [r for r in rows if r["scan_id"] in args.scan]
        if args.stage:
            rows = [r for r in rows if r["stage"] == args.stage]
        if not rows:
            raise ValueError("No locked test scans match the selection.")
        if args.list_scans:
            for row in rows:
                print(f"{row['scan_id']}\t{row['stage']}")
            return
        if not args.scan and not args.stage:
            parser.error("Choose --scan or --stage. Use --list-scans to find test scan IDs.")

        targets = {r["stage"] for r in rows} if args.target == "recorded" else {args.target}
        if args.against in targets:
            raise ValueError("--against must differ from the target stage for every selected scan.")

        results = args.run / "locked_test" / RESULT_SUBDIR
        metrics = read_json(results / "metrics.json")
        if (metrics["assessment"] != "locked test" or metrics["cohort_sha256"] != digest(cohort_path)
                or tuple(metrics["class_order"]) != tuple(STAGES)):
            raise ValueError("Locked test results and cohort do not match.")
        with (results / "predictions.csv").open(encoding="utf-8-sig", newline="") as source:
            predictions = {(r["mode"], r["scan_id"]): r for r in csv.DictReader(source)}
        # Load only heads from your own run directory. Joblib is a pickle-based format.
        models = {mode: joblib.load(results / f"{mode}_all_development_head.joblib")
                  for mode in args.modes}
        keys = {mode: signature(read_json(args.run / "features" / f"{mode.split('_', 1)[0]}_config.json"))
                for mode in models}
        failed = []
        for row in rows:
            try:
                save_scan(args, row, models, keys, predictions)
            except OSError as exc:
                failed.append(row["scan_id"])
                print(f"File error for {row['scan_id']}: {exc}", file=sys.stderr, flush=True)
        if failed:
            parser.exit(2, f"File errors for {len(failed)} of {len(rows)} scans. "
                           f"Retry these with --scan and, for output errors, a different --out folder:\n"
                           f"{', '.join(failed)}\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
