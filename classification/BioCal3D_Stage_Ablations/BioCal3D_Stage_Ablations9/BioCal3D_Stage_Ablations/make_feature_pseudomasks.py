"""Local-feature visualizations and exploratory plaque pseudo masks for stage ablations.

Copy beside stage_ablation.py. This script never imports torch or modifies stage heads.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from images_2d import coverage_mask
from stage_ablation import MODES, feature_path, read_json
from study_data import digest, seed_for, signature, write_csv, write_json

VERSION = "feature-pseudomasks-1.0"
STAGES = ("baseline", "partial", "clean")


def load_local(run, row, mode, min_coverage):
    path = feature_path(run, mode, row)
    config = read_json(run / "features" / f"{mode.split('_', 1)[0]}_config.json")
    expected = signature([signature(config), row["geometry_sha256"], row["row_id"]])
    with np.load(path, allow_pickle=False) as z:
        if z["key"].item() != expected:
            raise ValueError(f"Stale or mismatched features: {path}")
        if "local" not in z:
            raise ValueError(f"Local features missing: {path}. Repeat the original extraction with --spatial and the same settings.")
        local = np.asarray(z["local"], dtype=np.float32)
        if mode.startswith("dinov3"):
            coverage = np.asarray(z["coverage"], dtype=np.float32)
            if local.ndim != 3 or local.shape[1:] != coverage.shape:
                raise ValueError(f"Invalid DINO feature layout: {path}")
            x = local.reshape(local.shape[0], -1).T
            eligible = coverage.ravel() >= min_coverage
            meta = {"shape": coverage.shape, "coverage": coverage}
            weights = coverage.ravel()[eligible]
        else:
            if local.ndim != 2:
                raise ValueError(f"Invalid Concerto feature layout: {path}")
            x = local
            eligible = np.ones(len(x), bool)
            xyz = ((z["xyz_scaled"] * z["scale_mm"]) @ z["rotation"].T + z["center_mm"])
            if xyz.shape != (len(x), 3):
                raise ValueError(f"Invalid point mapping: {path}")
            meta = {"xyz_mm": xyz, "face_index": z["face_index"], "barycentric": z["barycentric"]}
            weights = np.ones(len(x), dtype=np.float32)
    if not eligible.any() or not np.isfinite(x).all() or not np.isfinite(weights).all():
        raise ValueError(f"Empty or nonfinite spatial features: {path}")
    return x[eligible], eligible, weights, meta


def render(run, row):
    folder = run / "renders" / row["partition"] / "images" / row["scanner"] / row["sample_name"]
    rgb = np.array(Image.open(folder / "rgb.png").convert("RGB"))
    valid = np.array(Image.open(folder / "valid_mask.png").convert("L")) > 0
    with np.load(folder / "maps.npz", allow_pickle=False) as z:
        if z["source_geometry_sha256"].item() != row["geometry_sha256"]:
            raise ValueError(f"Render geometry mismatch: {folder}")
        center, field = z["center_mm"], float(z["field_mm"])
    if rgb.shape[:2] != valid.shape:
        raise ValueError("Render and valid-mask dimensions disagree")
    return rgb, valid, center, field


def reduced(model, x):
    return model["pca"].transform(normalize(x)).astype(np.float32)


def cluster_vectors(model, x):
    return normalize(reduced(model, x))


def sampled(x, weights, count, rng):
    if len(x) <= count:
        return np.arange(len(x))
    return rng.choice(len(x), count, replace=False, p=weights / weights.sum(dtype=np.float64))


def load_seed_table(path, allowed_rows):
    if not path:
        return {}
    allowed = {r["scan_id"] for r in allowed_rows}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scan, mode = row["scan_id"], row["mode"]
            if scan not in allowed:
                raise ValueError(f"Seed scan {scan} is outside the training specimens. Validation/test seeds cannot fit the model.")
            if mode not in MODES or (mode, scan) in result:
                raise ValueError(f"Invalid or duplicate seed row: {mode}, {scan}")
            file = Path(row["path"])
            result[mode, scan] = file if file.is_absolute() else path.parent / file
    return result


def seed_labels(file, meta, eligible, run, row):
    if "shape" in meta:
        # Explicit 0=clean, 128=ignore, 255=plaque. A binary mask labels ALL its black pixels clean.
        raw = np.array(Image.open(file).convert("L"))
        _, valid, _, _ = render(run, row)
        if raw.shape != valid.shape or not np.isin(raw, [0, 128, 255]).all():
            raise ValueError(f"Seed PNG must match the render and contain only 0/128/255: {file}")
        p = coverage_mask((raw == 255) & valid, *meta["shape"])
        c = coverage_mask((raw == 0) & valid, *meta["shape"])
        cov = meta["coverage"]
        labels = np.full(meta["shape"], -1, np.int8)
        labels[p >= .9 * cov] = 1
        labels[c >= .9 * cov] = 0
        return labels.ravel()[eligible]
    with np.load(file, allow_pickle=False) as z:
        labels = z["labels"]
        if (labels.shape != eligible.shape or not np.isin(labels, [-1, 0, 1]).all()
                or "xyz_mm" not in z or not np.allclose(z["xyz_mm"], meta["xyz_mm"], atol=1e-6)):
            raise ValueError(f"Point seeds require labels=-1/0/1 and matching xyz_mm in native cache order: {file}")
        return labels[eligible].astype(np.int8)


def cluster_association(records, k):
    """Equal-scan, equal-stage and equal-scanner association; never labels all baseline points plaque."""
    scanner_scores, details = [], []
    for scanner in sorted({r["scanner"] for r in records}):
        groups = {s: [r["hist"] for r in records if r["scanner"] == scanner and r["stage"] == s]
                  for s in STAGES}
        if any(not v for v in groups.values()):
            raise ValueError(f"Need all three training stages for scanner {scanner}")
        means = {s: np.mean(v, axis=0) for s, v in groups.items()}
        nonclean = (means["baseline"] + means["partial"]) / 2
        clean = means["clean"]
        # Small neutral prior prevents a rare cluster becoming spuriously certain.
        score = (nonclean + .01) / (nonclean + clean + .02)
        scanner_scores.append(score)
        for j in range(k):
            details.append({"scanner": scanner, "cluster": j, "baseline_fraction": float(means["baseline"][j]),
                            "partial_fraction": float(means["partial"][j]), "clean_fraction": float(clean[j]),
                            "nonclean_association_score": float(score[j])})
    return np.mean(scanner_scores, axis=0), details


def train(args):
    manifest = read_json(args.run / "cohort.json")
    rows = manifest["rows"]
    train_rows = [r for r in rows if r["partition"] == "development" and r["fold"] != args.validation_fold]
    val_rows = [r for r in rows if r["partition"] == "development" and r["fold"] == args.validation_fold]
    if not train_rows or not val_rows:
        raise ValueError("Choose an existing development validation fold")
    if {r["physical_id"] for r in train_rows} & {r["physical_id"] for r in val_rows + [r for r in rows if r["partition"] == "test"]}:
        raise ValueError("Physical specimens overlap training and assessment")
    if args.model_dir.exists():
        raise ValueError("Model output already exists. Choose a new --model-dir.")
    seeds = load_seed_table(args.seeds_csv, train_rows)
    missing_modes = {mode for mode, _ in seeds} - set(args.modes)
    if missing_modes:
        raise ValueError(f"Seed modes not requested: {missing_modes}")
    models, reports = {}, []
    for mode in args.modes:
        print(f"{mode}: collecting training features ({len(train_rows)} scans)", flush=True)
        samples = []
        provenance = {}
        for row in train_rows:
            x, _, w, _ = load_local(args.run, row, mode, args.min_coverage)
            ids = sampled(x, w, args.samples_per_scan, np.random.default_rng(seed_for(row["scan_id"], args.seed)))
            samples.append(x[ids])
            provenance[str(row["row_id"])] = digest(feature_path(args.run, mode, row))
        pool = normalize(np.concatenate(samples))
        dimensions = min(args.pca_dimensions, pool.shape[1], len(pool) - 1)
        if dimensions < 3 or len(pool) < args.clusters:
            raise ValueError("Too few features for requested PCA/clusters")
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=args.seed)
        coords = pca.fit_transform(pool).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=args.clusters, n_init=3, random_state=args.seed,
                             batch_size=2048, max_iter=100).fit(normalize(coords))
        model = {"pca": pca, "kmeans": km, "rgb_low": np.quantile(coords[:, :3], .02, axis=0),
                 "rgb_high": np.quantile(coords[:, :3], .98, axis=0), "min_coverage": args.min_coverage,
                 "mode": mode, "seed_head": None}
        records, seed_x, seed_y, seed_w = [], [], [], []
        for row in train_rows:
            x, eligible, w, meta = load_local(args.run, row, mode, args.min_coverage)
            z = normalize(reduced(model, x))
            labels = km.predict(z)
            counts = np.bincount(labels, weights=w, minlength=args.clusters)
            records.append({"scanner": row["scanner"], "stage": row["stage"], "hist": counts / counts.sum()})
            file = seeds.get((mode, row["scan_id"]))
            if file:
                y = seed_labels(file, meta, eligible, args.run, row)
                keep = np.flatnonzero(y >= 0)
                rng = np.random.default_rng(seed_for(row["scan_id"], args.seed))
                if len(keep) > args.samples_per_scan:
                    keep = rng.choice(keep, args.samples_per_scan, replace=False)
                if len(keep):
                    seed_x.append(z[keep]); seed_y.append(y[keep])
                    seed_w.append(np.full(len(keep), 1 / len(keep)))
        association, detail = cluster_association(records, args.clusters)
        model["cluster_scores"] = association
        method = "stage_association"
        if any(m == mode for m, _ in seeds):
            if not seed_y or len(np.unique(np.concatenate(seed_y))) != 2:
                raise ValueError(f"{mode}: need both clean and plaque seed labels")
            y, w = np.concatenate(seed_y), np.concatenate(seed_w)
            w *= len(w) / w.sum()
            model["seed_head"] = LogisticRegression(C=1, class_weight="balanced", max_iter=2000,
                                                    random_state=args.seed).fit(np.concatenate(seed_x), y, sample_weight=w)
            method = "weak_seed_classifier"
        model.update(version=VERSION, cohort_sha256=digest(args.run / "cohort.json"),
                     config_sha256=digest(args.run / "features" / f"{mode.split('_', 1)[0]}_config.json"),
                     validation_fold=args.validation_fold, method=method,
                     training_scan_ids=[r["scan_id"] for r in train_rows],
                     training_physical_ids=sorted({r["physical_id"] for r in train_rows}),
                     training_cache_sha256=provenance,
                     seed_sha256={scan: digest(file) for (m, scan), file in seeds.items() if m == mode})
        models[mode] = model
        reports.extend(dict(mode=mode, **r) for r in detail)
        print(f"{mode}: fitted PCA/clusters; pseudo-score method={method}", flush=True)
    args.model_dir.mkdir(parents=True)
    for mode, model in models.items():
        joblib.dump(model, args.model_dir / f"{mode}.joblib")
    write_csv(args.model_dir / "cluster_stage_association.csv", reports)
    write_json(args.model_dir / "manifest.json", {
        "version": VERSION, "cohort_sha256": digest(args.run / "cohort.json"), "modes": args.modes,
        "validation_fold": args.validation_fold, "seed": args.seed, "clusters": args.clusters,
        "pca_dimensions": args.pca_dimensions, "samples_per_scan": args.samples_per_scan,
        "min_coverage": args.min_coverage,
        "training_physical_ids": sorted({r["physical_id"] for r in train_rows}),
        "validation_physical_ids": sorted({r["physical_id"] for r in val_rows}),
        "meaning": "Exploratory local-feature pseudo labels; stage association or weak seeds are not ground truth."
    })


def project(values, meta, eligible, valid, center, field):
    """Nearest native patch/point; no smooth display turned into artificial boundaries."""
    h, w = valid.shape
    vals = np.asarray(values)
    tail = vals.shape[1:]
    out = np.full((h, w) + tail, np.nan, dtype=np.float32)
    if "shape" in meta:
        grid = np.full(np.prod(meta["shape"]), np.nan)
        if tail:
            for ch in range(tail[0]):
                grid[:] = np.nan; grid[eligible] = vals[:, ch]
                out[..., ch] = np.asarray(Image.fromarray(grid.reshape(meta["shape"]).astype(np.float32)).resize((w, h), Image.Resampling.NEAREST))
        else:
            grid[eligible] = vals
            out = np.asarray(Image.fromarray(grid.reshape(meta["shape"]).astype(np.float32)).resize((w, h), Image.Resampling.NEAREST)).copy()
    else:
        yy, xx = np.where(valid)
        xy = np.column_stack(((xx + .5) * field / w + center[0] - field / 2,
                              center[1] + field / 2 - (yy + .5) * field / h))
        # Same flat visible-surface approximation as stage-heatmap previews; retain native 3D output too.
        distance, ids = cKDTree(meta["xyz_mm"][:, :2]).query(xy)
        keep = distance <= .35
        out[yy[keep], xx[keep]] = vals[ids[keep]]
    out[~valid] = np.nan
    return out


def native_labels(score, low, high):
    labels = np.full(len(score), -1, np.int8)
    labels[score <= low] = 0
    labels[score >= high] = 1
    return labels


def point_ply(path, xyz, score, labels, clusters, rgb):
    header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n" +
              "property float x\nproperty float y\nproperty float z\n" +
              "property uchar red\nproperty uchar green\nproperty uchar blue\n" +
              "property float pseudo_score\nproperty int pseudo_label\nproperty int cluster\nend_header\n")
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header)
        np.savetxt(f, np.column_stack((xyz, np.rint(rgb * 255), score, labels, clusters)),
                   fmt="%.7f %.7f %.7f %d %d %d %.7g %d %d")


def predict(args):
    info = read_json(args.model_dir / "manifest.json")
    if info["cohort_sha256"] != digest(args.run / "cohort.json"):
        raise ValueError("This model belongs to a different cohort")
    modes = args.modes or info["modes"]
    if set(modes) - set(info["modes"]):
        raise ValueError("Requested modes were not fitted")
    rows = read_json(args.run / "cohort.json")["rows"]
    if args.partition == "validation":
        rows = [r for r in rows if r["partition"] == "development" and r["fold"] == info["validation_fold"]]
    else:
        rows = [r for r in rows if r["partition"] == "test"]
    if args.scan:
        unknown = set(args.scan) - {r["scan_id"] for r in rows}
        if unknown:
            raise ValueError(f"Unknown scan IDs in selected partition: {sorted(unknown)}")
        rows = [r for r in rows if r["scan_id"] in args.scan]
    if args.scanner:
        rows = [r for r in rows if r["scanner"] in args.scanner]
    if args.limit:
        # Round-robin across scanner/stage so a small preview is not just one scanner.
        groups = {}
        for row in rows:
            groups.setdefault((row["scanner"], row["stage"]), []).append(row)
        rows = [group[i] for i in range(max((len(v) for v in groups.values()), default=0))
                for group in groups.values() if i < len(group)][:args.limit]
    if not rows:
        raise ValueError("No selected assessment scans")
    if args.out.exists():
        raise ValueError("Output already exists. Choose a new --out to preserve earlier results.")
    models = {mode: joblib.load(args.model_dir / f"{mode}.joblib") for mode in modes}
    for mode, model in models.items():
        file = args.run / "features" / f"{mode.split('_', 1)[0]}_config.json"
        if model["config_sha256"] != digest(file):
            raise ValueError(f"Feature extraction settings changed for {mode}")
        if {r["physical_id"] for r in rows} & set(model["training_physical_ids"]):
            raise ValueError("Assessment specimens overlap the pseudo-model training set")
    args.out.mkdir(parents=True)
    summaries = []
    for row in rows:
        rgb, valid, center, field = render(args.run, row)
        fig, axes = plt.subplots(len(modes), 5, figsize=(15, 3.2 * len(modes)), squeeze=False, layout="constrained")
        fig.suptitle(f"{row['scan_id']} | {row['stage']} | exploratory pseudo masks", fontweight="bold")
        for axrow, mode in zip(axes, modes):
            model = models[mode]
            x, eligible, weights, meta = load_local(args.run, row, mode, model["min_coverage"])
            coords = reduced(model, x)
            z = normalize(coords)
            clusters = model["kmeans"].predict(z)
            score = (model["seed_head"].predict_proba(z)[:, 1] if model["seed_head"] is not None
                     else model["cluster_scores"][clusters])
            labels = native_labels(score, args.low, args.high)
            colors = np.clip((coords[:, :3] - model["rgb_low"]) /
                             np.maximum(model["rgb_high"] - model["rgb_low"], 1e-9), 0, 1)
            pca_img = project(colors, meta, eligible, valid, center, field)
            cluster_img = project(clusters, meta, eligible, valid, center, field)
            score_img = project(score, meta, eligible, valid, center, field)
            covered = valid & np.isfinite(score_img)
            ternary = np.full(valid.shape, 128, np.uint8)
            ternary[covered & (score_img <= args.low)] = 0
            ternary[covered & (score_img >= args.high)] = 255
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row["scan_id"]))
            unique = hashlib.sha256(str(row["row_id"]).encode()).hexdigest()[:8]
            folder = args.out / f"{name}_{unique}" / mode
            folder.mkdir(parents=True)
            Image.fromarray(rgb).save(folder / "original.png")
            Image.fromarray(valid.astype(np.uint8) * 255).save(folder / "valid_mask.png")
            Image.fromarray(covered.astype(np.uint8) * 255).save(folder / "covered_mask.png")
            Image.fromarray(ternary).save(folder / "pseudo_mask.png")
            Image.fromarray(np.rint(np.nan_to_num(pca_img) * 255).astype(np.uint8)).save(folder / "pca.png")
            native = {"eligible": eligible, "score": score.astype(np.float32), "pseudo_label": labels,
                      "cluster": clusters, "pca_rgb": colors, "score_image": score_img,
                      "method": np.array(model["method"]), "low": args.low, "high": args.high,
                      "geometry_sha256": np.array(row["geometry_sha256"]),
                      "feature_cache_sha256": np.array(digest(feature_path(args.run, mode, row))), **meta}
            np.savez_compressed(folder / "native_features_map.npz", **native)
            if "xyz_mm" in meta:
                point_colors = np.tile([.6, .6, .6], (len(labels), 1))
                point_colors[labels == -1] = [.2, .7, 1]
                point_colors[labels == 1] = [1, .75, 0]
                point_ply(folder / "pseudo_points.ply", meta["xyz_mm"], score, labels, clusters, point_colors)
            axrow[0].imshow(rgb); axrow[0].set_title(mode + "\nOriginal RGB")
            axrow[1].imshow(np.nan_to_num(pca_img)); axrow[1].set_title("Local features: shared PCA")
            axrow[2].imshow(np.ma.masked_invalid(cluster_img), cmap=plt.get_cmap("tab20", model["kmeans"].n_clusters),
                            vmin=-.5, vmax=model["kmeans"].n_clusters - .5)
            axrow[2].set_title("Feature clusters (arbitrary IDs)")
            im = axrow[3].imshow(np.ma.masked_invalid(score_img), cmap="magma", vmin=0, vmax=1)
            axrow[3].set_title("Weak-seed score" if model["seed_head"] is not None else "Nonclean-stage association")
            fig.colorbar(im, ax=axrow[3], shrink=.7)
            overlay = np.zeros((*valid.shape, 4), np.float32)
            overlay[covered & (ternary == 255)] = [1, .75, 0, .8]
            overlay[covered & (ternary == 128)] = [.2, .7, 1, .3]
            axrow[4].imshow(rgb); axrow[4].imshow(overlay)
            axrow[4].set_title("Pseudo mask\nGold: plaque-like; blue: uncertain")
            for ax in axrow:
                ax.set_axis_off()
            total = weights.sum()
            plaque = float(weights[labels == 1].sum() / total)
            uncertain = float(weights[labels == -1].sum() / total)
            clean = float(weights[labels == 0].sum() / total)
            summaries.append({"scan_id": row["scan_id"], "physical_id": row["physical_id"], "scanner": row["scanner"],
                              "stage": row["stage"], "partition": args.partition, "mode": mode, "method": model["method"],
                              "plaque_like_fraction": plaque, "uncertain_fraction": uncertain, "clean_like_fraction": clean,
                              "fraction_basis": "covered patch area" if "shape" in meta else "area-sampled points",
                              "uncovered_projected_roi_fraction": float(1 - covered.sum() / valid.sum()),
                              "mean_pseudo_score": float(np.average(score, weights=weights)),
                              "low": args.low, "high": args.high})
        destination = args.out / f"{name}_{unique}" / "comparison.png"
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        print(f"Saved {destination}", flush=True)
    write_csv(args.out / "summary.csv", summaries)
    write_json(args.out / "prediction_manifest.json", {
        "version": VERSION, "cohort_sha256": digest(args.run / "cohort.json"),
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "model_sha256": {m: digest(args.model_dir / f"{m}.joblib") for m in modes},
        "partition": args.partition, "low": args.low, "high": args.high,
        "meaning": "Scores are not calibrated plaque probabilities; fractions describe candidate pseudo labels, not validated plaque coverage."
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit PCA/clusters on development training folds only")
    fit.add_argument("--run", type=Path, required=True)
    fit.add_argument("--model-dir", type=Path, required=True)
    fit.add_argument("--modes", nargs="+", choices=MODES, default=["dinov3_rgb", "concerto_rgb"])
    fit.add_argument("--validation-fold", type=int, default=0)
    fit.add_argument("--clusters", type=int, default=8)
    fit.add_argument("--pca-dimensions", type=int, default=32)
    fit.add_argument("--samples-per-scan", type=int, default=256)
    fit.add_argument("--min-coverage", type=float, default=.5)
    fit.add_argument("--seeds-csv", type=Path)
    fit.add_argument("--seed", type=int, default=123)
    pred = commands.add_parser("predict", help="Write shared-PCA maps, clusters, scores, and ternary pseudo masks")
    pred.add_argument("--run", type=Path, required=True)
    pred.add_argument("--model-dir", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)
    pred.add_argument("--partition", choices=["validation", "test"], default="validation")
    pred.add_argument("--modes", nargs="+", choices=MODES)
    pred.add_argument("--scan", nargs="+")
    pred.add_argument("--scanner", nargs="+")
    pred.add_argument("--limit", type=int)
    pred.add_argument("--low", type=float, default=.35)
    pred.add_argument("--high", type=float, default=.65)
    args = parser.parse_args()
    if args.command == "fit":
        if args.clusters < 2 or args.pca_dimensions < 3 or args.samples_per_scan < 4 or not 0 < args.min_coverage <= 1:
            parser.error("Require clusters>=2, PCA dimensions>=3, samples/scan>=4, 0<min-coverage<=1")
    elif not 0 <= args.low < args.high <= 1 or (args.limit is not None and args.limit <= 0):
        parser.error("Require 0<=low<high<=1 and a positive limit")
    try:
        (train if args.command == "fit" else predict)(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
