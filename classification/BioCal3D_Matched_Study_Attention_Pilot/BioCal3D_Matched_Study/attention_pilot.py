"""Run a specimen-matched attention pilot without loading the frozen backbones."""

import argparse
import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from attention_mil import clean_threshold, fit, forward, normalize_local
from images_2d import coverage_mask, pool_numpy, prepare_images
from make_feature_pseudomasks import project, render
from matched_figures import gallery_selection
from matched_study import INPUTS, MATURE_SCOPE, SCOPES, balanced, fit_fixed, metric_rows, study_rows, y
from stage_ablation import feature_path, read_json, save_npz
from study_data import STAGES, digest, seed_for, signature, write_csv, write_json

FAMILY = {"dinov3": "DINOv3", "concerto": "Concerto"}
HEADS = ("logistic", "uniform", "attention")
CONTRASTS = (("attention", "logistic"), ("attention", "uniform"))


def load_cache(run, row, mode, config_key):
    path = feature_path(run, mode, row)
    if not path.is_file():
        raise ValueError(f"Missing {mode} cache for {row['scan_id']}. Run extract for "
                         f"--partition {row['partition']} --spatial.")
    with np.load(path, allow_pickle=False) as cache:
        if cache["key"].item() != signature([config_key, row["geometry_sha256"], row["row_id"]]):
            raise ValueError(f"Feature cache identity mismatch: {path}")
        pooled = cache["pooled"]
        if pooled.ndim != 1 or not np.isfinite(pooled).all():
            raise ValueError(f"Invalid pooled features: {path}")
        if "local" not in cache:
            raise ValueError(f"Spatial features missing: {path}. Rerun the original extraction "
                             f"with --partition {row['partition']} --spatial.")
        local = cache["local"]
        if mode.startswith("dinov3_"):
            coverage = cache["coverage"]
            if (local.ndim != 3 or local.shape[1:] != coverage.shape
                    or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1))):
                raise ValueError(f"Invalid patch layout: {path}")
            reconstructed = pool_numpy(local, coverage)
            eligible = coverage.ravel() >= .05
            features = local.reshape(local.shape[0], -1).T[eligible]
            prior = coverage.ravel()[eligible].astype(np.float32)
            meta = {"shape": coverage.shape, "coverage": coverage}
        else:
            if local.ndim != 2 or not len(local):
                raise ValueError(f"Invalid point layout: {path}")
            reconstructed = np.concatenate([local.mean(axis=0), local.max(axis=0)])
            eligible = np.ones(len(local), dtype=bool)
            features = local
            prior = np.ones(len(local), dtype=np.float32)
            xyz = (cache["xyz_scaled"] * cache["scale_mm"]) @ cache["rotation"].T + cache["center_mm"]
            face, barycentric = cache["face_index"], cache["barycentric"]
            if (xyz.shape != (len(local), 3) or not np.isfinite(xyz).all()
                    or face.shape != (len(local),) or barycentric.shape != xyz.shape
                    or np.any(face < 0) or not np.isfinite(barycentric).all()):
                raise ValueError(f"Invalid point correspondence: {path}")
            meta = {"xyz_mm": xyz, "face_index": face, "barycentric": barycentric}
    if (not features.shape[1] or not np.isfinite(features).all()
            or reconstructed.shape != pooled.shape
            or not np.allclose(reconstructed, pooled, atol=2e-5, rtol=2e-5)):
        raise ValueError(f"Spatial features do not reproduce the cached logistic input: {path}")
    return pooled, features.astype(np.float32, copy=False), prior, eligible, meta


def development_features(args, rows, mode, key):
    bags, embeddings = [], []
    for i, row in enumerate(rows, 1):
        pooled, features, prior, _, _ = load_cache(args.run, row, mode, key)
        if args.max_locations and len(features) > args.max_locations:
            rng = np.random.default_rng(seed_for(row["row_id"], args.seed))
            selected = rng.choice(len(features), args.max_locations, replace=False)
            features, prior = features[selected], prior[selected]
        bags.append((normalize_local(features), prior))
        embeddings.append(pooled)
        if i == 1 or i % 50 == 0 or i == len(rows):
            print(f"{mode}: read {i}/{len(rows)} development scans", flush=True)
    return bags, np.stack(embeddings).astype(np.float64)


def check_existing_baseline(run, partition, assessment, mode, probabilities):
    results = run / "matched_results" / partition
    existing = results / "predictions.csv"
    if not existing.exists() or not (results / "settings.json").exists():
        return "Refit the matched logistic recipe"
    settings = read_json(results / "settings.json")
    if settings["cohort_sha256"] != digest(run / "cohort.json"):
        raise ValueError(f"Existing matched predictions use a different cohort: {results}")
    with existing.open(encoding="utf-8-sig", newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["mode"] == mode]
    indexed = {r["scan_id"]: r for r in rows}
    if set(indexed) != {r["scan_id"] for r in assessment} or len(rows) != len(indexed):
        raise ValueError(f"Existing matched predictions do not cover {mode} on these scans.")
    for row, p in zip(assessment, probabilities):
        saved = indexed[row["scan_id"]]
        if (saved["truth"] != row["stage"] or saved["specimen"] != row["physical_id"]
                or not np.allclose(p, [float(saved[f"p_{stage}"]) for stage in STAGES],
                                   atol=1e-5, rtol=1e-5)):
            raise ValueError(f"Refit differs from the saved matched baseline for {mode}, "
                             f"{row['scan_id']}. Use its original scikit-learn environment.")
    return "Refit the matched logistic recipe and verified saved probabilities"


def fit_and_predict(args, rows, assessment, folds, mode, key, dest):
    bags, features = development_features(args, rows, mode, key)
    labels = y(rows)
    predictions = {head: np.full((len(assessment), len(STAGES)), np.nan) for head in HEADS}
    metadata, fitted_attention = [], None
    is_test = args.partition == "test"
    for fold in [None] if is_test else folds:
        training = [i for i, row in enumerate(rows) if is_test or row["fold"] != fold]
        training_bags = [bags[i] for i in training]
        train_labels = labels[training]
        model_name = "all_development" if is_test else f"except_fold_{fold}"
        model_dir = dest / "models" / mode / model_name
        model_dir.mkdir(parents=True)
        print(f"{mode}: fitting {model_name} on {len(training)} scans", flush=True)
        logistic = fit_fixed(features[training], train_labels)
        joblib.dump(logistic, model_dir / "logistic.joblib")
        models = {}
        for name in HEADS[1:]:
            models[name], history = fit(
                training_bags, train_labels, args.seed, args.epochs, args.learning_rate,
                args.batch_size, args.proxy_strength,
                hidden_dim=args.attention_hidden_dim if name == "attention" else 0,
            )
            save_npz(model_dir / f"{name}.npz", **models[name], training_loss=history)
        threshold = clean_threshold(models["attention"], training_bags, train_labels,
                                    args.clean_quantile)
        info = {
            "mode": mode, "validation_fold": fold, "threshold": threshold,
            "training_scans": len(training),
            "training_stage_counts": dict(zip(STAGES, np.bincount(train_labels).tolist())),
            "training_specimens": sorted({rows[i]["physical_id"] for i in training}),
            "assessment_specimens": sorted({r["physical_id"] for r in assessment
                                            if is_test or r["fold"] == fold}),
        }
        if set(info["training_specimens"]) & set(info["assessment_specimens"]):
            raise ValueError(f"Specimen leakage in {model_name}.")
        write_json(model_dir / "training.json", info)
        metadata.append(info)
        with threadpool_limits(limits=1, user_api="blas"):
            for i, row in enumerate(assessment):
                if not is_test and row["fold"] != fold:
                    continue
                pooled, local, prior, _, _ = load_cache(args.run, row, mode, key)
                predictions["logistic"][i] = logistic.predict_proba(pooled[None].astype(np.float64))[0]
                bag = normalize_local(local), prior
                for name in HEADS[1:]:
                    _, _, logits, _ = forward(bag, models[name])
                    predictions[name][i] = softmax(logits[:-1])
        if is_test:
            fitted_attention = models["attention"], threshold
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError(f"Missing or invalid held-out predictions for {mode}.")
    verification = check_existing_baseline(args.run, args.partition, assessment, mode,
                                           predictions["logistic"])
    return predictions, metadata, fitted_attention, verification


def ensure_render_root(args, rows, temporary):
    folders = [args.run / "renders" / row["partition"] / "images" /
               row["scanner"] / row["sample_name"] for row in rows]
    required = ("rgb.png", "valid_mask.png", "maps.npz")
    if all(all((folder / file).is_file() for file in required) for folder in folders):
        return args.run
    # Concerto can create the same raw projection from the original mesh if
    # the DINO extraction has not already populated the shared render cache.
    dino_file = args.run / "features" / "dinov3_config.json"
    config = read_json(dino_file) if dino_file.exists() else {}
    source = [dict(row, mesh_path=str(args.snapshot / row["relative_mesh_path"]))
              if args.snapshot else row for row in rows]
    for row in source:
        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"Cannot render changed geometry for {row['scan_id']}.")
    prepare_images(source, temporary / "renders" / "test", size=config.get("image_size", 448),
                   field_mm=config.get("field_mm", 16.), accelerate=False)
    return temporary


def safe_scan_folder(row):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", row["scan_id"])
    return f"{name}_{signature(row['row_id'])[:8]}"


def save_mask(path, mask):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def draw_example(folder, rgb, mask, maps, mode, row, predictions, target, against, fraction):
    figure, axes = plt.subplots(1, 5, figsize=(14, 3.4), layout="constrained")
    predicted = "  |  ".join(f"{head} {STAGES[int(p.argmax())]} ({p.max():.0%})"
                             for head, p in predictions.items())
    figure.suptitle(f"{row['scan_id']} · {mode} · recorded {row['stage']}\n{predicted}",
                    fontweight="bold")
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    for ax, values, title in zip(axes[1:4], np.moveaxis(maps, -1, 0), (
        "Attention gain\nlog₂ versus ROI prior",
        "Local plaque-like logit\nrelative to clean threshold",
        f"Signed stage evidence\n{STAGES[target]} vs {STAGES[against]}",
    )):
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(np.quantile(finite, .99)), 1e-6) if finite.size else 1.
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_invalid(values), cmap="coolwarm", vmin=-bound, vmax=bound)
        ax.set_title(title)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (1., .76, 0., .74)
    axes[4].imshow(rgb)
    axes[4].imshow(overlay)
    axes[4].set_title(f"Candidate plaque mask\n{fraction:.1%} of projected ROI")
    for ax in axes:
        ax.set_axis_off()
    figure.savefig(folder / "example.png", dpi=150, facecolor="white")
    plt.close(figure)


def export_masks(args, dest, rows, fitted, config, predictions, render_root):
    indexed = {
        row["scan_id"]: {
            f"{mode}/{head}": {"prediction": STAGES[int(p[i].argmax())]}
            for mode, heads in predictions.items() for head, p in heads.items()
        } for i, row in enumerate(rows)
    }
    gallery = {r["scan_id"]: reason for r, reason in gallery_selection(rows, indexed)}
    key = signature(config)
    summary = []
    for i, row in enumerate(rows):
        scan = row["scan_id"]
        rgb, roi, center, field = render(render_root, row)
        if not roi.any() or not np.isfinite(center).all() or not np.isfinite(field) or field <= 0:
            raise ValueError(f"Invalid projection for {scan}.")
        if config["encoder"] == "dinov3" and (
            roi.shape != (config["image_size"],) * 2 or not np.isclose(field, config["field_mm"])
        ):
            raise ValueError(f"Render dimensions or field of view disagree with the DINOv3 cache for {scan}.")
        reason = gallery.get(scan, "")
        for mode, (model, threshold) in fitted.items():
            _, local, prior, eligible, meta = load_cache(args.run, row, mode, key)
            if "coverage" in meta and not np.allclose(
                coverage_mask(roi, *meta["shape"]), meta["coverage"], atol=1e-6, rtol=0
            ):
                raise ValueError(f"Render ROI disagrees with cached patch coverage for {scan}, {mode}.")
            with threadpool_limits(limits=1, user_api="blas"):
                bag = normalize_local(local), prior
                x, _ = bag
                attention, _, logits, _ = forward(bag, model)
                score = x @ model["classifier"][:, -1] + model["bias"][-1]
                stage_order = np.argsort(-logits[:-1], kind="stable")
                target, against = map(int, stage_order[:2])
                margin = model["classifier"][:, target] - model["classifier"][:, against]
                evidence = attention * (x @ margin)
            gain = np.log2(np.maximum(attention / (prior / prior.sum()), 1e-30))
            if reason:
                projected = project(np.column_stack([gain, score - threshold, evidence]),
                                    meta, eligible, roi, center, field)
                local_map = projected[..., 1]
            else:
                local_map = project(score - threshold, meta, eligible, roi, center, field)
            supported = np.isfinite(local_map)
            if not supported.any():
                raise ValueError(f"No spatial locations project into the ROI for {scan}, {mode}.")
            mask = supported & (local_map > 0)
            fraction = float(mask.sum() / roi.sum())
            folder = dest / "masks" / safe_scan_folder(row) / mode
            folder.mkdir(parents=True)
            save_mask(folder / "candidate_mask.png", mask)
            save_mask(folder / "covered_mask.png", supported)
            save_mask(folder / "valid_mask.png", roi)
            save_npz(folder / "native_map.npz", eligible=eligible, local_proxy_logit=score,
                     candidate_plaque=score > threshold, clean_logit_threshold=threshold,
                     attention_weight=attention, attention_log2_gain=gain,
                     stage_evidence=evidence, target_stage=STAGES[target],
                     against_stage=STAGES[against], stage_logit_margin=logits[target] - logits[against],
                     stage_margin_bias=model["bias"][target] - model["bias"][against],
                     geometry_sha256=row["geometry_sha256"], **meta)
            if reason:
                draw_example(folder, rgb, mask, projected, mode, row,
                             {head: p[i] for head, p in predictions[mode].items()},
                             target, against, fraction)
            summary.append({
                "scan_id": scan, "specimen": row["physical_id"], "scanner": row["scanner"],
                "recorded_stage": row["stage"], "mode": mode, "threshold": threshold,
                "projected_roi_fraction": fraction,
                "projected_supported_fraction": float(supported.sum() / roi.sum()),
                "projected_area_mm2": float(mask.sum() * field**2 / roi.size),
                "native_plaque_fraction": float(prior[score > threshold].sum() / prior.sum()),
                "gallery_reason": reason, "path": folder.relative_to(dest).as_posix(),
            })
    write_csv(dest / "mask_summary.csv", summary)
    write_csv(dest / "gallery_index.csv", [r for r in summary if r["gallery_reason"]])



def paired_deltas(rows, predictions, seed):
    """Match the original study's group-stratified whole-specimen bootstrap."""
    truth = y(rows)
    group = np.array([r["group"] for r in rows])
    specimen = np.array([r["physical_id"] for r in rows])
    scanner = np.array([r["scanner"] for r in rows])
    classes = np.arange(len(STAGES))
    results = []
    for scope in (*SCOPES, MATURE_SCOPE):
        subset = (group >= 3 if scope == MATURE_SCOPE else
                  np.ones(len(rows), bool) if scope == SCOPES[0] else scanner == scope)
        ids = np.unique(specimen[subset])
        indices = [np.flatnonzero(subset & (specimen == pid)) for pid in ids]
        by_group = np.array([group[index[0]] for index in indices])
        rng = np.random.default_rng(seed)
        strata = [np.flatnonzero(by_group == g) for g in np.unique(by_group)]
        draws = np.concatenate([rng.choice(s, size=(2000, len(s)), replace=True)
                                for s in strata], axis=1)
        for mode, heads in predictions.items():
            estimates, resamples = {}, {}
            for head, p in heads.items():
                guess = p.argmax(axis=1)
                counts = np.stack([confusion_matrix(truth[i], guess[i], labels=classes) for i in indices])
                estimates[head] = balanced(counts.sum(axis=0))
                resamples[head] = balanced(counts[draws].sum(axis=1))
            for head, reference in CONTRASTS:
                low, high = np.quantile(resamples[head] - resamples[reference], [.025, .975])
                results.append(dict(scope=scope, mode=mode, comparison=head, reference=reference,
                                    specimens=len(ids),
                                    delta_balanced_accuracy=float(estimates[head] - estimates[reference]),
                                    ci95_low=float(low), ci95_high=float(high)))
    return results


def run(args, encoder):
    manifest, dev, test, folds = study_rows(args.run)
    assessment = dev if args.partition == "development" else test
    config_file = args.run / "features" / f"{encoder}_config.json"
    config = read_json(config_file)
    if (config["encoder"] != encoder or config["adapter_hashes"].get("stage_ablation.py")
            != digest(Path(__file__).with_name("stage_ablation.py"))):
        raise ValueError(f"Use {encoder} features extracted by the supplied matched study.")
    available = INPUTS[FAMILY[encoder]][:3 if config.get("harmonization_sha256") else 2]
    modes = args.modes or list(available)
    if set(modes) - set(available):
        raise ValueError(f"Requested feature modes unavailable in this run. Available: {', '.join(available)}")
    key = signature(config)
    script_names = ("attention_pilot.py", "attention_mil.py", "matched_study.py")
    settings = dict(
        cohort_sha256=digest(args.run / "cohort.json"), config_signature=key,
        encoder=encoder, modes=modes, stage_order=STAGES, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, proxy_strength=args.proxy_strength,
        attention_hidden_dim=args.attention_hidden_dim,
        clean_quantile=args.clean_quantile, max_locations=args.max_locations,
        analysis_sha256=signature([digest(Path(__file__).with_name(name)) for name in script_names]),
    )
    root = args.out or args.run / "attention_results" / encoder
    dest = root / args.partition
    if dest.exists():
        raise ValueError(f"Output already exists: {dest}. Use a new --out for another pilot.")
    is_test = args.partition == "test"
    development_settings = root / "development" / "settings.json"
    if is_test and (not development_settings.is_file() or read_json(development_settings) != settings):
        raise ValueError("Complete development first with identical modes, settings and output root.")
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".attention-", dir=root) as tmp:
        working = Path(tmp) / "result"
        working.mkdir()
        predictions, fitted, fits, verified = {}, {}, [], {}
        for mode in modes:
            predictions[mode], records, fitted[mode], verified[mode] = fit_and_predict(
                args, dev, assessment, folds, mode, key, working,
            )
            fits.extend(records)

        flat = {f"{mode}/{head}": p for mode, heads in predictions.items() for head, p in heads.items()}
        summaries, stages, _, matrices = metric_rows(assessment, flat, manifest["split_seed"])
        deltas = paired_deltas(assessment, predictions, manifest["split_seed"])
        write_csv(working / "summary.csv", summaries)
        write_csv(working / "per_stage_metrics.csv", stages)
        write_csv(working / "paired_deltas.csv", deltas)
        write_json(working / "confusion_matrices.json", dict(class_order=STAGES, matrices=matrices))
        write_csv(working / "predictions.csv", [
            dict(mode=mode, head=head, scan_id=row["scan_id"], specimen=row["physical_id"],
                 scanner=row["scanner"], group=row["group"], fold=row["fold"], truth=row["stage"],
                 prediction=STAGES[int(p.argmax())],
                 **{f"p_{stage}": float(p[j]) for j, stage in enumerate(STAGES)})
            for mode, heads in predictions.items() for head, probabilities in heads.items()
            for row, p in zip(assessment, probabilities)
        ])
        if is_test:
            render_root = ensure_render_root(args, assessment, Path(tmp))
            export_masks(args, working, assessment, fitted, config, predictions, render_root)
        write_json(working / "fit_manifest.json", {"fits": fits, "baseline": verified})
        write_json(working / "settings.json", settings)
        working.rename(dest)
    print(f"Saved {dest}", flush=True)


def main(encoder):
    parser = argparse.ArgumentParser(
        description=f"{FAMILY[encoder]} attention pooling on matched frozen spatial features."
    )
    parser.add_argument("--run", type=Path, required=True, help="Existing matched study run directory.")
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    parser.add_argument("--out", type=Path, help="Optional new results root for this encoder.")
    parser.add_argument("--modes", nargs="+", choices=INPUTS[FAMILY[encoder]],
                        help="Defaults to both raw/no-colour modes plus harmonised if cached.")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--proxy-strength", type=float, default=.5)
    parser.add_argument("--attention-hidden-dim", type=int, default=32,
                        help="Hidden units in the ReLU attention scorer (default: 32).")
    parser.add_argument("--clean-quantile", type=float, default=.95)
    parser.add_argument("--max-locations", type=int, default=1024 if encoder == "dinov3" else 512,
                        help="Training locations per scan. Zero uses all. Inference uses all.")
    parser.add_argument("--snapshot", type=Path, help="Optional mesh root for missing renders.")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.max_locations < 0 or args.attention_hidden_dim < 1
            or not 0 <= args.seed < 2**32 or not 0 < args.clean_quantile < 1
            or not all(np.isfinite(v) and v > 0 for v in (args.learning_rate, args.proxy_strength))):
        parser.error("Choose positive finite training settings, a valid seed and a quantile between 0 and 1.")
    if args.modes and len(set(args.modes)) != len(args.modes):
        parser.error("Each mode may be requested only once.")
    try:
        run(args, encoder)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
