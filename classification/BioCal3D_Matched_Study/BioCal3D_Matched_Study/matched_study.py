"""Run the specimen-matched BioCal3D stage benchmark using existing encoder caches."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

import segment_color_boosting as boosting
from compare_mask_stage_classification import boosted_mask, describe_mask
from segment_color_gmm import segment
from stage_ablation import head, load_embeddings, read_json
from study_data import SCANNERS, STAGES, digest, signature, write_csv, write_json

LEAD_SPECIMEN = "BCG3-1"
LEAD_SCAN = "BCG3T1-1"
INPUTS = {
    "GMM": ("gmm_gray", "gmm_rgb", "gmm_harmonized"),
    "Boosting": ("boost_gray", "boost_rgb", "boost_harmonized"),
    "DINOv3": ("dinov3_gray", "dinov3_rgb", "dinov3_harmonized"),
    "Concerto": ("concerto_geometry", "concerto_rgb", "concerto_harmonized"),
}
SCOPES = ("All scanners", *SCANNERS)
MATURE_SCOPE = "Mature only (all scanners)"
SOURCE_LABELS = ("No colour", "Raw colour", "Harmonised colour")
CONTRASTS = ("Raw − no colour", "Harmonised − raw")
VARIANTS = ("gray", "rgb", "harmonized")


def mode_config(run):
    configs = [read_json(run / "features" / f"{encoder}_config.json")
               for encoder in ("dinov3", "concerto")]
    hashes = [config.get("harmonization_sha256") for config in configs]
    if hashes[0] != hashes[1] or configs[0].get("colour_matrices") != configs[1].get("colour_matrices"):
        raise ValueError("Both encoders must use the same fixed scanner harmonisation mapping.")
    current = Path(__file__).resolve().parent
    for config in configs:
        if config["adapter_hashes"].get("stage_ablation.py") != digest(current / "stage_ablation.py"):
            raise ValueError("Use encoder caches extracted by this matched-study package in a new run.")
    variants = VARIANTS if hashes[0] else VARIANTS[:2]
    return variants, configs, hashes[0]


def study_rows(run):
    manifest = read_json(run / "cohort.json")
    dev = [row for row in manifest["rows"] if row["partition"] == "development"]
    test = [row for row in manifest["rows"] if row["partition"] == "test"]
    if not dev or not test:
        raise ValueError("The matched study requires development and test partitions.")
    if len({row["scan_id"] for row in dev + test}) != len(dev + test):
        raise ValueError("Scan identities are not unique.")
    assignment = {}
    for row in dev + test:
        pid = row["physical_id"]
        value = (row["partition"], row["fold"])
        if assignment.setdefault(pid, value) != value:
            raise ValueError(f"Scans of {pid} disagree on split or fold.")
    if assignment.get(LEAD_SPECIMEN) != ("test", None):
        raise ValueError(f"Pin {LEAD_SPECIMEN} to the test set during prepare.")
    if {row["scanner"] for row in test if row["sample_name"] == LEAD_SCAN} != set(SCANNERS):
        raise ValueError(f"{LEAD_SCAN} must have a test scan from each scanner.")
    folds = sorted({row["fold"] for row in dev})
    if len(folds) < 3 or any({r["stage"] for r in dev if r["fold"] == f} != set(STAGES) for f in folds):
        raise ValueError("Need at least three development folds, each containing all three stages.")
    return manifest, dev, test, folds


def render_settings(variant):
    return ("renders_harmonized" if variant == "harmonized" else "renders",
            "gray.png" if variant == "gray" else "rgb.png")


def render(run, row, variant):
    return boosting.read_render(run, row, *render_settings(variant))


def verify_renders(run, rows, variants, config):
    for row in rows:
        for directory in ("renders", "renders_harmonized") if "harmonized" in variants else ("renders",):
            folder = run / directory / row["partition"] / "images" / row["scanner"] / row["sample_name"]
            meta = read_json(folder / "metadata.json")
            expected_matrix = config["colour_matrices"][row["scanner"]] if directory == "renders_harmonized" else None
            if (meta.get("color_matrix") != expected_matrix
                    or meta["source_geometry_sha256"] != row["geometry_sha256"]
                    or meta["size"] != config["image_size"] or meta["field_mm"] != config["field_mm"]
                    or any(digest(folder / name) != sha for name, sha in meta["file_hashes"].items())):
                raise ValueError(f"Render cache identity mismatch: {folder}")


def mask_matrix(run, rows, variant, model=None):
    gray = variant == "gray"
    output = []
    for row in rows:
        image, roi = render(run, row, variant)
        if model is None:
            mask, _ = segment(image, roi, .95, luminance_only=gray)
        else:
            mask = boosted_mask(image, roi, model)
        output.append(describe_mask(mask, roi))
    return np.asarray(output)


def y(rows):
    return np.array([STAGES.index(row["stage"]) for row in rows])


def fit_fixed(features, labels):
    with threadpool_limits(limits=2):
        fitted = head().fit(features, labels)
    if not np.array_equal(fitted.classes_, np.arange(len(STAGES))):
        raise ValueError("Every training fold must contain all three stages.")
    return fitted


def train_booster(args, rows, variant, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    render_dir, image_name = render_settings(variant)
    boosting.fit(SimpleNamespace(
        run=args.run, scanner=None, stage=None, scan=[r["scan_id"] for r in rows],
        model=path, trees=args.trees, pixels_per_scan=args.pixels_per_scan,
        render_dir=render_dir, image_name=image_name, luminance_only=variant == "gray",
        allow_empty=True,
    ))
    return joblib.load(path)


def boost_predictions(args, dev, test, folds, variant, dest):
    by_fold = {fold: [r for r in dev if r["fold"] == fold] for fold in folds}
    boosters = {}
    feature_cache = {}

    def model_without(omitted):
        if omitted not in boosters:
            training = [r for r in dev if r["fold"] not in omitted]
            suffix = "all_development" if not omitted else "except_" + "_".join(map(str, omitted))
            path = dest / "models" / f"boost_{variant}_{suffix}.joblib"
            print(f"Fitting {variant} mask booster on {len(training)} scans ({suffix}).", flush=True)
            boosters[omitted] = train_booster(args, training, variant, path)
        return boosters[omitted]

    def fold_features(fold, omitted):
        key = (fold, omitted)
        if key not in feature_cache:
            feature_cache[key] = mask_matrix(args.run, by_fold[fold], variant, model_without(omitted))
        return feature_cache[key]

    if args.partition == "test":
        # Train the stage head on development masks predicted by models that
        # excluded the corresponding specimens. Use an all-development booster on test.
        oof = np.vstack([fold_features(fold, (fold,)) for fold in folds])
        training = [row for fold in folds for row in by_fold[fold]]
        fitted = fit_fixed(oof, y(training))
        test_features = mask_matrix(args.run, test, variant, model_without(()))
        joblib.dump(fitted, dest / "models" / f"boost_{variant}_stage.joblib")
        return fitted.predict_proba(test_features)

    predictions = np.full((len(dev), len(STAGES)), np.nan)
    for outer in folds:
        inner = [fold for fold in folds if fold != outer]
        # Inner cross-fitting prevents the outer validation fold from influencing
        # either the validation masks or the stage head's mask features.
        features = np.vstack([fold_features(fold, tuple(sorted((outer, fold)))) for fold in inner])
        training = [row for fold in inner for row in by_fold[fold]]
        fitted = fit_fixed(features, y(training))
        selected = np.array([row["fold"] == outer for row in dev])
        predictions[selected] = fitted.predict_proba(fold_features(outer, (outer,)))
    return predictions


def fixed_predictions(args, dev, test, folds, mode, features, test_features, dest):
    labels = y(dev)
    if args.partition == "test":
        fitted = fit_fixed(features, labels)
        joblib.dump(fitted, dest / "models" / f"{mode}_stage.joblib")
        return fitted.predict_proba(test_features)

    predictions = np.full((len(dev), len(STAGES)), np.nan)
    for fold in folds:
        validation = np.array([row["fold"] == fold for row in dev])
        fitted = fit_fixed(features[~validation], labels[~validation])
        predictions[validation] = fitted.predict_proba(features[validation])
    return predictions


def balanced(cm):
    return (np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(axis=-1)).mean(axis=-1)


def metric_rows(rows, prediction, seed):
    labels = y(rows)
    specimen = np.array([row["physical_id"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    scanners = np.array([row["scanner"] for row in rows])
    summaries, stages, deltas, matrices = [], [], [], {}
    for scope in (*SCOPES, MATURE_SCOPE):
        selected = groups >= 3 if scope == MATURE_SCOPE else (
            np.ones(len(rows), bool) if scope == SCOPES[0] else scanners == scope)
        scope_rows = np.flatnonzero(selected)
        ids = np.unique(specimen[selected])
        group_by_id = np.array([groups[np.flatnonzero(specimen == pid)[0]] for pid in ids])
        rng = np.random.default_rng(seed)
        strata = [np.flatnonzero(group_by_id == g) for g in np.unique(group_by_id)]
        draws = np.concatenate([rng.choice(s, size=(2000, len(s)), replace=True) for s in strata], axis=1)
        matrices[scope], bootstrap = {}, {}
        for mode, probabilities in prediction.items():
            guess = probabilities.argmax(axis=1)
            cm = confusion_matrix(labels[selected], guess[selected], labels=np.arange(len(STAGES)))
            if not np.all(cm.sum(axis=1)):
                raise ValueError(f"Missing stage support in {scope}.")
            matrices[scope][mode] = cm.tolist()
            by_specimen = np.stack([
                confusion_matrix(labels[subset], guess[subset], labels=np.arange(len(STAGES)))
                for pid in ids
                for subset in [scope_rows[specimen[scope_rows] == pid]]
            ])
            bootstrap[mode] = balanced(by_specimen[draws].sum(axis=1))
            true_positive = cm.diagonal()
            recall = true_positive / cm.sum(axis=1)
            precision = np.divide(true_positive, cm.sum(axis=0), out=np.zeros(len(STAGES)), where=cm.sum(axis=0) != 0)
            f1 = 2 * true_positive / (cm.sum(axis=1) + cm.sum(axis=0))
            summaries.append(dict(scope=scope, mode=mode, scans=int(selected.sum()), specimens=len(ids),
                                  balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()),
                                  accuracy=float(true_positive.sum() / cm.sum()),
                                  macro_precision=float(precision.mean())))
            stages.extend(dict(scope=scope, mode=mode, stage=stage, support=int(cm[i].sum()),
                               recall=float(recall[i]), precision=float(precision[i]), f1=float(f1[i]))
                          for i, stage in enumerate(STAGES))
        for family, modes in INPUTS.items():
            for i, (reference, new) in enumerate(zip(modes, modes[1:])):
                if new not in bootstrap:
                    continue
                samples = bootstrap[new] - bootstrap[reference]
                low, high = np.quantile(samples, [.025, .975])
                point = balanced(np.array(matrices[scope][new])) - balanced(np.array(matrices[scope][reference]))
                deltas.append(dict(scope=scope, family=family, contrast=CONTRASTS[i], comparison=new,
                                   reference=reference, specimens=len(ids), delta_balanced_accuracy=float(point),
                                   ci95_low=float(low), ci95_high=float(high)))
    return summaries, stages, deltas, matrices


def run_study(args):
    manifest, dev, test, folds = study_rows(args.run)
    variants, configs, calibration = mode_config(args.run)
    settings = dict(cohort_sha256=digest(args.run / "cohort.json"), configs=configs,
                    harmonization_sha256=calibration, trees=args.trees, pixels_per_scan=args.pixels_per_scan,
                    analysis_sha256=signature([digest(Path(__file__).with_name(name)) for name in (
                        "matched_study.py", "segment_color_gmm.py", "segment_color_boosting.py",
                        "compare_mask_stage_classification.py")]))
    dev_directory = args.run / "matched_results" / "development"
    dest = args.run / "matched_results" / args.partition
    if (dest / "settings.json").exists():
        raise ValueError(f"Analysis already completed at {dest}.")
    is_test = args.partition == "test"
    if is_test and signature(read_json(dev_directory / "settings.json")) != signature(settings):
        raise ValueError("Complete development with identical cohort, mapping, encoder and model settings first.")
    if is_test and not calibration:
        raise ValueError("The final benchmark requires harmonised inputs. Complete a full development run first.")
    assessment = test if is_test else dev
    verify_renders(args.run, dev + (test if is_test else []), variants, configs[0])
    (dest / "models").mkdir(parents=True, exist_ok=True)

    predictions = {}
    for family in ("DINOv3", "Concerto"):
        for mode in INPUTS[family][:len(variants)]:
            print(f"Fitting {mode} stage classifier.", flush=True)
            # Match the precision used for the exact spatial decomposition.
            features = load_embeddings(args.run, dev, mode).astype(np.float64)
            heldout = load_embeddings(args.run, test, mode).astype(np.float64) if is_test else None
            predictions[mode] = fixed_predictions(args, dev, test, folds, mode, features, heldout, dest)
    for variant in variants:
        mode = f"gmm_{variant}"
        print(f"Scoring {mode} masks.", flush=True)
        features = mask_matrix(args.run, dev, variant)
        heldout = mask_matrix(args.run, test, variant) if is_test else None
        predictions[mode] = fixed_predictions(args, dev, test, folds, mode, features, heldout, dest)
        predictions[f"boost_{variant}"] = boost_predictions(args, dev, test, folds, variant, dest)
    if any(p.shape != (len(assessment), len(STAGES)) or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Missing or invalid held-out predictions.")
    summaries, stages, deltas, matrices = metric_rows(assessment, predictions, manifest["split_seed"])
    write_csv(dest / "summary.csv", summaries)
    write_csv(dest / "per_stage_metrics.csv", stages)
    write_csv(dest / "paired_deltas.csv", deltas)
    write_json(dest / "confusion_matrices.json", dict(class_order=STAGES, matrices=matrices))
    write_csv(dest / "predictions.csv", [
        dict(mode=mode, scan_id=row["scan_id"], specimen=row["physical_id"], scanner=row["scanner"],
             group=row["group"], fold=row["fold"], truth=row["stage"], prediction=STAGES[int(p.argmax())],
             **{f"p_{stage}": float(p[i]) for i, stage in enumerate(STAGES)})
        for mode, probabilities in predictions.items()
        for row, p in zip(assessment, probabilities)
    ])
    write_json(dest / "settings.json", settings)
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--pixels-per-scan", type=int, default=1024)
    args = parser.parse_args()
    if args.trees < 1 or args.pixels_per_scan < 32:
        parser.error("Choose at least one tree and 32 pixels per scan.")
    try:
        run_study(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
