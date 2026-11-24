"""Shared mask fitting, feature loading and performance metrics for BioCal3D."""

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
from study_data import SCANNERS, STAGES, digest

LEAD_SPECIMEN = "BCG3-1"
LEAD_SCAN = "BCG3T1-1"
INPUTS = {
    "GMM": ("gmm_gray", "gmm_rgb"),
    "Boosting": ("boost_gray", "boost_rgb"),
    "DINOv3": ("dinov3_gray", "dinov3_rgb"),
    "Concerto": ("concerto_geometry", "concerto_rgb"),
}
SCOPES = ("All scanners", *SCANNERS)
MATURE_SCOPE = "Mature only (all scanners)"
SOURCE_LABELS = ("No colour", "Raw colour")
CONTRASTS = ("Raw − no colour",)
VARIANTS = ("gray", "rgb")


def mode_config(run):
    configs = [read_json(run / "features" / f"{encoder}_config.json")
               for encoder in ("dinov3", "concerto")]
    if any(config.get("harmonization_sha256") is not None for config in configs):
        raise ValueError("This study requires unharmonised encoder caches.")
    current = Path(__file__).resolve().parent
    for config in configs:
        if config["adapter_hashes"].get("stage_ablation.py") != digest(current / "stage_ablation.py"):
            raise ValueError("Use encoder caches extracted by this matched-study package in a new run.")
    return configs


def render_settings(variant):
    return "renders", "gray.png" if variant == "gray" else "rgb.png"


def render(run, row, variant):
    return boosting.read_render(run, row, *render_settings(variant))


def verify_renders(run, rows, config):
    for row in rows:
        folder = run / "renders" / row["partition"] / "images" / row["scanner"] / row["sample_name"]
        meta = read_json(folder / "metadata.json")
        if (meta.get("color_matrix") is not None
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
        raise ValueError("The training set must contain all three stages.")
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

