"""Train paper-style attention on the completed BioCal3D train/test split."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import softmax

from explain_stage_classifier import image_values, load_cache
from fixed_study import context, fitted_settings, model_folder, test_folder
from images_2d import read_image, to_grayscale
from make_stage_heatmaps import render_inputs
from matched_study import verify_renders
from models_2d import Encoder, commit
from paper_attention import attention_to_unit_range, fit, forward, normalize, refine_local
from paper_pilot_gallery import export_panel, write_index
from stage_ablation import feature_path, read_json, save_npz
from study_data import STAGES, digest, make_surface, seed_for, signature, write_csv, write_json

FAMILIES = {"dinov3": ("dinov3_gray", "dinov3_rgb"), "concerto": ("concerto_geometry", "concerto_rgb")}
MODES = tuple(mode for modes in FAMILIES.values() for mode in modes)
GAMMA = .75
PAPER_THRESHOLD = .5


def completed_baseline(run):
    _, roles, settings = context(run)
    test = test_folder(run)
    completed = read_json(test / "settings.json")
    fitted_settings(run, settings)
    if (completed["cohort_sha256"] != settings["protocol"]["cohort_sha256"]
            or completed["training_models_sha256"] != digest(model_folder(run) / "settings.json")):
        raise ValueError("The baseline gallery does not belong to the saved train/test split and models.")
    if not (test / "examples" / "index.html").is_file():
        raise ValueError("Refresh the baseline first with run_train_test_on_this_pc.py --gallery-only.")
    held_out = {row["scan_id"]: row for row in roles["test"]}
    expected_maps = {f"{mode}_plaque_mean_logit_contribution" for mode in MODES}
    for scan in completed["examples"]:
        row = held_out[scan]
        with np.load(test / "examples" / f"{scan}_maps.npz", allow_pickle=False) as maps:
            if not expected_maps.issubset(maps.files):
                raise ValueError("Refresh the baseline gallery to export its mean-pooling score maps.")
        for mode in MODES:
            with np.load(feature_path(run, mode, row), allow_pickle=False) as cache:
                if "local" not in cache.files:
                    raise ValueError(f"The baseline spatial cache for {scan} {mode} is missing.")
    return roles, settings, completed


def metadata_for(run, mode, options, roles):
    source = Path(__file__).resolve().parent
    return dict(method="three_stage_attention_pooling_v1", classes=list(STAGES),
                mode=mode, cohort_sha256=digest(run / "cohort.json"),
                config_sha256=digest(run / "features" / f"{mode.split('_')[0]}_config.json"),
                training_scan_ids=[row["scan_id"] for row in roles["training"]], epochs=options.epochs,
                max_locations=options.max_locations, hidden_dim=options.hidden_dim,
                implementation_sha256=signature([digest(source / name) for name in ("paper_attention.py", "paper_pilot.py")]))


def native_bag(local, coverage):
    if coverage is None:
        return local, np.ones(len(local), dtype=np.float32), np.ones(len(local), dtype=bool)
    weights = np.asarray(coverage, dtype=np.float32).ravel()
    return local.reshape(local.shape[0], -1).T, weights, weights >= .05


def sample_bag(local, coverage, row, limit):
    x, weights, eligible = native_bag(local, coverage)
    indexes = np.flatnonzero(eligible)
    if not len(indexes):
        raise ValueError(f"No covered patches or points in {row['scan_id']}.")
    if len(indexes) > limit:
        rng = np.random.default_rng(seed_for(row["row_id"], 751))
        indexes = np.sort(rng.choice(indexes, size=limit, replace=False))
    sampled = normalize(x[indexes])
    if not np.isfinite(sampled).all():
        raise ValueError(f"Non-finite local features in {row['scan_id']}.")
    return sampled, weights[indexes]


def saved_native(run, row, mode, config):
    path = feature_path(run, mode, row)
    with np.load(path, allow_pickle=False) as cache:
        expected = signature([signature(config), row["geometry_sha256"], row["row_id"]])
        if cache["key"].item() != expected:
            raise ValueError(f"Baseline feature provenance differs at {path}.")
        if "local" not in cache:
            return None
        local = cache["local"]
        return local, cache["coverage"] if mode.startswith("dinov3") else None


class FrozenTrainingEncoder:
    """Use the baseline's existing encoder, transforms and checkpoint on demand."""

    def __init__(self, run, family, config, repo, checkpoint):
        import torch
        self.torch, self.run, self.family, self.config = torch, run, family, config
        device = config["device"]
        source = Path(__file__).resolve().parent
        for name, sha in config["adapter_hashes"].items():
            if digest(source / name) != sha:
                raise ValueError(f"The installed {name} differs from the baseline run.")
        if (digest(checkpoint) != config["checkpoint_sha256"] or commit(repo) != config["upstream_commit"]
                or str(torch.__version__) != config["torch"]):
            raise ValueError(f"{family} checkpoint, checkout or PyTorch environment differs from the baseline run.")
        if family == "dinov3":
            self.model = Encoder(family, {family: {"repo": str(repo), "checkpoint": str(checkpoint)}}, device)
            if self.model.meta["encoder_state_sha256"] != config["encoder_state_sha256"]:
                raise ValueError("DINOv3 model state differs from the baseline.")
        else:
            from concerto_backend import ConcertoEncoder
            surface = config["surface"]
            self.model = ConcertoEncoder(SimpleNamespace(
                repo=repo, checkpoint=checkpoint, grid_size=config["voxel_mm"] / surface["scale_mm"],
                attention_patch=config["attention_patch"]), device)

    def native(self, row, mode):
        if self.family == "dinov3":
            directory = self.run / "renders" / row["partition"] / "images" / row["scanner"] / row["sample_name"]
            image, roi = read_image(dict(row, image_dir=str(directory)), "gray" if mode.endswith("_gray") else "rgb")
            _, local, coverage = self.model.extract(image, roi)
            return local, coverage

        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"Geometry changed for {row['scan_id']}.")
        surface = make_surface(row, self.config["surface"], "concerto")
        data = {key: self.torch.from_numpy(surface[key])[None]
                for key in ("coord", "color", "normal", "grid_origin")}
        self.torch.manual_seed(seed_for(row["row_id"]))
        with self.torch.inference_mode():
            features, _ = self.model(data, use_color=mode != "concerto_geometry")
        local = features[0].float().cpu().numpy()
        if len(local) != len(surface["coord"]):
            raise ValueError(f"Point correspondence changed for {row['scan_id']}.")
        return local, None


def train(run, out, family, repo, checkpoint, options):
    roles, settings, _ = completed_baseline(run)
    rows = roles["training"]
    config = next(config for config in settings["configs"] if config["encoder"] == family)
    if family == "dinov3":
        verify_renders(run, rows, config)
    labels = np.array([STAGES.index(row["stage"]) for row in rows])
    encoder = None
    models = out / "models"
    for mode in FAMILIES[family]:
        metadata = metadata_for(run, mode, options, roles)
        path = models / f"{mode}.npz"
        description = models / f"{mode}.json"
        if path.is_file() and description.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if saved["training_signature"].item() != signature(metadata):
                    raise ValueError(f"{path} uses a different pilot configuration. Choose a new --out folder.")
            print(f"Reusing fitted attention model: {mode}", flush=True)
            continue
        bags = []
        for i, row in enumerate(rows, 1):
            native = saved_native(run, row, mode, config)
            if native is None:
                if encoder is None:
                    if repo is None or checkpoint is None:
                        raise ValueError(f"Supply --repo and --checkpoint to extract missing {family} training features.")
                    encoder = FrozenTrainingEncoder(run, family, config, repo, checkpoint)
                native = encoder.native(row, mode)
            bags.append(sample_bag(*native, row, options.max_locations))
            if i == 1 or i % 48 == 0 or i == len(rows):
                print(f"{mode}: prepared {i}/{len(rows)} training scans", flush=True)
        model, losses = fit(bags, labels, seed_for(mode, 751), options.epochs, options.hidden_dim)
        write_json(description, dict(metadata, loss=losses,
                                     training_stage_counts=dict(Counter(row["stage"] for row in rows))))
        save_npz(path, **model, training_signature=np.array(signature(metadata)))
        print(f"Fitted {mode} · loss {losses[0]:.4f} to {losses[-1]:.4f}", flush=True)


def paper_maps(run, row, mode, model, roi, raw, center, field):
    cache = load_cache(run, row, mode)
    eligible = cache["eligible"]
    x = normalize(cache["x"][eligible])
    weights = cache["w"][eligible].astype(np.float32)
    attention, _, logits, _ = forward((x, weights), model)
    probabilities = softmax(logits)
    prediction = STAGES[int(probabilities.argmax())]
    native = np.zeros(len(eligible), dtype=np.float32)
    native[eligible] = attention
    projected = image_values(cache, native, roi, center, field)
    support = roi & np.isfinite(projected)
    if not support.any():
        raise ValueError(f"No projected attention support for {row['scan_id']} {mode}.")
    unit = attention_to_unit_range(projected, support)
    luminance = None if mode == "concerto_geometry" else to_grayscale(raw)[..., 0].astype(np.float32) / 255.
    blended = unit if luminance is None else GAMMA * unit + (1 - GAMMA) * (1 - luminance)
    # Relative attention alone can also highlight clean evidence. Gate the
    # candidate map on the prediction of the plaque-stage scan classifier.
    score = refine_local(blended, support, luminance)
    if prediction == "clean":
        score[support] = 0.
    return prediction, probabilities, np.where(support, projected, np.nan).astype(np.float32), unit, score


def compare(run, out, mean_threshold, paper_threshold):
    roles, _, completed = completed_baseline(run)
    rows = {row["scan_id"]: row for row in roles["test"]}
    baseline = test_folder(run) / "examples"
    models = {}
    for mode in MODES:
        model_dir = out / "models"
        meta = read_json(model_dir / f"{mode}.json")
        options = SimpleNamespace(epochs=meta["epochs"], max_locations=meta["max_locations"],
                                  hidden_dim=meta["hidden_dim"])
        expected = metadata_for(run, mode, options, roles)
        with np.load(model_dir / f"{mode}.npz", allow_pickle=False) as model:
            if model["training_signature"].item() != signature(expected) or any(
                meta[key] != value for key, value in expected.items()
            ):
                raise ValueError(f"{mode} model belongs to a different split or encoder cache.")
            models[mode] = {key: model[key] for key in model.files if key != "training_signature"}

    gallery = out / "examples"
    gallery.mkdir(parents=True, exist_ok=True)
    metrics, panels = [], []
    for scan in completed["examples"]:
        row = rows[scan]
        raw, roi, center, field = render_inputs(run, row)
        data = {}
        with np.load(baseline / f"{scan}_maps.npz", allow_pickle=False) as maps:
            if not np.array_equal(roi, maps["valid_roi"]) or maps["recorded_stage"].item() != row["stage"]:
                raise ValueError(f"Baseline ROI or stage differs for {scan}.")
            for mode in MODES:
                mean = maps[f"{mode}_plaque_mean_logit_contribution"]
                prediction, probabilities, raw_attention, attention, paper = paper_maps(
                    run, row, mode, models[mode], roi, raw, center, field)
                shared = roi & np.isfinite(mean) & np.isfinite(paper)
                if not shared.any():
                    raise ValueError(f"No shared evaluation support for {scan} {mode}.")
                mean = np.where(shared, mean, np.nan).astype(np.float32)
                paper = np.where(shared, paper, np.nan).astype(np.float32)
                data[mode] = dict(mean=mean, attention=np.where(shared, attention, np.nan),
                                  attention_raw=np.where(shared, raw_attention, np.nan),
                                  paper=paper, prediction=prediction, stage_probabilities=probabilities)
                a, b = (mean > mean_threshold), (paper > paper_threshold)
                overlap, union = int((a & b).sum()), int((a | b).sum())
                total = int(a.sum() + b.sum())
                metrics.append(dict(scan_id=scan, stage=row["stage"], mode=mode, predicted_stage=prediction,
                                    **{f"p_{stage}": float(probabilities[i]) for i, stage in enumerate(STAGES)},
                                    shared_pixels=int(shared.sum()),
                                    valid_roi_pixels=int(roi.sum()), mean_threshold=mean_threshold,
                                    paper_threshold=paper_threshold, mean_fraction=float(a.sum()/shared.sum()),
                                    paper_fraction=float(b.sum()/shared.sum()),
                                    dice=2*overlap/total if total else "", iou=overlap/union if union else ""))
        export_panel(gallery, row, raw, data, mean_threshold, paper_threshold)
        panels.append((scan, row["stage"]))
        print(f"Compared held-out example {scan}", flush=True)
    write_csv(out / "mask_agreement.csv", metrics)
    write_index(gallery / "index.html", panels)
    print(f"Open the comparison: {gallery / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "compare"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-locations", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--mean-threshold", type=float, default=0.)
    parser.add_argument("--paper-threshold", type=float, default=PAPER_THRESHOLD)
    args = parser.parse_args()
    try:
        if (min(args.epochs, args.max_locations, args.hidden_dim) < 1
                or not np.isfinite(args.mean_threshold) or not 0 <= args.paper_threshold <= 1):
            raise ValueError("Invalid training settings or mask thresholds.")
        if args.action == "train":
            if not args.family:
                raise ValueError("Specify --family for training.")
            train(args.run, args.out, args.family, args.repo, args.checkpoint, args)
        else:
            compare(args.run, args.out, args.mean_threshold, args.paper_threshold)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
