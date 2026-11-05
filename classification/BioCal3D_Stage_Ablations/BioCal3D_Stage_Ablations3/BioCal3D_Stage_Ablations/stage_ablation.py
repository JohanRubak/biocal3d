"""Matched frozen DINOv3 and Concerto ablations on BioCal3D."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from study_data import (
    SCANNERS, STAGES, digest, geometry, load_snapshot, make_surface, score,
    seed_for, signature, write_csv, write_json,
)

MODES = ("dinov3_rgb", "dinov3_gray", "concerto_geometry", "concerto_rgb")
PAIRS = (("dinov3_rgb", "dinov3_gray"), ("concerto_rgb", "concerto_geometry"))
RESULT_SUBDIR = "all_groups_three_stage"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_id(value):
    match = re.fullmatch(r"(?:BCG(\d+)[_-](\d+)|G(\d+)_S(\d+))", value, re.I)
    if not match:
        raise ValueError(f"Invalid split specimen ID: {value}")
    group, specimen = (int(part) for part in match.groups() if part is not None)
    return f"BCG{group}-{specimen}"


def make_partition(rows, args):
    groups = {row["physical_id"]: row["group"] for row in rows}
    if args.split_json:
        mapping = {}
        for part, ids in read_json(args.split_json)["physical_ids"].items():
            if part not in ("train", "val", "test", "development"):
                raise ValueError(f"Unexpected split name {part!r}.")
            for value in ids:
                pid = normalize_id(value)
                if pid in mapping:
                    raise ValueError(f"Duplicate specimen in split file: {pid}")
                mapping[pid] = part
        missing = set(groups) - mapping.keys()
        if missing:
            raise ValueError(f"Split file lacks specimens: {sorted(missing)}")
        test = {pid for pid in groups if mapping[pid] == "test"}
        if not test:
            raise ValueError("Existing split contains no test specimens in the eligible cohort.")
    elif not args.cv_only:
        test = set()
        for group in sorted(set(groups.values())):
            ids = sorted(pid for pid, g in groups.items() if g == group)
            np.random.default_rng(seed_for(f"holdout:G{group}", args.seed)).shuffle(ids)
            test.update(ids[:2])
    else:
        test = set()

    development = sorted(set(groups) - test)
    strata = [groups[pid] for pid in development]
    counts = Counter(strata)
    if set(counts) != set(groups.values()) or min(counts.values(), default=0) < args.folds:
        raise ValueError(f"Every retained group needs at least {args.folds} development specimens. Counts: {counts}")
    if test and {groups[pid] for pid in test} != set(groups.values()):
        raise ValueError("The locked test set has no specimen from at least one retained group.")
    fold_by_id = {}
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (_, validation) in enumerate(splitter.split(development, strata)):
        fold_by_id.update({development[i]: fold for i in validation})
    assigned = []
    for row in rows:
        pid = row["physical_id"]
        assigned.append(dict(row, partition="test" if pid in test else "development", fold=fold_by_id.get(pid)))
    return assigned


def prepare(args):
    if (args.run / "cohort.json").exists():
        raise ValueError("This run's cohort is already locked. Choose a new --run directory.")
    source = args.approved_root or args.snapshot
    if args.approved_root:
        source = source.resolve()
        if not (source / "dataset_index.json").is_file():
            raise ValueError("Select the main Approved ROIs_reoriented dataset folder containing dataset_index.json.")
        if (source / "INCOMPLETE_ORIENTATION.json").exists():
            raise ValueError("The selected orientation export is incomplete.")
        stamp = source / "scanner_orientation.json"
        if stamp.is_file() and read_json(stamp).get("status") != "complete":
            raise ValueError("The selected orientation export is incomplete.")
        if args.run.resolve().is_relative_to(source):
            raise ValueError("Put the output run directory outside the approved ROI dataset.")
    rows, audit, excluded = load_snapshot(source, allow_incomplete=True, use_saved_splits=False, approved_export=bool(args.approved_root))
    requested = set(range(1, 7))
    selected = rows
    seen = {(row["scanner"], row["group"], row["specimen_number"], row["timepoint"]) for row in selected}
    expected = {(s, g, n, t) for s in SCANNERS for g in requested for n in range(1, 9) for t in range(2 if g <= 2 else 3)}
    if seen != expected:
        raise ValueError(f"Incomplete or unexpected eligible ROI cohort. Missing: {sorted(expected-seen)}. "
                         f"Unexpected: {sorted(seen-expected)}. Correct the snapshot before comparing inputs.")
    if args.approved_root and len({row["mesh_path"] for row in selected}) != len(selected):
        raise ValueError("The approved index maps multiple scan identities to the same geometry file.")
    for row in selected:
        meta_path = Path(row["folder"]) / "metadata.json"
        if args.approved_root and meta_path.is_file():
            meta = read_json(meta_path)
            if str(meta.get("geometry_review_status", "")).strip().lower() in {"pending", "flagged", "rejected", "draft"}:
                raise ValueError(f"{row['scan_id']} is not an approved geometry.")
            if meta.get("geometry_sha256") and meta["geometry_sha256"] != row["geometry_sha256"]:
                raise ValueError(f"{row['scan_id']} geometry does not match its approved metadata hash.")
        rgb = geometry(row)[3]
        if rgb is None or not rgb.max() > 0:
            raise ValueError(f"{row['scan_id']} lacks vertex RGB needed for the shared four-way cohort.")
    source_root = Path(audit["index"]).parent
    for row in selected:
        row["relative_mesh_path"] = Path(row["mesh_path"]).relative_to(source_root).as_posix()
    selected = make_partition(selected, args)
    counts = {}
    for key in ("development", "test"):
        part = [r for r in selected if r["partition"] == key]
        counts[key] = {"scans": len(part), "specimens": len({r["physical_id"] for r in part}),
                       "scans_by_group": dict(Counter(r["group"] for r in part))}
    args.run.mkdir(parents=True, exist_ok=True)
    write_csv(args.run / "manifest.csv", selected)
    assignment = {row["physical_id"]: {k: row[k] for k in ("physical_id", "group", "partition", "fold")}
                  for row in selected}
    write_csv(args.run / "specimen_split.csv", [assignment[pid] for pid in sorted(assignment)])
    if excluded:
        write_csv(args.run / "snapshot_non_ok.csv", excluded)
    write_json(args.run / "cohort.json", {
        "snapshot_audit": audit, "input_kind": "approved_reoriented" if args.approved_root else "snapshot", "eligible_groups": sorted(requested), "split_seed": args.seed,
        "policy": "imported_holdout" if args.split_json else "cv_only" if args.cv_only else "new_holdout",
        "source_split_sha256": digest(args.split_json) if args.split_json else None,
        "folds": args.folds, "counts": counts, "rows": selected,
    })
    print(json.dumps(counts, indent=2))


def cohort(args):
    rows = read_json(args.run / "cohort.json")["rows"]
    return [row for row in rows if row["partition"] == args.partition]


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def feature_path(run, mode, row):
    return run / "features" / mode / row["partition"] / f"{row['row_id']}.npz"


def extract(args):
    import torch
    from images_2d import prepare_images, read_image
    from models_2d import Encoder, commit

    rows = cohort(args)
    if args.snapshot:
        snapshot_root = args.snapshot if args.snapshot.is_dir() else args.snapshot.parent
        for row in rows:
            row["mesh_path"] = str(snapshot_root.resolve() / row["relative_mesh_path"])
    if not rows:
        raise ValueError(f"No scans in {args.partition}.")
    if args.smoke:
        rows = rows[:1]
    for row in rows:
        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"Snapshot geometry changed after the cohort was locked: {row['mesh_path']}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    source = Path(__file__).resolve().parent
    provenance = {
        "encoder": args.encoder, "checkpoint_sha256": digest(args.checkpoint), "upstream_commit": commit(args.repo),
        "device": args.device, "torch": str(torch.__version__), "numpy": np.__version__,
        "adapter_hashes": {name: digest(source / name) for name in
                           ("stage_ablation.py", "study_data.py", "images_2d.py", "models_2d.py", "concerto_backend.py")},
    }
    if args.encoder == "dinov3":
        provenance.update(image_size=args.image_size, field_mm=args.field_mm, background=127,
                          pooling="ROI weighted mean + covered patch max")
        if args.image_size < 16 or args.image_size % 16:
            raise ValueError("--image-size must be a positive multiple of 16 for ViT-S/16.")
        if not (args.repo / "hubconf.py").is_file():
            raise FileNotFoundError(args.repo / "hubconf.py")
        encoder = Encoder("dinov3", {"dinov3": {"repo": str(args.repo), "checkpoint": str(args.checkpoint)}},
                          args.device)
        provenance["encoder_state_sha256"] = encoder.meta["encoder_state_sha256"]
    else:
        from concerto_backend import ConcertoEncoder
        if not (args.repo / "concerto").is_dir():
            raise FileNotFoundError(args.repo / "concerto")
        settings = dict(align_plane=True, scale_mm=10.0, sampling_seed=123, sampling="surface",
                        points=args.points, inputs="geometry_rgb")
        provenance.update(surface=settings, voxel_mm=args.voxel_mm, attention_patch=args.attention_patch,
                          normals="mesh normals in both conditions", pooling="sampled surface mean + max")
        encoder = ConcertoEncoder(SimpleNamespace(repo=args.repo, checkpoint=args.checkpoint,
                                                  grid_size=args.voxel_mm / settings["scale_mm"],
                                                  attention_patch=args.attention_patch), args.device)

    config_file = args.run / "features" / f"{args.encoder}_config.json"
    config_key = signature(provenance)
    if config_file.exists() and signature(read_json(config_file)) != config_key:
        raise ValueError(f"Encoder or preprocessing settings changed. Use a new run directory. See {config_file}")
    if not args.smoke:
        write_json(config_file, provenance)
    if args.encoder == "dinov3":
        rows = prepare_images(rows, args.run / ("smoke_renders" if args.smoke else "renders") / args.partition,
                              size=args.image_size, field_mm=args.field_mm, accelerate=False)

    modes = MODES[:2] if args.encoder == "dinov3" else MODES[2:]
    for i, row in enumerate(rows, 1):
        paths = {mode: feature_path(args.run, mode, row) for mode in modes}
        key = signature([config_key, row["geometry_sha256"], row["row_id"]])
        pending = []
        for mode, path in paths.items():
            if args.smoke or not path.exists():
                pending.append(mode)
            else:
                with np.load(path, allow_pickle=False) as previous:
                    if previous["key"].item() != key:
                        raise ValueError(f"Stale feature cache: {path}")
                    if args.spatial and "local" not in previous:
                        pending.append(mode)
        if not pending:
            continue
        if args.encoder == "concerto":
            surface = make_surface(row, settings, "concerto")
            data = {name: torch.from_numpy(surface[name])[None] for name in
                    ("coord", "color", "normal", "grid_origin")}
            nvox = len(np.unique(np.floor((surface["coord"] - surface["grid_origin"]) /
                                          encoder.grid_size), axis=0))
            if nvox < 32:
                raise ValueError(f"Only {nvox} occupied voxels in {row['scan_id']}. Check mm units and voxel spacing.")
        for mode in pending:
            metadata = {}
            if args.encoder == "dinov3":
                image, mask = read_image(row, "gray" if mode.endswith("_gray") else "rgb")
                pooled, local, coverage = encoder.extract(image, mask)
                if args.spatial:
                    metadata = {"local": local, "coverage": coverage,
                                "image_dir": np.array(row["image_dir"])}
            else:
                with torch.inference_mode():
                    local, _ = encoder(data, use_color=mode == "concerto_rgb")
                    local = local[0].float().cpu().numpy()
                if local.shape[0] != len(surface["coord"]):
                    raise ValueError("Concerto output does not match the sampled points.")
                pooled = np.concatenate((local.mean(axis=0), local.max(axis=0)))
                metadata = {"occupied_voxels": nvox}
                if args.spatial:
                    metadata.update(local=local, xyz_scaled=surface["coord"], face_index=surface["face_index"],
                                    barycentric=surface["barycentric"], center_mm=surface["center_mm"],
                                    rotation=surface["rotation"], scale_mm=settings["scale_mm"])
            if not np.isfinite(pooled).all():
                raise ValueError(f"Non-finite pooled features for {row['scan_id']} {mode}")
            if args.smoke:
                print(mode, row["scan_id"], "embedding size", len(pooled), metadata.get("occupied_voxels", ""))
            else:
                save_npz(paths[mode], pooled=pooled, key=key, **metadata)
        if i == 1 or i % 20 == 0 or i == len(rows):
            print(f"{args.encoder} {args.partition}: {i}/{len(rows)}", flush=True)


def load_embeddings(run, rows, mode):
    model = mode.split("_", 1)[0]
    key = signature(read_json(run / "features" / f"{model}_config.json"))
    features = []
    for row in rows:
        path = feature_path(run, mode, row)
        if not path.is_file():
            raise ValueError(f"Missing {mode} embedding for {row['scan_id']}. Run extract for {row['partition']}.")
        with np.load(path, allow_pickle=False) as data:
            if data["key"].item() != signature([key, row["geometry_sha256"], row["row_id"]]):
                raise ValueError(f"Embedding identity mismatch: {path}")
            feature = data["pooled"]
            if feature.ndim != 1 or not np.isfinite(feature).all():
                raise ValueError(f"Invalid embedding: {path}")
            features.append(feature)
    return np.stack(features)


def head():
    # Fit standardization inside each fold. The regularization and solver are
    # fixed across all modalities, so no validation labels select a setting.
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))


def comparisons(rows, predictions, stages, seed, repeats=2000):
    labels = np.array([stages.index(r["stage"]) for r in rows])
    specimen_ids = np.array([row["physical_id"] for row in rows])
    ids = sorted(set(specimen_ids))
    subsets = [specimen_ids == pid for pid in ids]
    pid_group = {row["physical_id"]: row["group"] for row in rows}
    rng = np.random.default_rng(seed)
    strata = [np.array([i for i, pid in enumerate(ids) if (pid_group[pid] <= 2) == fresh])
              for fresh in (True, False)]
    draws = np.concatenate([rng.choice(stratum, size=(repeats, len(stratum)), replace=True)
                            for stratum in strata if len(stratum)], axis=1)
    output = []
    for first, second in PAIRS:
        changes = (predictions[first].argmax(1) == labels).astype(float)
        changes -= (predictions[second].argmax(1) == labels)
        by_specimen = np.full((len(subsets), len(stages)), np.nan)
        for i, subset in enumerate(subsets):
            for c in range(len(stages)):
                values = changes[subset & (labels == c)]
                if len(values):
                    by_specimen[i, c] = values.mean()
        recalls = np.nanmean(by_specimen, axis=0)
        resampled_recalls = np.nanmean(by_specimen[draws], axis=1)
        supports = np.isfinite(by_specimen).sum(axis=0)
        contrasts = [("balanced_accuracy", recalls.mean(), resampled_recalls.mean(axis=1), supports.min())]
        contrasts.extend((f"{stage}_recall", recalls[i], resampled_recalls[:, i], supports[i])
                         for i, stage in enumerate(stages))
        for metric, delta, resamples, support in contrasts:
            interval = np.quantile(resamples, [.025, .975]) if support >= 3 else None
            output.append({
                "higher_input": first, "reference_input": second, "metric": metric,
                "paired_delta": float(delta), "specimen_count": len(subsets),
                "minimum_specimens_per_relevant_stage": int(support),
                "ci95": interval.tolist() if interval is not None else None,
                "bootstrap_resamples": repeats,
                "interval_scope": "Whole-specimen resampling stratified by fresh/mature, conditional on these fits and specimens.",
            })
    return output


def fit(args):
    import joblib
    import sklearn

    manifest = read_json(args.run / "cohort.json")
    if set(manifest["eligible_groups"]) != set(range(1, 7)):
        raise ValueError("This pooled study requires all six groups. Prepare a new all-group run.")
    stages = STAGES
    rows = manifest["rows"]
    dev = [r for r in rows if r["partition"] == "development"]
    heldout = [r for r in rows if r["partition"] == "test"]
    labels = np.array([stages.index(r["stage"]) for r in dev])
    destination = args.run / ("locked_test" if args.test else "development_cv") / RESULT_SUBDIR
    if (destination / "metrics.json").exists():
        raise ValueError(f"Evaluation already completed at {destination}. Use an independent run for a new evaluation.")
    if args.test and not heldout:
        raise ValueError("This CV-only run has no locked test partition.")
    if args.test and not (args.run / "development_cv" / RESULT_SUBDIR / "metrics.json").is_file():
        raise ValueError("Complete development CV and freeze your study settings before evaluating the test partition.")
    assessment = heldout if args.test else dev
    features = {mode: load_embeddings(args.run, dev, mode) for mode in MODES}
    test_features = {mode: load_embeddings(args.run, heldout, mode) for mode in MODES} if args.test else {}
    destination.mkdir(parents=True, exist_ok=True)
    predictions, output_rows, metrics = {}, [], {}
    for mode, feature in features.items():
        fitted = head().fit(feature, labels)
        if args.test:
            probabilities = fitted.predict_proba(test_features[mode])
        else:
            probabilities = np.full((len(dev), len(stages)), np.nan)
            fold_by_row = np.array([r["fold"] for r in dev])
            for fold in range(manifest["folds"]):
                validation = fold_by_row == fold
                model = head().fit(feature[~validation], labels[~validation])
                probabilities[validation] = model.predict_proba(feature[validation])
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite predictions or incomplete development fold assignment.")
        joblib.dump(fitted, destination / f"{mode}_all_development_head.joblib")
        predictions[mode] = probabilities
        metrics[mode] = {"pooled": score(assessment, probabilities, stages), "by_scanner": {}}
        for scanner in SCANNERS:
            selected = np.array([row["scanner"] == scanner for row in assessment])
            if selected.any():
                metrics[mode]["by_scanner"][scanner] = score(
                    [row for row, include in zip(assessment, selected) if include], probabilities[selected], stages
                )
        for row, p in zip(assessment, probabilities):
            output_rows.append(dict(mode=mode, scan_id=row["scan_id"], physical_id=row["physical_id"],
                                    scanner=row["scanner"], group=row["group"], fold=row["fold"],
                                    truth=row["stage"], prediction=stages[int(p.argmax())],
                                    **{f"p_{stage}": float(p[j]) for j, stage in enumerate(stages)}))
    contrasts = comparisons(assessment, predictions, stages, manifest["split_seed"])
    write_csv(destination / "predictions.csv", output_rows)
    write_csv(destination / "paired_deltas.csv", contrasts)
    write_csv(destination / "summary.csv", [
        dict(mode=mode, scanner=scanner, balanced_accuracy=value["balanced_accuracy"],
             macro_f1=value["macro_f1"],
             **{f"{stage}_recall": value["recall"][i] for i, stage in enumerate(stages)},
             specimens=value["specimens"], scans=value["scans"])
        for mode, item in metrics.items()
        for scanner, value in [("all", item["pooled"]), *item["by_scanner"].items()]
    ])
    write_json(destination / "metrics.json", {
        "head": "Training-only StandardScaler + logistic regression, L2 C=1.0, balanced class weights",
        "sklearn": sklearn.__version__, "class_order": stages,
        "assessment": "locked test" if args.test else "specimen out-of-fold development",
        "cohort_sha256": digest(args.run / "cohort.json"), "metrics": metrics, "paired_deltas": contrasts,
    })
    print(f"Saved {destination}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="Audit the snapshot and lock one shared specimen split.")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--approved-root", type=Path, help="Path to Approved ROIs_reoriented.")
    source.add_argument("--snapshot", type=Path, help="Alternate prepared snapshot for an explicitly separate run.")
    prep.add_argument("--run", type=Path, required=True)
    split = prep.add_mutually_exclusive_group()
    split.add_argument("--split-json", type=Path, help="Keep the existing physical_ids.test assignment locked.")
    split.add_argument("--new-holdout", action="store_true", help="Reserve two specimens from each retained group (default).")
    split.add_argument("--cv-only", action="store_true", help="Use all eligible specimens in grouped CV, without a test holdout.")
    prep.add_argument("--seed", type=int, default=20261104)
    prep.add_argument("--folds", type=int, default=3)
    prep.set_defaults(action=prepare)

    feature = commands.add_parser("extract", help="Extract one encoder's paired ablations.")
    feature.add_argument("--run", type=Path, required=True)
    feature.add_argument("--snapshot", type=Path, help="Alternate path to the same snapshot, for Windows/WSL path mapping.")
    feature.add_argument("--encoder", choices=("dinov3", "concerto"), required=True)
    feature.add_argument("--repo", type=lambda x: Path(x).expanduser().resolve(), required=True)
    feature.add_argument("--checkpoint", type=lambda x: Path(x).expanduser().resolve(), required=True)
    feature.add_argument("--partition", choices=("development", "test"), default="development")
    feature.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    feature.add_argument("--smoke", action="store_true", help="Check both modes on one scan without saving embeddings.")
    feature.add_argument("--spatial", action="store_true", help="Also cache patch/point features and surface correspondence.")
    feature.add_argument("--image-size", type=int, default=448)
    feature.add_argument("--field-mm", type=float, default=16.0)
    feature.add_argument("--points", type=int, default=8192)
    feature.add_argument("--voxel-mm", type=float, default=0.10)
    feature.add_argument("--attention-patch", type=int, default=128)
    feature.set_defaults(action=extract)

    train = commands.add_parser("fit", help="Fit all four heads and report paired metrics.")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--test", action="store_true", help="Refit on all development scans and evaluate the locked test.")
    train.set_defaults(action=fit)

    args = parser.parse_args()
    if args.command == "prepare" and args.folds < 2:
        parser.error("Choose at least two folds.")
    if args.command == "extract" and min(args.points, args.voxel_mm, args.attention_patch, args.field_mm) <= 0:
        parser.error("Point count, voxel size, attention patch and field of view must be positive.")
    try:
        args.action(args)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
