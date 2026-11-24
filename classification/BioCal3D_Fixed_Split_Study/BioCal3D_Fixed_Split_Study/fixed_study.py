"""Fit once on the fixed training specimens and report validation and test separately."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

import matched_study as benchmark
from matched_figures import examples, gallery_data, performance
from fixed_split import LEAD_SCAN, split_rows
from scanner_harmonization import fit_chart, fit_training, read_matrices
from stage_ablation import read_json
from study_data import SCANNERS, STAGES, digest, signature, write_csv, write_json

MODES = tuple(mode for variants in benchmark.INPUTS.values() for mode in variants)
MODEL_FILES = tuple(f"{mode}_stage.joblib" for mode in MODES) + tuple(
    f"boost_{variant}_training.joblib" for variant in benchmark.VARIANTS)
IMPLEMENTATION = ("fixed_study.py", "fixed_split.py", "scanner_harmonization.py", "matched_study.py",
                  "segment_color_gmm.py", "segment_color_boosting.py", "compare_mask_stage_classification.py")


def model_folder(run):
    return run / "matched_results" / "models"


def prepare_mapping(run, swatches=None):
    cohort = run / "cohort.json"
    roles = split_rows(read_json(cohort))
    summary = {role: {"specimens": len({r["physical_id"] for r in rows}), "scans": len(rows)}
               for role, rows in roles.items()}
    print("Fixed specimen split: " + ", ".join(
        f"{role} {n['specimens']} specimens / {n['scans']} scans" for role, n in summary.items()), flush=True)
    split_file = run / "fixed_split.csv"
    if not split_file.exists():
        write_csv(split_file, [dict(partition=role, specimen=r["physical_id"], group=r["group"],
                                   scanner=r["scanner"], stage=r["stage"], scan_id=r["scan_id"])
                              for role, rows in roles.items() for r in rows])
    output = run / "colour_mapping.json"
    if output.is_file():
        mapping = read_json(output)
        expected_scope = "independent_calibration" if swatches else "fixed_split_training"
        if (mapping.get("fit_scope") != expected_scope
                or (swatches and mapping.get("calibration_sha256") != digest(swatches))):
            raise ValueError("The existing run uses a different colour map. Choose a new --run folder.")
        read_matrices(output, cohort)
        print(f"Reusing the frozen colour map: {output}", flush=True)
        return
    if swatches:
        fit_chart(SimpleNamespace(swatches=swatches, out=output, reference="TRIOS5"))
    else:
        fit_training(cohort, output)
    read_matrices(output, cohort)
    print(f"Frozen colour map: {output}", flush=True)


def context(run, trees, pixels_per_scan):
    cohort = run / "cohort.json"
    manifest = read_json(cohort)
    roles = split_rows(manifest)
    variants, configs, calibration = benchmark.mode_config(run)
    mapping = run / "colour_mapping.json"
    if variants != benchmark.VARIANTS or calibration != digest(mapping):
        raise ValueError("Both encoder caches must use this run's frozen colour map.")
    read_matrices(mapping, cohort)
    root = Path(__file__).resolve().parent
    settings = dict(cohort_sha256=digest(cohort), configs=configs, harmonization_sha256=calibration,
                    trees=trees, pixels_per_scan=pixels_per_scan,
                    implementation_sha256=signature([digest(root / name) for name in IMPLEMENTATION]),
                    training_scan_ids=sorted(row["scan_id"] for row in roles["training"]))
    return manifest, roles, settings


def fit(args):
    run = args.run
    _, roles, settings = context(run, args.trees, args.pixels_per_scan)
    dest = model_folder(run)
    marker = dest / "settings.json"
    if marker.is_file():
        trained_settings(run, settings)
        print(f"Reusing the fitted training models: {dest}", flush=True)
        return

    rows = roles["training"]
    benchmark.verify_renders(run, rows, benchmark.VARIANTS, settings["configs"][0])
    dest.mkdir(parents=True, exist_ok=True)
    labels = benchmark.y(rows)
    for family in ("DINOv3", "Concerto"):
        for mode in benchmark.INPUTS[family]:
            print(f"Fitting {mode} on the fixed training scans.", flush=True)
            features = benchmark.load_embeddings(run, rows, mode).astype(np.float64)
            joblib.dump(benchmark.fit_fixed(features, labels), dest / f"{mode}_stage.joblib")
    for variant in benchmark.VARIANTS:
        print(f"Fitting the {variant} GMM stage classifier.", flush=True)
        features = benchmark.mask_matrix(run, rows, variant)
        joblib.dump(benchmark.fit_fixed(features, labels), dest / f"gmm_{variant}_stage.joblib")
        print(f"Fitting the {variant} mask booster and stage classifier.", flush=True)
        model = benchmark.train_booster(args, rows, variant, dest / f"boost_{variant}_training.joblib")
        # The two booster stages use the same fixed training specimens.
        # Validation/test masks and scores are generated only after both fits.
        features = benchmark.mask_matrix(run, rows, variant, model)
        joblib.dump(benchmark.fit_fixed(features, labels), dest / f"boost_{variant}_stage.joblib")
    write_json(marker, dict(study=settings, files={name: digest(dest / name) for name in MODEL_FILES}))
    print(f"Saved the fixed training models: {dest}", flush=True)


def trained_settings(run, settings):
    dest = model_folder(run)
    trained = read_json(dest / "settings.json")
    if signature(trained["study"]) != signature(settings):
        raise ValueError("The saved models were fitted with different study settings. Use a new --run folder.")
    if set(trained["files"]) != set(MODEL_FILES) or any(
        digest(dest / filename) != sha for filename, sha in trained["files"].items()
    ):
        raise ValueError("A fitted model file is missing or differs from the saved training manifest.")
    return trained


def score(args):
    run, role = args.run, args.partition
    manifest, roles, settings = context(run, args.trees, args.pixels_per_scan)
    trained_settings(run, settings)
    models = model_folder(run)
    model_sha = digest(models / "settings.json")
    dest = run / "matched_results" / role
    marker = dest / "settings.json"
    if role == "test":
        validation = read_json(run / "matched_results" / "validation" / "settings.json")
        if validation["training_models_sha256"] != model_sha:
            raise ValueError("Complete validation with the same frozen models before testing.")
    if marker.is_file():
        if read_json(marker)["training_models_sha256"] != model_sha:
            raise ValueError(f"Saved {role} results belong to a different fitted model set.")
        print(f"Reusing {role} results: {dest}", flush=True)
        return

    rows = roles[role]
    benchmark.verify_renders(run, rows, benchmark.VARIANTS, settings["configs"][0])
    predictions = {}
    for family, modes in benchmark.INPUTS.items():
        for variant, mode in zip(benchmark.VARIANTS, modes):
            head = joblib.load(models / f"{mode}_stage.joblib")
            if family in ("DINOv3", "Concerto"):
                features = benchmark.load_embeddings(run, rows, mode).astype(np.float64)
            else:
                booster = None
                if family == "Boosting":
                    booster = joblib.load(models / f"boost_{variant}_training.joblib")
                    if sorted(booster["scans"]) != settings["training_scan_ids"]:
                        raise ValueError(f"The {variant} booster training set does not match the fixed split.")
                features = benchmark.mask_matrix(run, rows, variant, booster)
            predictions[mode] = head.predict_proba(features)
    if any(p.shape != (len(rows), len(STAGES)) or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError(f"Missing or invalid predictions on {role}.")

    summaries, stages, deltas, matrices = benchmark.metric_rows(rows, predictions, manifest["split_seed"])
    write_csv(dest / "summary.csv", summaries)
    write_csv(dest / "per_stage_metrics.csv", stages)
    write_csv(dest / "paired_deltas.csv", deltas)
    write_json(dest / "confusion_matrices.json", dict(class_order=STAGES, matrices=matrices))
    write_csv(dest / "predictions.csv", [
        dict(mode=mode, scan_id=row["scan_id"], specimen=row["physical_id"], scanner=row["scanner"],
             group=row["group"], partition=role, truth=row["stage"], prediction=STAGES[int(p.argmax())],
             **{f"p_{stage}": float(p[i]) for i, stage in enumerate(STAGES)})
        for mode, probabilities in predictions.items()
        for row, p in zip(rows, probabilities)
    ])
    figures = dest / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    performance(SimpleNamespace(partition=role), dest, figures)
    write_json(marker, dict(partition=role, cohort_sha256=settings["cohort_sha256"],
                            training_models_sha256=model_sha, scans=len(rows)))
    print(f"Saved {role} results: {dest}", flush=True)


def select_gallery(run):
    dest = run / "matched_results" / "test"
    _, predicted, selections = gallery_data(run, dest)
    lead = [row for row, _ in selections if row["sample_name"] == LEAD_SCAN]
    if len(lead) != len(SCANNERS) or {row["scanner"] for row in lead} != set(SCANNERS):
        raise ValueError(f"Expected all four scanner views of {LEAD_SCAN} in the test gallery.")
    records = [dict(scan_id=row["scan_id"], reason=reason, scanner=row["scanner"],
                    sample_name=row["sample_name"],
                    correct_conditions=sum(p["prediction"] == row["stage"] for p in predicted[row["scan_id"]].values()),
                    total_conditions=len(predicted[row["scan_id"]]))
               for row, reason in selections]
    write_json(dest / "gallery_selection.json", dict(examples=records))
    print("Selected test examples: " + ", ".join(row["scan_id"] for row in records), flush=True)


def render_gallery(run):
    dest = run / "matched_results" / "test"
    out = dest / "examples"
    out.mkdir(parents=True, exist_ok=True)
    examples(SimpleNamespace(run=run), dest, out, model_folder(run))
    write_json(out / "complete.json", dict(test_predictions_sha256=digest(dest / "predictions.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("calibrate", "fit", "score", "select-gallery", "gallery"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--swatches", type=Path)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--pixels-per-scan", type=int, default=1024)
    args = parser.parse_args()
    try:
        if args.trees < 1 or args.pixels_per_scan < 32:
            raise ValueError("Choose at least one tree and 32 pixels per training scan.")
        if args.action == "calibrate":
            prepare_mapping(args.run, args.swatches)
        elif args.action == "fit":
            fit(args)
        elif args.action == "score":
            score(args)
        elif args.action == "select-gallery":
            select_gallery(args.run)
        else:
            render_gallery(args.run)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
