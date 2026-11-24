"""Fit the raw-colour comparison once, then score and render held-out examples."""

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

import matched_study as benchmark
from fixed_split import LEAD_SCAN, split_rows
from matched_figures import examples, gallery_selection, performance, read_csv
from stage_ablation import read_json
from study_data import SCANNERS, STAGES, digest, signature, write_csv, write_json

MODES = tuple(mode for variants in benchmark.INPUTS.values() for mode in variants)
NEURAL = tuple(mode for family in ("DINOv3", "Concerto") for mode in benchmark.INPUTS[family])
MODEL_FILES = tuple(f"{mode}_stage.joblib" for mode in MODES) + tuple(
    f"boost_{variant}_training.joblib" for variant in benchmark.VARIANTS)
IMPLEMENTATION = ("fixed_study.py", "fixed_split.py", "matched_study.py",
                  "segment_color_gmm.py", "segment_color_boosting.py", "compare_mask_stage_classification.py")


def model_folder(run):
    return run / "matched_results" / "models"


def test_folder(run):
    return run / "matched_results" / "test"


def prepare_protocol(run, smoke):
    cohort = run / "cohort.json"
    roles = split_rows(read_json(cohort), smoke)
    protocol = dict(cohort_sha256=digest(cohort), smoke=smoke, trees=2 if smoke else 80,
                    pixels_per_scan=128 if smoke else 1024,
                    training_scan_ids=sorted(row["scan_id"] for row in roles["training"]),
                    test_scan_ids=sorted(row["scan_id"] for row in roles["test"]))
    destination = run / "protocol.json"
    if destination.is_file() and read_json(destination) != protocol:
        raise ValueError("This run folder has a different split or smoke setting. Choose a fresh --run folder.")
    if not destination.exists():
        write_csv(run / "fixed_split.csv", [
            dict(partition=role, specimen=row["physical_id"], group=row["group"],
                 scanner=row["scanner"], stage=row["stage"], scan_id=row["scan_id"])
            for role, rows in roles.items() for row in rows
        ])
        write_json(destination, protocol)
    name = "Smoke check" if smoke else "Fixed train/test split"
    print(name + ": " + ", ".join(
        f"{role} {len({r['physical_id'] for r in rows})} specimens / {len(rows)} scans"
        for role, rows in roles.items()), flush=True)


def context(run):
    cohort = run / "cohort.json"
    manifest = read_json(cohort)
    protocol = read_json(run / "protocol.json")
    roles = split_rows(manifest, protocol["smoke"])
    if (protocol["cohort_sha256"] != digest(cohort)
            or any(protocol[f"{role}_scan_ids"] != sorted(row["scan_id"] for row in rows)
                   for role, rows in roles.items())):
        raise ValueError("The saved protocol differs from the current specimen assignment.")
    configs = benchmark.mode_config(run)
    root = Path(__file__).resolve().parent
    settings = dict(protocol=protocol, configs=configs,
                    implementation_sha256=signature([digest(root / name) for name in IMPLEMENTATION]))
    return manifest, roles, settings


def fitted_settings(run, settings):
    dest = model_folder(run)
    trained = read_json(dest / "settings.json")
    if signature(trained["study"]) != signature(settings):
        raise ValueError("The saved models use a different protocol. Choose a fresh --run folder.")
    if set(trained["files"]) != set(MODEL_FILES) or any(
        digest(dest / filename) != sha for filename, sha in trained["files"].items()
    ):
        raise ValueError("A fitted model file is missing or differs from its manifest.")


def fit(run):
    _, roles, settings = context(run)
    dest = model_folder(run)
    marker = dest / "settings.json"
    if marker.is_file():
        fitted_settings(run, settings)
        print(f"Reusing the fitted training models: {dest}", flush=True)
        return
    rows = roles["training"]
    benchmark.verify_renders(run, rows, settings["configs"][0])
    dest.mkdir(parents=True, exist_ok=True)
    labels = benchmark.y(rows)
    for mode in NEURAL:
        print(f"Fitting {mode} on the training scans.", flush=True)
        features = benchmark.load_embeddings(run, rows, mode).astype(np.float64)
        joblib.dump(benchmark.fit_fixed(features, labels), dest / f"{mode}_stage.joblib")
    protocol = settings["protocol"]
    options = SimpleNamespace(run=run, trees=protocol["trees"], pixels_per_scan=protocol["pixels_per_scan"])
    for variant in benchmark.VARIANTS:
        print(f"Fitting the {variant} GMM and boosting stage classifiers.", flush=True)
        features = benchmark.mask_matrix(run, rows, variant)
        joblib.dump(benchmark.fit_fixed(features, labels), dest / f"gmm_{variant}_stage.joblib")
        booster = benchmark.train_booster(options, rows, variant, dest / f"boost_{variant}_training.joblib")
        features = benchmark.mask_matrix(run, rows, variant, booster)
        joblib.dump(benchmark.fit_fixed(features, labels), dest / f"boost_{variant}_stage.joblib")
    write_json(marker, dict(study=settings, files={name: digest(dest / name) for name in MODEL_FILES}))


def neural_scores(run, rows, models):
    return {
        mode: joblib.load(models / f"{mode}_stage.joblib").predict_proba(
            benchmark.load_embeddings(run, rows, mode).astype(np.float64))
        for mode in NEURAL
    }


def prediction_records(rows, predictions):
    if set(predictions) != set(MODES) or any(
        p.shape != (len(rows), len(STAGES)) or not np.isfinite(p).all() for p in predictions.values()
    ):
        raise ValueError("Missing or invalid test predictions.")
    return [
        dict(mode=mode, scan_id=row["scan_id"], specimen=row["physical_id"], scanner=row["scanner"],
             group=row["group"], partition="test", truth=row["stage"], prediction=STAGES[int(p.argmax())],
             **{f"p_{stage}": float(p[i]) for i, stage in enumerate(STAGES)})
        for mode in MODES
        for row, p in zip(rows, predictions[mode])
    ]


def by_scan(records):
    output = defaultdict(dict)
    for record in records:
        scan, mode = record["scan_id"], record["mode"]
        if mode in output[scan]:
            raise ValueError(f"Duplicate prediction for {scan}, {mode}.")
        output[scan][mode] = record
    return output


def shortlist(run):
    _, roles, settings = context(run)
    fitted_settings(run, settings)
    models = model_folder(run)
    rows = roles["test"]
    dest = test_folder(run)
    plan = dest / "gallery_selection.json"
    provisional = dest / "_preview_predictions.csv"
    model_sha = digest(models / "settings.json")
    if plan.is_file() and provisional.is_file():
        saved = read_json(plan)
        if saved["models_sha256"] == model_sha and saved["preview_sha256"] == digest(provisional):
            print("Reusing the test gallery selection.", flush=True)
            return
    benchmark.verify_renders(run, rows, settings["configs"][0])
    predictions = neural_scores(run, rows, models)
    for variant in benchmark.VARIANTS:
        booster = joblib.load(models / f"boost_{variant}_training.joblib")
        if sorted(booster["scans"]) != settings["protocol"]["training_scan_ids"]:
            raise ValueError(f"The {variant} booster was fitted on a different training set.")
        for family, model in (("gmm", None), ("boost", booster)):
            mode = f"{family}_{variant}"
            head = joblib.load(models / f"{mode}_stage.joblib")
            predictions[mode] = head.predict_proba(benchmark.mask_matrix(run, rows, variant, model))

    records = prediction_records(rows, predictions)
    selected = gallery_selection(rows, by_scan(records))
    lead = [row for row, _ in selected if row["sample_name"] == LEAD_SCAN]
    if len(lead) != len(SCANNERS) or {row["scanner"] for row in lead} != set(SCANNERS):
        raise ValueError(f"Expected all four scanner views of {LEAD_SCAN} in the test gallery.")
    write_csv(provisional, records)
    write_json(plan, dict(models_sha256=model_sha, preview_sha256=digest(provisional), examples=[
        dict(scan_id=row["scan_id"], scanner=row["scanner"], sample_name=row["sample_name"], reason=reason)
        for row, reason in selected
    ]))
    print("Selected test examples: " + ", ".join(row["scan_id"] for row, _ in selected), flush=True)


def finish(run):
    manifest, roles, settings = context(run)
    fitted_settings(run, settings)
    models = model_folder(run)
    dest = test_folder(run)
    rows = roles["test"]
    plan = read_json(dest / "gallery_selection.json")
    provisional = dest / "_preview_predictions.csv"
    if (plan["models_sha256"] != digest(models / "settings.json")
            or plan["preview_sha256"] != digest(provisional)):
        raise ValueError("The test gallery plan differs from the fitted models or provisional predictions.")
    preview = by_scan(read_csv(provisional))
    # Read the latest test caches after the selected scans have been upgraded
    # to spatial caches. The final table and the heatmaps use the same inputs.
    predictions = neural_scores(run, rows, models)
    for mode in MODES:
        if mode not in NEURAL:
            predictions[mode] = np.array([[float(preview[row["scan_id"]][mode][f"p_{stage}"]) for stage in STAGES]
                                          for row in rows])
    records = prediction_records(rows, predictions)
    indexed = by_scan(records)
    lookup = {row["scan_id"]: row for row in rows}
    selections = []
    for entry in plan["examples"]:
        scan = entry["scan_id"]
        row = lookup[scan]
        correct = sum(record["prediction"] == row["stage"] for record in indexed[scan].values())
        reason = ("Manuscript example" if row["sample_name"] == LEAD_SCAN else
                  "All conditions correct" if correct == len(MODES) else
                  "All conditions incorrect" if not correct else "Correct and incorrect conditions")
        selections.append((row, reason))
    verify = [row for row, _ in selections]
    benchmark.verify_renders(run, verify, settings["configs"][0])
    out = dest / "examples"
    out.mkdir(parents=True, exist_ok=True)
    is_smoke = settings["protocol"]["smoke"]
    title = "Smoke check" if is_smoke else "Held-out test"
    examples(run, out, models, indexed, selections, title=title)

    summaries, stages, deltas, matrices = benchmark.metric_rows(rows, predictions, manifest["split_seed"])
    write_csv(dest / "summary.csv", summaries)
    write_csv(dest / "per_stage_metrics.csv", stages)
    write_csv(dest / "paired_deltas.csv", deltas)
    write_json(dest / "confusion_matrices.json", dict(class_order=STAGES, matrices=matrices))
    write_csv(dest / "predictions.csv", records)
    figures = dest / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    performance(SimpleNamespace(partition=title), dest, figures)
    write_json(dest / "settings.json", dict(partition="test", smoke=is_smoke,
                                             cohort_sha256=settings["protocol"]["cohort_sha256"],
                                             training_models_sha256=plan["models_sha256"], scans=len(rows),
                                             examples=[row["scan_id"] for row, _ in selections]))
    print(f"{title} example panels: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "shortlist", "finish"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Prepare the isolated smoke protocol.")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare_protocol(args.run, args.smoke)
        elif args.action == "fit":
            fit(args.run)
        elif args.action == "shortlist":
            shortlist(args.run)
        else:
            finish(args.run)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
