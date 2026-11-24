"""Refresh a completed train/test gallery from saved models and spatial caches."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from fixed_split import LEAD_SCAN
from fixed_study import MODES, by_scan, context, fitted_settings, model_folder, test_folder
from matched_figures import examples, read_csv
from stage_ablation import read_json
from study_data import digest


def refresh(run):
    _, roles, settings = context(run)
    fitted_settings(run, settings)
    models = model_folder(run)
    test = test_folder(run)
    result = read_json(test / "settings.json")
    if (result["cohort_sha256"] != settings["protocol"]["cohort_sha256"]
            or result["training_models_sha256"] != digest(models / "settings.json")):
        raise ValueError("The completed test results belong to different fitted models or specimens.")

    predicted = by_scan(read_csv(test / "predictions.csv"))
    rows = {row["scan_id"]: row for row in roles["test"]}
    selections = []
    for scan in result["examples"]:
        row = rows[scan]
        records = predicted[scan]
        if set(records) != set(MODES) or any(record["truth"] != row["stage"] for record in records.values()):
            raise ValueError(f"Missing or inconsistent saved predictions for {scan}.")
        correct = sum(record["prediction"] == row["stage"] for record in records.values())
        reason = ("Manuscript example" if row["sample_name"] == LEAD_SCAN else
                  "All conditions correct" if correct == len(MODES) else
                  "All conditions incorrect" if not correct else "Correct and incorrect conditions")
        selections.append((row, reason))

    title = "Smoke check" if settings["protocol"]["smoke"] else "Held-out test"
    out = test / "examples"
    out.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="gallery_refresh_", dir=test) as temporary:
        staging = Path(temporary)
        examples(run, staging, models, predicted, selections, title=title)
        for path in staging.iterdir():
            path.replace(out / path.name)
    print(f"Open the interactive gallery: {out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        refresh(args.run)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
