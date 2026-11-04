"""Map stage-related colour evidence with weakly supervised gradient boosting."""

import argparse
import csv
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
from PIL import Image, ImageCms
from scipy import ndimage as ndi
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from segment_color_gmm import REFERENCE_DIAMETER, SCANNERS, STAGES, save_comparison


def select_rows(args, partition):
    manifest = json.loads((args.run / "cohort.json").read_text(encoding="utf-8-sig"))
    rows = [row for row in manifest["rows"] if row["partition"] == partition
            and (args.scanner is None or row["scanner"] == args.scanner)
            and (args.stage is None or row["stage"] == args.stage)]
    if args.scan:
        missing = set(args.scan) - {row["scan_id"] for row in rows}
        if missing:
            raise ValueError(f"Scan IDs outside the selected cohort: {', '.join(sorted(missing))}")
        rows = [row for row in rows if row["scan_id"] in args.scan]
    if not rows:
        raise ValueError(f"No matching {partition} scans.")
    return sorted(rows, key=lambda row: row["scan_id"])


def scan_folder(scan_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scan_id).strip(".")


def read_render(run, row):
    folder = run / "renders" / row["partition"] / "images" / row["scanner"] / row["sample_name"]
    with Image.open(folder / "rgb.png") as source:
        image = source.convert("RGB")
    with Image.open(folder / "valid_mask.png") as source:
        roi = np.asarray(source.convert("L")) > 0
    if roi.shape != (image.height, image.width) or roi.sum() < 32:
        raise ValueError(f"Invalid render ROI for {row['scan_id']}.")
    return image, roi


def pixel_features(image, roi):
    """Local Lab contrast at two scales."""
    scale = np.sqrt(4 * roi.sum() / np.pi) / REFERENCE_DIAMETER
    support = roi.astype(float)
    lab = np.asarray(ImageCms.profileToProfile(
        image, ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), outputMode="LAB"
    ), dtype=np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    masked_lab = lab * support[..., None]

    def local_average(sigma):
        weights = ndi.gaussian_filter(support, sigma * scale)
        values = ndi.gaussian_filter(masked_lab, (sigma * scale, sigma * scale, 0))
        return values / np.maximum(weights[..., None], 1e-8)

    fine = local_average(1.5)
    mid = local_average(5)
    broad = local_average(18)
    distance = ndi.distance_transform_edt(np.pad(roi, 1))[1:-1, 1:-1]
    eligible = distance >= max(1.0, 7 * scale)
    if not eligible.any():
        raise ValueError("No pixels remain inside the eligible ROI.")
    features = np.column_stack(((fine - mid)[eligible], (fine - broad)[eligible])).astype(np.float32)
    return features, eligible, scale


def color_gate(features):
    """Use local dark-red contrast as a spatial prior, independent of stage."""
    return (features[:, 3] < -.3) & (features[:, 4] > .75)


def training_data(args):
    groups = {}
    for row in select_rows(args, "development"):
        group = groups.setdefault((row["physical_id"], row["scanner"]), {})
        if row["stage"] in group:
            raise ValueError(f"Duplicate stage for {row['physical_id']} on {row['scanner']}.")
        group[row["stage"]] = row

    rng = np.random.default_rng(3)
    rows, samples, earlier, later = [], [], [], []
    for group in groups.values():
        ordered = [group[stage] for stage in STAGES if stage in group]
        if len(ordered) < 2:
            continue
        first = len(rows)
        for row in ordered:
            image, roi = read_render(args.run, row)
            features, _, _ = pixel_features(image, roi)
            pick = rng.choice(len(features), min(len(features), args.pixels_per_scan), replace=False)
            samples.append(features[pick])
            rows.append(row)
        earlier.extend(range(first, len(rows) - 1))
        later.extend(range(first + 1, len(rows)))
    if not earlier:
        raise ValueError("No matched development scans with two or more cleaning stages.")
    if not any(row["stage"] == "clean" for row in rows):
        raise ValueError("Include final-clean development scans to set the preview threshold.")
    counts = np.array([len(sample) for sample in samples])
    print(f"Using {len(rows)} development scans and {len(earlier)} matched stage comparisons.",
          flush=True)
    return rows, np.concatenate(samples), counts, np.array(earlier), np.array(later)


def pool_scores(scores, starts, counts, bag):
    """A scan score combines the mean and a smooth maximum of its pixel scores."""
    average = np.add.reduceat(scores, starts) / counts
    maximum = np.maximum.reduceat(scores, starts)
    exponential = np.exp(scores - maximum[bag])
    totals = np.add.reduceat(exponential, starts)
    smooth_maximum = maximum + np.log(totals / counts)
    score = .7 * average + .3 * smooth_maximum
    gradient = .7 / counts[bag] + .3 * exponential / totals[bag]
    return score, gradient


def fit(args):
    rows, features, counts, earlier, later = training_data(args)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    bag = np.repeat(np.arange(len(rows)), counts)
    scores = np.zeros(len(features))
    gate = color_gate(features)
    margin = .25
    trees = []

    def ranking_loss(burden):
        return float(np.logaddexp(0, margin - burden[earlier] + burden[later]).mean())

    with threadpool_limits(limits=2):
        burden, pool_gradient = pool_scores(scores, starts, counts, bag)
        loss = ranking_loss(burden)
        for iteration in range(1, args.trees + 1):
            weights = expit(margin - burden[earlier] + burden[later])
            scan_residual = (np.bincount(earlier, weights=weights, minlength=len(rows))
                             - np.bincount(later, weights=weights, minlength=len(rows)))
            residual = scan_residual[bag] * pool_gradient * counts.mean() * gate
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=32, random_state=3)
            tree.fit(features, residual)
            update = tree.predict(features) * gate
            step = .15
            for _ in range(7):
                trial = scores + step * update
                trial_burden, trial_gradient = pool_scores(trial, starts, counts, bag)
                trial_loss = ranking_loss(trial_burden)
                if trial_loss < loss - 1e-9:
                    break
                step /= 2
            else:
                break
            scores, burden, pool_gradient, loss = trial, trial_burden, trial_gradient, trial_loss
            trees.append((step, tree))
            if iteration == 1 or iteration % 10 == 0:
                ordered = np.mean(burden[earlier] > burden[later])
                print(f"Tree {iteration:3d}  training ranking loss {loss:.4f}"
                      f"  correctly ordered pairs {ordered:.1%}", flush=True)
    if not trees:
        raise ValueError("No tree improved the stage ranking. Check the training selection.")

    clean = np.array([row["stage"] == "clean" for row in rows])
    threshold = max(0., float(np.quantile(scores[clean[bag]], .99)))
    artifact = {
        "version": 2,
        "trees": trees,
        "threshold": threshold,
        "specimens": sorted({row["physical_id"] for row in rows}),
        "scans": [row["scan_id"] for row in rows],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=".boost-", dir=args.model.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(artifact, temporary)
        temporary.replace(args.model)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.model} with {len(trees)} trees. Preview cutoff: {threshold:.4f}.")


def predict(args):
    # Joblib files contain executable pickle data. Only load a model you created.
    artifact = joblib.load(args.model)
    if artifact["version"] != 2:
        raise ValueError("Unrecognized model format. Refit with this script.")
    rows = select_rows(args, args.partition)
    training_specimens = set(artifact["specimens"])
    held_out = [row for row in rows if row["physical_id"] not in training_specimens]
    skipped = len(rows) - len(held_out)
    if skipped and args.scan:
        raise ValueError("A requested scan belongs to a specimen used during fitting.")
    if not held_out:
        raise ValueError("No held-out specimens match. Try --partition test.")
    if skipped:
        print(f"Skipped {skipped} scans of specimens used during fitting.")
    rows = held_out[:args.limit]
    output_rows = []
    threshold = artifact["threshold"] if args.threshold is None else args.threshold
    for row in rows:
        image, roi = read_render(args.run, row)
        features, eligible, scale = pixel_features(image, roi)
        scores = np.zeros(len(features))
        for step, tree in artifact["trees"]:
            scores += step * tree.predict(features)
        candidate = np.zeros_like(roi)
        candidate[eligible] = (scores > threshold) & color_gate(features)
        components, _ = ndi.label(candidate)
        sizes = np.bincount(components.ravel())
        candidate &= sizes[components] >= max(1, round(10 * scale ** 2))

        output = args.out / scan_folder(row["scan_id"])
        output.mkdir(parents=True, exist_ok=True)
        mask = Image.fromarray(candidate.astype(np.uint8) * 255)
        mask.save(output / "candidate_mask.png")
        overlay = np.array(image)
        overlay[candidate] = np.rint(.4 * overlay[candidate] + .6 * np.array([255, 226, 94]))
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output / "overlay.png")
        save_comparison(image, overlay_image, mask, output / "comparison.png")
        fraction = float(candidate.sum() / roi.sum())
        output_rows.append({
            "scan_id": row["scan_id"], "scanner": row["scanner"], "stage": row["stage"],
            "partition": row["partition"], "score_threshold": threshold,
            "candidate_fraction": fraction,
            "output_folder": str(output),
        })
        print(f"{row['scan_id']}: {fraction:.1%} of projected ROI", flush=True)

    summary = args.out / "summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("fit", help="Fit using matched development cleaning stages.")
    training.set_defaults(action=fit, stage=None, scan=None)
    training.add_argument("--pixels-per-scan", type=int, default=1024)
    training.add_argument("--trees", type=int, default=80)
    prediction = commands.add_parser("predict", help="Write masks and three-panel comparisons.")
    prediction.set_defaults(action=predict)
    prediction.add_argument("--partition", choices=("development", "test"), default="test")
    prediction.add_argument("--out", type=Path, default=Path("boosting_preview"))
    prediction.add_argument("--threshold", type=float, help="Override the fitted raw score cutoff.")
    prediction.add_argument("--stage", choices=STAGES)
    prediction.add_argument("--scan", nargs="+", help="One or more exact scan IDs.")
    prediction.add_argument("--limit", type=int, help="Maximum scans after filtering.")
    for command in (training, prediction):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--scanner", choices=SCANNERS)

    args = parser.parse_args()
    if args.command == "fit":
        if args.pixels_per_scan < 32 or args.trees < 1:
            parser.error("--pixels-per-scan must be at least 32 and --trees must be positive.")
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive.")
        if args.threshold is not None and not np.isfinite(args.threshold):
            parser.error("--threshold must be finite.")
    try:
        args.action(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
