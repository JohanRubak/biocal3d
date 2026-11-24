"""Fit scanner-relative affine RGB maps from training scans or independent swatches."""

import argparse
import csv
from pathlib import Path

import numpy as np

from fixed_split import split_rows
from stage_ablation import read_json
from study_data import SCANNERS, canonical_scanner, digest, make_surface, write_json


def read_matrices(path, cohort_path=None):
    mapping = read_json(path)
    scope = mapping.get("fit_scope")
    if mapping.get("schema") != 1 or scope not in ("independent_calibration", "fixed_split_training"):
        raise ValueError("Use a schema-1 mapping from independent swatches or the fixed training split.")
    if scope == "fixed_split_training":
        if cohort_path is None or mapping.get("cohort_sha256") != digest(cohort_path):
            raise ValueError("The training-only colour map belongs to a different cohort.")
        training = split_rows(read_json(cohort_path))["training"]
        if mapping.get("training_scan_ids") != sorted(row["scan_id"] for row in training):
            raise ValueError("The colour map must be fitted on exactly the fixed training scans.")
    if set(mapping["matrices"]) != set(SCANNERS):
        raise ValueError("The colour mapping must cover the four study scanners.")
    matrices = {key: np.asarray(value, dtype=float) for key, value in mapping["matrices"].items()}
    if any(matrix.shape != (3, 4) or not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Each scanner matrix must be finite and shaped 3 by 4.")
    return matrices


def fit_chart(args):
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}")
    patches = {scanner: {} for scanner in SCANNERS}
    with args.swatches.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            scanner, patch = canonical_scanner(row["scanner"]), row["patch_id"].strip()
            rgb = np.array([float(row[channel]) for channel in ("r", "g", "b")])
            if not patch or patch in patches[scanner] or not np.all((rgb >= 0) & (rgb <= 255)):
                raise ValueError(f"Invalid or repeated calibration row for {scanner}, {patch}.")
            patches[scanner][patch] = rgb / 255
    keys = sorted(patches[args.reference])
    if len(keys) < 6 or any(set(by_patch) != set(keys) for by_patch in patches.values()):
        raise ValueError("Supply at least six corresponding patch IDs from each scanner.")
    target = np.stack([patches[args.reference][key] for key in keys])
    identity = np.eye(4)[:, :3]
    matrices = {}
    for scanner, by_patch in patches.items():
        design = np.column_stack((np.stack([by_patch[key] for key in keys]), np.ones(len(keys))))
        if np.linalg.matrix_rank(design) != 4:
            raise ValueError(f"Calibration patches for {scanner} do not span RGB and an intercept.")
        # Shrink toward identity to limit extreme corrections on a small chart.
        weight = np.sqrt(.01 * len(keys))
        coeffs, *_ = np.linalg.lstsq(np.vstack((design, weight * np.eye(4))),
                                    np.vstack((target, weight * identity)), rcond=None)
        matrices[scanner] = (identity if scanner == args.reference else coeffs).T.tolist()
    write_json(args.out, dict(schema=1, fit_scope="independent_calibration", reference=args.reference,
                              calibration_sha256=digest(args.swatches), patches=len(keys), matrices=matrices))


def fit_training(cohort_path, output, reference="TRIOS5"):
    training = split_rows(read_json(cohort_path))["training"]
    settings = dict(align_plane=False, scale_mm=10., sampling_seed=123,
                    sampling="surface", points=4096, inputs="geometry_rgb")
    quantiles = (.10, .25, .50, .75, .90)
    measurements = {scanner: {} for scanner in SCANNERS}
    for i, row in enumerate(training, start=1):
        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"Training geometry changed after preparation: {row['scan_id']}.")
        surface = make_surface(row, settings, "concerto")
        if not surface["has_rgb"]:
            raise ValueError(f"Training scan has no RGB: {row['scan_id']}.")
        scanner, sample = row["scanner"], row["sample_name"]
        if sample in measurements[scanner]:
            raise ValueError(f"Duplicate training specimen-stage for {scanner}: {sample}.")
        measurements[scanner][sample] = np.quantile(surface["display_rgb"], quantiles, axis=0)
        if i == 1 or i % 32 == 0 or i == len(training):
            print(f"Training colour measurements: {i}/{len(training)} scans.", flush=True)

    samples = sorted(measurements[reference])
    if not samples or any(set(anchors) != set(samples) for anchors in measurements.values()):
        raise ValueError("Harmonisation requires the same training specimen-stages on all four scanners.")
    target = np.concatenate([measurements[reference][sample] for sample in samples])
    matrices, diagnostics = {}, {}
    for scanner, anchors in measurements.items():
        source = np.concatenate([anchors[sample] for sample in samples])
        gain, offset = np.ones(3), np.zeros(3)
        if scanner != reference:
            for channel in range(3):
                design = np.column_stack((source[:, channel], np.ones(len(source))))
                regularization = np.sqrt(.01 * len(source)) * np.eye(2)
                slope, _ = np.linalg.lstsq(np.vstack((design, regularization)),
                                           np.r_[target[:, channel], regularization[:, 0]], rcond=None)[0]
                gain[channel] = np.clip(slope, .5, 2.)
                offset[channel] = np.median(target[:, channel] - gain[channel] * source[:, channel])
        matrices[scanner] = np.column_stack((np.diag(gain), offset)).tolist()
        transformed = source * gain + offset
        diagnostics[scanner] = {
            "mean_absolute_anchor_error_raw": float(np.mean(np.abs(source - target))),
            "mean_absolute_anchor_error_harmonized": float(np.mean(np.abs(np.clip(transformed, 0, 1) - target))),
            "fraction_mapped_anchors_outside_srgb": float(np.mean((transformed < 0) | (transformed > 1))),
        }
    write_json(output, dict(schema=1, fit_scope="fixed_split_training", reference=reference,
                            method="Per-channel affine RGB from matched specimen-stage surface quantiles",
                            cohort_sha256=digest(cohort_path),
                            training_scan_ids=sorted(row["scan_id"] for row in training),
                            surface_samples_per_scan=settings["points"], quantiles=quantiles,
                            diagnostics=diagnostics, matrices=matrices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swatches", type=Path, required=True, help="CSV with scanner,patch_id,r,g,b in 0–255 sRGB.")
    parser.add_argument("--reference", choices=SCANNERS, default="TRIOS5")
    parser.add_argument("--out", type=Path, required=True)
    fit_chart(parser.parse_args())
