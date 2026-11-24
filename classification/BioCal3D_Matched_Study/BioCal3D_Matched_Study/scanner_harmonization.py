"""Fit a fixed scanner-relative affine map from independent matched swatches."""

import argparse
import csv
from pathlib import Path

import numpy as np

from stage_ablation import read_json
from study_data import SCANNERS, canonical_scanner, digest, write_json


def read_matrices(path):
    mapping = read_json(path)
    if mapping.get("schema") != 1 or mapping.get("fit_scope") != "independent_calibration":
        raise ValueError("Use a schema-1 mapping fitted on independent calibration data.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swatches", type=Path, required=True, help="CSV with scanner,patch_id,r,g,b in 0–255 sRGB.")
    parser.add_argument("--reference", choices=SCANNERS, default="TRIOS5")
    parser.add_argument("--out", type=Path, required=True)
    fit_chart(parser.parse_args())
