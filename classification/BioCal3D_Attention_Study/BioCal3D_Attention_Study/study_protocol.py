"""The registered specimen split and input conditions for the BioCal3D study."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from study_data import SCANNERS, STAGES, digest, load_snapshot, seed_for, write_csv, write_json
from study_io import read_json

LEAD_SPECIMEN = "BCG3-1"
LEAD_SCAN = "BCG3T1-1"
SEED = 20261104
ROLES = ("train", "validation", "test")
INPUTS = {
    "dinov3_gray": ("dinov3", "gray", "DINOv3 · grayscale"),
    "dinov3_rgb": ("dinov3", "rgb", "DINOv3 · raw RGB"),
    "concerto_geometry": ("concerto", "geometry", "Concerto · geometry"),
    "concerto_rgb": ("concerto", "rgb", "Concerto · geometry + raw RGB"),
    "dinov3_harmonized": ("dinov3", "harmonized", "DINOv3 · harmonised RGB"),
    "concerto_harmonized": ("concerto", "harmonized", "Concerto · geometry + harmonised RGB"),
}


def input_modes(harmonized=False, family=None):
    return [mode for mode, (encoder, variant, _) in INPUTS.items()
            if (family is None or family == encoder) and (variant != "harmonized" or harmonized)]


def is_lead(row):
    return row["sample_name"] == LEAD_SCAN


def assign(rows):
    assignment = {}
    for group in range(1, 7):
        ids = sorted({row["physical_id"] for row in rows if row["group"] == group})
        if len(ids) != 8:
            raise ValueError(f"Expected eight physical specimens in group {group}.")
        np.random.default_rng(seed_for(f"holdout:G{group}", SEED)).shuffle(ids)
        if LEAD_SPECIMEN in ids:
            ids.remove(LEAD_SPECIMEN)
            ids.insert(0, LEAD_SPECIMEN)
        holdout, development = ids[:2], ids[2:]
        np.random.default_rng(seed_for(f"validation:G{group}", SEED)).shuffle(development)
        assignment.update({pid: "test" for pid in holdout})
        assignment[development[0]] = "validation"
        assignment.update({pid: "train" for pid in development[1:]})
    return [dict(row, role=assignment[row["physical_id"]]) for row in rows]


def prepare(run, roi, smoke=False):
    roi = roi.resolve()
    index = roi / "dataset_index.json"
    if not index.is_file():
        raise ValueError(f"Choose the Approved ROIs_reoriented folder containing dataset_index.json: {roi}")
    if run.resolve().is_relative_to(roi):
        raise ValueError("Choose a run folder outside the approved dataset.")
    stamp = roi / "scanner_orientation.json"
    if (roi / "INCOMPLETE_ORIENTATION.json").exists() or (
        stamp.is_file() and read_json(stamp).get("status") != "complete"
    ):
        raise ValueError("The orientation export is incomplete.")
    path = run / "cohort.json"
    if path.is_file():
        saved = read_json(path)
        if saved["smoke"] != smoke or Path(saved["audit"]["index"]).resolve() != index:
            raise ValueError("This run is assigned to a different dataset or split size. Choose a fresh run folder.")
        if saved["audit"]["index_sha256"] != digest(index):
            raise ValueError("The approved dataset index has changed. Use a fresh run folder.")
        rows = saved["rows"]
    else:
        rows, audit, excluded = load_snapshot(roi, use_saved_splits=False, approved_export=True)
        if len(rows) != 512 or len({row["mesh_path"] for row in rows}) != 512:
            raise ValueError("Expected 512 unique approved meshes.")
        for row in rows:
            meta = Path(row["folder"]) / "metadata.json"
            if meta.is_file():
                info = read_json(meta)
                if str(info.get("geometry_review_status", "")).lower() in {"pending", "flagged", "rejected", "draft"}:
                    raise ValueError(f"Unapproved geometry: {row['scan_id']}")
                if info.get("geometry_sha256") and info["geometry_sha256"] != row["geometry_sha256"]:
                    raise ValueError(f"Mesh differs from its approved metadata: {row['scan_id']}")
        rows = assign(rows)
        if smoke:
            ids = {LEAD_SPECIMEN}
            for role in ("train", "validation"):
                ids.add(min(r["physical_id"] for r in rows if r["role"] == role and r["group"] >= 3))
            rows = [row for row in rows if row["physical_id"] in ids]
        write_csv(run / "split.csv", [{k: r[k] for k in ("scan_id", "physical_id", "group", "scanner", "stage", "role")}
                                     for r in rows])
        if excluded:
            write_csv(run / "excluded.csv", excluded)
        write_json(path, dict(audit=audit, seed=SEED, smoke=smoke, rows=rows))

    partitions = {role: [row for row in rows if row["role"] == role] for role in ROLES}
    ids = {role: {row["physical_id"] for row in part} for role, part in partitions.items()}
    if len(set().union(*ids.values())) != sum(map(len, ids.values())):
        raise ValueError("Physical specimens span multiple partitions.")
    expected = (1, 1, 1) if smoke else (30, 6, 12)
    if tuple(len(ids[role]) for role in ROLES) != expected:
        raise ValueError("The saved split has unexpected specimen counts.")
    if {row["scanner"] for row in partitions["test"] if is_lead(row)} != set(SCANNERS):
        raise ValueError(f"{LEAD_SCAN} must appear on all four scanners in test.")
    for role, part in partitions.items():
        if any({r["stage"] for r in part if r["scanner"] == scanner} != set(STAGES) for scanner in SCANNERS):
            raise ValueError(f"Missing cleaning stage in {role}.")
        print(f"{role}: {len(ids[role])} specimens, {len(part)} scans, {dict(Counter(r['stage'] for r in part))}",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--harmonized-roi", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.run, args.roi, args.smoke)
        if args.harmonized_roi is not None:
            from harmonized_inputs import prepare_harmonized
            prepare_harmonized(args.run, args.harmonized_roi)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
