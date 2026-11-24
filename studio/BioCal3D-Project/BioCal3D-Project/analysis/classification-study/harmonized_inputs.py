"""Pair precomputed harmonized ROI colours with the locked raw cohort."""
from pathlib import Path

import numpy as np

from study_data import digest, geometry, load_snapshot, signature, write_json
from study_io import read_json


def paired_mesh(raw, corrected, mesh=None):
    for row in (raw, corrected):
        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"ROI changed since it was indexed: {row['mesh_path']}")
    if mesh is None:
        mesh = geometry(raw)
    updated = geometry(corrected)
    if not all(np.array_equal(mesh[i], updated[i]) for i in (0, 1)):
        raise ValueError(f"Harmonized ROI must preserve the approved vertices and faces: {raw['scan_id']}")
    if updated[3] is None or updated[3].max() <= 0:
        raise ValueError(f"Missing harmonized vertex RGB: {raw['scan_id']}")
    return (*mesh[:3], updated[3], *mesh[4:])


def prepare_harmonized(run, roi):
    index = roi.resolve() / "dataset_index.json"
    if not read_json(index).get("config", {}).get("harmonized_roi_export"):
        raise ValueError(f"Choose the output of export_harmonized_rois.py: {roi}")
    raw = read_json(run / "cohort.json")
    corrected, audit, _ = load_snapshot(roi, use_saved_splits=False, approved_export=True)
    by_id = {row["row_id"]: row for row in corrected}
    rows = []
    for row in raw["rows"]:
        match = by_id.get(row["row_id"])
        if not match or any(row[key] != match[key] for key in ("scan_id", "scanner", "sample_name",
                                                             "physical_id", "group", "stage")):
            raise ValueError(f"Harmonized dataset does not match the locked scan: {row['scan_id']}")
        rows.append(match)
    result = dict(index=str(index), index_sha256=audit["index_sha256"],
                  cohort_sha256=digest(run / "cohort.json"), rows=rows)
    manifest = run / "harmonized_cohort.json"
    if manifest.exists():
        if read_json(manifest) != result:
            raise ValueError("The paired harmonized dataset has changed. Choose a fresh --run folder.")
    else:
        for row, match in zip(raw["rows"], rows):
            paired_mesh(row, match)
        write_json(manifest, result)
    print(f"Paired {len(rows)} harmonized scans to the existing specimen split.", flush=True)


def harmonized_inputs(run, roi):
    manifest = read_json(run / "harmonized_cohort.json")
    index = roi.resolve() / "dataset_index.json"
    if (Path(manifest["index"]).resolve() != index or manifest["index_sha256"] != digest(index)
            or manifest["cohort_sha256"] != digest(run / "cohort.json")):
        raise ValueError("The harmonized ROI index or locked cohort differs from the paired run.")
    rows = {row["row_id"]: row for row in manifest["rows"]}
    provenance = dict(roi_index_sha256=manifest["index_sha256"],
                      paired_geometry_sha256=signature({k: r["geometry_sha256"] for k, r in rows.items()}))
    return rows, provenance
