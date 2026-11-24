"""Assign complete specimens to training or test and define the smoke subset."""

from collections import Counter

from study_data import SCANNERS, STAGES

LEAD_SPECIMEN = "BCG3-1"
LEAD_SCAN = "BCG3T1-1"


def split_rows(manifest, smoke=False):
    partitions = {"training": [], "test": []}
    by_specimen, seen = {}, set()
    for row in manifest["rows"]:
        scan = row["scan_id"]
        if scan in seen:
            raise ValueError(f"Duplicate scan ID: {scan}")
        seen.add(scan)
        pid = row["physical_id"]
        if row["partition"] not in ("development", "test"):
            raise ValueError(f"Unexpected saved partition for {scan}.")
        role = "test" if row["partition"] == "test" else "training"
        if by_specimen.setdefault(pid, role) != role:
            raise ValueError(f"Specimen {pid} spans more than one partition.")
        partitions[role].append(row)
    counts = Counter(by_specimen.values())
    if counts != {"training": 36, "test": 12}:
        raise ValueError(f"Expected 36 training and 12 test specimens. Found {dict(counts)}.")
    lead = [row for row in partitions["test"] if row["sample_name"] == LEAD_SCAN]
    if (by_specimen.get(LEAD_SPECIMEN) != "test" or len(lead) != len(SCANNERS)
            or {row["scanner"] for row in lead} != set(SCANNERS)):
        raise ValueError(f"Reserve the complete specimen {LEAD_SPECIMEN} and all views of {LEAD_SCAN} for test.")
    if smoke:
        train_example = min(row["physical_id"] for row in partitions["training"] if row["group"] >= 3)
        partitions = {
            "training": [row for row in partitions["training"] if row["physical_id"] == train_example],
            "test": [row for row in partitions["test"] if row["physical_id"] == LEAD_SPECIMEN],
        }
    for role, rows in partitions.items():
        for scanner in SCANNERS:
            if {row["stage"] for row in rows if row["scanner"] == scanner} != set(STAGES):
                raise ValueError(f"{role} lacks a cleaning stage on {scanner}.")
    return partitions
