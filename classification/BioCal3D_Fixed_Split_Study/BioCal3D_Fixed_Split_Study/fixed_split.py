"""Select one validation group from the previously saved specimen assignment."""

from collections import Counter

from study_data import SCANNERS, STAGES

VALIDATION_GROUP = 0
LEAD_SPECIMEN = "BCG3-1"
LEAD_SCAN = "BCG3T1-1"


def split_rows(manifest):
    partitions = {"training": [], "validation": [], "test": []}
    by_specimen = {}
    seen = set()
    for row in manifest["rows"]:
        scan = row["scan_id"]
        if scan in seen:
            raise ValueError(f"Duplicate scan ID: {scan}")
        seen.add(scan)
        pid = row["physical_id"]
        if row["partition"] == "test" and row["fold"] is None:
            role = "test"
        elif row["partition"] == "development" and isinstance(row["fold"], int):
            role = "validation" if row["fold"] == VALIDATION_GROUP else "training"
        else:
            raise ValueError(f"Unexpected saved partition for {scan}.")
        if by_specimen.setdefault(pid, role) != role:
            raise ValueError(f"Specimen {pid} spans more than one partition.")
        partitions[role].append(row)
    counts = Counter(by_specimen.values())
    if counts != {"training": 24, "validation": 12, "test": 12}:
        raise ValueError(f"Expected 24/12/12 specimens in the fixed split. Found {dict(counts)}.")
    for role, rows in partitions.items():
        for scanner in SCANNERS:
            if {row["stage"] for row in rows if row["scanner"] == scanner} != set(STAGES):
                raise ValueError(f"{role} lacks a cleaning stage on {scanner}.")
    lead = [row for row in partitions["test"] if row["sample_name"] == LEAD_SCAN]
    if (by_specimen.get(LEAD_SPECIMEN) != "test" or len(lead) != len(SCANNERS)
            or {row["scanner"] for row in lead} != set(SCANNERS)):
        raise ValueError(f"Reserve the complete specimen {LEAD_SPECIMEN} and all views of {LEAD_SCAN} for test.")
    return partitions
