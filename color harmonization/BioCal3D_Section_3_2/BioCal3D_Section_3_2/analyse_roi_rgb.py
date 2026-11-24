"""Compare RGB distributions in the full extracted regions on approved ROI maps."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_harmonization import rgb_values
from investigate_approved_rois import SCANNERS, STAGES, Figures, discover, plt

METRICS = ("mean_rgb_distance", "rgb_sd_distance", "sliced_wasserstein")
PAIRS = list(combinations(SCANNERS, 2))
SCOPES = ("all", *STAGES)


def load_regions(raw_path, corrected_path):
    if raw_path.resolve() == corrected_path.resolve():
        raise ValueError("Choose separate raw and harmonized maps.")
    with np.load(raw_path, allow_pickle=False) as raw, np.load(corrected_path, allow_pickle=False) as corrected:
        valid = raw["valid"].astype(bool)
        raw_rgb, corrected_rgb = raw["rgb"], corrected["rgb"]
        if valid.ndim != 2 or raw_rgb.shape != valid.shape + (3,) or corrected_rgb.shape != raw_rgb.shape:
            raise ValueError("The RGB maps and validity mask must share the same grid.")
        for key in ("valid", "fov_mm"):
            if not np.array_equal(raw[key], corrected[key]):
                raise ValueError(f"Raw and harmonized maps differ in {key}.")
        for key in ("vertices", "faces", "height_mm"):
            if key in raw and key in corrected and not np.array_equal(raw[key], corrected[key], equal_nan=True):
                raise ValueError(f"Raw and harmonized maps differ in {key}.")
        if valid.sum() < 2:
            raise ValueError("The extracted ROI contains fewer than two valid pixels.")
        return rgb_values(raw_rgb[valid]), rgb_values(corrected_rgb[valid]), valid


def describe(rgb, indices, directions):
    return rgb.mean(0), rgb.std(0), np.sort(rgb[indices] @ directions.T, axis=0)


def distance(a, b):
    return np.array([np.linalg.norm(a[0]-b[0]), np.linalg.norm(a[1]-b[1]),
                     np.abs(a[2]-b[2]).mean()])


def benchmark(features, inventory):
    references, scanner_rows = {}, []
    for (cohort, stage), stage_inventory in inventory.groupby(["cohort", "stage"]):
        specimens = sorted(set(stage_inventory.physical_id))
        for scanner in SCANNERS:
            scans = [features[specimen, scanner, stage][0] for specimen in specimens
                     if (specimen, scanner, stage) in features]
            values = [distance(a, b) for a, b in combinations(scans, 2)]
            refs = np.mean(values, axis=0) if values else np.full(len(METRICS), np.nan)
            references[cohort, scanner, stage] = refs
            scanner_rows.extend(dict(cohort=cohort, stage=stage, scanner=scanner, metric=m,
                                     raw_reference=v, n_specimens=len(scans), n_distinct_pairs=len(values))
                                for m, v in zip(METRICS, refs))
    pair_rows = []
    for row in inventory[["physical_id", "cohort", "stage"]].drop_duplicates().itertuples():
        for a, b in PAIRS:
            x = features.get((row.physical_id, a, row.stage))
            y = features.get((row.physical_id, b, row.stage))
            if x is None or y is None:
                continue
            ref = (references[row.cohort, a, row.stage] + references[row.cohort, b, row.stage])/2
            for metric, raw, corrected, target in zip(METRICS, distance(x[0], y[0]), distance(x[1], y[1]), ref):
                pair_rows.append(dict(physical_id=row.physical_id, cohort=row.cohort, stage=row.stage,
                                      scanner_a=a, scanner_b=b, metric=metric, raw=raw,
                                      corrected=corrected, raw_within_reference=target))
    columns = ["physical_id", "cohort", "stage", "scanner_a", "scanner_b", "metric",
               "raw", "corrected", "raw_within_reference"]
    return pd.DataFrame(pair_rows, columns=columns), pd.DataFrame(scanner_rows)


def summarize(paired, inventory, args, cohort="all"):
    rows = []
    rng = np.random.default_rng(args.seed)
    values = ["raw", "corrected", "raw_within_reference"]
    eligible = inventory.groupby(["physical_id", "stage"]).scanner.agg(set).reset_index()
    for scope in SCOPES:
        part = paired if scope == "all" else paired[paired.stage == scope]
        scans = eligible if scope == "all" else eligible[eligible.stage == scope]
        for a, b in PAIRS:
            expected = scans.loc[scans.scanner.map(lambda scanners: {a, b} <= scanners).astype(bool)]
            selected = part[(part.scanner_a == a) & (part.scanner_b == b)]
            for metric in METRICS:
                records = selected[selected.metric == metric]
                per_specimen = records.groupby("physical_id")[values].mean()
                raw, corrected = per_specimen[["raw", "corrected"]].mean()
                change = per_specimen.corrected.to_numpy() - per_specimen.raw.to_numpy()
                lo, hi = np.full(2, np.nan)
                if len(change) >= 2:
                    boot = rng.choice(change, size=(args.bootstrap, len(change))).mean(axis=1)
                    lo, hi = np.quantile(boot, [.025, .975])
                # Ratios share the same reference-eligible observations for both conditions.
                ref_records = records[records.raw_within_reference > 0]
                ref_specimens = ref_records.groupby("physical_id")[values].mean()
                ref_raw, ref_corrected, ref = ref_specimens.mean()
                rows.append(dict(cohort=cohort, scope=scope, scanner_a=a, scanner_b=b, metric=metric,
                                 available_specimens=expected.physical_id.nunique(),
                                 available_scan_pairs=len(expected), n_specimens=len(per_specimen),
                                 n_scan_pairs=len(records), n_ratio_specimens=len(ref_specimens),
                                 n_ratio_scan_pairs=len(ref_records), raw=raw, corrected=corrected,
                                 corrected_minus_raw=corrected-raw, ci_low=lo, ci_high=hi,
                                 reduction_percent=100*(raw-corrected)/raw if raw > 0 else np.nan,
                                 raw_within_reference=ref, raw_ratio=ref_raw/ref if ref > 0 else np.nan,
                                 corrected_ratio=ref_corrected/ref if ref > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--corrected-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--projections", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    if args.samples < 2 or args.projections < 1 or args.bootstrap < 100 or args.dpi < 72:
        p.error("Use at least 2 samples, 1 projection, 100 bootstrap draws and 72 DPI.")
    args.raw_root = args.raw_root.resolve()
    args.corrected_root = args.corrected_root.resolve()
    inventory = discover(args.raw_root).sort_values("scan_id")
    if len(inventory) != 512:
        print(f"Found {len(inventory)} raw scans. Full study inventory is 512.")
    rng = np.random.default_rng(args.seed)
    directions = rng.normal(size=(args.projections, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    features, qc = {}, []
    args.output.mkdir(parents=True, exist_ok=True)
    for row in inventory.itertuples():
        if row.review_status in {"pending", "flagged", "rejected"}:
            raise ValueError(f"{row.scan_id} is not an approved ROI.")
        relative = Path(row.sample_dir).relative_to(args.raw_root)
        raw_path = args.raw_root / relative / "maps.npz"
        corrected_path = args.corrected_root / relative / "maps.npz"
        try:
            raw, corrected, valid = load_regions(raw_path, corrected_path)
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"{row.scan_id}: {error}") from error
        qc.append(dict(scan_id=row.scan_id, physical_id=row.physical_id, scanner=row.scanner,
                       cohort=row.cohort, stage=row.stage, valid_pixels=int(valid.sum())))
        indices = rng.choice(len(raw), size=args.samples, replace=len(raw) < args.samples)
        features[row.physical_id, row.scanner, row.stage] = (
            describe(raw, indices, directions), describe(corrected, indices, directions))
    paired, references = benchmark(features, inventory)
    summary = summarize(paired, inventory, args)
    cohort_summary = pd.concat([
        summarize(paired[paired.cohort == cohort], part, args, cohort)
        for cohort, part in inventory.groupby("cohort")
    ], ignore_index=True)
    cohort_summary = cohort_summary[~((cohort_summary.cohort == "fresh") &
                                     (cohort_summary.scope == "partial"))]

    pd.DataFrame(qc).to_csv(args.output / "roi_mask_qc.csv", index=False)
    references.to_csv(args.output / "roi_raw_within_scanner.csv", index=False)
    paired.to_csv(args.output / "roi_paired_distances.csv", index=False)
    summary.to_csv(args.output / "roi_before_after_summary.csv", index=False)
    pd.DataFrame([vars(args)]).to_csv(args.output / "roi_settings.csv", index=False)
    cohort_summary.to_csv(args.output / "roi_cohort_before_after_summary.csv", index=False)
    from manuscript_figures import swd_bar_figure, supporting_rgb_tables
    swd_bar_figure(cohort_summary, Figures(args.output / "manuscript_figures", args))
    supporting_rgb_tables(cohort_summary, args.output)
    print(f"Included {len(features)} of {len(inventory)} scans using all valid ROI pixels.")


if __name__ == "__main__":
    main()
