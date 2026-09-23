"""Exploratory scanner-relative colour normalization with fixed offsets.

Input: BioCal3D analysis_dataset.csv from the pixel-colour endpoint pipeline.
Use: python normalize_scanner_color_offsets.py
     python normalize_scanner_color_offsets.py --input PATH/analysis_dataset.csv

The statistics remain ROI FEATURE normalization. When saved map_npz files are
available, additional pixel-map renderings apply the same fixed L*a*b* offset
for visual inspection only; the source maps are never overwritten.
Each offset comes solely from comparisons of the same physical sample at the
same stage. A fixed offset cancels exactly in all within-scanner stage changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold


FEATURES = ("lab_L_mean", "lab_a_mean", "lab_b_mean", "lab_chroma_mean")
KEY = ("physical_sample_id", "stage")
STAGES = ("Baseline", "Partial", "Clean")

# Same default data root and analysis output as validate_precalibration_endpoints.py.
# Change this one path if you move your BioCal3D data.
DATA_ROOT = Path(r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected")
DEFAULT_INPUT = (DATA_ROOT / "_pixel_color_analysis" / "precalibration_validation"
                 / "tables" / "analysis_dataset.csv")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


def load_rows(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    required = {"scanner", "physical_sample_id", "stage", "analysis_ok", *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    rejected = int((~as_bool(frame.analysis_ok)).sum())
    frame = frame.loc[as_bool(frame.analysis_ok)].copy()
    for name in ("scanner", "physical_sample_id", "stage"):
        frame[name] = frame[name].astype(str).str.strip()
    frame = frame.loc[frame.stage.isin(STAGES)].copy()
    frame = frame.loc[frame.scanner.ne("") & frame.physical_sample_id.ne("")]
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    duplicate = frame.duplicated([*KEY, "scanner"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, [*KEY, "scanner"]].head(8).to_dict("records")
        raise ValueError(f"Duplicate physical sample / stage / scanner rows: {examples}")
    if frame.empty:
        raise ValueError("No QC-passing rows with recognized stages.")
    return frame, rejected


def estimate_offsets(rows: pd.DataFrame, reference: str, min_pairs: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference not in set(rows.scanner):
        raise ValueError(f"Reference {reference!r} absent; available: {sorted(rows.scanner.unique())}")
    ref = rows.loc[rows.scanner.eq(reference), [*KEY, *FEATURES]]
    paired = rows.loc[~rows.scanner.eq(reference), [*KEY, "scanner", *FEATURES]].merge(
        ref, on=list(KEY), how="inner", suffixes=("_scanner", "_reference"), validate="many_to_one")
    paired_records: list[dict] = []
    offsets: list[dict] = []
    for scanner in sorted(rows.scanner.unique()):
        for feature in FEATURES:
            if scanner == reference:
                offsets.append(dict(scanner=scanner, feature=feature, offset_to_subtract=0.0,
                                    n_matched_pairs=0, n_stages=0, status="reference"))
                continue
            subset = paired.loc[paired.scanner.eq(scanner), [*KEY,
                feature + "_scanner", feature + "_reference"]].dropna()
            if len(subset) < min_pairs:
                raise ValueError(f"{scanner}/{feature}: only {len(subset)} matched pairs; "
                                 f"require at least {min_pairs}. Cannot fit a reliable offset.")
            differences_by_stage = {}
            for stage, stage_rows in subset.groupby("stage"):
                differences_by_stage[stage] = float(np.median(
                    stage_rows[feature + "_scanner"] - stage_rows[feature + "_reference"]))
            # Equal weight for each available STAGE, regardless of its sample count.
            # The same correction is subsequently applied to ALL stages.
            offset = float(np.median(list(differences_by_stage.values())))
            offsets.append(dict(scanner=scanner, feature=feature, offset_to_subtract=offset,
                                n_matched_pairs=len(subset), n_stages=len(differences_by_stage),
                                status="estimated"))
            for record in subset.itertuples(index=False, name=None):
                sample, stage, scanner_value, ref_value = record
                paired_records.append(dict(physical_sample_id=sample, stage=stage, scanner=scanner,
                                           reference_scanner=reference, feature=feature,
                                           scanner_value=scanner_value, reference_value=ref_value,
                                           raw_difference=scanner_value - ref_value,
                                           corrected_difference=scanner_value - ref_value - offset))
    return pd.DataFrame(offsets), pd.DataFrame(paired_records)


def stage_change_audit(rows: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    records = []
    for feature in FEATURES:
        corrected = feature + "_offset_normalized"
        for (sample, scanner), group in rows.groupby(["physical_sample_id", "scanner"]):
            indexed = group.set_index("stage")
            for first, second in (("Baseline", "Partial"), ("Partial", "Clean"),
                                  ("Baseline", "Clean")):
                if first not in indexed.index or second not in indexed.index:
                    continue
                values = indexed.loc[[first, second], [feature, corrected]]
                if values.isna().any().any():
                    continue
                original_delta = float(values.loc[second, feature] - values.loc[first, feature])
                corrected_delta = float(values.loc[second, corrected] - values.loc[first, corrected])
                records.append(dict(physical_sample_id=sample, scanner=scanner, feature=feature,
                                    comparison=f"{second} - {first}", raw_change=original_delta,
                                    corrected_change=corrected_delta,
                                    difference=corrected_delta - original_delta,
                                    unchanged=bool(np.isclose(corrected_delta, original_delta,
                                                               atol=tolerance, rtol=0))))
    result = pd.DataFrame(records)
    if result.empty or not result.unchanged.all():
        raise AssertionError("At least one within-scanner stage change was altered or none were auditable.")
    return result


def paired_agreement(paired: pd.DataFrame) -> pd.DataFrame:
    # These are descriptive summaries of matched comparisons, not independent pixel samples.
    records = []
    for (scanner, feature), group in paired.groupby(["scanner", "feature"]):
        for label, column in (("raw", "raw_difference"), ("corrected", "corrected_difference")):
            values = group[column].to_numpy(dtype=float)
            records.append(dict(scanner=scanner, feature=feature, version=label,
                                n_matched_sample_stages=len(values),
                                median_scanner_minus_reference=float(np.median(values)),
                                median_absolute_difference=float(np.median(np.abs(values))),
                                mean_difference=float(np.mean(values)),
                                sd_difference=float(np.std(values, ddof=1)) if len(values) > 1 else np.nan))
    return pd.DataFrame(records)


def holdout_evaluation(rows: pd.DataFrame, reference: str, min_pairs: int,
                       n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit only on training physical samples; evaluate matched scans on held-out samples."""
    groups = rows.physical_sample_id.to_numpy()
    n_folds = min(n_folds, len(np.unique(groups)))
    if n_folds < 2:
        raise ValueError("At least two different physical samples are needed for holdout evaluation.")
    predictions = []
    trained_offsets = []
    cv = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, test_idx) in enumerate(cv.split(rows, groups=groups), start=1):
        train = rows.iloc[train_idx]
        test = rows.iloc[test_idx]
        assert set(train.physical_sample_id).isdisjoint(test.physical_sample_id)
        learned, _ = estimate_offsets(train, reference, min_pairs)
        trained_offsets.append(learned.assign(fold=fold))
        reference_rows = test.loc[test.scanner.eq(reference), [*KEY, *FEATURES]]
        matches = test.loc[~test.scanner.eq(reference), [*KEY, "scanner", *FEATURES]].merge(
            reference_rows, on=list(KEY), how="inner", validate="many_to_one",
            suffixes=("_scanner", "_reference"))
        for offset in learned.itertuples(index=False):
            if offset.scanner == reference:
                continue
            for sample, stage, observed, target in matches.loc[
                matches.scanner.eq(offset.scanner),
                [*KEY, offset.feature + "_scanner", offset.feature + "_reference"]
            ].dropna().itertuples(index=False, name=None):
                predictions.append(dict(fold=fold, physical_sample_id=sample, stage=stage,
                                        scanner=offset.scanner, feature=offset.feature,
                                        reference_scanner=reference, reference_value=target,
                                        raw_scanner_value=observed,
                                        normalized_scanner_value=observed - offset.offset_to_subtract,
                                        training_offset=offset.offset_to_subtract,
                                        raw_difference=observed - target,
                                        corrected_difference=observed - target - offset.offset_to_subtract))
    held = pd.DataFrame(predictions)
    if held.empty:
        raise ValueError("No matched held-out scanner pairs to evaluate.")
    summary = paired_agreement(held)
    summary = summary.rename(columns={"n_matched_sample_stages": "n_heldout_sample_stages"})
    return held, summary, pd.concat(trained_offsets, ignore_index=True)


def make_figures(rows: pd.DataFrame, held: pd.DataFrame, summary: pd.DataFrame,
                 reference: str, destination: Path) -> str | None:
    destination.mkdir(parents=True, exist_ok=True)
    labels = {"lab_L_mean": "L* (lightness)", "lab_a_mean": "a* (redness)",
              "lab_b_mean": "b*", "lab_chroma_mean": "C*ab (chroma)"}
    colours = {"LABscanner": "#a35e6f", "TRIOS3": "#3175ab", "TRIOS5": "#28a377",
               "iTERO": "#d58b24"}
    selected = ("lab_a_mean", "lab_chroma_mean", "lab_L_mean")
    scanners = sorted(held.scanner.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, feature in zip(axes, selected):
        for pos, scanner in enumerate(scanners):
            subset = summary.loc[summary.scanner.eq(scanner) & summary.feature.eq(feature)]
            before = float(subset.loc[subset.version.eq("raw"), "median_absolute_difference"].iloc[0])
            after = float(subset.loc[subset.version.eq("corrected"), "median_absolute_difference"].iloc[0])
            ax.plot([pos, pos], [before, after], color=colours.get(scanner, "#555"), lw=2)
            ax.scatter(pos, before, facecolors="white", edgecolors=colours.get(scanner, "#555"), s=95, lw=2, zorder=3)
            ax.scatter(pos, after, color=colours.get(scanner, "#555"), s=70, zorder=3)
        ax.set_xticks(range(len(scanners)), scanners, rotation=25, ha="right")
        ax.set_title(labels[feature])
        ax.set_ylabel("Median absolute difference to " + reference)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("Held-out physical samples: open = raw; solid = offset corrected")
    fig.savefig(destination / "heldout_agreement.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, feature in zip(axes, selected):
        subset = held.loc[held.feature.eq(feature)]
        positions = np.arange(len(STAGES))
        for scanner in scanners:
            vals = subset.loc[subset.scanner.eq(scanner)]
            medians = [vals.loc[vals.stage.eq(stage), "raw_difference"].median() for stage in STAGES]
            ax.plot(positions, medians, "o-", label=scanner, color=colours.get(scanner))
        ax.axhline(0, lw=1, color="black", alpha=.4)
        ax.set_xticks(positions, STAGES)
        ax.set_title(labels[feature])
        ax.set_ylabel("Median scanner minus " + reference)
        ax.grid(axis="y", alpha=.22)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Stage dependence of scanner differences (held-out matches)\n"
                 "A constant offset shifts each line vertically; it cannot flatten it")
    fig.savefig(destination / "scanner_difference_by_stage.png", dpi=180)
    plt.close(fig)

    # A concrete physical sample with all three stages and most scanner observations.
    mature = rows.loc[rows.stage.isin(STAGES)]
    counts = mature.groupby("physical_sample_id").agg(stages=("stage", "nunique"),
                                                     observations=("scanner", "count"))
    eligible = counts.loc[counts.stages.eq(3)].sort_values("observations", ascending=False)
    if eligible.empty:
        return None
    example = str(eligible.index[0])
    sample_rows = mature.loc[mature.physical_sample_id.eq(example)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for row_i, feature in enumerate(("lab_a_mean", "lab_chroma_mean")):
        for col_i, column in enumerate((feature, feature + "_offset_normalized")):
            ax = axes[row_i, col_i]
            for scanner, scan in sample_rows.groupby("scanner"):
                y = [scan.loc[scan.stage.eq(stage), column].iloc[0]
                     if scan.stage.eq(stage).any() else np.nan for stage in STAGES]
                ax.plot(range(3), y, "o-", label=scanner,
                        color=colours.get(scanner), lw=2)
            ax.set_xticks(range(3), STAGES)
            ax.set_ylabel(labels[feature])
            ax.set_title("Raw" if col_i == 0 else "Fixed scanner offset corrected")
            ax.grid(axis="y", alpha=.22)
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Example: {example} | Same sample, same stage, four scanners\n"
                 "Within each scanner, line slopes must stay unchanged")
    fig.savefig(destination / "sample_trajectory_raw_vs_corrected.png", dpi=180)
    plt.close(fig)
    return example


def lab_to_display_rgb(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the sRGB/D65 conversion used by analyze_pixel_color_maps.py.

    Returns displayable RGB and a per-pixel mask of out-of-gamut colours BEFORE
    clipping, so the visual example cannot silently hide clipping.
    """
    lab = np.asarray(lab, dtype=float)
    delta = 6 / 29
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    coords = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(coords > delta, coords ** 3, 3 * delta ** 2 * (coords - 4 / 29))
    xyz *= np.array([0.95047, 1.0, 1.08883])
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    linear = xyz @ np.linalg.inv(matrix).T
    outside = np.isfinite(linear).all(axis=-1) & ((linear < 0).any(axis=-1) | (linear > 1).any(axis=-1))
    rgb = np.where(linear <= 0.0031308, 12.92 * linear,
                   1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1), outside


def pixel_examples(rows: pd.DataFrame, offsets: pd.DataFrame, output: Path,
                   preferred_sample: str | None = None) -> dict:
    """Create actual sample images if map_npz points to accessible saved maps."""
    if "map_npz" not in rows.columns:
        return {"status": "No map_npz column; pixel images unavailable"}
    available = rows.copy()
    available["map_npz"] = available.map_npz.fillna("").astype(str)
    available = available.loc[available.map_npz.map(lambda p: bool(p) and Path(p).is_file())]
    if available.empty:
        return {"status": "No accessible map_npz files; run on the original data computer"}
    counts = available.groupby("physical_sample_id").agg(n_stages=("stage", "nunique"),
                                                         n_maps=("scanner", "size"))
    if preferred_sample is not None:
        if preferred_sample not in counts.index:
            raise ValueError(f"Requested example sample {preferred_sample!r} has no accessible maps.")
        chosen = preferred_sample
    else:
        chosen = str(counts.sort_values(["n_stages", "n_maps"], ascending=False).index[0])
    sample = available.loc[available.physical_sample_id.eq(chosen)]
    scanners = sorted(sample.scanner.unique())
    stages = [stage for stage in STAGES if stage in set(sample.stage)]
    offset_lookup = offsets.set_index(["scanner", "feature"]).offset_to_subtract
    fig, axes = plt.subplots(len(stages) * 2, len(scanners), figsize=(3.2 * len(scanners), 3.0 * len(stages) * 2),
                             squeeze=False, constrained_layout=True)
    report = []
    for stage_i, stage in enumerate(stages):
        for scanner_i, scanner in enumerate(scanners):
            pair = sample.loc[sample.scanner.eq(scanner) & sample.stage.eq(stage)]
            upper, lower = axes[2 * stage_i, scanner_i], axes[2 * stage_i + 1, scanner_i]
            for ax in (upper, lower):
                ax.axis("off")
            if pair.empty:
                continue
            map_path = Path(pair.iloc[0].map_npz)
            try:
                with np.load(map_path, allow_pickle=False) as source:
                    lab = np.asarray(source["lab"], dtype=float)
                    valid = np.asarray(source["valid_mask"], dtype=bool)
                    original_rgb = np.asarray(source["rgb"], dtype=float)
                if lab.ndim != 3 or lab.shape[-1] != 3 or valid.shape != lab.shape[:2] or original_rgb.shape != lab.shape:
                    raise ValueError("Unexpected map dimensions")
                valid = valid & np.isfinite(lab).all(axis=-1) & np.isfinite(original_rgb).all(axis=-1)
                if not valid.any():
                    raise ValueError("No valid coloured pixels")
                corrected_lab = lab.copy()
                for index, feature in enumerate(("lab_L_mean", "lab_a_mean", "lab_b_mean")):
                    corrected_lab[..., index] -= float(offset_lookup.loc[scanner, feature])
                corrected_rgb, clipped = lab_to_display_rgb(corrected_lab)
                for ax, img in ((upper, original_rgb), (lower, corrected_rgb)):
                    display = np.ones_like(img)
                    display[valid] = np.clip(img[valid], 0, 1)
                    ax.imshow(display, origin="lower", interpolation="nearest")
                upper.set_title(f"{stage} | {scanner} | original", fontsize=10)
                lower.set_title(f"{stage} | {scanner} | corrected", fontsize=10)
                if scanner_i == 0:
                    upper.set_ylabel(stage + "\noriginal", fontsize=10)
                    lower.set_ylabel(stage + "\ncorrected", fontsize=10)
                report.append(dict(sample=chosen, stage=stage, scanner=scanner,
                                   map_npz=str(map_path), valid_pixels=int(valid.sum()),
                                   corrected_out_of_gamut_fraction=float(np.mean(clipped[valid])), status="ok"))
            except (OSError, ValueError, KeyError) as error:
                report.append(dict(sample=chosen, stage=stage, scanner=scanner,
                                   map_npz=str(map_path), valid_pixels=0,
                                   corrected_out_of_gamut_fraction=np.nan,
                                   status=f"{type(error).__name__}: {error}"))
    plt.suptitle(f"Actual saved pixel maps: {chosen}\n"
                 "Same scanner offsets at every stage; clipped colours are counted separately")
    png = output / f"pixel_example_{chosen}_original_vs_corrected.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)
    pd.DataFrame(report).to_csv(output / "pixel_example_gamut_audit.csv", index=False)
    return {"status": "saved", "sample": chosen, "figure": str(png),
            "n_successful_maps": sum(row["status"] == "ok" for row in report),
            "max_out_of_gamut_fraction": max((row["corrected_out_of_gamut_fraction"]
                                              for row in report if row["status"] == "ok"), default=None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Existing analysis_dataset.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--reference", default="TRIOS3", help="Scanner used as relative reference (default: TRIOS3)")
    parser.add_argument("--output", type=Path, help="Output directory (default: sibling scanner_offset_normalization)")
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum matched sample/stage pairs per scanner and feature")
    parser.add_argument("--folds", type=int, default=5, help="Physical-sample holdout folds (default: 5)")
    parser.add_argument("--example-sample", default=None,
                        help="Optional physical_sample_id for pixel figure, e.g. BCG3-8; default selects a well-covered sample")
    args = parser.parse_args()
    if args.min_pairs < 2:
        parser.error("--min-pairs must be at least 2")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}\n"
                     "Edit DEFAULT_INPUT at the top of this script if your analysis "
                     "was saved in another folder, or pass --input explicitly.")
    rows, qc_rejected = load_rows(args.input)
    offsets, paired = estimate_offsets(rows, args.reference, args.min_pairs)
    for offset in offsets.itertuples(index=False):
        mask = rows.scanner.eq(offset.scanner)
        rows.loc[mask, offset.feature + "_offset_normalized"] = (
            rows.loc[mask, offset.feature] - offset.offset_to_subtract)
    changes = stage_change_audit(rows)
    held, held_summary, cv_offsets = holdout_evaluation(rows, args.reference, args.min_pairs,
                                                         args.folds)
    output = args.output or args.input.parent / "scanner_offset_normalization"
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "analysis_dataset_with_offset_columns.csv", index=False)
    offsets.to_csv(output / "scanner_offsets.csv", index=False)
    paired.to_csv(output / "matched_scanner_comparisons.csv", index=False)
    paired_agreement(paired).to_csv(output / "matched_agreement_before_after.csv", index=False)
    changes.to_csv(output / "stage_changes_unchanged_audit.csv", index=False)
    held.to_csv(output / "heldout_matched_comparisons.csv", index=False)
    held_summary.to_csv(output / "heldout_agreement_before_after.csv", index=False)
    cv_offsets.to_csv(output / "heldout_training_offsets.csv", index=False)
    example = make_figures(rows, held, held_summary, args.reference, output / "figures")
    pixel_report = pixel_examples(rows, offsets, output / "figures", args.example_sample)
    summary = dict(input=str(args.input.resolve()), reference_scanner=args.reference,
                   corrected_features=list(FEATURES), n_qc_rows=len(rows),
                   n_qc_rejected=qc_rejected, n_matched_comparisons=len(paired),
                   n_stage_change_checks=len(changes), maximum_stage_change_error=float(changes.difference.abs().max()),
                   holdout_folds=min(args.folds, rows.physical_sample_id.nunique()),
                   n_heldout_matched_comparisons=len(held), example_sample=example,
                   pixel_example=pixel_report,
                   interpretation="Relative feature-level scanner offset, plus optional pixel-map visualizations. Source pixel maps are not modified; this is not calibrated colour.",
                   warning="The CSV's corrected chroma is an independently offset-adjusted endpoint; chroma computed from corrected pixel a* and b* can change differently. Original map files and scanner-specific cluster fractions remain unchanged.")
    (output / "README_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to: {output.resolve()}")
    print(f"Reference: {args.reference} | QC rows: {len(rows)} | QC rejected: {qc_rejected}")
    print(f"Matched sample-stage-feature pairs: {len(paired)}")
    print(f"Stage changes checked: {len(changes)} | maximum error: {summary['maximum_stage_change_error']:.3g}")
    print(f"Held-out matched feature comparisons: {len(held)} across {summary['holdout_folds']} physical-sample folds")
    print(f"Pixel-map example: {pixel_report['status']}")
    if pixel_report.get("max_out_of_gamut_fraction", 0):
        print(f"Largest out-of-gamut fraction in example: {pixel_report['max_out_of_gamut_fraction']:.1%}")
    print("This fixed-offset correction cannot change paired within-scanner stage effects.")


if __name__ == "__main__":
    main()
