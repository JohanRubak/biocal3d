"""BioCal3D pre-calibration validation of pixel-colour endpoints.

This script is deliberately run *before* scanner colour calibration.  It does
not normalize colours and it does not replace the mixed-model analysis in
``analyze_pixel_color_statistics.py``.  Instead, it checks whether the chosen
ROI-level endpoints and their biological changes are stable enough to carry
forward into the later calibration analysis.

The physical sample is the resampling/statistical unit.  Pixels and scanners
are never treated as independent biological replicates.

Main checks
-----------
1. Reproducible QC audit with fixed map-coverage and completeness rules.
2. Baseline/Partial/Clean paired changes with sample-clustered bootstrap CIs.
3. Leave-one-scanner-out sensitivity of paired biological changes.
4. Fresh (12 h) versus mature (24 h) Baseline-to-Clean change comparison.
5. Stability of local spatial features at 0.5 and 1.0 mm.
6. Exploratory associations between paired colour and geometry changes.
7. Optional comparison of tables produced at other raster resolutions.

Typical use
-----------
    python validate_precalibration_endpoints.py

With explicit files::

    python validate_precalibration_endpoints.py \
        --pixel-table pixel_color_features.csv \
        --pixel-qc pixel_color_qc.csv \
        --radial-table radial_color_features.csv \
        --master-table master_feature_table.csv

Optional resolution sensitivity, after rerunning Step 1-3 into separate
folders::

    python validate_precalibration_endpoints.py \
        --sensitivity-table grid128=PATH/128/pixel_color_features.csv \
        --sensitivity-table grid512=PATH/512/pixel_color_features.csv

Dependencies
------------
numpy, pandas, scipy and matplotlib.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)

STAGE_ORDER = ["Baseline", "Partial", "Clean"]
KNOWN_SCANNER_ORDER = ["LABscanner", "iTERO", "TRIOS3", "TRIOS5"]

ENDPOINT_ROLES = {
    "lab_a_mean": "PRIMARY",
    "lab_chroma_mean": "PRIMARY",
    "color_cluster_1_fraction": "SECONDARY",
    "lab_L_mean": "SECONDARY",
    "lab_a_std": "EXPLORATORY",
    "lab_chroma_std": "EXPLORATORY",
    "lab_a_local_sd_0.5mm_mean": "EXPLORATORY",
    "lab_chroma_local_sd_0.5mm_mean": "EXPLORATORY",
    "lab_ab_joint_entropy": "EXPLORATORY",
}

ENDPOINT_LABELS = {
    "lab_a_mean": "Mean a* (redness)",
    "lab_chroma_mean": "Mean C*ab (chroma)",
    "color_cluster_1_fraction": "Reddest-cluster fraction",
    "lab_L_mean": "Mean L* (lightness)",
    "lab_a_std": "Within-ROI a* SD",
    "lab_chroma_std": "Within-ROI chroma SD",
    "lab_a_local_sd_0.5mm_mean": "Local a* SD (0.5 mm)",
    "lab_chroma_local_sd_0.5mm_mean": "Local chroma SD (0.5 mm)",
    "lab_ab_joint_entropy": "a*-b* joint entropy",
}

TRANSITIONS = [
    ("Baseline", "Clean", "Clean - Baseline"),
    ("Baseline", "Partial", "Partial - Baseline"),
    ("Partial", "Clean", "Clean - Partial"),
]

SPATIAL_SCALE_PAIRS = [
    (
        "lab_a_local_sd_0.5mm_mean",
        "lab_a_local_sd_1mm_mean",
        "Local a* SD mean",
    ),
    (
        "lab_a_local_sd_0.5mm_p90",
        "lab_a_local_sd_1mm_p90",
        "Local a* SD p90",
    ),
    (
        "lab_chroma_local_sd_0.5mm_mean",
        "lab_chroma_local_sd_1mm_mean",
        "Local chroma SD mean",
    ),
    (
        "lab_chroma_local_sd_0.5mm_p90",
        "lab_chroma_local_sd_1mm_p90",
        "Local chroma SD p90",
    ),
]

GEOMETRY_CHANGE_FEATURES = [
    "geometry__plane_rmse_mm",
    "geometry__native_Sa_mm",
    "geometry__native_Sq_mm",
    "geometry__native_height_p95_minus_p05_mm",
    "geometry__surface_to_projected_area_ratio",
    "geometry__area_weighted_normal_angle_median_deg",
    "geometry__grid_Sa_mm",
    "geometry__grid_Sq_mm",
    "geometry__grid_height_p95_minus_p05_mm",
    "geometry__grid_rms_slope",
]

GEOMETRY_LABELS = {
    "geometry__plane_rmse_mm": "Plane RMSE",
    "geometry__native_Sa_mm": "Native Sa",
    "geometry__native_Sq_mm": "Native Sq",
    "geometry__native_height_p95_minus_p05_mm": "Native height P95-P05",
    "geometry__surface_to_projected_area_ratio": "Surface/projected area",
    "geometry__area_weighted_normal_angle_median_deg": "Normal-angle median",
    "geometry__grid_Sa_mm": "Grid Sa",
    "geometry__grid_Sq_mm": "Grid Sq",
    "geometry__grid_height_p95_minus_p05_mm": "Grid height P95-P05",
    "geometry__grid_rms_slope": "Grid RMS slope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pixel-table", type=Path, default=None)
    parser.add_argument("--pixel-qc", type=Path, default=None)
    parser.add_argument("--radial-table", type=Path, default=None)
    parser.add_argument(
        "--master-table",
        type=Path,
        default=None,
        help="Optional combined colour/geometry/environment master table.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Endpoint columns to validate instead of the predefined family.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.75,
        help="Fixed minimum standardized-map coverage (default: 0.75).",
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=2000,
        help="Physical-sample bootstrap iterations (default: 2000).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sensitivity-table",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Optional alternative pixel table; may be repeated.",
    )
    return parser.parse_args()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.casefold().isin(
        ["true", "1", "yes", "ok"]
    )


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def endpoint_label(feature: str) -> str:
    return ENDPOINT_LABELS.get(
        feature, feature.replace("lab_", "").replace("_", " ")
    )


def holm_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.size, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    p = values[finite]
    order = np.argsort(p)
    ordered = p[order]
    adjusted_ordered = np.maximum.accumulate(
        ordered * (len(ordered) - np.arange(len(ordered)))
    )
    adjusted_ordered = np.clip(adjusted_ordered, 0, 1)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    result[np.flatnonzero(finite)] = adjusted
    return result


def scanner_order(data: pd.DataFrame) -> list[str]:
    present = [str(x) for x in data["scanner"].dropna().unique()]
    ordered = [x for x in KNOWN_SCANNER_ORDER if x in present]
    return ordered + sorted(set(present) - set(ordered))


def discover_optional(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def clustered_bootstrap_ci(
    data: pd.DataFrame,
    value: str,
    cluster: str,
    statistic: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    work = data[[cluster, value]].dropna()
    clusters = work[cluster].astype(str).unique()
    if len(clusters) < 3:
        return np.nan, np.nan
    grouped = {key: group[value].to_numpy(float) for key, group in work.groupby(cluster)}
    estimates = np.empty(iterations, dtype=float)
    reducer = np.nanmedian if statistic == "median" else np.nanmean
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([grouped[key] for key in sampled])
        estimates[index] = reducer(values)
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def simple_bootstrap_ci(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    statistic: str = "median",
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan
    reducer = np.nanmedian if statistic == "median" else np.nanmean
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = reducer(rng.choice(values, size=len(values), replace=True))
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def safe_wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) < 3:
        return np.nan, np.nan
    try:
        result = wilcoxon(nonzero, alternative="two-sided", method="auto")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def paired_differences(
    data: pd.DataFrame, feature: str, before: str, after: str
) -> pd.DataFrame:
    columns = ["physical_sample_id", "scanner", "stage", feature]
    work = data[columns].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work[work["stage"].isin([before, after])]
    pivot = work.pivot_table(
        index=["physical_sample_id", "scanner"],
        columns="stage",
        values=feature,
        aggfunc="median",
        observed=True,
    )
    if before not in pivot or after not in pivot:
        return pd.DataFrame(
            columns=["physical_sample_id", "scanner", "before", "after", "change"]
        )
    pivot = pivot.dropna(subset=[before, after]).reset_index()
    pivot["before"] = pivot[before]
    pivot["after"] = pivot[after]
    pivot["change"] = pivot[after] - pivot[before]
    return pivot[["physical_sample_id", "scanner", "before", "after", "change"]]


# =============================================================================
# INPUT PREPARATION AND QC
# =============================================================================

def build_qc_audit(
    features: pd.DataFrame,
    qc: pd.DataFrame,
    endpoints: list[str],
    coverage_threshold: float,
) -> pd.DataFrame:
    keys = [
        "scanner", "group", "timepoint_number", "sample_number", "sample_id",
        "physical_sample_id", "stage",
    ]
    available_keys = [key for key in keys if key in features and key in qc]
    qc_columns = [
        column for column in [
            "analysis_ok", "qc_reasons", "analysis_error", "colour_source",
            "input_vertices", "coloured_vertices", "saved_roi_radius_mm",
            "map_coverage", "source_ply", "source_texture",
        ] if column in qc
    ]
    merged = features.merge(
        qc[available_keys + qc_columns],
        on=available_keys,
        how="left",
        suffixes=("", "_qc"),
        validate="one_to_one",
    )

    original_ok = (
        bool_series(merged["analysis_ok_qc"])
        if "analysis_ok_qc" in merged
        else bool_series(merged.get("analysis_ok", pd.Series(True, index=merged.index)))
    )
    coverage_column = "map_coverage_qc" if "map_coverage_qc" in merged else "map_coverage"
    coverage = pd.to_numeric(merged.get(coverage_column), errors="coerce")

    endpoint_missing = merged[endpoints].apply(
        lambda column: pd.to_numeric(column, errors="coerce").isna()
    ).any(axis=1)
    duplicate = merged.duplicated(
        [column for column in ["scanner", "sample_id"] if column in merged],
        keep=False,
    )

    reasons = []
    for index in merged.index:
        row_reasons = []
        if not bool(original_ok.loc[index]):
            row_reasons.append("upstream_analysis_failed")
        if not np.isfinite(coverage.loc[index]):
            row_reasons.append("missing_map_coverage")
        elif coverage.loc[index] < coverage_threshold:
            row_reasons.append("low_map_coverage")
        if endpoint_missing.loc[index]:
            row_reasons.append("missing_selected_endpoint")
        if duplicate.loc[index]:
            row_reasons.append("duplicate_scanner_sample_row")
        reasons.append(";".join(row_reasons))

    audit_columns = [
        column for column in [
            *keys, "colour_source", "input_vertices", "coloured_vertices",
            "saved_roi_radius_mm", coverage_column, "qc_reasons", "analysis_error",
        ] if column in merged
    ]
    audit = merged[audit_columns].copy()
    if coverage_column != "map_coverage":
        audit = audit.rename(columns={coverage_column: "map_coverage"})
    audit["validation_ok"] = [not bool(value) for value in reasons]
    audit["validation_reasons"] = reasons
    return audit


def robust_outlier_flags(data: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    rows = []
    for feature in endpoints:
        values = pd.to_numeric(data[feature], errors="coerce")
        for (scanner, stage), indices in data.groupby(
            ["scanner", "stage"], observed=True
        ).groups.items():
            group_values = values.loc[indices]
            median = float(group_values.median())
            mad = float((group_values - median).abs().median())
            if not np.isfinite(mad) or mad <= 1e-12:
                continue
            robust_z = 0.67448975 * (group_values - median) / mad
            for index in group_values.index[np.abs(robust_z) > 6]:
                rows.append({
                    "scanner": scanner,
                    "stage": stage,
                    "sample_id": data.loc[index, "sample_id"],
                    "physical_sample_id": data.loc[index, "physical_sample_id"],
                    "feature": feature,
                    "feature_label": endpoint_label(feature),
                    "value": finite_float(values.loc[index]),
                    "robust_z_within_scanner_stage": finite_float(robust_z.loc[index]),
                    "action": "review_only_not_automatically_excluded",
                })
    return pd.DataFrame(rows)


# =============================================================================
# PAIRED BIOLOGICAL EFFECTS
# =============================================================================

def paired_effect_tables(
    data: pd.DataFrame,
    endpoints: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    scanner_rows = []
    sample_change_rows = []

    for feature in endpoints:
        for before, after, transition in TRANSITIONS:
            differences = paired_differences(data, feature, before, after)
            if differences.empty:
                continue
            differences["feature"] = feature
            differences["transition"] = transition
            sample_change_rows.append(differences)

            sample_level = differences.groupby("physical_sample_id", as_index=False).agg(
                change=("change", "median")
            )
            statistic, p_value = safe_wilcoxon(sample_level["change"].to_numpy())
            median_ci = simple_bootstrap_ci(
                sample_level["change"].to_numpy(), iterations, rng, "median"
            )
            mean_ci = simple_bootstrap_ci(
                sample_level["change"].to_numpy(), iterations, rng, "mean"
            )
            sample_sd = float(sample_level["change"].std(ddof=1))
            standardized = (
                float(sample_level["change"].mean()) / sample_sd
                if np.isfinite(sample_sd) and sample_sd > 1e-12 else np.nan
            )
            scanner_medians = differences.groupby("scanner")["change"].median()
            nonzero_scanner_medians = scanner_medians[np.abs(scanner_medians) > 1e-12]
            if len(nonzero_scanner_medians):
                positive_fraction = float((nonzero_scanner_medians > 0).mean())
                direction_consistency = max(positive_fraction, 1 - positive_fraction)
            else:
                direction_consistency = np.nan

            pooled_rows.append({
                "feature": feature,
                "feature_label": endpoint_label(feature),
                "role": ENDPOINT_ROLES.get(feature, "USER_SELECTED"),
                "transition": transition,
                "n_scanner_level_pairs": len(differences),
                "n_physical_samples": sample_level["physical_sample_id"].nunique(),
                "n_scanners": differences["scanner"].nunique(),
                "mean_change": float(sample_level["change"].mean()),
                "mean_ci95_low": mean_ci[0],
                "mean_ci95_high": mean_ci[1],
                "median_change": float(sample_level["change"].median()),
                "median_ci95_low": median_ci[0],
                "median_ci95_high": median_ci[1],
                "paired_standardized_mean_change": standardized,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
                "scanner_direction_consistency": direction_consistency,
            })

            for scanner, subset in differences.groupby("scanner"):
                values = subset["change"].to_numpy(float)
                statistic, p_value = safe_wilcoxon(values)
                median_ci = simple_bootstrap_ci(values, iterations, rng, "median")
                scanner_rows.append({
                    "feature": feature,
                    "feature_label": endpoint_label(feature),
                    "role": ENDPOINT_ROLES.get(feature, "USER_SELECTED"),
                    "transition": transition,
                    "scanner": scanner,
                    "n_pairs": len(values),
                    "mean_change": float(np.mean(values)),
                    "median_change": float(np.median(values)),
                    "median_ci95_low": median_ci[0],
                    "median_ci95_high": median_ci[1],
                    "fraction_positive": float(np.mean(values > 0)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                })

    pooled = pd.DataFrame(pooled_rows)
    by_scanner = pd.DataFrame(scanner_rows)
    sample_changes = (
        pd.concat(sample_change_rows, ignore_index=True)
        if sample_change_rows else pd.DataFrame()
    )
    if not pooled.empty:
        pooled["wilcoxon_p_holm"] = holm_adjust(pooled["wilcoxon_p"])
    if not by_scanner.empty:
        by_scanner["wilcoxon_p_holm"] = holm_adjust(by_scanner["wilcoxon_p"])
    return pooled, by_scanner, sample_changes


def leave_one_scanner_out(
    data: pd.DataFrame,
    endpoints: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    scanners = scanner_order(data)
    scenarios: list[tuple[str, list[str]]] = [("ALL_SCANNERS", scanners)]
    scenarios.extend((scanner, [x for x in scanners if x != scanner]) for scanner in scanners)

    for feature in endpoints:
        for omitted, included in scenarios:
            subset = data[data["scanner"].isin(included)]
            differences = paired_differences(subset, feature, "Baseline", "Clean")
            if differences.empty:
                continue
            sample_level = differences.groupby("physical_sample_id", as_index=False).agg(
                change=("change", "median")
            )
            statistic, p_value = safe_wilcoxon(sample_level["change"].to_numpy())
            ci = simple_bootstrap_ci(
                sample_level["change"].to_numpy(), iterations, rng, "median"
            )
            rows.append({
                "feature": feature,
                "feature_label": endpoint_label(feature),
                "role": ENDPOINT_ROLES.get(feature, "USER_SELECTED"),
                "omitted_scanner": omitted,
                "included_scanners": ";".join(included),
                "n_physical_samples": len(sample_level),
                "median_clean_minus_baseline": float(sample_level["change"].median()),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "fraction_positive": float((sample_level["change"] > 0).mean()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"])
        baseline = result[result["omitted_scanner"] == "ALL_SCANNERS"].set_index("feature")
        result["relative_to_all_scanners"] = result.apply(
            lambda row: (
                row["median_clean_minus_baseline"]
                / baseline.loc[row["feature"], "median_clean_minus_baseline"]
                if row["feature"] in baseline.index
                and abs(baseline.loc[row["feature"], "median_clean_minus_baseline"]) > 1e-12
                else np.nan
            ),
            axis=1,
        )
    return result


def maturity_comparison(
    data: pd.DataFrame,
    endpoints: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sample_group = (
        data[["physical_sample_id", "group"]]
        .drop_duplicates("physical_sample_id")
        .assign(
            maturity=lambda frame: np.where(
                pd.to_numeric(frame["group"], errors="coerce") <= 2,
                "Fresh (12 h)",
                "Mature (24 h)",
            )
        )
    )
    rows = []
    for feature in endpoints:
        differences = paired_differences(data, feature, "Baseline", "Clean")
        sample_level = differences.groupby("physical_sample_id", as_index=False).agg(
            change=("change", "median")
        ).merge(sample_group[["physical_sample_id", "maturity"]], on="physical_sample_id")
        fresh = sample_level.loc[sample_level["maturity"] == "Fresh (12 h)", "change"].to_numpy()
        mature = sample_level.loc[sample_level["maturity"] == "Mature (24 h)", "change"].to_numpy()
        if len(fresh) >= 3 and len(mature) >= 3:
            test = mannwhitneyu(fresh, mature, alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic, p_value = np.nan, np.nan
        fresh_ci = simple_bootstrap_ci(fresh, iterations, rng, "median")
        mature_ci = simple_bootstrap_ci(mature, iterations, rng, "median")
        rows.append({
            "feature": feature,
            "feature_label": endpoint_label(feature),
            "role": ENDPOINT_ROLES.get(feature, "USER_SELECTED"),
            "n_fresh_samples": len(fresh),
            "fresh_median_change": float(np.median(fresh)) if len(fresh) else np.nan,
            "fresh_ci95_low": fresh_ci[0],
            "fresh_ci95_high": fresh_ci[1],
            "n_mature_samples": len(mature),
            "mature_median_change": float(np.median(mature)) if len(mature) else np.nan,
            "mature_ci95_low": mature_ci[0],
            "mature_ci95_high": mature_ci[1],
            "median_difference_mature_minus_fresh": (
                float(np.median(mature) - np.median(fresh))
                if len(fresh) and len(mature) else np.nan
            ),
            "mann_whitney_statistic": statistic,
            "mann_whitney_p": p_value,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["mann_whitney_p_holm"] = holm_adjust(result["mann_whitney_p"])
    return result


# =============================================================================
# SPATIAL, RADIAL AND GEOMETRY SENSITIVITY
# =============================================================================

def spatial_scale_sensitivity(
    data: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for small, large, label in SPATIAL_SCALE_PAIRS:
        if small not in data or large not in data:
            continue
        paired = data[[small, large]].apply(pd.to_numeric, errors="coerce").dropna()
        correlation = spearmanr(paired[small], paired[large]) if len(paired) >= 3 else None
        row = {
            "feature_family": label,
            "small_scale_feature": small,
            "large_scale_feature": large,
            "n_rows": len(paired),
            "spearman_rho_raw_values": float(correlation.statistic) if correlation else np.nan,
            "spearman_p_raw_values": float(correlation.pvalue) if correlation else np.nan,
        }
        for feature, prefix in [(small, "scale_0_5mm"), (large, "scale_1mm")]:
            differences = paired_differences(data, feature, "Baseline", "Clean")
            sample_level = differences.groupby("physical_sample_id", as_index=False).agg(
                change=("change", "median")
            )
            ci = simple_bootstrap_ci(
                sample_level["change"].to_numpy(), iterations, rng, "median"
            )
            row[f"{prefix}_n_samples"] = len(sample_level)
            row[f"{prefix}_median_change"] = (
                float(sample_level["change"].median()) if len(sample_level) else np.nan
            )
            row[f"{prefix}_ci95_low"] = ci[0]
            row[f"{prefix}_ci95_high"] = ci[1]
            row[f"{prefix}_fraction_positive"] = (
                float((sample_level["change"] > 0).mean()) if len(sample_level) else np.nan
            )
        small_change = paired_differences(data, small, "Baseline", "Clean").rename(
            columns={"change": "small_change"}
        )
        large_change = paired_differences(data, large, "Baseline", "Clean").rename(
            columns={"change": "large_change"}
        )
        change_pair = small_change.merge(
            large_change,
            on=["physical_sample_id", "scanner"],
            suffixes=("_small", "_large"),
        )
        if len(change_pair) >= 3:
            change_correlation = spearmanr(
                change_pair["small_change"], change_pair["large_change"]
            )
            row["spearman_rho_paired_changes"] = float(change_correlation.statistic)
            row["spearman_p_paired_changes"] = float(change_correlation.pvalue)
            row["paired_change_sign_agreement"] = float(
                np.mean(
                    np.sign(change_pair["small_change"])
                    == np.sign(change_pair["large_change"])
                )
            )
        else:
            row["spearman_rho_paired_changes"] = np.nan
            row["spearman_p_paired_changes"] = np.nan
            row["paired_change_sign_agreement"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def radial_stage_changes(
    radial: pd.DataFrame | None,
    endpoints: list[str],
) -> pd.DataFrame:
    if radial is None or radial.empty:
        return pd.DataFrame()
    rows = []
    radial_features = [
        feature for feature in endpoints
        if feature in radial and feature in {"lab_a_mean", "lab_chroma_mean", "lab_L_mean", "color_cluster_1_fraction"}
    ]
    for feature in radial_features:
        for ring, subset in radial.groupby("ring"):
            differences = paired_differences(subset, feature, "Baseline", "Clean")
            sample_level = differences.groupby("physical_sample_id", as_index=False).agg(
                change=("change", "median")
            )
            statistic, p_value = safe_wilcoxon(sample_level["change"].to_numpy())
            rows.append({
                "feature": feature,
                "feature_label": endpoint_label(feature),
                "ring": ring,
                "n_physical_samples": len(sample_level),
                "mean_clean_minus_baseline": (
                    float(sample_level["change"].mean()) if len(sample_level) else np.nan
                ),
                "median_clean_minus_baseline": (
                    float(sample_level["change"].median()) if len(sample_level) else np.nan
                ),
                "fraction_positive": (
                    float((sample_level["change"] > 0).mean()) if len(sample_level) else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"])
    return result


def merge_master_variables(data: pd.DataFrame, master: pd.DataFrame | None) -> pd.DataFrame:
    if master is None or master.empty:
        return data.copy()
    preferred_keys = ["scanner", "group", "timepoint_number", "sample_number"]
    keys = [key for key in preferred_keys if key in data and key in master]
    if len(keys) < 3:
        fallback = ["scanner", "sample_id"]
        keys = [key for key in fallback if key in data and key in master]
    if len(keys) < 2:
        warnings.warn("Master table could not be joined: insufficient common keys")
        return data.copy()
    extra = [
        column for column in master
        if column.startswith("geometry__")
        or column in {"light_avg", "humidity_avg", "temperature_avg", "session_id"}
    ]
    extra = [column for column in extra if column not in data]
    if not extra:
        return data.copy()
    lookup = master[keys + extra].drop_duplicates(keys)
    return data.merge(lookup, on=keys, how="left", validate="many_to_one")


def colour_geometry_associations(
    data: pd.DataFrame,
    endpoints: list[str],
) -> pd.DataFrame:
    geometry = [feature for feature in GEOMETRY_CHANGE_FEATURES if feature in data]
    if not geometry:
        return pd.DataFrame()
    rows = []
    for colour in endpoints:
        colour_change = paired_differences(data, colour, "Baseline", "Clean").rename(
            columns={"change": "colour_change"}
        )
        for geometric in geometry:
            geometry_change = paired_differences(
                data, geometric, "Baseline", "Clean"
            ).rename(columns={"change": "geometry_change"})
            paired = colour_change.merge(
                geometry_change,
                on=["physical_sample_id", "scanner"],
                suffixes=("_colour", "_geometry"),
            )
            # Aggregate scanner repeats so the physical sample remains the unit.
            sample_level = paired.groupby("physical_sample_id", as_index=False).agg(
                colour_change=("colour_change", "median"),
                geometry_change=("geometry_change", "median"),
            ).dropna()
            if len(sample_level) >= 6:
                test = spearmanr(
                    sample_level["colour_change"], sample_level["geometry_change"]
                )
                rho, p_value = float(test.statistic), float(test.pvalue)
            else:
                rho, p_value = np.nan, np.nan
            rows.append({
                "colour_feature": colour,
                "colour_label": endpoint_label(colour),
                "role": ENDPOINT_ROLES.get(colour, "USER_SELECTED"),
                "geometry_feature": geometric,
                "geometry_label": GEOMETRY_LABELS.get(geometric, geometric),
                "n_physical_samples": len(sample_level),
                "spearman_rho": rho,
                "spearman_p": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["spearman_p_holm"] = holm_adjust(result["spearman_p"])
    return result


def parse_sensitivity_specs(specifications: list[str]) -> list[tuple[str, Path]]:
    result = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Sensitivity table must be LABEL=CSV, received: {specification}"
            )
        label, path = specification.split("=", 1)
        result.append((label.strip(), Path(path.strip())))
    return result


def configuration_sensitivity(
    base: pd.DataFrame,
    alternatives: list[tuple[str, Path]],
    endpoints: list[str],
) -> pd.DataFrame:
    if not alternatives:
        return pd.DataFrame(columns=[
            "configuration", "path", "feature", "feature_label",
            "n_matched_rows", "spearman_rho_raw_values",
            "base_median_clean_minus_baseline",
            "alternative_median_clean_minus_baseline",
            "same_change_direction", "alternative_over_base_effect",
        ])
    base_key = ["scanner", "physical_sample_id", "stage"]
    rows = []
    for label, path in alternatives:
        require_file(path, f"Sensitivity table '{label}'")
        alternative = pd.read_csv(path)
        missing_keys = [key for key in base_key if key not in alternative]
        if missing_keys:
            raise ValueError(f"{path} is missing keys: {missing_keys}")
        for feature in endpoints:
            if feature not in alternative:
                continue
            merged = base[base_key + [feature]].merge(
                alternative[base_key + [feature]],
                on=base_key,
                suffixes=("_base", "_alternative"),
            ).dropna()
            if len(merged) >= 3:
                test = spearmanr(
                    merged[f"{feature}_base"], merged[f"{feature}_alternative"]
                )
                rho = float(test.statistic)
            else:
                rho = np.nan
            base_change = paired_differences(base, feature, "Baseline", "Clean")
            alternative_change = paired_differences(
                alternative, feature, "Baseline", "Clean"
            )
            base_median = float(base_change["change"].median()) if len(base_change) else np.nan
            alternative_median = (
                float(alternative_change["change"].median())
                if len(alternative_change) else np.nan
            )
            rows.append({
                "configuration": label,
                "path": str(path),
                "feature": feature,
                "feature_label": endpoint_label(feature),
                "n_matched_rows": len(merged),
                "spearman_rho_raw_values": rho,
                "base_median_clean_minus_baseline": base_median,
                "alternative_median_clean_minus_baseline": alternative_median,
                "same_change_direction": (
                    bool(np.sign(base_median) == np.sign(alternative_median))
                    if np.isfinite(base_median) and np.isfinite(alternative_median)
                    else np.nan
                ),
                "alternative_over_base_effect": (
                    alternative_median / base_median
                    if np.isfinite(base_median) and abs(base_median) > 1e-12
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


# =============================================================================
# FIGURES AND REPORT
# =============================================================================

def save_qc_figure(audit: pd.DataFrame, output: Path, threshold: float) -> None:
    scanners = scanner_order(audit)
    values = [
        pd.to_numeric(audit.loc[audit["scanner"] == scanner, "map_coverage"], errors="coerce").dropna()
        for scanner in scanners
    ]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.boxplot(values, tick_labels=scanners, showfliers=True)
    axis.axhline(threshold, color="firebrick", linestyle="--", label=f"QC threshold = {threshold:.2f}")
    axis.set_ylabel("Standardized-map coverage")
    axis.set_ylim(0, 1.02)
    axis.set_title("Pixel-map QC by scanner")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_effect_forest(pooled: pd.DataFrame, output: Path) -> None:
    if pooled.empty:
        return
    endpoints = list(dict.fromkeys(pooled["feature"]))
    transitions = [x for x in [item[2] for item in TRANSITIONS] if x in set(pooled["transition"])]
    figure, axes = plt.subplots(
        1, len(transitions), figsize=(5.2 * len(transitions), 0.55 * len(endpoints) + 2.2),
        sharey=True, squeeze=False,
    )
    role_colours = {"PRIMARY": "#2E8B57", "SECONDARY": "#3B75AF", "EXPLORATORY": "#777777", "USER_SELECTED": "#7B4FA3"}
    for column, transition in enumerate(transitions):
        axis = axes[0, column]
        subset = pooled[pooled["transition"] == transition].set_index("feature").reindex(endpoints)
        y = np.arange(len(endpoints))
        for index, feature in enumerate(endpoints):
            row = subset.loc[feature]
            value = row["median_change"]
            low = row["median_ci95_low"]
            high = row["median_ci95_high"]
            axis.errorbar(
                value, index,
                xerr=[[value - low], [high - value]],
                fmt="o", capsize=3,
                color=role_colours.get(row["role"], "#333333"),
            )
        axis.axvline(0, color="black", linewidth=0.9, linestyle="--")
        axis.set_title(transition)
        axis.set_xlabel("Median paired change (95% bootstrap CI)")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(y)
        axis.set_yticklabels([endpoint_label(feature) for feature in endpoints])
        axis.invert_yaxis()
    figure.suptitle("BioCal3D paired endpoint effects\nPhysical sample is the resampling unit", y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_leave_one_out_figure(result: pd.DataFrame, output: Path) -> None:
    if result.empty:
        return
    feature_order = list(dict.fromkeys(result["feature"]))
    omitted_order = ["ALL_SCANNERS"] + [x for x in KNOWN_SCANNER_ORDER if x in set(result["omitted_scanner"])]
    omitted_order += sorted(set(result["omitted_scanner"]) - set(omitted_order))
    pivot = result.pivot(index="feature", columns="omitted_scanner", values="relative_to_all_scanners")
    pivot = pivot.reindex(index=feature_order, columns=omitted_order)
    matrix = pivot.to_numpy(float)
    figure, axis = plt.subplots(figsize=(1.35 * len(omitted_order) + 4.5, 0.52 * len(feature_order) + 2.2))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=0.5, vmax=1.5, aspect="auto")
    axis.set_xticks(np.arange(len(omitted_order)))
    axis.set_xticklabels(["All scanners" if x == "ALL_SCANNERS" else f"Without {x}" for x in omitted_order], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(feature_order)))
    axis.set_yticklabels([endpoint_label(x) for x in feature_order])
    axis.set_title("Leave-one-scanner-out effect stability\n1.0 = same median Clean - Baseline effect as all scanners")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Effect relative to all-scanner estimate")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_geometry_heatmap(result: pd.DataFrame, output: Path) -> None:
    if result.empty:
        return
    pivot = result.pivot(index="colour_feature", columns="geometry_feature", values="spearman_rho")
    colour_order = list(dict.fromkeys(result["colour_feature"]))
    geometry_order = [x for x in GEOMETRY_CHANGE_FEATURES if x in pivot.columns]
    pivot = pivot.reindex(index=colour_order, columns=geometry_order)
    matrix = pivot.to_numpy(float)
    figure, axis = plt.subplots(figsize=(1.05 * len(geometry_order) + 4.5, 0.55 * len(colour_order) + 2.3))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(geometry_order)))
    axis.set_xticklabels([GEOMETRY_LABELS.get(x, x) for x in geometry_order], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(colour_order)))
    axis.set_yticklabels([endpoint_label(x) for x in colour_order])
    axis.set_title("Exploratory correlation of paired colour and geometry changes")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=7.5)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Spearman rho (physical-sample median across scanners)")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    data: pd.DataFrame,
    audit: pd.DataFrame,
    pooled: pd.DataFrame,
    loo: pd.DataFrame,
    spatial: pd.DataFrame,
    associations: pd.DataFrame,
    endpoints: list[str],
) -> None:
    failed = audit.loc[~audit["validation_ok"]]
    primary = pooled[(pooled["role"] == "PRIMARY") & (pooled["transition"] == "Clean - Baseline")]
    lines = [
        "# BioCal3D pre-calibration validation",
        "",
        "This report validates endpoint stability without changing or normalizing scanner colour.",
        "Pixels and scanners are not treated as independent biological replicates.",
        "",
        "## Dataset",
        "",
        f"- Accepted scan-stage rows: **{len(data)}**",
        f"- Physical samples: **{data['physical_sample_id'].nunique()}**",
        f"- Scanners: **{data['scanner'].nunique()}**",
        f"- QC exclusions: **{len(failed)}**",
        "",
        "## Endpoint hierarchy",
        "",
        "| Endpoint | Role |",
        "|---|---|",
    ]
    lines.extend(
        f"| {endpoint_label(feature)} | {ENDPOINT_ROLES.get(feature, 'USER_SELECTED').title()} |"
        for feature in endpoints
    )
    lines.extend(["", "## Primary Clean - Baseline results", "", "| Endpoint | Median change | 95% CI | Holm p |", "|---|---:|---:|---:|"])
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['feature_label']} | {row['median_change']:.4g} | "
            f"[{row['median_ci95_low']:.4g}, {row['median_ci95_high']:.4g}] | "
            f"{row['wilcoxon_p_holm']:.3g} |"
        )
    if not loo.empty:
        unstable = loo[(loo["omitted_scanner"] != "ALL_SCANNERS") & ((loo["relative_to_all_scanners"] < 0.75) | (loo["relative_to_all_scanners"] > 1.25))]
        unstable_primary = unstable[unstable["role"] == "PRIMARY"]
        lines.extend([
            "",
            "## Leave-one-scanner-out check",
            "",
            f"- Primary endpoint/scanner omissions changing the median effect by more than 25%: **{len(unstable_primary)}**.",
            f"- All endpoint/scanner omissions changing the median effect by more than 25%: **{len(unstable)}**.",
            "- Inspect `tables/leave_one_scanner_out.csv` before interpreting scanner independence.",
        ])
    if not spatial.empty:
        lines.extend([
            "",
            "## Spatial-scale check",
            "",
            "Local spatial features are considered stable only when 0.5 and 1.0 mm versions retain similar sample ranking and paired-change direction.",
        ])
    if not associations.empty:
        significant = int((associations["spearman_p_holm"] < 0.05).sum())
        lines.extend([
            "",
            "## Colour-geometry analysis",
            "",
            f"- Holm-significant exploratory change-change associations: **{significant}**.",
            "- These associations are exploratory and do not establish localized plaque-volume loss.",
        ])
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Keep uncalibrated results as the primary transparent analysis until the physical target is available.",
        "- Do not automatically remove robust-z outliers; they are flagged for QC review only.",
        "- A significant stage effect with unstable leave-one-scanner-out magnitude remains scanner-dependent.",
        "- Scanner-specific colour-cluster fractions remain relative, not absolute calibrated plaque fractions.",
        "- Rerun this exact validation after adding scanner 5 and again after physical colour calibration.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    if not (0 < args.coverage_threshold <= 1):
        raise ValueError("--coverage-threshold must be in (0, 1]")
    if args.bootstrap_iterations < 200:
        raise ValueError("Use at least 200 bootstrap iterations")

    data_root = args.data_root
    pixel_table = args.pixel_table or (
        data_root / "_pixel_color_analysis" / "tables" / "pixel_color_features.csv"
    )
    pixel_qc = args.pixel_qc or (
        data_root / "_pixel_color_analysis" / "tables" / "pixel_color_qc.csv"
    )
    radial_table = args.radial_table or discover_optional([
        data_root / "_pixel_color_analysis" / "tables" / "radial_color_features.csv"
    ])
    master_table = args.master_table or discover_optional([
        data_root / "_feature_robustness_analysis" / "tables" / "master_feature_table.csv",
        data_root / "_raw_color_analysis" / "step_D_feature_robustness" / "tables" / "master_feature_table.csv",
    ])
    output_root = args.output or (
        data_root / "_pixel_color_analysis" / "precalibration_validation"
    )
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(require_file(pixel_table, "Pixel-colour feature table"))
    qc = pd.read_csv(require_file(pixel_qc, "Pixel-colour QC table"))
    radial = pd.read_csv(radial_table) if radial_table is not None else None
    master = pd.read_csv(master_table) if master_table is not None else None

    required = ["scanner", "stage", "physical_sample_id", "sample_id"]
    missing = [column for column in required if column not in features]
    if missing:
        raise ValueError(f"Pixel table is missing required columns: {missing}")

    endpoints = args.features or list(ENDPOINT_ROLES)
    missing_endpoints = [feature for feature in endpoints if feature not in features]
    if missing_endpoints:
        raise ValueError(f"Selected endpoints are missing: {missing_endpoints}")

    audit = build_qc_audit(
        features, qc, endpoints, args.coverage_threshold
    )
    accepted_keys = set(
        zip(
            audit.loc[audit["validation_ok"], "scanner"].astype(str),
            audit.loc[audit["validation_ok"], "sample_id"].astype(str),
        )
    )
    keep = [
        (str(scanner), str(sample_id)) in accepted_keys
        for scanner, sample_id in zip(features["scanner"], features["sample_id"])
    ]
    analysis = features.loc[keep].copy()
    analysis["stage"] = pd.Categorical(
        analysis["stage"].astype(str), categories=STAGE_ORDER, ordered=True
    )
    analysis = merge_master_variables(analysis, master)

    hierarchy = pd.DataFrame([
        {
            "feature": feature,
            "feature_label": endpoint_label(feature),
            "role": ENDPOINT_ROLES.get(feature, "USER_SELECTED"),
            "recommended_interpretation": (
                "Primary biological endpoint"
                if ENDPOINT_ROLES.get(feature) == "PRIMARY"
                else "Secondary supportive endpoint"
                if ENDPOINT_ROLES.get(feature) == "SECONDARY"
                else "Exploratory; require technical sensitivity checks"
            ),
        }
        for feature in endpoints
    ])

    rng = np.random.default_rng(args.random_state)
    pooled, by_scanner, sample_changes = paired_effect_tables(
        analysis, endpoints, args.bootstrap_iterations, rng
    )
    loo = leave_one_scanner_out(
        analysis, endpoints, args.bootstrap_iterations, rng
    )
    maturity = maturity_comparison(
        analysis, endpoints, args.bootstrap_iterations, rng
    )
    spatial = spatial_scale_sensitivity(
        analysis, args.bootstrap_iterations, rng
    )
    radial_changes = radial_stage_changes(radial, endpoints)
    outliers = robust_outlier_flags(analysis, endpoints)
    associations = colour_geometry_associations(analysis, endpoints)
    configurations = configuration_sensitivity(
        analysis,
        parse_sensitivity_specs(args.sensitivity_table),
        endpoints,
    )

    tables = {
        "endpoint_hierarchy.csv": hierarchy,
        "qc_audit.csv": audit,
        "analysis_dataset.csv": analysis,
        "paired_effects.csv": pooled,
        "paired_effects_by_scanner.csv": by_scanner,
        "paired_change_rows.csv": sample_changes,
        "leave_one_scanner_out.csv": loo,
        "fresh_vs_mature_cleaning_change.csv": maturity,
        "spatial_scale_sensitivity.csv": spatial,
        "radial_cleaning_changes.csv": radial_changes,
        "review_only_robust_outliers.csv": outliers,
        "colour_geometry_change_associations.csv": associations,
        "configuration_sensitivity.csv": configurations,
    }
    for filename, table in tables.items():
        table.to_csv(table_root / filename, index=False)

    save_qc_figure(audit, figure_root / "pixel_map_qc_by_scanner.png", args.coverage_threshold)
    save_effect_forest(pooled, figure_root / "paired_endpoint_effects.png")
    save_leave_one_out_figure(loo, figure_root / "leave_one_scanner_out.png")
    save_geometry_heatmap(
        associations, figure_root / "colour_geometry_change_correlations.png"
    )
    write_report(
        output_root / "PRECALIBRATION_VALIDATION_REPORT.md",
        analysis,
        audit,
        pooled,
        loo,
        spatial,
        associations,
        endpoints,
    )

    print("=" * 78)
    print("BIOCAL3D PRE-CALIBRATION VALIDATION COMPLETE")
    print("=" * 78)
    print(f"Pixel input:       {pixel_table}")
    print(f"QC input:          {pixel_qc}")
    print(f"Accepted rows:     {len(analysis)} / {len(features)}")
    print(f"Physical samples:  {analysis['physical_sample_id'].nunique()}")
    print(f"Endpoints:         {len(endpoints)}")
    print(f"Output:            {output_root}")


if __name__ == "__main__":
    main()
