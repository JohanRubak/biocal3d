"""
BioCal3D six-part exploratory endpoint investigation.

This script extends the pixel-colour endpoint analysis with three questions:

1. Do scanners rank sample-level cleaning changes similarly?
2. Do mature samples follow the expected Baseline -> Partial -> Clean trajectory?
3. How much variation is associated with sample, scanner, scanner-by-stage, and
   residual variation?
4. How do fresh and mature baseline colour and final residual signal differ?
5. Which endpoints provide independent information?
6. Can simple models predict stage for a held-out sample or scanner?

The input is the ``analysis_dataset.csv`` produced by the previous BioCal3D
pixel-colour validation script. Statistical resampling and paired summaries use
the physical sample as the biological unit.

Example
-------
python BioCal3D_endpoint_investigations.py \
    --input "C:\\path\\to\\analysis_dataset.csv" \
    --output "C:\\path\\to\\scanner_agreement_analysis"

Dependencies
------------
numpy, pandas, scipy, matplotlib, scikit-learn. statsmodels is needed for
adjusted regressions and mixed models; those outputs are marked unavailable if
statsmodels is not installed.
"""

from __future__ import annotations

import argparse
import itertools
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_DATA_ROOT = Path.cwd()

# Set this to a concrete Path if you prefer running the file without arguments.
# When it is None, the script searches DEFAULT_DATA_ROOT recursively and uses the
# most recently modified analysis_dataset.csv beneath the current directory.
DEFAULT_INPUT_FILE: Path | None = None

DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "_scanner_agreement_analysis"


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

RANDOM_SEED = 20260922
N_BOOTSTRAP = 2_000
FIGURE_DPI = 220
PREDICTORS = ("lab_a_mean", "lab_chroma_mean", "lab_L_mean")
ENDPOINT_FAMILY = (
    "lab_a_mean", "lab_chroma_mean", "color_cluster_1_fraction", "lab_L_mean",
    "lab_a_std", "lab_chroma_std", "lab_a_local_sd_0.5mm_mean",
    "lab_chroma_local_sd_0.5mm_mean", "lab_ab_joint_entropy",
)


@dataclass(frozen=True)
class Endpoint:
    column: str
    label: str
    expected_direction: str  # "decrease" or "increase"
    role: str


ENDPOINTS = (
    Endpoint("lab_a_mean", "Mean a* (redness)", "decrease", "PRIMARY"),
    Endpoint("lab_chroma_mean", "Mean C*ab (chroma)", "decrease", "PRIMARY"),
    Endpoint(
        "color_cluster_1_fraction",
        "Reddest-cluster fraction",
        "decrease",
        "SECONDARY",
    ),
    Endpoint("lab_L_mean", "Mean L* (lightness)", "increase", "SECONDARY"),
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Six exploratory BioCal3D endpoint investigations from analysis_dataset.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to analysis_dataset.csv. If omitted, search the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap replicates for ICC confidence intervals (default {N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default {RANDOM_SEED}).",
    )
    return parser.parse_args()


def resolve_input_file(path: Path | None) -> Path:
    if path is not None:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    candidates = list(DEFAULT_DATA_ROOT.rglob("analysis_dataset.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No analysis_dataset.csv was found. Pass it explicitly with "
            "--input C:\\path\\to\\analysis_dataset.csv"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        print("Multiple analysis_dataset.csv files found. Using the newest:")
        for candidate in candidates[:5]:
            marker = "  ->" if candidate == candidates[0] else "    "
            print(f"{marker} {candidate}")
    return candidates[0].resolve()


def make_output_directories(root: Path) -> tuple[Path, Path]:
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return name or "endpoint"


def canonical_stage(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"baseline", "t0", "0", "0.0"} or "base" in text:
        return "Baseline"
    if text in {"partial", "t1", "1", "1.0"} or "partial" in text:
        return "Partial"
    if text in {"clean", "t2", "2", "2.0"} or "clean" in text:
        return "Clean"
    return None


def robust_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def validate_and_prepare(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scanner",
        "group",
        "stage",
        "physical_sample_id",
        *(endpoint.column for endpoint in ENDPOINTS),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    out = data.copy()
    if "analysis_ok" in out.columns:
        out = out.loc[robust_bool(out["analysis_ok"])].copy()
    else:
        warnings.warn("No analysis_ok column found: QC exclusions cannot be enforced.", stacklevel=2)

    out["scanner"] = out["scanner"].astype(str).str.strip()
    out["physical_sample_id"] = out["physical_sample_id"].astype(str).str.strip()
    out["stage"] = out["stage"].map(canonical_stage)
    out = out.loc[out["stage"].notna()].copy()

    out["group"] = pd.to_numeric(out["group"], errors="coerce")
    out = out.loc[out["group"].notna()].copy()
    if out.empty:
        raise ValueError("No usable data after stage, group, and QC filtering.")
    out["growth_class"] = np.where(out["group"] <= 2, "Fresh", "Mature")

    for endpoint in ENDPOINTS:
        out[endpoint.column] = pd.to_numeric(out[endpoint.column], errors="coerce")
    for column in ENDPOINT_FAMILY:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    duplicate_keys = ["physical_sample_id", "scanner", "stage"]
    duplicates = out.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        warnings.warn(
            f"Found {count} rows with duplicated sample/scanner/stage keys. "
            "Numeric endpoints will be aggregated by their median.",
            stacklevel=2,
        )
        # Resolve duplicated identifiers uniformly for every downstream analysis.
        keys = ["physical_sample_id", "scanner", "stage"]
        numerical = out.select_dtypes(include=[np.number]).columns.difference(keys).tolist()
        out = out.groupby(keys, as_index=False).agg({
            **{column: "median" for column in numerical},
            "growth_class": "first",
        })

    return out


def save_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def holm_adjust(values: pd.Series) -> np.ndarray:
    """Holm correction with original input order preserved."""
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


# =============================================================================
# CLEAN - BASELINE CHANGE SCORES
# =============================================================================


def calculate_change_scores(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    index_cols = ["physical_sample_id", "scanner"]

    for endpoint in ENDPOINTS:
        pivot = data.pivot_table(
            index=index_cols,
            columns="stage",
            values=endpoint.column,
            aggfunc="median",
        )
        if not {"Baseline", "Clean"}.issubset(pivot.columns):
            continue

        complete = pivot.dropna(subset=["Baseline", "Clean"]).copy()
        complete["change"] = complete["Clean"] - complete["Baseline"]
        complete = complete.reset_index()

        for row in complete.itertuples(index=False):
            records.append(
                {
                    "physical_sample_id": row.physical_sample_id,
                    "scanner": row.scanner,
                    "feature": endpoint.column,
                    "feature_label": endpoint.label,
                    "role": endpoint.role,
                    "expected_direction": endpoint.expected_direction,
                    "baseline": row.Baseline,
                    "clean": row.Clean,
                    "clean_minus_baseline": row.change,
                }
            )

    return pd.DataFrame.from_records(records)


# =============================================================================
# CROSS-SCANNER AGREEMENT
# =============================================================================


def icc_two_way(values: np.ndarray) -> tuple[float, float]:
    """Return ICC(A,1) absolute agreement and ICC(C,1) consistency.

    Rows are physical samples and columns are scanners. The implementation
    follows the standard two-way ANOVA mean-square formulation.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return np.nan, np.nan
    n, k = values.shape
    if n < 3 or k < 2 or not np.isfinite(values).all():
        return np.nan, np.nan

    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)

    ss_rows = k * np.square(row_means - grand).sum()
    ss_cols = n * np.square(col_means - grand).sum()
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ss_error = np.square(residual).sum()

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator_absolute = (
        ms_rows
        + (k - 1) * ms_error
        + k * (ms_cols - ms_error) / n
    )
    denominator_consistency = ms_rows + (k - 1) * ms_error

    icc_absolute = (
        (ms_rows - ms_error) / denominator_absolute
        if denominator_absolute != 0
        else np.nan
    )
    icc_consistency = (
        (ms_rows - ms_error) / denominator_consistency
        if denominator_consistency != 0
        else np.nan
    )
    return float(icc_absolute), float(icc_consistency)


def bootstrap_icc(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    absolute, consistency = icc_two_way(values)
    n = values.shape[0]
    boot_absolute: list[float] = []
    boot_consistency: list[float] = []

    for _ in range(max(0, n_bootstrap)):
        indices = rng.integers(0, n, size=n)
        abs_b, con_b = icc_two_way(values[indices, :])
        if np.isfinite(abs_b):
            boot_absolute.append(abs_b)
        if np.isfinite(con_b):
            boot_consistency.append(con_b)

    def interval(samples: Iterable[float]) -> tuple[float, float]:
        array = np.asarray(list(samples), dtype=float)
        if array.size < 20:
            return np.nan, np.nan
        low, high = np.percentile(array, [2.5, 97.5])
        return float(low), float(high)

    abs_low, abs_high = interval(boot_absolute)
    con_low, con_high = interval(boot_consistency)
    return {
        "icc_absolute_agreement": absolute,
        "icc_absolute_ci95_low": abs_low,
        "icc_absolute_ci95_high": abs_high,
        "icc_consistency": consistency,
        "icc_consistency_ci95_low": con_low,
        "icc_consistency_ci95_high": con_high,
        "n_valid_bootstrap_absolute": len(boot_absolute),
        "n_valid_bootstrap_consistency": len(boot_consistency),
    }


def reliability_label(value: float) -> str:
    if not np.isfinite(value):
        return "Unavailable"
    if value < 0.50:
        return "Poor"
    if value < 0.75:
        return "Moderate"
    if value < 0.90:
        return "Good"
    return "Excellent"


def scanner_agreement(
    changes: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise_records: list[dict[str, object]] = []
    icc_records: list[dict[str, object]] = []
    bland_altman_records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for endpoint in ENDPOINTS:
        subset = changes.loc[changes["feature"] == endpoint.column]
        pivot = subset.pivot_table(
            index="physical_sample_id",
            columns="scanner",
            values="clean_minus_baseline",
            aggfunc="median",
        ).sort_index(axis=1)

        scanners = list(pivot.columns)
        for scanner_a, scanner_b in itertools.combinations(scanners, 2):
            paired = pivot[[scanner_a, scanner_b]].dropna()
            n_pairs = len(paired)
            if n_pairs >= 3:
                rho, p_value = spearmanr(
                    paired[scanner_a].to_numpy(),
                    paired[scanner_b].to_numpy(),
                )
            else:
                rho, p_value = np.nan, np.nan

            difference = paired[scanner_a] - paired[scanner_b]
            pair_mean = (paired[scanner_a] + paired[scanner_b]) / 2
            bias = float(difference.mean()) if n_pairs else np.nan
            sd_difference = float(difference.std(ddof=1)) if n_pairs > 1 else np.nan
            loa_low = bias - 1.96 * sd_difference if np.isfinite(sd_difference) else np.nan
            loa_high = bias + 1.96 * sd_difference if np.isfinite(sd_difference) else np.nan

            pairwise_records.append(
                {
                    "feature": endpoint.column,
                    "feature_label": endpoint.label,
                    "role": endpoint.role,
                    "scanner_a": scanner_a,
                    "scanner_b": scanner_b,
                    "n_physical_samples": n_pairs,
                    "spearman_rho": rho,
                    "spearman_p": p_value,
                    "mean_change_scanner_a": paired[scanner_a].mean(),
                    "mean_change_scanner_b": paired[scanner_b].mean(),
                    "mean_difference_a_minus_b": bias,
                    "sd_difference": sd_difference,
                    "loa95_low": loa_low,
                    "loa95_high": loa_high,
                }
            )

            for sample_id, mean_value, difference_value in zip(
                paired.index,
                pair_mean,
                difference,
                strict=True,
            ):
                bland_altman_records.append(
                    {
                        "feature": endpoint.column,
                        "feature_label": endpoint.label,
                        "physical_sample_id": sample_id,
                        "scanner_a": scanner_a,
                        "scanner_b": scanner_b,
                        "pair_mean": mean_value,
                        "difference_a_minus_b": difference_value,
                        "bias": bias,
                        "loa95_low": loa_low,
                        "loa95_high": loa_high,
                    }
                )

        complete = pivot.dropna(axis=0, how="any")
        icc_result = bootstrap_icc(complete.to_numpy(), n_bootstrap, rng)
        icc_records.append(
            {
                "feature": endpoint.column,
                "feature_label": endpoint.label,
                "role": endpoint.role,
                "n_physical_samples_complete": len(complete),
                "n_scanners": complete.shape[1],
                "scanners": ";".join(map(str, complete.columns)),
                **icc_result,
                "absolute_agreement_interpretation": reliability_label(
                    icc_result["icc_absolute_agreement"]
                ),
                "consistency_interpretation": reliability_label(
                    icc_result["icc_consistency"]
                ),
            }
        )

    pairwise = pd.DataFrame(pairwise_records)
    finite = pairwise["spearman_p"].notna()
    pairwise["spearman_p_holm"] = np.nan
    if finite.any():
        pairwise.loc[finite, "spearman_p_holm"] = holm_adjust(pairwise.loc[finite, "spearman_p"])

    return (
        pairwise,
        pd.DataFrame(icc_records),
        pd.DataFrame(bland_altman_records),
    )


# =============================================================================
# MONOTONIC BIOLOGICAL TRAJECTORIES
# =============================================================================


def trajectory_flags(
    baseline: pd.Series,
    partial: pd.Series,
    clean: pd.Series,
    direction: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if direction == "decrease":
        baseline_to_partial = partial < baseline
        partial_to_clean = clean < partial
        baseline_to_clean = clean < baseline
    elif direction == "increase":
        baseline_to_partial = partial > baseline
        partial_to_clean = clean > partial
        baseline_to_clean = clean > baseline
    else:
        raise ValueError(f"Unknown expected direction: {direction}")

    fully_monotonic = baseline_to_partial & partial_to_clean
    return baseline_to_partial, partial_to_clean, baseline_to_clean, fully_monotonic


def monotonic_trajectory_analysis(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = data.loc[data["growth_class"] == "Mature"].copy()
    detail_records: list[dict[str, object]] = []

    for endpoint in ENDPOINTS:
        pivot = mature.pivot_table(
            index=["physical_sample_id", "scanner"],
            columns="stage",
            values=endpoint.column,
            aggfunc="median",
        )
        if not {"Baseline", "Partial", "Clean"}.issubset(pivot.columns):
            continue
        complete = pivot.dropna(subset=["Baseline", "Partial", "Clean"]).copy()
        btp, ptc, btc, full = trajectory_flags(
            complete["Baseline"],
            complete["Partial"],
            complete["Clean"],
            endpoint.expected_direction,
        )

        for index, row in complete.iterrows():
            sample_id, scanner = index
            detail_records.append(
                {
                    "physical_sample_id": sample_id,
                    "scanner": scanner,
                    "feature": endpoint.column,
                    "feature_label": endpoint.label,
                    "role": endpoint.role,
                    "expected_direction": endpoint.expected_direction,
                    "baseline": row["Baseline"],
                    "partial": row["Partial"],
                    "clean": row["Clean"],
                    "expected_baseline_to_partial": bool(btp.loc[index]),
                    "expected_partial_to_clean": bool(ptc.loc[index]),
                    "expected_baseline_to_clean": bool(btc.loc[index]),
                    "fully_monotonic": bool(full.loc[index]),
                }
            )

        # Add a physical-sample summary based on the median across available scanners.
        sample_stage = mature.pivot_table(
            index="physical_sample_id",
            columns="stage",
            values=endpoint.column,
            aggfunc="median",
        )
        sample_complete = sample_stage.dropna(
            subset=["Baseline", "Partial", "Clean"]
        ).copy()
        btp, ptc, btc, full = trajectory_flags(
            sample_complete["Baseline"],
            sample_complete["Partial"],
            sample_complete["Clean"],
            endpoint.expected_direction,
        )
        for sample_id, row in sample_complete.iterrows():
            detail_records.append(
                {
                    "physical_sample_id": sample_id,
                    "scanner": "ALL_SCANNERS_MEDIAN",
                    "feature": endpoint.column,
                    "feature_label": endpoint.label,
                    "role": endpoint.role,
                    "expected_direction": endpoint.expected_direction,
                    "baseline": row["Baseline"],
                    "partial": row["Partial"],
                    "clean": row["Clean"],
                    "expected_baseline_to_partial": bool(btp.loc[sample_id]),
                    "expected_partial_to_clean": bool(ptc.loc[sample_id]),
                    "expected_baseline_to_clean": bool(btc.loc[sample_id]),
                    "fully_monotonic": bool(full.loc[sample_id]),
                }
            )

    details = pd.DataFrame(detail_records)
    if details.empty:
        return details, pd.DataFrame()

    summary = (
        details.groupby(
            ["feature", "feature_label", "role", "scanner"],
            as_index=False,
        )
        .agg(
            n_complete_trajectories=("fully_monotonic", "size"),
            fraction_expected_baseline_to_partial=(
                "expected_baseline_to_partial",
                "mean",
            ),
            fraction_expected_partial_to_clean=(
                "expected_partial_to_clean",
                "mean",
            ),
            fraction_expected_baseline_to_clean=(
                "expected_baseline_to_clean",
                "mean",
            ),
            fraction_fully_monotonic=("fully_monotonic", "mean"),
        )
    )
    return details, summary


# =============================================================================
# MIXED MODELS AND VARIANCE DECOMPOSITION
# =============================================================================


def fit_with_fallback(model, reml: bool):
    attempts = (
        ("lbfgs", 1_000),
        ("powell", 2_000),
    )
    errors: list[str] = []
    for method, maxiter in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )
            return result, method, ""
        except Exception as exc:  # pragma: no cover - depends on local data geometry
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    return None, "", " | ".join(errors)


def mixed_models_and_variance(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if smf is None:
        warnings.warn(
            "statsmodels is unavailable: variance decomposition and adjusted "
            "fresh/mature regressions will be skipped. Install statsmodels to run them.",
            stacklevel=2,
        )
        status = pd.DataFrame(
            [
                {"feature": e.column, "feature_label": e.label, "model": model,
                 "fit_succeeded": False, "converged": False, "fit_method": "",
                 "error": "statsmodels not installed"}
                for e in ENDPOINTS
                for model in ("fixed_stage_scanner_interaction", "crossed_variance_components")
            ]
        )
        return pd.DataFrame(), pd.DataFrame(), status
    fixed_records: list[dict[str, object]] = []
    variance_records: list[dict[str, object]] = []
    model_status_records: list[dict[str, object]] = []

    for endpoint in ENDPOINTS:
        columns = [
            endpoint.column,
            "stage",
            "scanner",
            "growth_class",
            "physical_sample_id",
        ]
        frame = data[columns].dropna().copy()
        frame["stage"] = pd.Categorical(
            frame["stage"],
            categories=["Baseline", "Partial", "Clean"],
            ordered=True,
        )
        frame["growth_class"] = pd.Categorical(
            frame["growth_class"],
            categories=["Fresh", "Mature"],
        )

        # Model 1: fixed scanner effects and stage-by-scanner interaction,
        # with a random intercept for the physical sample.
        fixed_formula = (
            f"{endpoint.column} ~ C(stage) * C(scanner) + C(growth_class)"
        )
        try:
            fixed_model = smf.mixedlm(
                fixed_formula,
                data=frame,
                groups=frame["physical_sample_id"],
                re_formula="1",
            )
            fixed_result, fixed_method, fixed_error = fit_with_fallback(
                fixed_model, reml=False
            )
        except Exception as exc:
            fixed_result = None
            fixed_method = ""
            fixed_error = f"{type(exc).__name__}: {exc}"

        if fixed_result is not None:
            ci = fixed_result.conf_int()
            for term in fixed_result.fe_params.index:
                fixed_records.append(
                    {
                        "feature": endpoint.column,
                        "feature_label": endpoint.label,
                        "role": endpoint.role,
                        "term": term,
                        "estimate": fixed_result.fe_params[term],
                        "standard_error": fixed_result.bse_fe[term],
                        "z_value": fixed_result.tvalues[term],
                        "p_value": fixed_result.pvalues[term],
                        "ci95_low": ci.loc[term, 0],
                        "ci95_high": ci.loc[term, 1],
                        "n_rows": len(frame),
                        "n_physical_samples": frame["physical_sample_id"].nunique(),
                        "fit_method": fixed_method,
                        "converged": bool(fixed_result.converged),
                    }
                )

        model_status_records.append(
            {
                "feature": endpoint.column,
                "feature_label": endpoint.label,
                "model": "fixed_stage_scanner_interaction",
                "fit_succeeded": fixed_result is not None,
                "converged": (
                    bool(fixed_result.converged) if fixed_result is not None else False
                ),
                "fit_method": fixed_method,
                "error": fixed_error,
            }
        )

        # Model 2: crossed variance components. Scanner and scanner-by-stage
        # estimates are exploratory because the current dataset has four scanners.
        vc_frame = frame.copy()
        vc_frame["all_observations"] = "all"
        vc_frame["scanner_stage"] = (
            vc_frame["scanner"].astype(str)
            + "::"
            + vc_frame["stage"].astype(str)
        )
        vc_formula = {
            "physical_sample": "0 + C(physical_sample_id)",
            "scanner": "0 + C(scanner)",
            "scanner_stage": "0 + C(scanner_stage)",
        }
        variance_formula = f"{endpoint.column} ~ C(stage) + C(growth_class)"
        try:
            variance_model = smf.mixedlm(
                variance_formula,
                data=vc_frame,
                groups=vc_frame["all_observations"],
                re_formula="0",
                vc_formula=vc_formula,
            )
            variance_result, variance_method, variance_error = fit_with_fallback(
                variance_model, reml=True
            )
        except Exception as exc:
            variance_result = None
            variance_method = ""
            variance_error = f"{type(exc).__name__}: {exc}"

        if variance_result is not None:
            names = list(variance_result.model.exog_vc.names)
            components = dict(zip(names, variance_result.vcomp, strict=True))
            components["residual"] = float(variance_result.scale)
            total_variance = float(sum(components.values()))
            for component in [
                "physical_sample",
                "scanner",
                "scanner_stage",
                "residual",
            ]:
                value = float(components.get(component, np.nan))
                variance_records.append(
                    {
                        "feature": endpoint.column,
                        "feature_label": endpoint.label,
                        "role": endpoint.role,
                        "component": component,
                        "variance": value,
                        "variance_fraction": (
                            value / total_variance
                            if np.isfinite(value) and total_variance > 0
                            else np.nan
                        ),
                        "total_variance": total_variance,
                        "n_rows": len(vc_frame),
                        "n_physical_samples": vc_frame[
                            "physical_sample_id"
                        ].nunique(),
                        "n_scanners": vc_frame["scanner"].nunique(),
                        "fit_method": variance_method,
                        "converged": bool(variance_result.converged),
                        "interpretation_note": (
                            "Scanner variance is exploratory because only a small "
                            "number of scanner systems is currently available."
                        ),
                    }
                )

        model_status_records.append(
            {
                "feature": endpoint.column,
                "feature_label": endpoint.label,
                "model": "crossed_variance_components",
                "fit_succeeded": variance_result is not None,
                "converged": (
                    bool(variance_result.converged)
                    if variance_result is not None
                    else False
                ),
                "fit_method": variance_method,
                "error": variance_error,
            }
        )

    fixed = pd.DataFrame(fixed_records)
    if not fixed.empty:
        finite = fixed["p_value"].notna()
        fixed["p_value_holm"] = np.nan
        if finite.any():
            fixed.loc[finite, "p_value_holm"] = holm_adjust(fixed.loc[finite, "p_value"])

    return fixed, pd.DataFrame(variance_records), pd.DataFrame(model_status_records)


# =============================================================================
# FRESH VERSUS MATURE, BASELINE SEVERITY, AND RESIDUAL CLEAN SIGNAL
# =============================================================================


def fit_clustered_ols(frame: pd.DataFrame, formula: str, feature: str, analysis: str) -> dict:
    """Cluster-robust uncertainty across repeat scanner measurements of a sample."""
    record = {"feature": feature, "analysis": analysis, "formula": formula,
              "n_rows": len(frame),
              "n_physical_samples": frame["physical_sample_id"].nunique(),
              "coefficient": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
              "p_value": np.nan, "status": "statsmodels not installed"}
    if smf is None or len(frame) < 10 or frame["physical_sample_id"].nunique() < 8:
        if smf is not None:
            record["status"] = "insufficient observations"
        return record
    try:
        result = smf.ols(formula, data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["physical_sample_id"]}
        )
        term = "C(growth_class)[T.Mature]"
        record.update({
            "coefficient": float(result.params[term]),
            "ci95_low": float(result.conf_int().loc[term, 0]),
            "ci95_high": float(result.conf_int().loc[term, 1]),
            "p_value": float(result.pvalues[term]),
            "status": "ok",
        })
    except Exception as exc:
        record["status"] = f"{type(exc).__name__}: {exc}"
    return record


def growth_comparisons(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = []
    regressions = []
    for e in ENDPOINTS:
        wide = data.pivot_table(
            index=["physical_sample_id", "scanner", "growth_class"],
            columns="stage", values=e.column, aggfunc="median"
        ).reset_index()
        for stage in ("Baseline", "Clean"):
            if stage not in wide.columns:
                continue
            # One independent observation per physical sample for nonparametric testing.
            per_sample = wide.groupby(["physical_sample_id", "growth_class"], as_index=False)[stage].median()
            fresh = per_sample.loc[per_sample.growth_class == "Fresh", stage].dropna()
            mature = per_sample.loc[per_sample.growth_class == "Mature", stage].dropna()
            p = float(mannwhitneyu(fresh, mature, alternative="two-sided").pvalue) if min(len(fresh), len(mature)) >= 3 else np.nan
            comparisons.append({
                "feature": e.column, "feature_label": e.label, "stage": stage,
                "n_fresh_samples": len(fresh), "n_mature_samples": len(mature),
                "median_fresh": fresh.median(), "median_mature": mature.median(),
                "median_difference_mature_minus_fresh": mature.median() - fresh.median(),
                "mann_whitney_p": p,
                "note": "Unadjusted physical-sample medians, exploratory; group and session may be confounded",
            })
        if {"Baseline", "Clean"}.issubset(wide.columns):
            baseline = wide.dropna(subset=["Baseline"])
            final = wide.dropna(subset=["Baseline", "Clean"])
            regressions.append(fit_clustered_ols(
                baseline, 'Q("Baseline") ~ C(growth_class) + C(scanner)', e.column,
                "baseline_severity_adjusted_for_scanner"
            ))
            regressions.append(fit_clustered_ols(
                final, 'Q("Clean") ~ Q("Baseline") + C(growth_class) + C(scanner)',
                e.column, "clean_residual_adjusted_for_baseline_and_scanner"
            ))
    comparison = pd.DataFrame(comparisons)
    if not comparison.empty:
        comparison["mann_whitney_p_holm"] = np.nan
        valid = comparison.mann_whitney_p.notna()
        if valid.any():
            comparison.loc[valid, "mann_whitney_p_holm"] = holm_adjust(comparison.loc[valid, "mann_whitney_p"])
    regression = pd.DataFrame(regressions)
    if not regression.empty and regression.p_value.notna().any():
        regression["p_value_holm"] = np.nan
        valid = regression.p_value.notna()
        regression.loc[valid, "p_value_holm"] = holm_adjust(regression.loc[valid, "p_value"])
    return comparison, regression


# =============================================================================
# ENDPOINT REDUNDANCY: ONE CLEANING CHANGE PER PHYSICAL SAMPLE
# =============================================================================


def endpoint_redundancy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = [name for name in ENDPOINT_FAMILY if name in data.columns]
    wide = data.pivot_table(index=["physical_sample_id", "scanner"], columns="stage",
                            values=available, aggfunc="median")
    records = []
    for name in available:
        if (name, "Baseline") not in wide.columns or (name, "Clean") not in wide.columns:
            continue
        series = wide[(name, "Clean")] - wide[(name, "Baseline")]
        for (sample, scanner), value in series.dropna().items():
            records.append({"physical_sample_id": sample, "scanner": scanner,
                            "feature": name, "change": value})
    long = pd.DataFrame(records)
    if long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sample_median = long.pivot_table(index="physical_sample_id", columns="feature",
                                     values="change", aggfunc="median")
    correlations = []
    for a, b in itertools.combinations(sample_median.columns, 2):
        pairs = sample_median[[a, b]].dropna()
        rho, p = spearmanr(pairs[a], pairs[b]) if len(pairs) >= 6 else (np.nan, np.nan)
        correlations.append({"feature_a": a, "feature_b": b, "n_physical_samples": len(pairs),
                             "spearman_rho": rho, "spearman_p": p})
    correlation = pd.DataFrame(correlations)
    if not correlation.empty:
        correlation["spearman_p_holm"] = np.nan
        valid = correlation.spearman_p.notna()
        if valid.any():
            correlation.loc[valid, "spearman_p_holm"] = holm_adjust(correlation.loc[valid, "spearman_p"])
    # Complete physical samples only. Standardization and PCA are exploratory,
    # not used to select supervised predictors or to claim independent validation.
    complete = sample_median.dropna()
    if len(complete) < 6 or complete.shape[1] < 2:
        return correlation, pd.DataFrame(), pd.DataFrame()
    standardized = StandardScaler().fit_transform(complete)
    pca = PCA().fit(standardized)
    variance = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                             "explained_variance_ratio": pca.explained_variance_ratio_,
                             "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
                             "n_physical_samples": len(complete)})
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings.insert(0, "feature", complete.columns.to_list())
    return correlation, variance, loadings


# =============================================================================
# EXPLORATORY STAGE PREDICTION; MATURE SAMPLES ONLY
# =============================================================================


def predict_stage(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = data.loc[data.growth_class == "Mature", [
        "physical_sample_id", "scanner", "stage", *PREDICTORS
    ]].dropna(subset=["stage"]).copy()
    # Resolve accidental duplicate scan identifiers before cross-validation.
    mature = mature.groupby(["physical_sample_id", "scanner", "stage"], as_index=False)[list(PREDICTORS)].median()
    mature = mature.dropna(subset=list(PREDICTORS), how="all")
    stages = ["Baseline", "Partial", "Clean"]
    if mature.physical_sample_id.nunique() < 6 or not set(stages).issubset(mature.stage.unique()):
        return pd.DataFrame(), pd.DataFrame()
    X = mature[list(PREDICTORS)]
    y = mature.stage.to_numpy()
    sample_groups = mature.physical_sample_id.to_numpy()
    scanner_groups = mature.scanner.to_numpy()
    cv_methods = {
        "held_out_physical_samples": GroupKFold(n_splits=min(5, len(np.unique(sample_groups)))).split(X, y, sample_groups),
        "held_out_scanner_same_samples_elsewhere": LeaveOneGroupOut().split(X, y, scanner_groups),
    }
    predictions = []
    folds = []
    for method, splits in cv_methods.items():
        for fold_number, (train, test) in enumerate(splits, start=1):
            if len(np.unique(y[train])) < len(stages):
                folds.append({"validation": method, "fold": fold_number,
                              "n_train": len(train), "n_test": len(test),
                              "status": "missing training class"})
                continue
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=2_000, class_weight="balanced"))
            model.fit(X.iloc[train], y[train])
            predicted = model.predict(X.iloc[test])
            fold_rows = mature.iloc[test]
            folds.append({
                "validation": method, "fold": fold_number, "n_train": len(train),
                "n_test": len(test), "n_test_samples": len(np.unique(sample_groups[test])),
                "held_out_scanner": ";".join(np.unique(scanner_groups[test])) if method.startswith("held_out_scanner") else "",
                "accuracy": accuracy_score(y[test], predicted),
                "balanced_accuracy": np.mean([
                    np.mean(predicted[y[test] == stage] == stage)
                    for stage in stages if np.any(y[test] == stage)
                ]),
                "status": "ok",
            })
            for row, target, estimate in zip(fold_rows.itertuples(index=False), y[test], predicted, strict=True):
                predictions.append({"validation": method, "fold": fold_number,
                                    "physical_sample_id": row.physical_sample_id,
                                    "scanner": row.scanner, "actual_stage": target,
                                    "predicted_stage": estimate})
    pred_df = pd.DataFrame(predictions)
    fold_df = pd.DataFrame(folds)
    if not pred_df.empty:
        for method, subset in pred_df.groupby("validation"):
            print(f"Stage prediction {method}: accuracy={accuracy_score(subset.actual_stage, subset.predicted_stage):.3f}; "
                  f"balanced accuracy={balanced_accuracy_score(subset.actual_stage, subset.predicted_stage):.3f}; "
                  f"n={len(subset)}")
    return pred_df, fold_df


# =============================================================================
# FIGURES
# =============================================================================


def plot_scanner_correlation_heatmaps(
    changes: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = None

    for axis, endpoint in zip(axes.flat, ENDPOINTS, strict=True):
        subset = changes.loc[changes["feature"] == endpoint.column]
        pivot = subset.pivot_table(
            index="physical_sample_id",
            columns="scanner",
            values="clean_minus_baseline",
            aggfunc="median",
        ).sort_index(axis=1)
        correlation = pivot.corr(method="spearman")
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        labels = list(correlation.columns)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(endpoint.label)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = correlation.iloc[row, column]
                colour = "white" if abs(value) > 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=10,
                )

    fig.suptitle(
        "Cross-scanner rank agreement of Clean - Baseline changes",
        fontsize=16,
    )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.78, label="Spearman correlation")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_bland_altman(
    bland_altman: pd.DataFrame,
    figures_dir: Path,
) -> None:
    for endpoint in ENDPOINTS:
        subset = bland_altman.loc[bland_altman["feature"] == endpoint.column]
        pairs = list(subset[["scanner_a", "scanner_b"]].drop_duplicates().itertuples(
            index=False, name=None
        ))
        if not pairs:
            continue

        n_columns = 3
        n_rows = int(np.ceil(len(pairs) / n_columns))
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(15, 4.3 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        for axis, (scanner_a, scanner_b) in zip(axes.flat, pairs):
            pair = subset.loc[
                (subset["scanner_a"] == scanner_a)
                & (subset["scanner_b"] == scanner_b)
            ]
            axis.scatter(
                pair["pair_mean"],
                pair["difference_a_minus_b"],
                alpha=0.75,
                s=36,
                color="#2f6f9f",
                edgecolor="white",
                linewidth=0.4,
            )
            bias = pair["bias"].iloc[0]
            low = pair["loa95_low"].iloc[0]
            high = pair["loa95_high"].iloc[0]
            axis.axhline(bias, color="black", linewidth=1.3, label=f"Bias {bias:.2f}")
            axis.axhline(low, color="#b33a3a", linestyle="--", linewidth=1)
            axis.axhline(high, color="#b33a3a", linestyle="--", linewidth=1)
            axis.axhline(0, color="0.7", linewidth=0.8)
            axis.set_title(f"{scanner_a} minus {scanner_b}")
            axis.set_xlabel("Mean change of scanner pair")
            axis.set_ylabel("Difference in change")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=9)

        for axis in axes.flat[len(pairs) :]:
            axis.set_visible(False)

        fig.suptitle(
            f"Bland-Altman agreement for {endpoint.label}",
            fontsize=16,
        )
        output = figures_dir / f"bland_altman_{safe_name(endpoint.column)}.png"
        fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output}")


def plot_monotonic_consistency(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    if summary.empty:
        return

    scanner_order = sorted(
        scanner
        for scanner in summary["scanner"].unique()
        if scanner != "ALL_SCANNERS_MEDIAN"
    )
    if "ALL_SCANNERS_MEDIAN" in set(summary["scanner"]):
        scanner_order.append("ALL_SCANNERS_MEDIAN")

    label_map = {endpoint.column: endpoint.label for endpoint in ENDPOINTS}
    feature_order = [endpoint.column for endpoint in ENDPOINTS]
    matrix = (
        summary.pivot_table(
            index="feature",
            columns="scanner",
            values="fraction_fully_monotonic",
            aggfunc="first",
        )
        .reindex(index=feature_order, columns=scanner_order)
    )

    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGn")
    axis.set_xticks(
        range(len(scanner_order)),
        [s.replace("ALL_SCANNERS_MEDIAN", "Median across scanners") for s in scanner_order],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(
        range(len(feature_order)),
        [label_map[feature] for feature in feature_order],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            text = "NA" if not np.isfinite(value) else f"{100 * value:.0f}%"
            colour = "white" if np.isfinite(value) and value > 0.62 else "black"
            axis.text(column, row, text, ha="center", va="center", color=colour)

    axis.set_title("Samples following the expected Baseline - Partial - Clean trajectory")
    fig.colorbar(image, ax=axis, label="Fraction fully monotonic")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_variance_components(
    variance: pd.DataFrame,
    path: Path,
) -> None:
    if variance.empty:
        return

    endpoint_order = [endpoint.column for endpoint in ENDPOINTS]
    label_map = {endpoint.column: endpoint.label for endpoint in ENDPOINTS}
    component_order = ["physical_sample", "scanner", "scanner_stage", "residual"]
    component_labels = {
        "physical_sample": "Physical sample",
        "scanner": "Scanner",
        "scanner_stage": "Scanner x stage",
        "residual": "Residual",
    }
    colours = {
        "physical_sample": "#4c78a8",
        "scanner": "#f58518",
        "scanner_stage": "#e45756",
        "residual": "#b8b8b8",
    }

    pivot = (
        variance.pivot_table(
            index="feature",
            columns="component",
            values="variance_fraction",
            aggfunc="first",
        )
        .reindex(index=endpoint_order, columns=component_order)
        .fillna(0)
    )

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(endpoint_order))
    bottom = np.zeros(len(endpoint_order))
    for component in component_order:
        values = pivot[component].to_numpy()
        axis.bar(
            x,
            values,
            bottom=bottom,
            label=component_labels[component],
            color=colours[component],
        )
        bottom += values

    axis.set_xticks(x, [label_map[feature] for feature in endpoint_order], rotation=25, ha="right")
    axis.set_ylabel("Fraction of model variance")
    axis.set_ylim(0, 1)
    axis.set_title("Exploratory variance decomposition of colour endpoints")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_endpoint_redundancy(correlation: pd.DataFrame, path: Path) -> None:
    if correlation.empty:
        return
    names = sorted(set(correlation.feature_a) | set(correlation.feature_b))
    values = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in correlation.itertuples(index=False):
        values.loc[row.feature_a, row.feature_b] = row.spearman_rho
        values.loc[row.feature_b, row.feature_a] = row.spearman_rho
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    image = axis.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=50, ha="right", fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, f"{values.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if abs(values.iloc[i, j]) > 0.6 else "black")
    axis.set_title("Endpoint redundancy: sample-median Clean - Baseline changes")
    fig.colorbar(image, ax=axis, label="Spearman correlation across physical samples")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stage_prediction(predictions: pd.DataFrame, path: Path) -> None:
    if predictions.empty:
        return
    labels = ["Baseline", "Partial", "Clean"]
    methods = list(predictions.validation.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5),
                             squeeze=False, constrained_layout=True)
    for ax, method in zip(axes.flat, methods, strict=True):
        subset = predictions.loc[predictions.validation == method]
        counts = confusion_matrix(subset.actual_stage, subset.predicted_stage, labels=labels)
        proportions = np.divide(counts, counts.sum(axis=1, keepdims=True),
                                out=np.zeros_like(counts, dtype=float),
                                where=counts.sum(axis=1, keepdims=True) > 0)
        ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(method.replace("_", " "), fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                        color="white" if proportions[i, j] > 0.55 else "black")
    fig.suptitle("Exploratory mature-sample stage prediction (cell labels = scan counts)")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_root = args.output.expanduser().resolve()
    tables_dir, figures_dir = make_output_directories(output_root)

    print(f"Input:  {input_file}")
    print(f"Output: {output_root}")

    raw = pd.read_csv(input_file)
    data = validate_and_prepare(raw)
    print(
        f"Rows used: {len(data):,}; physical samples: "
        f"{data['physical_sample_id'].nunique()}; scanners: {data['scanner'].nunique()}"
    )

    endpoint_hierarchy = pd.DataFrame(
        [
            {
                "feature": endpoint.column,
                "feature_label": endpoint.label,
                "role": endpoint.role,
                "expected_cleaning_direction": endpoint.expected_direction,
            }
            for endpoint in ENDPOINTS
        ]
    )
    save_csv(endpoint_hierarchy, tables_dir / "agreement_endpoint_hierarchy.csv")

    changes = calculate_change_scores(data)
    save_csv(changes, tables_dir / "clean_baseline_change_scores.csv")

    pairwise, icc, bland_altman = scanner_agreement(
        changes,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    save_csv(pairwise, tables_dir / "scanner_pairwise_agreement.csv")
    save_csv(icc, tables_dir / "scanner_icc.csv")
    save_csv(bland_altman, tables_dir / "bland_altman_values.csv")

    trajectory_details, trajectory_summary = monotonic_trajectory_analysis(data)
    save_csv(trajectory_details, tables_dir / "monotonic_trajectory_details.csv")
    save_csv(trajectory_summary, tables_dir / "monotonic_trajectory_summary.csv")
    if not trajectory_details.empty:
        discordant = trajectory_details.loc[
            ~trajectory_details["fully_monotonic"]
        ].copy()
        save_csv(discordant, tables_dir / "monotonic_trajectory_discordant.csv")

    fixed_effects, variance, model_status = mixed_models_and_variance(data)
    save_csv(fixed_effects, tables_dir / "mixed_model_fixed_effects.csv")
    save_csv(variance, tables_dir / "variance_components.csv")
    save_csv(model_status, tables_dir / "mixed_model_status.csv")

    growth_summary, growth_regression = growth_comparisons(data)
    save_csv(growth_summary, tables_dir / "fresh_mature_baseline_and_clean.csv")
    save_csv(growth_regression, tables_dir / "fresh_mature_adjusted_regression.csv")

    redundancy, pca_variance, pca_loadings = endpoint_redundancy(data)
    save_csv(redundancy, tables_dir / "endpoint_redundancy_correlations.csv")
    save_csv(pca_variance, tables_dir / "endpoint_pca_variance.csv")
    save_csv(pca_loadings, tables_dir / "endpoint_pca_loadings.csv")

    predictions, prediction_folds = predict_stage(data)
    save_csv(predictions, tables_dir / "stage_prediction_rows.csv")
    save_csv(prediction_folds, tables_dir / "stage_prediction_folds.csv")

    plot_scanner_correlation_heatmaps(
        changes,
        figures_dir / "scanner_change_correlation_heatmaps.png",
    )
    plot_bland_altman(bland_altman, figures_dir)
    plot_monotonic_consistency(
        trajectory_summary,
        figures_dir / "monotonic_trajectory_consistency.png",
    )
    plot_variance_components(
        variance,
        figures_dir / "variance_components.png",
    )
    plot_endpoint_redundancy(redundancy, figures_dir / "endpoint_redundancy.png")
    plot_stage_prediction(predictions, figures_dir / "stage_prediction_confusion.png")

    print("\nAnalysis complete.")
    print("Main interpretation:")
    print("  - ICC consistency: preservation of sample ranking across scanners.")
    print("  - ICC absolute agreement: numerical interchangeability of scanners.")
    print("  - Monotonic fraction: individual Baseline-Partial-Clean tracking.")
    print("  - Scanner variance estimates remain exploratory with few scanner systems.")
    print("  - Fresh/mature comparisons may reflect group or session differences.")
    print("  - Scanner holdout tests scanner transfer, not generalization to new samples.")


if __name__ == "__main__":
    main()
