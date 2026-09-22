"""BioCal3D Step 4: repeated-measures statistics for pixel-colour features.

This script consumes the outputs from ``analyze_pixel_color_maps.py`` and asks:

* Does cleaning stage change each ROI-level colour endpoint?
* Is there an overall scanner offset?
* Does the stage response differ by scanner?
* Do session-level light, humidity and temperature explain additional variation?

The physical specimen is the repeated-measures unit. Raster pixels are never
treated as independent observations. The default mixed model is::

    endpoint ~ stage + scanner + group + environment
               + random intercept for physical_sample_id

Scanner dependence is tested by comparing that model with::

    endpoint ~ stage * scanner + group + environment
               + random intercept for physical_sample_id

``group`` can be replaced by the coarser ``maturity`` adjustment from the
command line. Cluster fractions are analysed on an empirical-logit scale while
descriptives and paired changes remain on the original 0-1 fraction scale.

Default input
-------------
DATA_ROOT/_pixel_color_analysis/tables/pixel_color_features.csv

Default output
--------------
DATA_ROOT/_pixel_color_analysis/step_4_statistical_analysis/

Important outputs
-----------------
tables/endpoint_model_summary.csv
    Omnibus mixed-model evidence for stage, scanner, stage x scanner and the
    joint environmental contribution, with Holm-adjusted p-values.
tables/stage_scanner_descriptives.csv
    Raw-scale descriptive statistics by stage and scanner.
tables/paired_stage_changes.csv
    Paired Baseline-Clean, Baseline-Partial and Partial-Clean changes within
    each physical sample and scanner, including Wilcoxon tests.
tables/model_fixed_effects.csv
    Fixed-effect coefficients from additive and interaction models.
tables/environment_effects.csv
    Standardized environmental coefficients and drop-one likelihood-ratio
    tests. These are exploratory because environment varies by session.
figures/
    Primary endpoint stage plots, evidence matrix, direction heatmap and
    environmental coefficient overview.

Examples
--------
Run with the project defaults and environmental adjustment::

    python analyze_pixel_color_statistics.py

Use an explicit pixel table and previously prepared environment table::

    python analyze_pixel_color_statistics.py \
        --pixel-table pixel_color_features.csv \
        --environment-table environment_sessions_used.csv

Deliberately run without environmental covariates::

    python analyze_pixel_color_statistics.py --allow-no-environment

Dependencies
------------
numpy, pandas, scipy, matplotlib, statsmodels and patsy.
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
from scipy.stats import chi2, wilcoxon

try:
    import statsmodels.formula.api as smf
except ImportError:  # Allow --help and validation in lightweight environments.
    smf = None


# =============================================================================
# PATHS, DESIGN AND ENDPOINTS
# =============================================================================

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)

KNOWN_SCANNER_ORDER = ["LABscanner", "iTERO", "TRIOS3", "TRIOS5"]
STAGE_ORDER = ["Baseline", "Partial", "Clean"]
ALPHA = 0.05

GROUP_STAGE_MAP = {
    1: {0: "Baseline", 1: "Clean"},
    2: {0: "Baseline", 1: "Clean"},
    3: {0: "Baseline", 1: "Partial", 2: "Clean"},
    4: {0: "Baseline", 1: "Partial", 2: "Clean"},
    5: {0: "Baseline", 1: "Partial", 2: "Clean"},
}

ENVIRONMENT_LABELS = {
    "light_avg": "Ambient light",
    "humidity_avg": "Relative humidity",
    "temperature_avg": "Temperature",
}

# A deliberately focused family. Use --features to replace it.
DEFAULT_ENDPOINTS = [
    "lab_a_mean",
    "lab_chroma_mean",
    "color_cluster_1_fraction",
    "lab_L_mean",
    "lab_a_std",
    "lab_chroma_std",
    "lab_a_local_sd_0.5mm_mean",
    "lab_chroma_local_sd_0.5mm_mean",
    "lab_ab_joint_entropy",
]

PRIMARY_ENDPOINTS = [
    "lab_a_mean",
    "lab_chroma_mean",
    "color_cluster_1_fraction",
    "lab_L_mean",
]

FEATURE_LABELS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--pixel-table", type=Path, default=None,
        help="Explicit pixel_color_features.csv path.",
    )
    parser.add_argument(
        "--environment-table", type=Path, default=None,
        help=(
            "Optional Step C environment_session_qc.csv, "
            "environment_sessions_used.csv, or the original environmental Excel file."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: DATA_ROOT/_pixel_color_analysis/step_4_statistical_analysis",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Exact endpoint columns to analyse instead of the focused defaults.",
    )
    parser.add_argument(
        "--group-adjustment", choices=("group", "maturity"), default="group",
        help="Adjust for individual group (default) or only Fresh/Mature status.",
    )
    parser.add_argument(
        "--allow-no-environment", action="store_true",
        help="Allow a deliberately unadjusted run when no environmental table exists.",
    )
    parser.add_argument(
        "--include-qc-failures", action="store_true",
        help="Include analysis_ok=False rows (not recommended).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate and summarize inputs without fitting models or writing outputs.",
    )
    return parser.parse_args()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return (
        values.astype(str).str.strip().str.casefold().isin(["true", "1", "yes", "ok"])
    )


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").replace("0.5mm", "0.5 mm")


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def holm_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return output
    p_values = numeric[finite]
    order = np.argsort(p_values)
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        ordered * (len(ordered) - np.arange(len(ordered)))
    )
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    output[np.flatnonzero(finite)] = adjusted
    return output


def negative_log10_p(value: object) -> float:
    number = finite_float(value)
    if not np.isfinite(number):
        return np.nan
    return -math.log10(max(number, np.nextafter(0.0, 1.0)))


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return np.nan
    ranks = pd.Series(np.abs(differences)).rank(method="average").to_numpy(float)
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else np.nan


def stage_from_design(group: object, timepoint: object) -> str:
    group_number, time_number = int(float(group)), int(float(timepoint))
    try:
        return GROUP_STAGE_MAP[group_number][time_number]
    except KeyError as exc:
        raise ValueError(
            f"No biological-stage mapping for group={group_number}, "
            f"timepoint={time_number}."
        ) from exc


def scanner_order_from_data(data: pd.DataFrame) -> list[str]:
    observed = data["scanner"].dropna().astype(str).unique().tolist()
    known = [scanner for scanner in KNOWN_SCANNER_ORDER if scanner in observed]
    extra = sorted(set(observed) - set(known))
    return known + extra


# =============================================================================
# INPUT AND ENVIRONMENT
# =============================================================================

CORE_COLUMNS = [
    "scanner", "group", "timepoint_number", "sample_number", "sample_id",
    "physical_sample_id", "stage",
]


def prepare_pixel_table(
    raw: pd.DataFrame, include_qc_failures: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    require_columns(
        raw,
        ["scanner", "group", "timepoint_number", "sample_number", "sample_id",
         "physical_sample_id", "analysis_ok"],
        "Pixel feature table",
    )
    data = raw.copy()
    data["scanner"] = data["scanner"].astype(str).str.strip()
    data["sample_id"] = data["sample_id"].astype(str).str.strip().str.upper()
    data["physical_sample_id"] = (
        data["physical_sample_id"].astype(str).str.strip().str.upper()
    )
    for column in ["group", "timepoint_number", "sample_number"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(
        subset=["scanner", "group", "timepoint_number", "sample_number", "sample_id"]
    ).copy()
    data[["group", "timepoint_number", "sample_number"]] = data[
        ["group", "timepoint_number", "sample_number"]
    ].astype(int)

    derived_id = [
        f"BCG{group}-{sample}"
        for group, sample in zip(data["group"], data["sample_number"])
    ]
    id_mismatch = data["physical_sample_id"].ne(derived_id)
    if id_mismatch.any():
        examples = data.loc[
            id_mismatch,
            ["scanner", "sample_id", "physical_sample_id", "group", "sample_number"],
        ].head(10)
        raise ValueError(
            "physical_sample_id is inconsistent with group/sample_number:\n"
            + examples.to_string(index=False)
        )
    data["physical_sample_id"] = derived_id

    derived_stage = [
        stage_from_design(group, timepoint)
        for group, timepoint in zip(data["group"], data["timepoint_number"])
    ]
    if "stage" in data:
        supplied = data["stage"].astype(str).str.strip()
        mismatch = supplied.ne(pd.Series(derived_stage, index=data.index))
        mismatch &= data["stage"].notna()
        if mismatch.any():
            examples = data.loc[
                mismatch, ["scanner", "sample_id", "group", "timepoint_number", "stage"]
            ].head(10)
            raise ValueError(
                "Stage labels are inconsistent with the experimental design:\n"
                + examples.to_string(index=False)
            )
    data["stage"] = derived_stage
    data["maturity"] = np.where(data["group"].isin([1, 2]), "Fresh", "Mature")

    duplicates = data.duplicated(["scanner", "sample_id"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate scanner x sample_id rows found:\n"
            + data.loc[duplicates, ["scanner", "sample_id"]].head(12).to_string(index=False)
        )

    accepted = bool_series(data["analysis_ok"])
    excluded = data.loc[~accepted].copy()
    if not include_qc_failures:
        data = data.loc[accepted].copy()

    scanners = scanner_order_from_data(data)
    data["scanner"] = pd.Categorical(data["scanner"], categories=scanners, ordered=True)
    data["stage"] = pd.Categorical(data["stage"], categories=STAGE_ORDER, ordered=True)
    data["maturity"] = pd.Categorical(
        data["maturity"], categories=["Fresh", "Mature"], ordered=True
    )
    return data.reset_index(drop=True), excluded.reset_index(drop=True), scanners


def resolve_environment_path(data_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        data_root / "_raw_color_analysis" / "step_C_environment_effect"
        / "tables" / "environment_session_qc.csv",
        data_root / "_feature_robustness_analysis" / "tables"
        / "environment_sessions_used.csv",
        data_root / "_raw_color_analysis" / "step_C_environment_effect"
        / "tables" / "roi_color_with_environment.csv",
        data_root / "Environment information.xlsx",
    ]
    return next((path for path in candidates if path.is_file()), None)


def read_environment_file(path: Path) -> pd.DataFrame:
    require_file(path, "Environmental table")
    if path.suffix.casefold() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name=0)
        if {"group", "timepoint_number"} <= set(raw.columns):
            environment = raw.copy()
        elif raw.shape[1] >= 13:
            environment = pd.DataFrame({
                "session_id": raw.iloc[:, 0].astype("string").str.strip(),
                "light_avg": pd.to_numeric(raw.iloc[:, 4], errors="coerce").fillna(
                    raw.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                ),
                "humidity_avg": pd.to_numeric(raw.iloc[:, 8], errors="coerce").fillna(
                    raw.iloc[:, 5:8].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                ),
                "temperature_avg": pd.to_numeric(raw.iloc[:, 12], errors="coerce").fillna(
                    raw.iloc[:, 9:12].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                ),
            })
        else:
            raise ValueError(
                "Environmental Excel file needs named group/timepoint columns or "
                "the original 13+ column session/readings layout."
            )
    else:
        environment = pd.read_csv(path)

    if "group" not in environment or "timepoint_number" not in environment:
        if "session_id" not in environment:
            raise ValueError(
                "Environmental data need group/timepoint_number or session_id."
            )
        extracted = environment["session_id"].astype(str).str.extract(
            r"^BCG(?P<group>\d+)T(?P<timepoint_number>\d+)$"
        )
        environment["group"] = pd.to_numeric(extracted["group"], errors="coerce")
        environment["timepoint_number"] = pd.to_numeric(
            extracted["timepoint_number"], errors="coerce"
        )

    environment = environment.dropna(subset=["group", "timepoint_number"]).copy()
    environment["group"] = pd.to_numeric(environment["group"], errors="raise").astype(int)
    environment["timepoint_number"] = pd.to_numeric(
        environment["timepoint_number"], errors="raise"
    ).astype(int)
    variables = [column for column in ENVIRONMENT_LABELS if column in environment]
    if not variables:
        raise ValueError(
            "No environmental variables found. Expected: "
            + ", ".join(ENVIRONMENT_LABELS)
        )

    keys = ["group", "timepoint_number"]
    rows = []
    for key, subset in environment.groupby(keys, observed=True):
        row: dict[str, object] = {"group": key[0], "timepoint_number": key[1]}
        for variable in variables:
            values = pd.to_numeric(subset[variable], errors="coerce").dropna().unique()
            if len(values) > 1 and not np.allclose(values, values[0]):
                raise ValueError(f"Conflicting {variable} values for session {key}.")
            row[variable] = values[0] if len(values) else np.nan
        row["session_id"] = f"BCG{key[0]}T{key[1]}"
        if "all_calibrations_ok" in subset:
            values = subset["all_calibrations_ok"].dropna().unique()
            row["all_calibrations_ok"] = values[0] if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def merge_environment(
    data: pd.DataFrame, environment: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    variables = [column for column in ENVIRONMENT_LABELS if column in environment]
    merged = data.merge(
        environment,
        on=["group", "timepoint_number"],
        how="left",
        validate="many_to_one",
    )
    missing = merged[variables].isna().all(axis=1)
    if missing.any():
        sessions = (
            merged.loc[missing, ["group", "timepoint_number"]]
            .drop_duplicates().sort_values(["group", "timepoint_number"])
        )
        raise ValueError(
            "Environmental measurements are missing for sessions:\n"
            + sessions.to_string(index=False)
        )
    return merged, variables


# =============================================================================
# DESCRIPTIVE AND PAIRED ANALYSES
# =============================================================================

def stage_scanner_descriptives(
    data: pd.DataFrame, endpoints: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in endpoints:
        for (scanner, stage), subset in data.groupby(
            ["scanner", "stage"], observed=True
        ):
            values = pd.to_numeric(subset[feature], errors="coerce").dropna()
            if not len(values):
                continue
            rows.append({
                "feature": feature,
                "feature_label": feature_label(feature),
                "scanner": str(scanner),
                "stage": str(stage),
                "n": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    return pd.DataFrame(rows)


def paired_stage_changes(
    data: pd.DataFrame, endpoints: list[str], scanners: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in endpoints:
        selected = data[
            ["scanner", "physical_sample_id", "stage", feature]
        ].copy()
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
        for scanner in scanners:
            scanner_data = selected[selected["scanner"].astype(str) == scanner]
            wide = scanner_data.pivot_table(
                index="physical_sample_id", columns="stage", values=feature,
                aggfunc="first", observed=True,
            )
            for stage_a, stage_b, transition in TRANSITIONS:
                if stage_a not in wide or stage_b not in wide:
                    continue
                pairs = wide[[stage_a, stage_b]].dropna()
                if pairs.empty:
                    continue
                differences = (pairs[stage_b] - pairs[stage_a]).to_numpy(float)
                if np.any(differences != 0):
                    try:
                        statistic, p_value = wilcoxon(
                            differences, zero_method="wilcox", alternative="two-sided"
                        )
                    except ValueError:
                        statistic, p_value = np.nan, np.nan
                else:
                    statistic, p_value = 0.0, 1.0
                rows.append({
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "scanner": scanner,
                    "transition": transition,
                    "stage_A": stage_a,
                    "stage_B": stage_b,
                    "n_pairs": len(differences),
                    "stage_A_median": float(np.median(pairs[stage_a])),
                    "stage_B_median": float(np.median(pairs[stage_b])),
                    "mean_change": float(np.mean(differences)),
                    "median_change": float(np.median(differences)),
                    "change_q25": float(np.percentile(differences, 25)),
                    "change_q75": float(np.percentile(differences, 75)),
                    "fraction_increased": float(np.mean(differences > 0)),
                    "fraction_decreased": float(np.mean(differences < 0)),
                    "rank_biserial": rank_biserial_from_differences(differences),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                })
    output = pd.DataFrame(rows)
    if not output.empty:
        output["wilcoxon_p_holm"] = holm_adjust(output["wilcoxon_p"])
    return output


def clean_baseline_direction(
    changes: pd.DataFrame, feature: str,
) -> tuple[float, int, float]:
    selected = changes[
        (changes["feature"] == feature)
        & (changes["transition"] == "Clean - Baseline")
    ]
    medians = pd.to_numeric(selected["median_change"], errors="coerce").dropna()
    if medians.empty:
        return np.nan, 0, np.nan
    pooled_direction = float(np.sign(np.median(medians)))
    if pooled_direction == 0:
        return np.nan, len(medians), float(np.median(medians))
    consistency = float(np.mean(np.sign(medians) == pooled_direction))
    return consistency, len(medians), float(np.median(medians))


# =============================================================================
# MIXED MODELS
# =============================================================================

def transform_endpoint(data: pd.DataFrame, feature: str) -> tuple[pd.Series, str]:
    values = pd.to_numeric(data[feature], errors="coerce")
    if feature.endswith("_fraction"):
        if "map_valid_pixels" in data:
            n_pixels = pd.to_numeric(data["map_valid_pixels"], errors="coerce")
            epsilon = 0.5 / n_pixels.clip(lower=1)
        else:
            epsilon = pd.Series(1e-4, index=data.index)
        clipped = np.minimum(np.maximum(values, epsilon), 1.0 - epsilon)
        return np.log(clipped / (1.0 - clipped)), "empirical_logit"
    return values, "identity"


def prepare_model_data(
    master: pd.DataFrame,
    feature: str,
    environment_variables: list[str],
    adjustment: str,
    scanners: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str], str]:
    columns = [
        feature, "scanner", "stage", adjustment, "physical_sample_id",
        "group", "timepoint_number", *environment_variables,
    ]
    if "map_valid_pixels" in master:
        columns.append("map_valid_pixels")
    columns = list(dict.fromkeys(columns))
    data = master[columns].copy()
    data["value"], transformation = transform_endpoint(data, feature)

    model_environment = []
    model_to_original = {}
    for index, original in enumerate(environment_variables):
        values = pd.to_numeric(data[original], errors="coerce")
        standard_deviation = float(values.std(ddof=1))
        if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
            continue
        name = f"environment_z_{index}"
        data[name] = (values - values.mean()) / standard_deviation
        model_environment.append(name)
        model_to_original[name] = original

    required = [
        "value", "scanner", "stage", adjustment, "physical_sample_id",
        *model_environment,
    ]
    data = data.dropna(subset=required).copy()
    data["scanner"] = pd.Categorical(
        data["scanner"].astype(str), categories=scanners, ordered=True
    ).remove_unused_categories()
    data["stage"] = pd.Categorical(
        data["stage"].astype(str), categories=STAGE_ORDER, ordered=True
    ).remove_unused_categories()
    if adjustment == "maturity":
        data["maturity"] = pd.Categorical(
            data["maturity"].astype(str), categories=["Fresh", "Mature"], ordered=True
        ).remove_unused_categories()
    return data, model_environment, model_to_original, transformation


def build_formula(
    include_stage: bool,
    include_scanner: bool,
    include_interaction: bool,
    environment_terms: list[str],
    adjustment: str,
) -> str:
    terms = []
    if include_interaction:
        terms.append("C(stage, Treatment(reference='Baseline')) * C(scanner)")
    else:
        if include_stage:
            terms.append("C(stage, Treatment(reference='Baseline'))")
        if include_scanner:
            terms.append("C(scanner)")
    terms.append(f"C({adjustment})")
    terms.extend(environment_terms)
    return "value ~ " + " + ".join(terms)


def fit_mixed_model(formula: str, data: pd.DataFrame):
    if smf is None:
        raise ImportError(
            "statsmodels is required for Step 4. Install it in the BioCal3D "
            "environment with `pip install statsmodels patsy`."
        )
    errors = []
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula, data=data, groups=data["physical_sample_id"]
                )
                result = model.fit(
                    reml=False, method=method, maxiter=2500, disp=False
                )
            if not np.isfinite(result.llf):
                raise RuntimeError("non-finite log-likelihood")
            return result, method
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def likelihood_ratio_test(full_model, reduced_model) -> tuple[float, int, float]:
    statistic = max(0.0, float(2 * (full_model.llf - reduced_model.llf)))
    degrees = max(1, int(len(full_model.fe_params) - len(reduced_model.fe_params)))
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def fixed_effect_rows(feature: str, model_name: str, result, transformation: str):
    rows = []
    confidence = result.conf_int()
    for term, estimate in result.fe_params.items():
        rows.append({
            "feature": feature,
            "feature_label": feature_label(feature),
            "model": model_name,
            "outcome_transformation": transformation,
            "term": term,
            "estimate": finite_float(estimate),
            "standard_error": finite_float(result.bse_fe.get(term)),
            "wald_p": finite_float(result.pvalues.get(term)),
            "ci95_low": finite_float(confidence.loc[term, 0]) if term in confidence.index else np.nan,
            "ci95_high": finite_float(confidence.loc[term, 1]) if term in confidence.index else np.nan,
        })
    return rows


def analyse_endpoint(
    master: pd.DataFrame,
    feature: str,
    environment_variables: list[str],
    adjustment: str,
    scanners: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    data, environment_terms, model_to_original, transformation = prepare_model_data(
        master, feature, environment_variables, adjustment, scanners
    )
    if len(data) < 20:
        raise RuntimeError(f"Only {len(data)} complete observations")
    if data["physical_sample_id"].nunique() < 5:
        raise RuntimeError("Fewer than five physical samples")
    if data["stage"].nunique() < 2 or data["scanner"].nunique() < 2:
        raise RuntimeError("At least two stages and two scanners are required")
    if float(data["value"].std(ddof=1)) <= 1e-12:
        raise RuntimeError("Endpoint has essentially no variance")

    additive_formula = build_formula(
        True, True, False, environment_terms, adjustment
    )
    additive, additive_optimizer = fit_mixed_model(additive_formula, data)
    no_stage, _ = fit_mixed_model(
        build_formula(False, True, False, environment_terms, adjustment), data
    )
    no_scanner, _ = fit_mixed_model(
        build_formula(True, False, False, environment_terms, adjustment), data
    )
    interaction_formula = build_formula(
        True, True, True, environment_terms, adjustment
    )
    interaction, interaction_optimizer = fit_mixed_model(interaction_formula, data)

    stage_lr, stage_df, stage_p = likelihood_ratio_test(additive, no_stage)
    scanner_lr, scanner_df, scanner_p = likelihood_ratio_test(additive, no_scanner)
    interaction_lr, interaction_df, interaction_p = likelihood_ratio_test(
        interaction, additive
    )

    if environment_terms:
        no_environment, _ = fit_mixed_model(
            build_formula(True, True, False, [], adjustment), data
        )
        environment_lr, environment_df, environment_p = likelihood_ratio_test(
            additive, no_environment
        )
    else:
        environment_lr, environment_df, environment_p = np.nan, np.nan, np.nan

    environment_rows = []
    for term in environment_terms:
        reduced_terms = [candidate for candidate in environment_terms if candidate != term]
        reduced, _ = fit_mixed_model(
            build_formula(True, True, False, reduced_terms, adjustment), data
        )
        lr_statistic, lr_df, lr_p = likelihood_ratio_test(additive, reduced)
        original = model_to_original[term]
        environment_rows.append({
            "feature": feature,
            "feature_label": feature_label(feature),
            "outcome_transformation": transformation,
            "environment_variable": original,
            "environment_label": ENVIRONMENT_LABELS.get(original, original),
            "standardized_beta": finite_float(additive.fe_params.get(term)),
            "standard_error": finite_float(additive.bse_fe.get(term)),
            "wald_p": finite_float(additive.pvalues.get(term)),
            "drop_one_lr_statistic": lr_statistic,
            "drop_one_lr_df": lr_df,
            "drop_one_p": lr_p,
            "n_observations": len(data),
            "n_environment_sessions": data[["group", "timepoint_number"]]
            .drop_duplicates().shape[0],
        })

    coefficients = fixed_effect_rows(
        feature, "additive", additive, transformation
    ) + fixed_effect_rows(feature, "stage_x_scanner", interaction, transformation)

    model_result = {
        "feature": feature,
        "feature_label": feature_label(feature),
        "outcome_transformation": transformation,
        "n_observations": len(data),
        "n_physical_samples": data["physical_sample_id"].nunique(),
        "n_scanners": data["scanner"].nunique(),
        "n_stages": data["stage"].nunique(),
        "n_environment_sessions": data[["group", "timepoint_number"]]
        .drop_duplicates().shape[0],
        "group_adjustment": adjustment,
        "additive_optimizer": additive_optimizer,
        "interaction_optimizer": interaction_optimizer,
        "additive_converged": bool(getattr(additive, "converged", False)),
        "interaction_converged": bool(getattr(interaction, "converged", False)),
        "random_intercept_variance": finite_float(np.asarray(additive.cov_re)[0, 0]),
        "residual_variance": finite_float(additive.scale),
        "stage_lr_statistic": stage_lr,
        "stage_lr_df": stage_df,
        "stage_p": stage_p,
        "scanner_lr_statistic": scanner_lr,
        "scanner_lr_df": scanner_df,
        "scanner_p": scanner_p,
        "interaction_lr_statistic": interaction_lr,
        "interaction_lr_df": interaction_df,
        "interaction_p": interaction_p,
        "environment_joint_lr_statistic": environment_lr,
        "environment_joint_lr_df": environment_df,
        "environment_joint_p": environment_p,
        "additive_formula": additive_formula,
        "interaction_formula": interaction_formula,
    }
    return model_result, coefficients, environment_rows


def classify_endpoint(row: pd.Series) -> str:
    stage_p = finite_float(row.get("stage_p_holm"))
    interaction_p = finite_float(row.get("interaction_p_holm"))
    consistency = finite_float(row.get("clean_baseline_direction_consistency"))
    if str(row.get("model_status")) != "OK" or not np.isfinite(stage_p):
        return "MODEL_FAILED"
    if stage_p < ALPHA:
        if np.isfinite(interaction_p) and interaction_p < ALPHA:
            return "STAGE_SIGNAL_SCANNER_DEPENDENT"
        if np.isfinite(consistency) and consistency >= 0.75:
            return "ROBUST_STAGE_SIGNAL"
        return "STAGE_SIGNAL_INCONSISTENT"
    if finite_float(row.get("scanner_p_holm")) < ALPHA:
        return "MAINLY_SCANNER_EFFECT"
    return "NO_CLEAR_STAGE_SIGNAL"


# =============================================================================
# FIGURES
# =============================================================================

def save_primary_stage_plot(
    data: pd.DataFrame, endpoints: list[str], scanners: list[str], output: Path,
) -> None:
    selected = [feature for feature in PRIMARY_ENDPOINTS if feature in endpoints]
    if not selected:
        return
    figure, axes = plt.subplots(
        len(selected), len(scanners),
        figsize=(3.7 * len(scanners), 3.1 * len(selected)),
        squeeze=False, constrained_layout=True,
    )
    colors = {"Baseline": "#B2182B", "Partial": "#E6AB02", "Clean": "#2166AC"}
    rng = np.random.default_rng(42)
    for row_index, feature in enumerate(selected):
        for column_index, scanner in enumerate(scanners):
            axis = axes[row_index, column_index]
            scanner_data = data[data["scanner"].astype(str) == scanner]
            arrays, labels, positions = [], [], []
            for position, stage in enumerate(STAGE_ORDER, 1):
                values = pd.to_numeric(
                    scanner_data.loc[scanner_data["stage"].astype(str) == stage, feature],
                    errors="coerce",
                ).dropna().to_numpy(float)
                if not len(values):
                    continue
                arrays.append(values)
                labels.append(stage)
                positions.append(position)
                jitter = rng.normal(0, 0.045, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter, values,
                    s=12, alpha=0.40, color=colors[stage], edgecolor="none",
                )
            if arrays:
                boxes = axis.boxplot(
                    arrays, positions=positions, widths=0.48, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.3},
                )
                for patch, label in zip(boxes["boxes"], labels):
                    patch.set_facecolor(colors[label])
                    patch.set_alpha(0.28)
            axis.set_xticks(positions, labels, rotation=25)
            axis.grid(axis="y", alpha=0.2)
            if row_index == 0:
                axis.set_title(scanner)
            if column_index == 0:
                axis.set_ylabel(feature_label(feature))
    figure.suptitle("ROI-level colour endpoints by biological stage and scanner")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_evidence_matrix(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        return
    data = results.sort_values("stage_p_holm", na_position="last").reset_index(drop=True)
    columns = [
        "stage_p_holm", "scanner_p_holm", "interaction_p_holm",
        "environment_joint_p_holm",
    ]
    raw = np.asarray([
        [negative_log10_p(row.get(column)) for column in columns]
        + [finite_float(row.get("clean_baseline_direction_consistency"))]
        for _, row in data.iterrows()
    ])
    shown = raw.copy()
    shown[:, :4] = np.clip(shown[:, :4], 0, 10) / 10
    shown[:, 4] = np.clip(shown[:, 4], 0, 1)
    figure, axis = plt.subplots(figsize=(10.5, max(5.5, 0.55 * len(data) + 2)))
    image = axis.imshow(shown, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(5), [
        "Stage\nsignal", "Scanner\noffset", "Stage x scanner\ninteraction",
        "Environment\njoint", "Direction\nconsistency",
    ])
    axis.set_yticks(range(len(data)), [feature_label(value) for value in data["feature"]])
    axis.set_title("Pixel-colour endpoint evidence matrix\nBrighter = stronger evidence or consistency")
    for row_index, (_, row) in enumerate(data.iterrows()):
        labels = []
        for column in columns:
            value = finite_float(row.get(column))
            labels.append(f"{value:.3g}" if np.isfinite(value) else "NA")
        consistency = finite_float(row.get("clean_baseline_direction_consistency"))
        labels.append(f"{consistency:.0%}" if np.isfinite(consistency) else "NA")
        for column_index, label in enumerate(labels):
            axis.text(
                column_index, row_index, label, ha="center", va="center", fontsize=8,
                color="white" if shown[row_index, column_index] < 0.45 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.04)
    colorbar.set_label("Normalized evidence / consistency")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_direction_heatmap(
    changes: pd.DataFrame, endpoints: list[str], scanners: list[str], output: Path,
) -> None:
    selected = changes[changes["transition"] == "Clean - Baseline"]
    if selected.empty:
        return
    matrix = np.full((len(endpoints), len(scanners)), np.nan)
    annotations = np.full(matrix.shape, "", dtype=object)
    for row_index, feature in enumerate(endpoints):
        for column_index, scanner in enumerate(scanners):
            rows = selected[
                (selected["feature"] == feature) & (selected["scanner"] == scanner)
            ]
            if rows.empty:
                continue
            value = finite_float(rows.iloc[0]["median_change"])
            matrix[row_index, column_index] = value
            annotations[row_index, column_index] = f"{value:.3g}"
        finite = np.abs(matrix[row_index, np.isfinite(matrix[row_index])])
        if len(finite) and finite.max() > 0:
            matrix[row_index] /= finite.max()
    figure, axis = plt.subplots(figsize=(2.0 * len(scanners) + 3.5, max(5.5, 0.55 * len(endpoints) + 2)))
    image = axis.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(scanners)), scanners)
    axis.set_yticks(range(len(endpoints)), [feature_label(value) for value in endpoints])
    axis.set_title("Median paired Clean - Baseline change\nColour normalized within each endpoint; labels show raw change")
    for row_index in range(len(endpoints)):
        for column_index in range(len(scanners)):
            axis.text(column_index, row_index, annotations[row_index, column_index],
                      ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Normalized change within endpoint")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_environment_overview(environment: pd.DataFrame, output: Path) -> None:
    if environment.empty:
        return
    features = list(dict.fromkeys(environment["feature"]))
    variables = [value for value in ENVIRONMENT_LABELS if value in set(environment["environment_variable"])]
    matrix = np.full((len(features), len(variables)), np.nan)
    labels = np.full(matrix.shape, "", dtype=object)
    for row_index, feature in enumerate(features):
        for column_index, variable in enumerate(variables):
            rows = environment[
                (environment["feature"] == feature)
                & (environment["environment_variable"] == variable)
            ]
            if rows.empty:
                continue
            beta = finite_float(rows.iloc[0]["standardized_beta"])
            p_value = finite_float(rows.iloc[0]["drop_one_p_holm"])
            matrix[row_index, column_index] = beta
            labels[row_index, column_index] = (
                f"β={beta:.2g}\np={p_value:.2g}" if np.isfinite(p_value) else f"β={beta:.2g}"
            )
    limit = np.nanpercentile(np.abs(matrix), 95) if np.any(np.isfinite(matrix)) else 1
    limit = max(float(limit), 1e-9)
    figure, axis = plt.subplots(figsize=(8.5, max(5.5, 0.6 * len(features) + 2)))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(variables)), [ENVIRONMENT_LABELS[value] for value in variables])
    axis.set_yticks(range(len(features)), [feature_label(value) for value in features])
    axis.set_title("Exploratory session-level environmental coefficients")
    for row_index in range(len(features)):
        for column_index in range(len(variables)):
            axis.text(column_index, row_index, labels[row_index, column_index],
                      ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Standardized coefficient")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    data_root = args.data_root
    pixel_path = args.pixel_table or (
        data_root / "_pixel_color_analysis" / "tables" / "pixel_color_features.csv"
    )
    require_file(pixel_path, "Pixel-colour feature table")
    raw = pd.read_csv(pixel_path)
    data, excluded, scanners = prepare_pixel_table(raw, args.include_qc_failures)

    endpoints = args.features or [
        feature for feature in DEFAULT_ENDPOINTS if feature in data
    ]
    missing_endpoints = [feature for feature in endpoints if feature not in data]
    if missing_endpoints:
        raise ValueError("Requested endpoints not found: " + ", ".join(missing_endpoints))
    if not endpoints:
        raise ValueError("No requested/default pixel-colour endpoints were found")

    environment_path = resolve_environment_path(data_root, args.environment_table)
    environment = pd.DataFrame()
    environment_variables: list[str] = []
    if environment_path is not None and environment_path.is_file():
        environment = read_environment_file(environment_path)
        data, environment_variables = merge_environment(data, environment)
    elif not args.allow_no_environment:
        raise FileNotFoundError(
            "Environmental data were not found. Provide --environment-table, run "
            "the earlier environment step, or deliberately use --allow-no-environment."
        )

    print("=" * 78)
    print("BIOCAL3D STEP 4 - PIXEL-COLOUR REPEATED-MEASURES ANALYSIS")
    print("=" * 78)
    print(f"Pixel table: {pixel_path}")
    print(f"Accepted ROI rows: {len(data)}")
    print(f"Excluded QC rows: {len(excluded) if not args.include_qc_failures else 0}")
    print(f"Physical samples: {data['physical_sample_id'].nunique()}")
    print("Scanners: " + ", ".join(scanners))
    print("Groups: " + ", ".join(f"G{value}" for value in sorted(data["group"].unique())))
    print("Endpoints: " + ", ".join(endpoints))
    print(
        "Environment: "
        + (str(environment_path) if environment_variables else "not included")
    )
    if args.validate_only:
        print("Validation completed; no models or files were created.")
        return
    if smf is None:
        raise ImportError(
            "statsmodels/patsy are required. Install them in the active BioCal3D "
            "environment with `pip install statsmodels patsy`."
        )

    output_root = args.output or (
        data_root / "_pixel_color_analysis" / "step_4_statistical_analysis"
    )
    table_root, figure_root = output_root / "tables", output_root / "figures"
    for directory in (table_root, figure_root):
        directory.mkdir(parents=True, exist_ok=True)

    data.to_csv(table_root / "analysis_dataset.csv", index=False)
    excluded.to_csv(table_root / "excluded_qc_rows.csv", index=False)
    if not environment.empty:
        environment.to_csv(table_root / "environment_sessions_used.csv", index=False)

    descriptives = stage_scanner_descriptives(data, endpoints)
    changes = paired_stage_changes(data, endpoints, scanners)
    model_rows, coefficient_rows, environment_rows = [], [], []

    for index, feature in enumerate(endpoints, 1):
        print(f"[{index:02d}/{len(endpoints):02d}] {feature}")
        consistency, n_direction_scanners, pooled_scanner_median = (
            clean_baseline_direction(changes, feature)
        )
        row: dict[str, object] = {
            "feature": feature,
            "feature_label": feature_label(feature),
            "model_status": "OK",
            "model_error": "",
            "clean_baseline_direction_consistency": consistency,
            "n_scanners_with_clean_baseline_pairs": n_direction_scanners,
            "median_of_scanner_clean_baseline_changes": pooled_scanner_median,
        }
        try:
            result, coefficients, environmental = analyse_endpoint(
                data, feature, environment_variables, args.group_adjustment, scanners
            )
            row.update(result)
            coefficient_rows.extend(coefficients)
            environment_rows.extend(environmental)
        except Exception as exc:
            row["model_status"] = "FAILED"
            row["model_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  MODEL FAILED: {exc}")
            for column in [
                "stage_p", "scanner_p", "interaction_p", "environment_joint_p",
                "n_observations", "n_physical_samples", "n_scanners", "n_stages",
            ]:
                row[column] = np.nan
        model_rows.append(row)

    results = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    environment_effects = pd.DataFrame(environment_rows)

    for raw_column, adjusted_column in [
        ("stage_p", "stage_p_holm"),
        ("scanner_p", "scanner_p_holm"),
        ("interaction_p", "interaction_p_holm"),
        ("environment_joint_p", "environment_joint_p_holm"),
    ]:
        results[adjusted_column] = holm_adjust(
            pd.to_numeric(results.get(raw_column), errors="coerce")
        )
    results["interpretation_class"] = results.apply(classify_endpoint, axis=1)

    if not coefficients.empty:
        coefficients["wald_p_holm"] = holm_adjust(coefficients["wald_p"])
    if not environment_effects.empty:
        environment_effects["drop_one_p_holm"] = holm_adjust(
            environment_effects["drop_one_p"]
        )
        environment_effects["wald_p_holm"] = holm_adjust(
            environment_effects["wald_p"]
        )

    results = results.sort_values("stage_p_holm", na_position="last")
    results.to_csv(table_root / "endpoint_model_summary.csv", index=False)
    descriptives.to_csv(table_root / "stage_scanner_descriptives.csv", index=False)
    changes.to_csv(table_root / "paired_stage_changes.csv", index=False)
    coefficients.to_csv(table_root / "model_fixed_effects.csv", index=False)
    environment_effects.to_csv(table_root / "environment_effects.csv", index=False)

    save_primary_stage_plot(
        data, endpoints, scanners,
        figure_root / "primary_endpoints_by_stage_and_scanner.png",
    )
    save_evidence_matrix(results, figure_root / "endpoint_evidence_matrix.png")
    save_direction_heatmap(
        changes, endpoints, scanners,
        figure_root / "clean_vs_baseline_direction_heatmap.png",
    )
    save_environment_overview(
        environment_effects, figure_root / "environmental_coefficients.png"
    )

    class_counts = results["interpretation_class"].value_counts()
    notes = f"""BIOCAL3D STEP 4 - PIXEL-COLOUR STATISTICS

Inputs
------
Pixel features: {pixel_path}
Environment: {environment_path if environment_variables else 'Not included'}

Current data
------------
Accepted ROI rows: {len(data)}
QC exclusions: {len(excluded) if not args.include_qc_failures else 0}
Physical samples: {data['physical_sample_id'].nunique()}
Scanners: {', '.join(scanners)}
Groups: {', '.join(f'G{value}' for value in sorted(data['group'].unique()))}
Environmental sessions: {len(environment) if not environment.empty else 0}

Primary model
-------------
endpoint ~ stage + scanner + {args.group_adjustment} + environment
           + random physical-sample intercept

Scanner-dependence model
------------------------
endpoint ~ stage * scanner + {args.group_adjustment} + environment
           + random physical-sample intercept

Multiplicity
------------
Stage, scanner, interaction and joint-environment p-values are Holm-adjusted
across the selected endpoint family. Paired Wilcoxon p-values are Holm-adjusted
across all reported endpoint x scanner x transition comparisons.

Interpretation
--------------
The physical specimen is the repeated-measures unit. Pixels are not treated as
replicates. Cluster fractions use an empirical-logit model transformation but
remain on the raw fraction scale in descriptives and paired-change tables.

Environment varies across only group x stage sessions. Environmental terms are
nuisance adjustments and exploratory associations; their ROI-level mixed-model
p-values do not create additional independent environmental sessions and must
not be interpreted causally.

The scanner-specific colour clusters are relative colour phenotypes. Cluster 1
is the highest-a* cluster within each scanner, not a calibrated plaque class.

Classification counts
---------------------
{class_counts.to_string()}
"""
    (output_root / "README_results.txt").write_text(notes, encoding="utf-8")

    print("\nAnalysis complete")
    print(class_counts.to_string())
    print(f"Results: {output_root}")
    print("Start with:")
    print(f"  {table_root / 'endpoint_model_summary.csv'}")
    print(f"  {table_root / 'paired_stage_changes.csv'}")
    print(f"  {figure_root / 'endpoint_evidence_matrix.png'}")


if __name__ == "__main__":
    main()
