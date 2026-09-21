"""Combined BioCal3D colour/geometry feature robustness analysis.

This script is compatible with the outputs produced by:

* analyze_raw_scanner_colors.ipynb
  -> _raw_color_analysis/roi_color_summary.csv
* analyze_roi_geometry.py
  -> _geometry_analysis/tables/roi_geometry_features.csv
* analyze_environment_effect.py
  -> _raw_color_analysis/step_C_environment_effect/tables/
     environment_session_qc.csv

The analysis asks whether each ROI-level feature changes with biological
cleaning stage, whether scanners differ, and whether the stage response is
scanner-dependent. Ambient light, humidity and temperature are included as
session-level covariates and are also reported explicitly.

The physical specimen is the repeated-measures unit. Technical acquisition and
mesh-density variables are analysed separately from biological candidates.

Typical use
-----------
Run Steps A-C and the geometry analysis first, then run:

    python analyze_feature_robustness.py

To deliberately run without environmental adjustment:

    python analyze_feature_robustness.py --allow-no-environment

Dependencies
------------
numpy, pandas, scipy, matplotlib, statsmodels and patsy.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


# ============================================================================
# PATHS AND EXPERIMENTAL DESIGN
# ============================================================================

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)

SCANNER_ORDER = [
    "LABscanner",
    "iTERO",
    "TRIOS3",
    "TRIOS5",
]

GROUP_STAGE_MAP = {
    1: {0: "Baseline", 1: "Clean"},
    2: {0: "Baseline", 1: "Clean"},
    3: {0: "Baseline", 1: "Partial", 2: "Clean"},
    4: {0: "Baseline", 1: "Partial", 2: "Clean"},
    5: {0: "Baseline", 1: "Partial", 2: "Clean"},
}

STAGE_ORDER = ["Baseline", "Partial", "Clean"]

ALPHA = 0.05
MIN_SCANNERS_FOR_DIRECTION = 3
MIN_DIRECTION_CONSISTENCY = 0.75

ENVIRONMENT_VARIABLES = {
    "light_avg": "Ambient light",
    "humidity_avg": "Relative humidity",
    "temperature_avg": "Temperature",
}


# These names exactly match roi_color_summary.csv.
COLOR_FEATURES = [
    "n_color_points",
    "mean_R",
    "median_R",
    "std_R",
    "mean_G",
    "median_G",
    "std_G",
    "mean_B",
    "median_B",
    "std_B",
    "mean_L",
    "median_L",
    "std_L",
    "p10_L",
    "p90_L",
    "mean_a",
    "median_a",
    "std_a",
    "p10_a",
    "p90_a",
    "mean_b",
    "median_b",
    "std_b",
    "p10_b",
    "p90_b",
    "mean_chroma",
    "median_chroma",
    "median_hue_deg",
    "fraction_near_black",
    "fraction_near_white",
]

# These names exactly match ANALYSIS_METRICS in analyze_roi_geometry.py.
GEOMETRY_FEATURES = [
    "roi_vertices_per_nominal_mm2",
    "roi_faces_per_nominal_mm2",
    "median_nearest_neighbor_mm",
    "median_unique_edge_length_mm",
    "median_triangle_area_mm2",
    "median_triangle_edge_ratio",
    "median_triangle_min_angle_deg",
    "plane_rmse_mm",
    "native_Sa_mm",
    "native_Sq_mm",
    "native_height_p95_minus_p05_mm",
    "surface_to_projected_area_ratio",
    "area_weighted_normal_angle_median_deg",
    "grid_Sa_mm",
    "grid_Sq_mm",
    "grid_height_p95_minus_p05_mm",
    "grid_rms_slope",
]

# Acquisition/reconstruction variables are useful QC outcomes, but they are not
# interpreted as candidate measures of biological plaque.
TECHNICAL_FEATURES = {
    "color__n_color_points",
    "color__fraction_near_black",
    "color__fraction_near_white",
    "geometry__roi_vertices_per_nominal_mm2",
    "geometry__roi_faces_per_nominal_mm2",
    "geometry__median_nearest_neighbor_mm",
    "geometry__median_unique_edge_length_mm",
    "geometry__median_triangle_area_mm2",
    "geometry__median_triangle_edge_ratio",
    "geometry__median_triangle_min_angle_deg",
}


CLASSIFICATION_ORDER = {
    "ROBUST_CANDIDATE": 0,
    "STAGE_EFFECT_SCANNER_DEPENDENT": 1,
    "STAGE_EFFECT_INCONSISTENT": 2,
    "MAINLY_SCANNER_EFFECT": 3,
    "NO_CLEAR_STAGE_SIGNAL": 4,
    "MODEL_FAILED": 5,
    "TECHNICAL_QC_FEATURE": 6,
}

CLASSIFICATION_LABELS = {
    "ROBUST_CANDIDATE": "Robust candidate",
    "STAGE_EFFECT_SCANNER_DEPENDENT": "Stage effect; scanner-dependent",
    "STAGE_EFFECT_INCONSISTENT": "Stage effect; inconsistent direction",
    "MAINLY_SCANNER_EFFECT": "Mainly scanner effect",
    "NO_CLEAR_STAGE_SIGNAL": "No clear stage signal",
    "MODEL_FAILED": "Model failed",
    "TECHNICAL_QC_FEATURE": "Technical/QC feature",
}

CLASSIFICATION_COLORS = {
    "ROBUST_CANDIDATE": "#2E8B57",
    "STAGE_EFFECT_SCANNER_DEPENDENT": "#D95F02",
    "STAGE_EFFECT_INCONSISTENT": "#E6AB02",
    "MAINLY_SCANNER_EFFECT": "#7570B3",
    "NO_CLEAR_STAGE_SIGNAL": "#7F7F7F",
    "MODEL_FAILED": "#B2182B",
    "TECHNICAL_QC_FEATURE": "#4D4D4D",
}


# ============================================================================
# COMMAND LINE AND GENERAL HELPERS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--color-table", type=Path, default=None)
    parser.add_argument("--geometry-table", type=Path, default=None)
    parser.add_argument(
        "--environment-table",
        type=Path,
        default=None,
        help=(
            "Optional explicit Step C environment_session_qc.csv. If omitted, "
            "the standard Step C output is found automatically."
        ),
    )
    parser.add_argument(
        "--master-table",
        type=Path,
        default=None,
        help=(
            "Optional already-built master_feature_table.csv. This is useful "
            "for reruns; raw colour and geometry inputs are otherwise used."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-no-environment",
        action="store_true",
        help="Allow an explicitly unadjusted run if no Step C table exists.",
    )
    parser.add_argument(
        "--include-geometry-qc-failures",
        action="store_true",
        help="Keep geometry rows with analysis_ok=False (not recommended).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional exact prefixed feature names for a targeted rerun.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def holm_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna() & np.isfinite(numeric)
    if mask.any():
        output.loc[mask] = multipletests(
            numeric.loc[mask].to_numpy(dtype=float), method="holm"
        )[1]
    return output


def negative_log10_p(value: object) -> float:
    number = finite_float(value)
    if not np.isfinite(number):
        return np.nan
    return -math.log10(max(number, np.nextafter(0.0, 1.0)))


def pretty_feature_name(feature: str) -> str:
    text = feature.replace("color__", "Colour: ").replace(
        "geometry__", "Geometry: "
    )
    return text.replace("_", " ").replace(" mm2", " mm²")


def stage_from_group_timepoint(group: object, timepoint: object) -> str:
    group_number = int(float(group))
    time_number = int(float(timepoint))
    try:
        return GROUP_STAGE_MAP[group_number][time_number]
    except KeyError as exc:
        raise ValueError(
            f"No stage mapping for group={group_number}, timepoint={time_number}."
        ) from exc


def physical_sample_id(group: object, sample_number: object) -> str:
    return f"BCG{int(float(group))}-{int(float(sample_number))}"


def feature_role(feature: str) -> str:
    return "technical" if feature in TECHNICAL_FEATURES else "biological"


# ============================================================================
# BUILD MASTER TABLE FROM THE ACTUAL UPSTREAM OUTPUTS
# ============================================================================

KEY_COLUMNS = ["scanner", "sample_id"]
CORE_COLUMNS = [
    "scanner",
    "group",
    "timepoint_number",
    "sample_number",
    "sample_id",
    "physical_sample_id",
    "stage",
]


def normalize_identifiers(data: pd.DataFrame, label: str) -> pd.DataFrame:
    output = data.copy()
    require_columns(
        output,
        ["scanner", "sample_id", "group", "timepoint_number", "sample_number"],
        label,
    )

    output = output.dropna(
        subset=["scanner", "sample_id", "group", "timepoint_number", "sample_number"]
    ).copy()
    output["scanner"] = output["scanner"].astype(str).str.strip()
    output["sample_id"] = output["sample_id"].astype(str).str.strip().str.upper()
    output["group"] = pd.to_numeric(output["group"], errors="raise").astype(int)
    output["timepoint_number"] = pd.to_numeric(
        output["timepoint_number"], errors="raise"
    ).astype(int)
    output["sample_number"] = pd.to_numeric(
        output["sample_number"], errors="raise"
    ).astype(int)
    output = output[output["scanner"].isin(SCANNER_ORDER)].copy()

    derived_specimen = [
        physical_sample_id(group, sample)
        for group, sample in zip(output["group"], output["sample_number"])
    ]
    if "physical_sample_id" in output:
        supplied = output["physical_sample_id"].astype(str).str.strip().str.upper()
        mismatch = supplied.ne(pd.Series(derived_specimen, index=output.index))
        mismatch &= output["physical_sample_id"].notna()
        if mismatch.any():
            examples = output.loc[mismatch, KEY_COLUMNS + ["physical_sample_id"]].head()
            raise ValueError(
                f"{label} has inconsistent physical_sample_id values:\n"
                + examples.to_string(index=False)
            )
    output["physical_sample_id"] = derived_specimen

    derived_stage = [
        stage_from_group_timepoint(group, timepoint)
        for group, timepoint in zip(output["group"], output["timepoint_number"])
    ]
    if "stage" in output:
        supplied_stage = output["stage"].astype(str).str.strip()
        mismatch = supplied_stage.ne(pd.Series(derived_stage, index=output.index))
        mismatch &= output["stage"].notna()
        if mismatch.any():
            examples = output.loc[mismatch, KEY_COLUMNS + ["stage"]].head()
            raise ValueError(
                f"{label} has stage values inconsistent with GROUP_STAGE_MAP:\n"
                + examples.to_string(index=False)
            )
    output["stage"] = derived_stage

    duplicates = output.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = output.loc[duplicates, KEY_COLUMNS].sort_values(KEY_COLUMNS).head(12)
        raise ValueError(
            f"{label} contains duplicate scanner × sample_id rows:\n"
            + examples.to_string(index=False)
        )
    return output


def build_master_table(
    colour: pd.DataFrame,
    geometry: pd.DataFrame,
    include_geometry_qc_failures: bool,
) -> tuple[pd.DataFrame, list[str]]:
    colour = normalize_identifiers(colour, "ROI colour table")
    geometry = normalize_identifiers(geometry, "ROI geometry table")

    if "analysis_ok" in geometry and not include_geometry_qc_failures:
        if pd.api.types.is_bool_dtype(geometry["analysis_ok"]):
            accepted = geometry["analysis_ok"].fillna(False)
        else:
            accepted = (
                geometry["analysis_ok"]
                .astype(str)
                .str.strip()
                .str.casefold()
                .isin(["true", "1", "yes"])
            )
        rejected_count = int((~accepted).sum())
        if rejected_count:
            print(f"Geometry QC exclusions: {rejected_count}")
        geometry = geometry.loc[accepted].copy()

    available_colour = [feature for feature in COLOR_FEATURES if feature in colour]
    available_geometry = [
        feature for feature in GEOMETRY_FEATURES if feature in geometry
    ]
    missing_colour = sorted(set(COLOR_FEATURES) - set(available_colour))
    missing_geometry = sorted(set(GEOMETRY_FEATURES) - set(available_geometry))
    if missing_colour:
        print("WARNING: colour features absent: " + ", ".join(missing_colour))
    if missing_geometry:
        print("WARNING: geometry features absent: " + ", ".join(missing_geometry))
    if not available_colour:
        raise ValueError("No expected colour features were found.")
    if not available_geometry:
        raise ValueError("No expected geometry features were found.")

    colour_keep = CORE_COLUMNS + available_colour
    geometry_keep = CORE_COLUMNS + available_geometry
    if "analysis_ok" in geometry:
        geometry_keep.append("analysis_ok")

    colour_part = colour[colour_keep].rename(
        columns={feature: f"color__{feature}" for feature in available_colour}
    )
    geometry_part = geometry[geometry_keep].rename(
        columns={
            feature: f"geometry__{feature}" for feature in available_geometry
        }
    )

    merged = colour_part.merge(
        geometry_part,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("_color", "_geometry"),
        indicator=True,
    )

    if len(merged) != len(colour_part) or len(merged) != len(geometry_part):
        colour_keys = set(map(tuple, colour_part[KEY_COLUMNS].to_numpy()))
        geometry_keys = set(map(tuple, geometry_part[KEY_COLUMNS].to_numpy()))
        print(
            "WARNING: unmatched upstream rows were excluded: "
            f"colour-only={len(colour_keys - geometry_keys)}, "
            f"geometry-only={len(geometry_keys - colour_keys)}"
        )

    for column in [value for value in CORE_COLUMNS if value not in KEY_COLUMNS]:
        colour_column = f"{column}_color"
        geometry_column = f"{column}_geometry"
        if colour_column not in merged or geometry_column not in merged:
            continue
        left = merged[colour_column].astype(str)
        right = merged[geometry_column].astype(str)
        mismatch = left.ne(right) & merged[colour_column].notna() & merged[
            geometry_column
        ].notna()
        if mismatch.any():
            examples = merged.loc[
                mismatch, KEY_COLUMNS + [colour_column, geometry_column]
            ].head()
            raise ValueError(
                f"Colour/geometry metadata mismatch for {column}:\n"
                + examples.to_string(index=False)
            )
        merged[column] = merged[colour_column].combine_first(
            merged[geometry_column]
        )
        merged = merged.drop(columns=[colour_column, geometry_column])

    merged = merged.drop(columns=["_merge"])
    merged["scanner"] = pd.Categorical(
        merged["scanner"], categories=SCANNER_ORDER, ordered=True
    )
    merged["stage"] = pd.Categorical(
        merged["stage"], categories=STAGE_ORDER, ordered=True
    )
    merged["maturity"] = np.where(merged["group"].isin([1, 2]), "Fresh", "Mature")

    feature_columns = [
        column
        for column in merged.columns
        if column.startswith("color__") or column.startswith("geometry__")
    ]
    ordered = CORE_COLUMNS + [
        column for column in ["analysis_ok", "maturity"] if column in merged
    ]
    ordered += feature_columns
    return merged[ordered], feature_columns


def load_existing_master(path: Path) -> tuple[pd.DataFrame, list[str]]:
    require_file(path, "Master feature table")
    master = pd.read_csv(path)
    require_columns(master, CORE_COLUMNS, "Master feature table")
    master = normalize_identifiers(master, "Master feature table")
    master["scanner"] = pd.Categorical(
        master["scanner"], categories=SCANNER_ORDER, ordered=True
    )
    master["stage"] = pd.Categorical(
        master["stage"], categories=STAGE_ORDER, ordered=True
    )
    features = [
        column
        for column in master.columns
        if column.startswith("color__") or column.startswith("geometry__")
    ]
    if not features:
        raise ValueError("Master table contains no prefixed feature columns.")
    return master, features


# ============================================================================
# ENVIRONMENT TABLE COMPATIBILITY
# ============================================================================

def collapse_environment_sessions(environment: pd.DataFrame) -> pd.DataFrame:
    output = environment.copy()
    if "group" not in output or "timepoint_number" not in output:
        if "session_id" not in output:
            raise ValueError(
                "Environment table needs group/timepoint_number or session_id."
            )
        extracted = output["session_id"].astype(str).str.extract(
            r"^BCG(?P<group>\d+)T(?P<timepoint_number>\d+)$"
        )
        output["group"] = pd.to_numeric(extracted["group"], errors="coerce")
        output["timepoint_number"] = pd.to_numeric(
            extracted["timepoint_number"], errors="coerce"
        )

    output = output.dropna(subset=["group", "timepoint_number"]).copy()
    output["group"] = pd.to_numeric(output["group"], errors="raise").astype(int)
    output["timepoint_number"] = pd.to_numeric(
        output["timepoint_number"], errors="raise"
    ).astype(int)

    variables = [column for column in ENVIRONMENT_VARIABLES if column in output]
    if not variables:
        raise ValueError(
            "No Step C environmental variables found. Expected one or more of: "
            + ", ".join(ENVIRONMENT_VARIABLES)
        )

    extra = [
        column
        for column in ["session_id", "all_calibrations_ok"]
        if column in output
    ]
    keep = ["group", "timepoint_number", *variables, *extra]
    output = output[keep].copy()

    keys = ["group", "timepoint_number"]
    rows = []
    for key, subset in output.groupby(keys, dropna=False, observed=True):
        row = {"group": key[0], "timepoint_number": key[1]}
        for column in variables:
            values = pd.to_numeric(subset[column], errors="coerce").dropna().unique()
            if len(values) > 1 and not np.allclose(values, values[0]):
                raise ValueError(
                    f"Conflicting {column} values for group/timepoint {key}."
                )
            row[column] = values[0] if len(values) else np.nan
        if "session_id" in subset:
            values = subset["session_id"].dropna().astype(str).unique()
            row["session_id"] = values[0] if len(values) else f"BCG{key[0]}T{key[1]}"
        else:
            row["session_id"] = f"BCG{key[0]}T{key[1]}"
        if "all_calibrations_ok" in subset:
            values = subset["all_calibrations_ok"].dropna().unique()
            row["all_calibrations_ok"] = values[0] if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def resolve_environment_path(
    data_root: Path, explicit: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        data_root
        / "_raw_color_analysis"
        / "step_C_environment_effect"
        / "tables"
        / "environment_session_qc.csv",
        data_root
        / "_raw_color_analysis"
        / "step_C_environment_effect"
        / "tables"
        / "roi_color_with_environment.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def merge_environment(
    master: pd.DataFrame, environment_path: Path
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    require_file(environment_path, "Step C environmental table")
    environment = collapse_environment_sessions(pd.read_csv(environment_path))
    variables = [column for column in ENVIRONMENT_VARIABLES if column in environment]
    merged = master.merge(
        environment,
        on=["group", "timepoint_number"],
        how="left",
        validate="many_to_one",
    )
    unmatched = merged[variables].isna().all(axis=1)
    if unmatched.any():
        missing_sessions = (
            merged.loc[unmatched, ["group", "timepoint_number"]]
            .drop_duplicates()
            .sort_values(["group", "timepoint_number"])
        )
        raise ValueError(
            "Environmental data are absent for these group/timepoint sessions:\n"
            + missing_sessions.to_string(index=False)
        )
    return merged, variables, environment


# ============================================================================
# MIXED MODELS
# ============================================================================

def prepare_model_data(
    master: pd.DataFrame, feature: str, environment_variables: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    columns = [
        feature,
        "scanner",
        "stage",
        "group",
        "physical_sample_id",
        *environment_variables,
    ]
    data = master[columns].copy().rename(columns={feature: "value"})
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    model_environment = []
    model_to_original = {}
    for index, original in enumerate(environment_variables):
        values = pd.to_numeric(data[original], errors="coerce")
        standard_deviation = float(values.std())
        if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
            print(f"  Environment skipped (no variance): {original}")
            continue
        model_name = f"env_z_{index}"
        data[model_name] = (values - values.mean()) / standard_deviation
        model_environment.append(model_name)
        model_to_original[model_name] = original

    required = [
        "value",
        "scanner",
        "stage",
        "group",
        "physical_sample_id",
        *model_environment,
    ]
    data = data[required].dropna().copy()
    data["scanner"] = pd.Categorical(
        data["scanner"], categories=SCANNER_ORDER, ordered=True
    ).remove_unused_categories()
    data["stage"] = pd.Categorical(
        data["stage"], categories=STAGE_ORDER, ordered=True
    ).remove_unused_categories()
    return data, model_environment, model_to_original


def build_formula(
    include_stage: bool,
    include_scanner: bool,
    include_interaction: bool,
    environment_terms: list[str],
) -> str:
    terms = []
    if include_interaction:
        terms.append("C(stage) * C(scanner)")
    else:
        if include_stage:
            terms.append("C(stage)")
        if include_scanner:
            terms.append("C(scanner)")
    terms.append("C(group)")
    terms.extend(environment_terms)
    return "value ~ " + " + ".join(terms)


def fit_mixed_model(formula: str, data: pd.DataFrame):
    errors = []
    for method in ["lbfgs", "powell", "cg"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data["physical_sample_id"],
                )
                result = model.fit(
                    reml=False, method=method, maxiter=2000, disp=False
                )
            if not np.isfinite(result.llf):
                raise RuntimeError("non-finite log-likelihood")
            return result, method
        except Exception as exc:  # optimizers fail differently by feature
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def likelihood_ratio_test(full_model, reduced_model) -> tuple[float, int, float]:
    statistic = max(0.0, float(2.0 * (full_model.llf - reduced_model.llf)))
    df_difference = max(
        1, int(len(full_model.fe_params) - len(reduced_model.fe_params))
    )
    return statistic, df_difference, float(chi2.sf(statistic, df_difference))


def analyse_feature(
    master: pd.DataFrame,
    feature: str,
    environment_variables: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    data, environment_terms, model_to_original = prepare_model_data(
        master, feature, environment_variables
    )
    if len(data) < 20:
        raise RuntimeError(f"Only {len(data)} complete observations.")
    if data["physical_sample_id"].nunique() < 5:
        raise RuntimeError("Fewer than five physical specimens.")
    if data["stage"].nunique() < 2:
        raise RuntimeError("Fewer than two cleaning stages.")
    if data["scanner"].nunique() < 2:
        raise RuntimeError("Fewer than two scanners.")
    if float(data["value"].std()) <= 1e-12:
        raise RuntimeError("Feature has essentially no variance.")

    additive_formula = build_formula(True, True, False, environment_terms)
    additive, additive_optimizer = fit_mixed_model(additive_formula, data)
    no_stage, _ = fit_mixed_model(
        build_formula(False, True, False, environment_terms), data
    )
    no_scanner, _ = fit_mixed_model(
        build_formula(True, False, False, environment_terms), data
    )
    interaction, interaction_optimizer = fit_mixed_model(
        build_formula(True, True, True, environment_terms), data
    )

    stage_lr, stage_df, stage_p = likelihood_ratio_test(additive, no_stage)
    scanner_lr, scanner_df, scanner_p = likelihood_ratio_test(additive, no_scanner)
    interaction_lr, interaction_df, interaction_p = likelihood_ratio_test(
        interaction, additive
    )

    if environment_terms:
        no_environment, _ = fit_mixed_model(
            build_formula(True, True, False, []), data
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
            build_formula(True, True, False, reduced_terms), data
        )
        lr_statistic, lr_df, lr_p = likelihood_ratio_test(additive, reduced)
        environment_rows.append(
            {
                "feature": feature,
                "environment_variable": model_to_original[term],
                "environment_label": ENVIRONMENT_VARIABLES.get(
                    model_to_original[term], model_to_original[term]
                ),
                "standardized_beta": finite_float(additive.fe_params.get(term)),
                "standard_error": finite_float(additive.bse_fe.get(term)),
                "wald_p": finite_float(additive.pvalues.get(term)),
                "drop_one_lr_statistic": lr_statistic,
                "drop_one_lr_df": lr_df,
                "drop_one_p": lr_p,
                "n_observations": len(data),
                "n_environment_sessions": master.loc[data.index, [
                    "group", "timepoint_number"
                ]].drop_duplicates().shape[0],
            }
        )

    result = {
        "n_observations": len(data),
        "n_physical_samples": data["physical_sample_id"].nunique(),
        "n_scanners": data["scanner"].nunique(),
        "n_stages": data["stage"].nunique(),
        "mixedlm_optimizer_additive": additive_optimizer,
        "mixedlm_optimizer_interaction": interaction_optimizer,
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
    }
    return result, environment_rows


# ============================================================================
# DESCRIPTIVES AND CLASSIFICATION
# ============================================================================

def stage_descriptives(master: pd.DataFrame, feature: str) -> dict[str, float]:
    values = master[["stage", feature]].copy()
    values[feature] = pd.to_numeric(values[feature], errors="coerce")
    means = values.groupby("stage", observed=True)[feature].mean()
    medians = values.groupby("stage", observed=True)[feature].median()
    return {
        "baseline_mean": means.get("Baseline", np.nan),
        "partial_mean": means.get("Partial", np.nan),
        "clean_mean": means.get("Clean", np.nan),
        "baseline_median": medians.get("Baseline", np.nan),
        "partial_median": medians.get("Partial", np.nan),
        "clean_median": medians.get("Clean", np.nan),
    }


def clean_baseline_changes(
    master: pd.DataFrame, feature: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = master.loc[
        master["stage"].isin(["Baseline", "Clean"]),
        ["scanner", "physical_sample_id", "stage", feature],
    ].copy()
    selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    selected = selected.dropna(subset=[feature])
    wide = selected.pivot_table(
        index=["scanner", "physical_sample_id"],
        columns="stage",
        values=feature,
        aggfunc="mean",
        observed=True,
    )
    if "Baseline" not in wide or "Clean" not in wide:
        return {
            "n_clean_baseline_pairs": 0,
            "mean_clean_minus_baseline": np.nan,
            "median_clean_minus_baseline": np.nan,
            "direction_consistency": np.nan,
            "n_scanners_with_pairs": 0,
        }, []

    wide = wide.dropna(subset=["Baseline", "Clean"])
    wide["difference"] = wide["Clean"] - wide["Baseline"]
    scanner_rows = []
    scanner_medians = []
    for scanner in SCANNER_ORDER:
        if scanner in wide.index.get_level_values(0):
            values = wide.xs(scanner, level=0)["difference"].dropna().to_numpy(float)
        else:
            values = np.array([], dtype=float)
        median = float(np.median(values)) if len(values) else np.nan
        if np.isfinite(median):
            scanner_medians.append(median)
        scanner_rows.append(
            {
                "feature": feature,
                "scanner": scanner,
                "n_pairs": len(values),
                "mean_clean_minus_baseline": (
                    float(np.mean(values)) if len(values) else np.nan
                ),
                "median_clean_minus_baseline": median,
            }
        )

    pooled = wide["difference"].dropna().to_numpy(float)
    pooled_median = float(np.median(pooled)) if len(pooled) else np.nan
    if (
        len(scanner_medians) >= MIN_SCANNERS_FOR_DIRECTION
        and np.isfinite(pooled_median)
        and not np.isclose(pooled_median, 0.0)
    ):
        consistency = float(
            np.mean(np.sign(scanner_medians) == np.sign(pooled_median))
        )
    else:
        consistency = np.nan
    return {
        "n_clean_baseline_pairs": len(pooled),
        "mean_clean_minus_baseline": (
            float(np.mean(pooled)) if len(pooled) else np.nan
        ),
        "median_clean_minus_baseline": pooled_median,
        "direction_consistency": consistency,
        "n_scanners_with_pairs": len(scanner_medians),
    }, scanner_rows


def classify_feature(row: pd.Series) -> str:
    if row.get("feature_role") != "biological":
        return "TECHNICAL_QC_FEATURE"
    stage_p = finite_float(row.get("stage_p_holm"))
    scanner_p = finite_float(row.get("scanner_p_holm"))
    interaction_p = finite_float(row.get("interaction_p_holm"))
    consistency = finite_float(row.get("direction_consistency"))
    n_scanners = finite_float(row.get("n_scanners_with_pairs"))
    if not np.isfinite(stage_p):
        return "MODEL_FAILED"
    if stage_p < ALPHA:
        if np.isfinite(interaction_p) and interaction_p < ALPHA:
            return "STAGE_EFFECT_SCANNER_DEPENDENT"
        if (
            np.isfinite(consistency)
            and np.isfinite(n_scanners)
            and n_scanners >= MIN_SCANNERS_FOR_DIRECTION
            and consistency >= MIN_DIRECTION_CONSISTENCY
        ):
            return "ROBUST_CANDIDATE"
        return "STAGE_EFFECT_INCONSISTENT"
    if np.isfinite(scanner_p) and scanner_p < ALPHA:
        return "MAINLY_SCANNER_EFFECT"
    return "NO_CLEAR_STAGE_SIGNAL"


def scanner_adjustment_label(row: pd.Series) -> str:
    interaction_p = finite_float(row.get("interaction_p_holm"))
    scanner_p = finite_float(row.get("scanner_p_holm"))
    if np.isfinite(interaction_p) and interaction_p < ALPHA:
        return "scanner_changes_stage_response"
    if np.isfinite(scanner_p) and scanner_p < ALPHA:
        return "scanner_offset_present"
    return "no_clear_scanner_effect"


# ============================================================================
# FIGURES
# ============================================================================

def save_robustness_counts(biological: pd.DataFrame, output: Path) -> None:
    order = [
        "ROBUST_CANDIDATE",
        "STAGE_EFFECT_SCANNER_DEPENDENT",
        "STAGE_EFFECT_INCONSISTENT",
        "MAINLY_SCANNER_EFFECT",
        "NO_CLEAR_STAGE_SIGNAL",
        "MODEL_FAILED",
    ]
    counts = biological["robustness_class"].value_counts().reindex(order, fill_value=0)
    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(
        [CLASSIFICATION_LABELS[value] for value in order],
        counts.values,
        color=[CLASSIFICATION_COLORS[value] for value in order],
    )
    axis.set_ylabel("Number of biological features")
    axis.set_title("BioCal3D biological feature robustness classification")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            str(int(value)),
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_feature_overview(biological: pd.DataFrame, output: Path) -> None:
    data = biological.copy()
    data["stage_evidence"] = data["stage_p_holm"].map(negative_log10_p)
    data["_order"] = data["robustness_class"].map(CLASSIFICATION_ORDER)
    data = data.sort_values(
        ["_order", "stage_evidence"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)
    if data.empty:
        return
    figure, axis = plt.subplots(figsize=(14, max(8, 0.36 * len(data) + 2)))
    axis.axvline(
        -math.log10(ALPHA), color="black", linestyle="--", label="Holm p = 0.05"
    )
    for classification, subset in data.groupby("robustness_class", sort=False):
        axis.scatter(
            subset["stage_evidence"],
            subset.index,
            s=65,
            color=CLASSIFICATION_COLORS.get(classification, "#7F7F7F"),
            label=CLASSIFICATION_LABELS.get(classification, classification),
            zorder=3,
        )
    axis.set_yticks(np.arange(len(data)))
    axis.set_yticklabels([pretty_feature_name(value) for value in data["feature"]], fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("Stage evidence: −log10(Holm-adjusted p-value)")
    axis.set_title("BioCal3D biological feature robustness overview")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_evidence_matrix(biological: pd.DataFrame, output: Path) -> None:
    data = biological.copy()
    data["_order"] = data["robustness_class"].map(CLASSIFICATION_ORDER)
    data = data.sort_values(["_order", "stage_p_holm"], na_position="last")
    if data.empty:
        return
    raw = np.asarray(
        [
            [
                negative_log10_p(row.get("stage_p_holm")),
                negative_log10_p(row.get("scanner_p_holm")),
                negative_log10_p(row.get("interaction_p_holm")),
                negative_log10_p(row.get("environment_joint_p_holm")),
                finite_float(row.get("direction_consistency")),
            ]
            for _, row in data.iterrows()
        ],
        dtype=float,
    )
    matrix = raw.copy()
    matrix[:, :4] = np.clip(matrix[:, :4], 0, 10) / 10.0
    matrix[:, 4] = np.clip(matrix[:, 4], 0, 1)
    figure, axis = plt.subplots(figsize=(12, max(8, 0.36 * len(data) + 2)))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(np.arange(5))
    axis.set_xticklabels(
        [
            "Stage\nsignal",
            "Scanner\neffect",
            "Stage × scanner\ninteraction",
            "Environment\njoint effect",
            "Direction\nconsistency",
        ]
    )
    axis.set_yticks(np.arange(len(data)))
    axis.set_yticklabels([pretty_feature_name(value) for value in data["feature"]], fontsize=8.5)
    axis.set_title("BioCal3D evidence matrix\nBrighter = stronger evidence / consistency")
    display_columns = [
        "stage_p_holm",
        "scanner_p_holm",
        "interaction_p_holm",
        "environment_joint_p_holm",
    ]
    for row_index, (_, row) in enumerate(data.iterrows()):
        texts = []
        for column in display_columns:
            value = finite_float(row.get(column))
            texts.append(f"{value:.3g}" if np.isfinite(value) else "NA")
        consistency = finite_float(row.get("direction_consistency"))
        texts.append(f"{consistency:.0%}" if np.isfinite(consistency) else "NA")
        for column_index, label in enumerate(texts):
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row_index, column_index] < 0.45 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("Normalized evidence / consistency")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_direction_heatmap(
    biological: pd.DataFrame, scanner_changes: pd.DataFrame, output: Path
) -> None:
    candidates = biological[
        biological["robustness_class"].isin(
            [
                "ROBUST_CANDIDATE",
                "STAGE_EFFECT_SCANNER_DEPENDENT",
                "STAGE_EFFECT_INCONSISTENT",
            ]
        )
    ].sort_values("stage_p_holm")
    if candidates.empty:
        return
    features = candidates["feature"].tolist()
    matrix = np.full((len(features), len(SCANNER_ORDER)), np.nan)
    for row_index, feature in enumerate(features):
        subset = scanner_changes[scanner_changes["feature"] == feature]
        for column_index, scanner in enumerate(SCANNER_ORDER):
            values = subset.loc[
                subset["scanner"] == scanner, "median_clean_minus_baseline"
            ]
            if len(values):
                matrix[row_index, column_index] = finite_float(values.iloc[0])
        finite = np.abs(matrix[row_index, np.isfinite(matrix[row_index])])
        if len(finite) and finite.max() > 0:
            matrix[row_index] /= finite.max()
    figure, axis = plt.subplots(figsize=(10, max(7, 0.40 * len(features) + 2)))
    image = axis.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(SCANNER_ORDER)))
    axis.set_xticklabels(SCANNER_ORDER)
    axis.set_yticks(np.arange(len(features)))
    axis.set_yticklabels([pretty_feature_name(value) for value in features], fontsize=8.5)
    axis.set_title(
        "Clean − baseline direction by scanner\n(normalized separately within each feature)"
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Normalized clean − baseline change")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_environment_overview(environment_effects: pd.DataFrame, output: Path) -> None:
    biological = environment_effects[
        environment_effects["feature_role"] == "biological"
    ].copy()
    if biological.empty:
        return
    pivot = biological.pivot_table(
        index="feature",
        columns="environment_variable",
        values="drop_one_p_holm",
        aggfunc="first",
    )
    order = (
        biological.groupby("feature")["drop_one_p_holm"]
        .min()
        .sort_values()
        .index
    )
    pivot = pivot.reindex(order)
    columns = [column for column in ENVIRONMENT_VARIABLES if column in pivot]
    pivot = pivot[columns]
    if hasattr(pivot, "map"):
        evidence = pivot.map(negative_log10_p)
    else:  # pandas < 2.1
        evidence = pivot.applymap(negative_log10_p)
    matrix = evidence.to_numpy(float)
    matrix = np.clip(matrix, 0, 10)
    figure, axis = plt.subplots(figsize=(9, max(8, 0.36 * len(pivot) + 2)))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=10, cmap="magma")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels([ENVIRONMENT_VARIABLES[column] for column in columns])
    axis.set_yticks(np.arange(len(pivot)))
    axis.set_yticklabels([pretty_feature_name(value) for value in pivot.index], fontsize=8.5)
    axis.set_title("Environmental sensitivity of biological features")
    for row_index, feature in enumerate(pivot.index):
        for column_index, variable in enumerate(columns):
            p_value = finite_float(pivot.loc[feature, variable])
            axis.text(
                column_index,
                row_index,
                f"{p_value:.3g}" if np.isfinite(p_value) else "NA",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row_index, column_index] < 4.5 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("−log10(Holm-adjusted drop-one p-value), capped at 10")
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_root = args.output or data_root / "_feature_robustness_analysis"
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BIOCAL3D COMBINED FEATURE ROBUSTNESS ANALYSIS")
    print("=" * 80)

    if args.master_table is not None:
        master, feature_columns = load_existing_master(args.master_table)
        colour_path = None
        geometry_path = None
        print(f"Master input: {args.master_table}")
    else:
        colour_path = args.color_table or (
            data_root / "_raw_color_analysis" / "roi_color_summary.csv"
        )
        geometry_path = args.geometry_table or (
            data_root / "_geometry_analysis" / "tables" / "roi_geometry_features.csv"
        )
        require_file(colour_path, "ROI colour summary")
        require_file(geometry_path, "ROI geometry feature table")
        print(f"Colour input:   {colour_path}")
        print(f"Geometry input: {geometry_path}")
        master, feature_columns = build_master_table(
            pd.read_csv(colour_path),
            pd.read_csv(geometry_path),
            args.include_geometry_qc_failures,
        )

    environment_path = resolve_environment_path(data_root, args.environment_table)
    environment_variables: list[str] = []
    environment_sessions = pd.DataFrame()
    if environment_path is not None and environment_path.is_file():
        master, environment_variables, environment_sessions = merge_environment(
            master, environment_path
        )
        print(f"Environment input: {environment_path}")
        print("Environment variables: " + ", ".join(environment_variables))
    elif args.allow_no_environment:
        print("WARNING: running without environmental adjustment by explicit request.")
    else:
        expected = (
            data_root
            / "_raw_color_analysis"
            / "step_C_environment_effect"
            / "tables"
            / "environment_session_qc.csv"
        )
        raise FileNotFoundError(
            "Step C environmental output was not found. Run "
            "analyze_environment_effect.py first, provide --environment-table, "
            "or deliberately use --allow-no-environment. Expected: "
            f"{expected}"
        )

    if args.features is not None:
        missing = sorted(set(args.features) - set(feature_columns))
        if missing:
            raise ValueError("Requested features not found: " + ", ".join(missing))
        feature_columns = args.features

    master.to_csv(table_root / "master_feature_table.csv", index=False)
    if not environment_sessions.empty:
        environment_sessions.to_csv(
            table_root / "environment_sessions_used.csv", index=False
        )

    groups_present = sorted(master["group"].dropna().astype(int).unique())
    print("\nCurrent dataset:")
    print("  Groups present: " + ", ".join(f"G{value}" for value in groups_present))
    print(f"  Accepted scan/stage rows: {len(master)}")
    print(f"  Physical specimens: {master['physical_sample_id'].nunique()}")
    print(f"  Features selected: {len(feature_columns)}")

    result_rows = []
    scanner_change_rows = []
    environment_effect_rows = []

    for index, feature in enumerate(feature_columns, start=1):
        role = feature_role(feature)
        print(f"[{index:03d}/{len(feature_columns):03d}] {feature} ({role})")
        result = {
            "feature": feature,
            "feature_source": (
                "colour" if feature.startswith("color__") else "geometry"
            ),
            "feature_role": role,
            "model_status": "OK",
            "model_error": "",
        }
        result.update(stage_descriptives(master, feature))
        change_summary, scanner_rows = clean_baseline_changes(master, feature)
        result.update(change_summary)
        scanner_change_rows.extend(scanner_rows)
        try:
            model_result, environment_rows = analyse_feature(
                master, feature, environment_variables
            )
            result.update(model_result)
            for row in environment_rows:
                row["feature_source"] = result["feature_source"]
                row["feature_role"] = role
            environment_effect_rows.extend(environment_rows)
        except Exception as exc:
            result["model_status"] = "FAILED"
            result["model_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  MODEL FAILED: {exc}")
            for column in [
                "n_observations",
                "n_physical_samples",
                "n_scanners",
                "n_stages",
                "stage_lr_statistic",
                "stage_lr_df",
                "stage_p",
                "scanner_lr_statistic",
                "scanner_lr_df",
                "scanner_p",
                "interaction_lr_statistic",
                "interaction_lr_df",
                "interaction_p",
                "environment_joint_lr_statistic",
                "environment_joint_lr_df",
                "environment_joint_p",
            ]:
                result[column] = np.nan
        result_rows.append(result)

    results = pd.DataFrame(result_rows)
    scanner_changes = pd.DataFrame(scanner_change_rows)
    environment_effects = pd.DataFrame(environment_effect_rows)

    for column in [
        "stage_p_holm",
        "scanner_p_holm",
        "interaction_p_holm",
        "environment_joint_p_holm",
    ]:
        results[column] = np.nan
    p_pairs = [
        ("stage_p", "stage_p_holm"),
        ("scanner_p", "scanner_p_holm"),
        ("interaction_p", "interaction_p_holm"),
        ("environment_joint_p", "environment_joint_p_holm"),
    ]
    for role in ["biological", "technical"]:
        mask = results["feature_role"] == role
        for raw_column, adjusted_column in p_pairs:
            results.loc[mask, adjusted_column] = holm_adjust(
                results.loc[mask, raw_column]
            )

    if not environment_effects.empty:
        environment_effects["drop_one_p_holm"] = np.nan
        environment_effects["wald_p_holm"] = np.nan
        for (role, variable), indexes in environment_effects.groupby(
            ["feature_role", "environment_variable"]
        ).groups.items():
            environment_effects.loc[indexes, "drop_one_p_holm"] = holm_adjust(
                environment_effects.loc[indexes, "drop_one_p"]
            )
            environment_effects.loc[indexes, "wald_p_holm"] = holm_adjust(
                environment_effects.loc[indexes, "wald_p"]
            )
        environment_effects["environment_signal"] = np.where(
            environment_effects["drop_one_p_holm"] < ALPHA,
            "HOLM_SIGNIFICANT",
            "NO_CLEAR_EFFECT",
        )

    results["robustness_class"] = results.apply(classify_feature, axis=1)
    results["scanner_adjustment"] = results.apply(
        scanner_adjustment_label, axis=1
    )
    results["environment_adjustment"] = np.where(
        results["environment_joint_p_holm"] < ALPHA,
        "environment_association_detected",
        np.where(
            results["environment_joint_p_holm"].notna(),
            "no_clear_environment_association",
            "environment_not_tested",
        ),
    )
    results["_sort"] = results["robustness_class"].map(CLASSIFICATION_ORDER)
    results = results.sort_values(
        ["_sort", "stage_p_holm"], na_position="last"
    ).drop(columns="_sort")

    biological = results[results["feature_role"] == "biological"].copy()
    technical = results[results["feature_role"] == "technical"].copy()

    results.to_csv(table_root / "feature_robustness_summary.csv", index=False)
    biological.to_csv(
        table_root / "biological_feature_robustness_summary.csv", index=False
    )
    technical.to_csv(table_root / "technical_feature_summary.csv", index=False)
    scanner_changes.to_csv(
        table_root / "clean_vs_baseline_by_scanner.csv", index=False
    )
    environment_effects.to_csv(
        table_root / "environment_effects_by_feature.csv", index=False
    )

    save_robustness_counts(
        biological, figure_root / "biological_robustness_counts.png"
    )
    # Backward-compatible filename used by the previous script.
    save_robustness_counts(biological, figure_root / "robustness_class_counts.png")
    save_feature_overview(
        biological, figure_root / "biological_feature_robustness_overview.png"
    )
    save_evidence_matrix(
        biological, figure_root / "biological_feature_evidence_matrix.png"
    )
    save_direction_heatmap(
        biological,
        scanner_changes,
        figure_root / "clean_vs_baseline_direction_heatmap.png",
    )
    if not environment_effects.empty:
        save_environment_overview(
            environment_effects,
            figure_root / "environmental_effect_overview.png",
        )

    counts = biological["robustness_class"].value_counts()
    notes = f"""BIOCAL3D COMBINED FEATURE ROBUSTNESS ANALYSIS

Inputs
------
Colour: {colour_path if colour_path is not None else 'Loaded from master table'}
Geometry: {geometry_path if geometry_path is not None else 'Loaded from master table'}
Environment: {environment_path if environment_path is not None else 'Not included'}
Environment variables: {', '.join(environment_variables) if environment_variables else 'None'}

Current dataset
---------------
Groups present: {', '.join(f'G{value}' for value in groups_present)}
Accepted scan/stage rows: {len(master)}
Physical specimens: {master['physical_sample_id'].nunique()}
Biological candidate features: {len(biological)}
Technical/QC features: {len(technical)}

Primary mixed model
-------------------
feature ~ stage + scanner + group + light + humidity + temperature
          + random physical-specimen intercept

Scanner-dependence model
------------------------
feature ~ stage * scanner + group + light + humidity + temperature
          + random physical-specimen intercept

Environmental results
---------------------
The joint environment likelihood-ratio test is stored in the main feature
summary. Standardized coefficients and drop-one tests for light, humidity and
temperature are stored in environment_effects_by_feature.csv. Holm correction
is performed separately for each environmental variable and feature role.

Important limitation
--------------------
Environmental measurements are session-level (group x timepoint), not
specimen-level. Their associations are exploratory and may be partly confounded
with group and cleaning stage. Do not interpret repeated specimen/scanner rows
as independent environmental measurements.

Biological classification
-------------------------
ROBUST_CANDIDATE: adjusted stage p < {ALPHA}, no adjusted stage x scanner
interaction, and at least {MIN_DIRECTION_CONSISTENCY:.0%} of scanners have the
same clean-minus-baseline direction.

Technical mesh/acquisition variables are analysed separately and can never be
labelled robust biological candidates.

Results
-------
Robust candidates: {int(counts.get('ROBUST_CANDIDATE', 0))}
Stage effect, scanner-dependent: {int(counts.get('STAGE_EFFECT_SCANNER_DEPENDENT', 0))}
Stage effect, inconsistent: {int(counts.get('STAGE_EFFECT_INCONSISTENT', 0))}
Mainly scanner effect: {int(counts.get('MAINLY_SCANNER_EFFECT', 0))}
No clear stage signal: {int(counts.get('NO_CLEAR_STAGE_SIGNAL', 0))}
Failed biological models: {int(counts.get('MODEL_FAILED', 0))}
"""
    (output_root / "README_results.txt").write_text(notes, encoding="utf-8")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nBiological feature classification:")
    print(counts.to_string())
    print(f"\nOutput: {output_root}")
    print("Start with:")
    print(f"  {table_root / 'biological_feature_robustness_summary.csv'}")
    print(f"  {table_root / 'environment_effects_by_feature.csv'}")
    print(f"  {figure_root / 'biological_feature_robustness_overview.png'}")
    print(f"  {figure_root / 'environmental_effect_overview.png'}")


if __name__ == "__main__":
    main()
