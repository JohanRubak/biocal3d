"""Step C: environmental sensitivity and confounding analysis.

Inputs
------
1. _raw_color_analysis/roi_color_summary.csv
2. Environment information.xlsx
3. Step B tables (when available):
   - individual_stage_changes.csv
   - individual_stage_deltaE00.csv

The environmental variables are measured at session level. Consequently,
associations are calculated from session-level or transition-level summaries,
not by treating every specimen as an independent environmental observation.
"""

from itertools import combinations
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


# ============================================================
# DATA LOCATION
# ============================================================

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected"
)

# The environment variable is optional and mainly useful for testing or when
# the data is moved. If it is not set, the same path as Steps A and B is used.
DATA_ROOT = Path(os.environ.get("BIOCAL3D_DATA_ROOT", DEFAULT_DATA_ROOT))

RAW_ANALYSIS_ROOT = DATA_ROOT / "_raw_color_analysis"

ROI_INPUT_FILE = RAW_ANALYSIS_ROOT / "roi_color_summary.csv"
ENVIRONMENT_FILE = DATA_ROOT / "Environment information.xlsx"

STEP_B_TABLE_ROOT = (
    RAW_ANALYSIS_ROOT
    / "step_B_cleaning_effect"
    / "tables"
)

STAGE_CHANGE_FILE = STEP_B_TABLE_ROOT / "individual_stage_changes.csv"
STAGE_DELTA_E_FILE = STEP_B_TABLE_ROOT / "individual_stage_deltaE00.csv"

OUTPUT_ROOT = RAW_ANALYSIS_ROOT / "step_C_environment_effect"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"

for folder in [OUTPUT_ROOT, FIGURE_ROOT, TABLE_ROOT]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SCANNER_ORDER = [
    "LABscanner",
    "iTERO",
    "TRIOS3",
    "TRIOS5",
]

SCANNER_COLORS = {
    "LABscanner": "#4C78A8",
    "iTERO": "#F58518",
    "TRIOS3": "#54A24B",
    "TRIOS5": "#E45756",
}

GROUP_STAGE_MAP = {
    1: {0: "Baseline", 1: "Clean"},
    2: {0: "Baseline", 1: "Clean"},
    3: {0: "Baseline", 1: "Partial", 2: "Clean"},
    4: {0: "Baseline", 1: "Partial", 2: "Clean"},
    5: {0: "Baseline", 1: "Partial", 2: "Clean"},
}

COLOR_METRICS = {
    "median_L": "Median L*",
    "median_a": "Median a*",
    "median_b": "Median b*",
    "median_chroma": "Median C*ab",
}

ENVIRONMENT_VARIABLES = {
    "light_avg": "Ambient light",
    "humidity_avg": "Relative humidity (%)",
    "temperature_avg": "Temperature (°C)",
}

ENVIRONMENT_CHANGE_VARIABLES = {
    "delta_light": "Change in ambient light",
    "delta_humidity": "Change in relative humidity (% points)",
    "delta_temperature": "Change in temperature (°C)",
}

CSV_KWARGS = {
    "index": False,
    "encoding": "utf-8-sig",
}


# ============================================================
# HELPERS
# ============================================================


def require_columns(dataframe, columns, source_name):
    """Raise a readable error when an expected input column is absent."""

    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns:\n"
            + "\n".join(missing)
        )


def ciede2000(lab_A, lab_B):
    """Calculate scalar CIEDE2000 for two CIELAB triplets.

    This follows the Sharma et al. implementation notes with kL = kC = kH = 1.
    """

    L1, a1, b1 = map(float, lab_A)
    L2, a2, b2 = map(float, lab_B)

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (
        1.0
        - np.sqrt(
            C_bar**7
            / (C_bar**7 + 25.0**7)
        )
    )

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2
    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    if C1_prime == 0:
        h1_prime = 0.0
    if C2_prime == 0:
        h2_prime = 0.0

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    hue_difference = h2_prime - h1_prime
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0.0
    elif abs(hue_difference) <= 180.0:
        delta_h_prime = hue_difference
    elif hue_difference > 180.0:
        delta_h_prime = hue_difference - 360.0
    else:
        delta_h_prime = hue_difference + 360.0

    delta_H_prime = (
        2.0
        * np.sqrt(C1_prime * C2_prime)
        * np.sin(np.radians(delta_h_prime / 2.0))
    )

    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        h_bar_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360.0:
        h_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        h_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )

    delta_theta = 30.0 * np.exp(
        -((h_bar_prime - 275.0) / 25.0) ** 2
    )
    R_C = 2.0 * np.sqrt(
        C_bar_prime**7
        / (C_bar_prime**7 + 25.0**7)
    )
    S_L = 1.0 + (
        0.015 * (L_bar_prime - 50.0) ** 2
        / np.sqrt(20.0 + (L_bar_prime - 50.0) ** 2)
    )
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    L_term = delta_L_prime / S_L
    C_term = delta_C_prime / S_C
    H_term = delta_H_prime / S_H

    return float(
        np.sqrt(
            L_term**2
            + C_term**2
            + H_term**2
            + R_T * C_term * H_term
        )
    )


def safe_name(text):
    """Create a conservative filename fragment."""

    return (
        str(text)
        .replace("*", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("°", "deg")
        .replace("→", "to")
    )


def stage_to_timepoint(group, stage):
    """Convert a biological stage label back to its timepoint number."""

    mapping = GROUP_STAGE_MAP.get(int(group), {})
    reverse_mapping = {label: timepoint for timepoint, label in mapping.items()}

    if stage in reverse_mapping:
        return reverse_mapping[stage]

    stage_string = str(stage).strip()
    if stage_string.upper().startswith("T"):
        try:
            return int(stage_string[1:])
        except ValueError:
            pass

    return np.nan


def association_row(x, y, base_information):
    """Return descriptive linear, Pearson, and Spearman associations.

    The caller must first aggregate to the correct independent unit.
    """

    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    n = len(pair)
    n_unique_x = pair["x"].nunique()
    n_unique_y = pair["y"].nunique()

    result = {
        **base_information,
        "n_independent_conditions": n,
        "n_unique_environment_values": n_unique_x,
        "x_min": pair["x"].min() if n else np.nan,
        "x_max": pair["x"].max() if n else np.nan,
        "y_min": pair["y"].min() if n else np.nan,
        "y_max": pair["y"].max() if n else np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "evidence_flag": (
            "very limited: fewer than 6 independent conditions"
            if n < 6
            else "exploratory: no multiplicity correction"
        ),
    }

    if n >= 3 and n_unique_x >= 2 and n_unique_y >= 2:
        regression = linregress(pair["x"], pair["y"])
        pearson = pearsonr(pair["x"], pair["y"])
        spearman = spearmanr(pair["x"], pair["y"])

        result.update(
            {
                "slope": regression.slope,
                "intercept": regression.intercept,
                "r_squared": regression.rvalue**2,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )

    return result


def add_environment_to_transitions(dataframe, environment):
    """Attach session A/B conditions and their differences to Step B rows."""

    transition_df = dataframe.copy()
    transition_df["group"] = pd.to_numeric(
        transition_df["group"], errors="raise"
    ).astype(int)

    transition_df["timepoint_A"] = [
        stage_to_timepoint(group, stage)
        for group, stage in zip(
            transition_df["group"], transition_df["stage_A"]
        )
    ]
    transition_df["timepoint_B"] = [
        stage_to_timepoint(group, stage)
        for group, stage in zip(
            transition_df["group"], transition_df["stage_B"]
        )
    ]

    transition_df["session_A"] = [
        f"BCG{group}T{int(timepoint)}" if pd.notna(timepoint) else pd.NA
        for group, timepoint in zip(
            transition_df["group"], transition_df["timepoint_A"]
        )
    ]
    transition_df["session_B"] = [
        f"BCG{group}T{int(timepoint)}" if pd.notna(timepoint) else pd.NA
        for group, timepoint in zip(
            transition_df["group"], transition_df["timepoint_B"]
        )
    ]

    environment_columns = [
        "session_id",
        "light_avg",
        "humidity_avg",
        "temperature_avg",
        "all_calibrations_ok",
    ]

    environment_A = environment[environment_columns].rename(
        columns={column: f"{column}_A" for column in environment_columns}
    )
    environment_B = environment[environment_columns].rename(
        columns={column: f"{column}_B" for column in environment_columns}
    )

    transition_df = transition_df.merge(
        environment_A,
        left_on="session_A",
        right_on="session_id_A",
        how="left",
        validate="many_to_one",
    )
    transition_df = transition_df.merge(
        environment_B,
        left_on="session_B",
        right_on="session_id_B",
        how="left",
        validate="many_to_one",
    )

    transition_df["delta_light"] = (
        transition_df["light_avg_B"] - transition_df["light_avg_A"]
    )
    transition_df["delta_humidity"] = (
        transition_df["humidity_avg_B"] - transition_df["humidity_avg_A"]
    )
    transition_df["delta_temperature"] = (
        transition_df["temperature_avg_B"]
        - transition_df["temperature_avg_A"]
    )
    transition_df["both_calibrations_ok"] = (
        transition_df["all_calibrations_ok_A"].eq(True)
        & transition_df["all_calibrations_ok_B"].eq(True)
    )

    return transition_df


def save_scatter_plot(
    data,
    x_column,
    y_column,
    group_column,
    x_label,
    y_label,
    title,
    output_file,
    label_column=None,
):
    """Create a grouped scatter plot with exploratory fitted lines."""

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    groups = [
        value for value in SCANNER_ORDER if value in data[group_column].values
    ]
    groups += sorted(
        set(data[group_column].dropna().astype(str)) - set(groups)
    )

    fallback_palette = plt.get_cmap("tab10")

    for group_index, group in enumerate(groups):
        subset = data[data[group_column].astype(str) == str(group)].copy()
        subset[x_column] = pd.to_numeric(subset[x_column], errors="coerce")
        subset[y_column] = pd.to_numeric(subset[y_column], errors="coerce")
        subset = subset.dropna(subset=[x_column, y_column])

        if subset.empty:
            continue

        color = SCANNER_COLORS.get(
            str(group), fallback_palette(group_index % 10)
        )
        ax.scatter(
            subset[x_column],
            subset[y_column],
            label=str(group),
            color=color,
            s=46,
            alpha=0.85,
        )

        if subset[x_column].nunique() >= 2 and len(subset) >= 3:
            fit = linregress(subset[x_column], subset[y_column])
            x_line = np.linspace(
                subset[x_column].min(), subset[x_column].max(), 100
            )
            ax.plot(
                x_line,
                fit.intercept + fit.slope * x_line,
                color=color,
                linewidth=1.4,
                alpha=0.75,
            )

        if label_column is not None and label_column in subset.columns:
            for _, row in subset.iterrows():
                ax.annotate(
                    str(row[label_column]),
                    (row[x_column], row[y_column]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                    alpha=0.75,
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LOAD AND PREPARE ENVIRONMENTAL DATA
# ============================================================


def load_environment_file(path):
    """Read the current 16-column environmental workbook layout."""

    raw = pd.read_excel(path, sheet_name=0)

    if raw.shape[1] < 16:
        raise ValueError(
            "Environment information.xlsx must contain the session column, "
            "three readings plus an average for each environmental variable, "
            "and the three calibration columns (at least 16 columns total)."
        )

    environment = pd.DataFrame(
        {
            "session_id": raw.iloc[:, 0].astype("string").str.strip(),
            "light_1": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "light_2": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "light_3": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "light_avg_reported": pd.to_numeric(
                raw.iloc[:, 4], errors="coerce"
            ),
            "humidity_1": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
            "humidity_2": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
            "humidity_3": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
            "humidity_avg_reported": pd.to_numeric(
                raw.iloc[:, 8], errors="coerce"
            ),
            "temperature_1": pd.to_numeric(
                raw.iloc[:, 9], errors="coerce"
            ),
            "temperature_2": pd.to_numeric(
                raw.iloc[:, 10], errors="coerce"
            ),
            "temperature_3": pd.to_numeric(
                raw.iloc[:, 11], errors="coerce"
            ),
            "temperature_avg_reported": pd.to_numeric(
                raw.iloc[:, 12], errors="coerce"
            ),
            "calibration_prime": raw.iloc[:, 13].astype("string").str.strip(),
            "calibration_trios": raw.iloc[:, 14].astype("string").str.strip(),
            "calibration_lab": raw.iloc[:, 15].astype("string").str.strip(),
        }
    )

    environment = environment[
        environment["session_id"].notna()
        & environment["session_id"].str.match(r"^BCG\d+T\d+$", na=False)
    ].copy()

    if environment["session_id"].duplicated().any():
        duplicates = environment.loc[
            environment["session_id"].duplicated(keep=False), "session_id"
        ].tolist()
        raise ValueError(
            "Duplicate environmental session IDs found: "
            + ", ".join(map(str, duplicates))
        )

    extracted = environment["session_id"].str.extract(
        r"^BCG(?P<group>\d+)T(?P<timepoint_number>\d+)$"
    )
    environment["group"] = extracted["group"].astype(int)
    environment["timepoint_number"] = extracted["timepoint_number"].astype(int)

    for variable in ["light", "humidity", "temperature"]:
        repeat_columns = [f"{variable}_{number}" for number in range(1, 4)]
        environment[f"{variable}_avg_calculated"] = environment[
            repeat_columns
        ].mean(axis=1)
        environment[f"{variable}_sd"] = environment[repeat_columns].std(
            axis=1, ddof=1
        )
        environment[f"{variable}_range"] = (
            environment[repeat_columns].max(axis=1)
            - environment[repeat_columns].min(axis=1)
        )
        environment[f"{variable}_average_difference"] = (
            environment[f"{variable}_avg_reported"]
            - environment[f"{variable}_avg_calculated"]
        )
        environment[f"{variable}_avg"] = environment[
            f"{variable}_avg_reported"
        ].fillna(environment[f"{variable}_avg_calculated"])

    calibration_columns = [
        "calibration_prime",
        "calibration_trios",
        "calibration_lab",
    ]
    environment["all_calibrations_ok"] = environment[
        calibration_columns
    ].apply(
        lambda row: all(str(value).strip().lower() == "ok" for value in row),
        axis=1,
    )

    return environment.sort_values(
        ["group", "timepoint_number"]
    ).reset_index(drop=True)


print("=" * 80)
print("STEP C — ENVIRONMENTAL SENSITIVITY / CONFOUNDING ANALYSIS")
print("=" * 80)

environment_df = load_environment_file(ENVIRONMENT_FILE)
environment_df.to_csv(
    TABLE_ROOT / "environment_session_qc.csv",
    **CSV_KWARGS,
)

print(f"\nEnvironmental sessions loaded: {len(environment_df)}")
print(
    "Environmental session IDs: "
    + ", ".join(environment_df["session_id"].astype(str))
)


# Environmental-variable correlation uses only unique sessions.
environment_correlation_rows = []
for variable_A, variable_B in combinations(ENVIRONMENT_VARIABLES, 2):
    environment_correlation_rows.append(
        association_row(
            environment_df[variable_A],
            environment_df[variable_B],
            {
                "variable_A": variable_A,
                "variable_B": variable_B,
            },
        )
    )

pd.DataFrame(environment_correlation_rows).to_csv(
    TABLE_ROOT / "environment_variable_correlations.csv",
    **CSV_KWARGS,
)


# ============================================================
# ABSOLUTE COLOR ENVIRONMENTAL SENSITIVITY
# ============================================================

roi_df = pd.read_csv(ROI_INPUT_FILE)

require_columns(
    roi_df,
    [
        "scanner",
        "sample_id",
        "physical_sample_id",
        "group",
        "timepoint_number",
        *COLOR_METRICS.keys(),
    ],
    ROI_INPUT_FILE.name,
)

roi_df["group"] = pd.to_numeric(roi_df["group"], errors="raise").astype(int)
roi_df["timepoint_number"] = pd.to_numeric(
    roi_df["timepoint_number"], errors="raise"
).astype(int)
roi_df = roi_df[roi_df["scanner"].isin(SCANNER_ORDER)].copy()
roi_df["session_id"] = (
    "BCG"
    + roi_df["group"].astype(str)
    + "T"
    + roi_df["timepoint_number"].astype(str)
)

duplicate_mask = roi_df.duplicated(["sample_id", "scanner"], keep=False)
if duplicate_mask.any():
    duplicate_file = TABLE_ROOT / "duplicate_sample_scanner_rows.csv"
    roi_df.loc[duplicate_mask].sort_values(
        ["sample_id", "scanner"]
    ).to_csv(duplicate_file, **CSV_KWARGS)
    raise ValueError(
        "Duplicate sample_id × scanner rows were found. They were saved to "
        f"{duplicate_file}. Resolve them before running Step C."
    )

roi_environment_df = roi_df.merge(
    environment_df,
    on=["session_id", "group", "timepoint_number"],
    how="left",
    validate="many_to_one",
)

roi_environment_df.to_csv(
    TABLE_ROOT / "roi_color_with_environment.csv",
    **CSV_KWARGS,
)

unmatched_roi_sessions = (
    roi_environment_df.loc[
        roi_environment_df["light_avg"].isna(),
        ["session_id", "group", "timepoint_number"],
    ]
    .drop_duplicates()
    .sort_values(["group", "timepoint_number"])
)
unmatched_roi_sessions.to_csv(
    TABLE_ROOT / "roi_sessions_without_environment.csv",
    **CSV_KWARGS,
)

aggregation = {
    "sample_id": "nunique",
    **{metric: "median" for metric in COLOR_METRICS},
    **{variable: "first" for variable in ENVIRONMENT_VARIABLES},
    "all_calibrations_ok": "first",
}

scanner_session_df = (
    roi_environment_df.dropna(subset=list(ENVIRONMENT_VARIABLES))
    .groupby(
        ["scanner", "session_id", "group", "timepoint_number"],
        observed=True,
        as_index=False,
    )
    .agg(aggregation)
    .rename(columns={"sample_id": "n_scans"})
)

scanner_session_df.to_csv(
    TABLE_ROOT / "scanner_session_color_summary.csv",
    **CSV_KWARGS,
)

absolute_association_rows = []
for scanner, scanner_data in scanner_session_df.groupby(
    "scanner", observed=True
):
    for metric, metric_label in COLOR_METRICS.items():
        for variable, variable_label in ENVIRONMENT_VARIABLES.items():
            absolute_association_rows.append(
                association_row(
                    scanner_data[variable],
                    scanner_data[metric],
                    {
                        "analysis": "absolute scanner color",
                        "scanner": scanner,
                        "outcome": metric,
                        "outcome_label": metric_label,
                        "environment_variable": variable,
                        "environment_label": variable_label,
                        "independent_unit": "scanner × acquisition session",
                    },
                )
            )

for metric, metric_label in COLOR_METRICS.items():
    for variable, variable_label in ENVIRONMENT_VARIABLES.items():
        save_scatter_plot(
            scanner_session_df,
            variable,
            metric,
            "scanner",
            variable_label,
            metric_label,
            f"{metric_label} versus {variable_label}",
            FIGURE_ROOT
            / f"absolute_{safe_name(metric)}_vs_{safe_name(variable)}.png",
            label_column="session_id",
        )

absolute_association_df = pd.DataFrame(absolute_association_rows)
absolute_association_df.to_csv(
    TABLE_ROOT / "absolute_color_environment_associations.csv",
    **CSV_KWARGS,
)


# ============================================================
# MATCHED INTER-SCANNER DELTA E00
# ============================================================

scanner_pair_rows = []

for sample_id, sample_data in roi_environment_df.groupby(
    "sample_id", observed=True
):
    sample_data = sample_data.dropna(
        subset=["median_L", "median_a", "median_b"]
    ).copy()

    available_scanners = [
        scanner
        for scanner in SCANNER_ORDER
        if scanner in sample_data["scanner"].values
    ]

    for scanner_A, scanner_B in combinations(available_scanners, 2):
        row_A = sample_data.loc[sample_data["scanner"] == scanner_A].iloc[0]
        row_B = sample_data.loc[sample_data["scanner"] == scanner_B].iloc[0]

        lab_A = np.array(
            [row_A["median_L"], row_A["median_a"], row_A["median_b"]],
            dtype=float,
        )
        lab_B = np.array(
            [row_B["median_L"], row_B["median_a"], row_B["median_b"]],
            dtype=float,
        )

        delta_e = ciede2000(lab_A, lab_B)

        scanner_pair_rows.append(
            {
                "sample_id": sample_id,
                "physical_sample_id": row_A["physical_sample_id"],
                "group": row_A["group"],
                "timepoint_number": row_A["timepoint_number"],
                "session_id": row_A["session_id"],
                "scanner_A": scanner_A,
                "scanner_B": scanner_B,
                "scanner_pair": f"{scanner_A} vs {scanner_B}",
                "deltaE00": delta_e,
                "light_avg": row_A["light_avg"],
                "humidity_avg": row_A["humidity_avg"],
                "temperature_avg": row_A["temperature_avg"],
                "all_calibrations_ok": row_A["all_calibrations_ok"],
            }
        )

scanner_pair_sample_df = pd.DataFrame(scanner_pair_rows)
scanner_pair_sample_df.to_csv(
    TABLE_ROOT / "scanner_pair_sample_deltaE00.csv",
    **CSV_KWARGS,
)

if not scanner_pair_sample_df.empty:
    scanner_pair_session_df = (
        scanner_pair_sample_df.dropna(subset=list(ENVIRONMENT_VARIABLES))
        .groupby(
            [
                "scanner_A",
                "scanner_B",
                "scanner_pair",
                "session_id",
                "group",
                "timepoint_number",
            ],
            as_index=False,
        )
        .agg(
            n_pairs=("sample_id", "nunique"),
            median_deltaE00=("deltaE00", "median"),
            mean_deltaE00=("deltaE00", "mean"),
            sd_deltaE00=("deltaE00", "std"),
            light_avg=("light_avg", "first"),
            humidity_avg=("humidity_avg", "first"),
            temperature_avg=("temperature_avg", "first"),
            all_calibrations_ok=("all_calibrations_ok", "first"),
        )
    )
else:
    scanner_pair_session_df = pd.DataFrame()

scanner_pair_session_df.to_csv(
    TABLE_ROOT / "scanner_pair_session_deltaE00_summary.csv",
    **CSV_KWARGS,
)

scanner_pair_association_rows = []
if not scanner_pair_session_df.empty:
    for scanner_pair, pair_data in scanner_pair_session_df.groupby(
        "scanner_pair"
    ):
        for variable, variable_label in ENVIRONMENT_VARIABLES.items():
            scanner_pair_association_rows.append(
                association_row(
                    pair_data[variable],
                    pair_data["median_deltaE00"],
                    {
                        "analysis": "matched inter-scanner disagreement",
                        "scanner_pair": scanner_pair,
                        "outcome": "median_deltaE00",
                        "outcome_label": "Median paired CIEDE2000 ΔE00",
                        "environment_variable": variable,
                        "environment_label": variable_label,
                        "independent_unit": "scanner pair × acquisition session",
                    },
                )
            )

    for variable, variable_label in ENVIRONMENT_VARIABLES.items():
        save_scatter_plot(
            scanner_pair_session_df,
            variable,
            "median_deltaE00",
            "scanner_pair",
            variable_label,
            "Median paired CIEDE2000 ΔE00",
            f"Inter-scanner disagreement versus {variable_label}",
            FIGURE_ROOT
            / f"scanner_pair_deltaE00_vs_{safe_name(variable)}.png",
            label_column="session_id",
        )

pd.DataFrame(scanner_pair_association_rows).to_csv(
    TABLE_ROOT / "scanner_pair_environment_associations.csv",
    **CSV_KWARGS,
)


# ============================================================
# CLEANING-STAGE CHANGE VERSUS ENVIRONMENTAL CHANGE
# ============================================================

transition_environment_summary_frames = []

if STAGE_CHANGE_FILE.exists():
    individual_stage_change_df = pd.read_csv(STAGE_CHANGE_FILE)
    require_columns(
        individual_stage_change_df,
        [
            "group",
            "scanner",
            "physical_sample_id",
            "metric",
            "stage_A",
            "stage_B",
            "difference_B_minus_A",
        ],
        STAGE_CHANGE_FILE.name,
    )

    if "metric_label" not in individual_stage_change_df.columns:
        individual_stage_change_df["metric_label"] = (
            individual_stage_change_df["metric"].map(COLOR_METRICS)
            .fillna(individual_stage_change_df["metric"])
        )

    stage_change_environment_df = add_environment_to_transitions(
        individual_stage_change_df,
        environment_df,
    )
    stage_change_environment_df.to_csv(
        TABLE_ROOT / "individual_stage_changes_with_environment.csv",
        **CSV_KWARGS,
    )

    stage_change_summary_df = (
        stage_change_environment_df.dropna(
            subset=["delta_light", "delta_humidity", "delta_temperature"]
        )
        .groupby(
            [
                "group",
                "scanner",
                "metric",
                "metric_label",
                "stage_A",
                "stage_B",
                "session_A",
                "session_B",
                "delta_light",
                "delta_humidity",
                "delta_temperature",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_samples=("physical_sample_id", "nunique"),
            median_color_change=("difference_B_minus_A", "median"),
            mean_color_change=("difference_B_minus_A", "mean"),
            sd_color_change=("difference_B_minus_A", "std"),
        )
    )
    stage_change_summary_df["outcome_type"] = "Lab/chroma change"
    stage_change_summary_df.to_csv(
        TABLE_ROOT / "cleaning_transition_color_change_summary.csv",
        **CSV_KWARGS,
    )
    transition_environment_summary_frames.append(stage_change_summary_df)

    stage_change_association_rows = []
    for (scanner, metric), subset in stage_change_summary_df.groupby(
        ["scanner", "metric"], observed=True
    ):
        metric_label = COLOR_METRICS.get(metric, metric)
        for variable, variable_label in ENVIRONMENT_CHANGE_VARIABLES.items():
            stage_change_association_rows.append(
                association_row(
                    subset[variable],
                    subset["median_color_change"],
                    {
                        "analysis": "cleaning color change versus environmental change",
                        "scanner": scanner,
                        "outcome": metric,
                        "outcome_label": metric_label,
                        "environment_change_variable": variable,
                        "environment_change_label": variable_label,
                        "independent_unit": "group × stage transition × scanner",
                    },
                )
            )

    pd.DataFrame(stage_change_association_rows).to_csv(
        TABLE_ROOT / "cleaning_color_change_environment_associations.csv",
        **CSV_KWARGS,
    )
else:
    print(
        f"\nStep B file not found; skipping metric-change merge:\n{STAGE_CHANGE_FILE}"
    )


if STAGE_DELTA_E_FILE.exists():
    individual_stage_delta_e_df = pd.read_csv(STAGE_DELTA_E_FILE)
    require_columns(
        individual_stage_delta_e_df,
        [
            "group",
            "scanner",
            "physical_sample_id",
            "stage_A",
            "stage_B",
            "deltaE00",
        ],
        STAGE_DELTA_E_FILE.name,
    )

    stage_delta_e_environment_df = add_environment_to_transitions(
        individual_stage_delta_e_df,
        environment_df,
    )
    stage_delta_e_environment_df.to_csv(
        TABLE_ROOT / "individual_stage_deltaE00_with_environment.csv",
        **CSV_KWARGS,
    )

    stage_delta_e_summary_df = (
        stage_delta_e_environment_df.dropna(
            subset=["delta_light", "delta_humidity", "delta_temperature"]
        )
        .groupby(
            [
                "group",
                "scanner",
                "stage_A",
                "stage_B",
                "session_A",
                "session_B",
                "delta_light",
                "delta_humidity",
                "delta_temperature",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_samples=("physical_sample_id", "nunique"),
            median_deltaE00=("deltaE00", "median"),
            mean_deltaE00=("deltaE00", "mean"),
            sd_deltaE00=("deltaE00", "std"),
        )
    )
    stage_delta_e_summary_df["outcome_type"] = "Cleaning-stage ΔE00"
    stage_delta_e_summary_df.to_csv(
        TABLE_ROOT / "cleaning_transition_deltaE00_summary.csv",
        **CSV_KWARGS,
    )

    stage_delta_e_association_rows = []
    for scanner, subset in stage_delta_e_summary_df.groupby(
        "scanner", observed=True
    ):
        for variable, variable_label in ENVIRONMENT_CHANGE_VARIABLES.items():
            stage_delta_e_association_rows.append(
                association_row(
                    subset[variable],
                    subset["median_deltaE00"],
                    {
                        "analysis": "cleaning ΔE00 versus environmental change",
                        "scanner": scanner,
                        "outcome": "median_deltaE00",
                        "outcome_label": "Median cleaning-stage CIEDE2000 ΔE00",
                        "environment_change_variable": variable,
                        "environment_change_label": variable_label,
                        "independent_unit": "group × stage transition × scanner",
                    },
                )
            )

    for variable, variable_label in ENVIRONMENT_CHANGE_VARIABLES.items():
        save_scatter_plot(
            stage_delta_e_summary_df,
            variable,
            "median_deltaE00",
            "scanner",
            variable_label,
            "Median cleaning-stage CIEDE2000 ΔE00",
            f"Cleaning-stage ΔE00 versus {variable_label}",
            FIGURE_ROOT
            / f"cleaning_deltaE00_vs_{safe_name(variable)}.png",
        )

    pd.DataFrame(stage_delta_e_association_rows).to_csv(
        TABLE_ROOT / "cleaning_deltaE00_environment_associations.csv",
        **CSV_KWARGS,
    )
else:
    print(
        f"\nStep B file not found; skipping ΔE00-change merge:\n{STAGE_DELTA_E_FILE}"
    )


# ============================================================
# SAVE METHOD NOTES AND FINISH
# ============================================================

notes = """STEP C — ENVIRONMENTAL ANALYSIS NOTES

1. Environmental averages are used as predictors. The three raw readings are
   retained in environment_session_qc.csv to assess measurement stability.
2. Calibration status is treated as quality-control metadata, not a predictor.
3. Absolute-color associations use scanner × acquisition session summaries.
4. Inter-scanner ΔE00 is calculated internally from exactly matched sample IDs,
   then summarized by scanner pair × acquisition session before association.
5. Cleaning analyses read individual outputs from Step B, add the environmental
   change between stages, and aggregate by group × stage transition × scanner.
6. Current environmental analyses are exploratory. There are very few unique
   sessions/transitions, environmental factors overlap with cleaning stage, and
   some transitions within a group overlap. Do not interpret specimen count as
   the environmental sample size or use these results to normalize color yet.
7. Pearson/Spearman p-values are descriptive and are not multiplicity-adjusted.
"""

(OUTPUT_ROOT / "README.txt").write_text(notes, encoding="utf-8")

print("\n" + "=" * 80)
print("STEP C COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to:\n{OUTPUT_ROOT}")
print(
    "\nPrimary tables:\n"
    "- environment_session_qc.csv\n"
    "- scanner_session_color_summary.csv\n"
    "- absolute_color_environment_associations.csv\n"
    "- scanner_pair_session_deltaE00_summary.csv\n"
    "- scanner_pair_environment_associations.csv\n"
    "- cleaning_transition_color_change_summary.csv (when Step B exists)\n"
    "- cleaning_transition_deltaE00_summary.csv (when Step B exists)"
)
print(
    "\nInterpret all correlations as exploratory because the independent "
    "environmental sample size is the number of sessions/transitions, not "
    "the number of scanned specimens."
)
