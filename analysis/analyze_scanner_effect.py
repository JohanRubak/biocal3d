from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    f,
    t,
    shapiro,
    friedmanchisquare,
    wilcoxon,
    rankdata,
    probplot,
)

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices


# ============================================================
# DATA LOCATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected"
)

RAW_ANALYSIS_ROOT = DATA_ROOT / "_raw_color_analysis"

INPUT_FILE = RAW_ANALYSIS_ROOT / "roi_color_summary.csv"

OUTPUT_ROOT = (
    RAW_ANALYSIS_ROOT
    / "step_A_scanner_effect_final"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"
MODEL_ROOT = OUTPUT_ROOT / "model_summaries"

for folder in [
    OUTPUT_ROOT,
    FIGURE_ROOT,
    TABLE_ROOT,
    MODEL_ROOT,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# SETTINGS
# ============================================================

SCANNER_ORDER = [
    "LABscanner",
    "iTERO",
    "TRIOS3",
    "TRIOS5",
]

REFERENCE_SCANNER = "LABscanner"

METRICS = {
    "median_L": "Median L*",
    "median_a": "Median a*",
    "median_b": "Median b*",
    "median_chroma": "Median C*ab",
}

# For the non-parametric sensitivity analysis:
#
# Each physical specimen may have several timepoints.
# We average its timepoints within each scanner so that
# each physical specimen contributes exactly one observation
# per scanner.
PHYSICAL_LEVEL_AGGREGATION = "mean"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STEP A — FINAL SCANNER EFFECT ANALYSIS")
print("=" * 80)

print(f"\nInput file:\n{INPUT_FILE}")
print(f"\nRows loaded: {len(df)}")


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "scanner",
    "sample_id",
    "physical_sample_id",
    "group",
    "timepoint_number",
    "sample_number",
    "median_L",
    "median_a",
    "median_b",
    "median_chroma",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.dropna(
    subset=[
        "scanner",
        "sample_id",
        "physical_sample_id",
    ]
).copy()

df = df[
    df["scanner"].isin(SCANNER_ORDER)
].copy()

df["scanner"] = pd.Categorical(
    df["scanner"],
    categories=SCANNER_ORDER,
    ordered=True,
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_mask = df.duplicated(
    subset=[
        "sample_id",
        "scanner",
    ],
    keep=False,
)

if duplicate_mask.any():

    duplicates = (
        df.loc[
            duplicate_mask,
            [
                "physical_sample_id",
                "sample_id",
                "scanner",
            ],
        ]
        .sort_values(
            [
                "physical_sample_id",
                "sample_id",
                "scanner",
            ]
        )
    )

    duplicates.to_csv(
        TABLE_ROOT
        / "duplicate_sample_scanner_rows.csv",
        index=False,
    )

    print(duplicates)

    raise ValueError(
        "Duplicate sample_id × scanner measurements found."
    )


# ============================================================
# DESCRIBE DATA STRUCTURE
# ============================================================

print("\nRows per scanner:")

print(
    df["scanner"]
    .value_counts(sort=False)
)

print(
    "\nUnique physical specimens:",
    df["physical_sample_id"].nunique(),
)

print(
    "Unique sample/timepoint IDs:",
    df["sample_id"].nunique(),
)


sample_scanner_counts = (
    df.groupby(
        "sample_id",
        observed=True,
    )["scanner"]
    .nunique()
)


print(
    "\nNumber of scanners available per sample_id:"
)

print(
    sample_scanner_counts
    .value_counts()
    .sort_index()
)


matched_ids = (
    sample_scanner_counts[
        sample_scanner_counts >= 2
    ]
    .index
)

complete_ids = (
    sample_scanner_counts[
        sample_scanner_counts
        == len(SCANNER_ORDER)
    ]
    .index
)


print(
    "\nSample IDs measured by >=2 scanners:",
    len(matched_ids),
)

print(
    "Sample IDs measured by all four scanners:",
    len(complete_ids),
)


# ============================================================
# MATCHING TABLE
# ============================================================

matching_summary = (
    df[
        [
            "physical_sample_id",
            "sample_id",
            "group",
            "timepoint_number",
            "sample_number",
        ]
    ]
    .drop_duplicates()
    .copy()
)

matching_summary[
    "n_scanners"
] = (
    matching_summary[
        "sample_id"
    ]
    .map(sample_scanner_counts)
)

matching_summary.to_csv(
    TABLE_ROOT
    / "sample_matching_summary.csv",
    index=False,
)


# ============================================================
# DATA FOR PRIMARY ANALYSIS
# ============================================================

analysis_master = (
    df[
        df["sample_id"].isin(
            matched_ids
        )
    ]
    .copy()
)


# ============================================================
# PAIRED SCANNER PLOTS
# ============================================================

def paired_scanner_plot(
    dataframe,
    variable,
    ylabel,
    filename,
):

    plot_df = dataframe[
        dataframe[
            "sample_id"
        ].isin(
            complete_ids
        )
    ].copy()

    pivot = (
        plot_df
        .pivot(
            index="sample_id",
            columns="scanner",
            values=variable,
        )
        .reindex(
            columns=SCANNER_ORDER
        )
        .dropna()
    )

    if pivot.empty:
        return

    x = np.arange(
        len(SCANNER_ORDER)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # One line per exact sample/timepoint
    ax.plot(
        x,
        pivot.to_numpy().T,
        marker="o",
        linewidth=0.8,
        alpha=0.15,
    )

    # Overall mean
    mean_values = (
        pivot
        .mean(axis=0)
        .to_numpy()
    )

    ax.plot(
        x,
        mean_values,
        marker="o",
        linewidth=3,
        markersize=8,
        label="Mean",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        SCANNER_ORDER
    )

    ax.set_xlabel(
        "Scanner"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        f"Paired scanner measurements — {ylabel}\n"
        f"n = {len(pivot)} matched sample IDs"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


for variable, label in METRICS.items():

    paired_scanner_plot(
        analysis_master,
        variable,
        label,
        f"paired_{variable}.png",
    )


# ============================================================
# EFFECT SIZE FOR PAIRED WILCOXON
# ============================================================

def paired_rank_biserial(
    x,
    y,
):
    """
    Rank-biserial effect size.

    Positive:
        scanner B > scanner A

    Negative:
        scanner B < scanner A
    """

    differences = (
        np.asarray(y)
        - np.asarray(x)
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(differences)
    )

    positive = ranks[
        differences > 0
    ].sum()

    negative = ranks[
        differences < 0
    ].sum()

    denominator = (
        positive + negative
    )

    if denominator == 0:
        return 0.0

    return (
        positive - negative
    ) / denominator


# ============================================================
# RESULT CONTAINERS
# ============================================================

overall_results = []

pairwise_results = []

adjusted_means_results = []

diagnostic_results = []

outlier_results = []

friedman_results = []

wilcoxon_results = []

physical_level_rows = []


# ============================================================
# PRIMARY ANALYSIS
#
# Fixed-effect blocking model:
#
# Outcome ~ scanner + C(sample_id)
#
# sample_id:
#     blocks the exact specimen/timepoint
#
# physical_sample_id:
#     used for cluster-robust covariance
#
# Therefore scanner effects are estimated entirely from
# WITHIN-SAMPLE comparisons.
# ============================================================

for variable, label in METRICS.items():

    print("\n")
    print("=" * 80)
    print(label)
    print("=" * 80)

    analysis_df = (
        analysis_master[
            [
                "physical_sample_id",
                "sample_id",
                "scanner",
                variable,
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )


    # ========================================================
    # CLUSTER INFORMATION
    # ========================================================

    n_clusters = (
        analysis_df[
            "physical_sample_id"
        ]
        .nunique()
    )

    cluster_df = (
        n_clusters - 1
    )

    print(
        f"\nIndependent physical specimens: "
        f"{n_clusters}"
    )

    print(
        f"Cluster inference degrees of freedom: "
        f"{cluster_df}"
    )

    if n_clusters < 20:

        warnings.warn(
            "Fewer than 20 clusters. "
            "Cluster-robust inference may be unstable."
        )


    # ========================================================
    # FIT BLOCKED MODEL
    # ========================================================

    scanner_term = (
        "C("
        "scanner, "
        "Treatment("
        f"reference='{REFERENCE_SCANNER}'"
        ")"
        ")"
    )

    formula = (
        f"{variable} "
        f"~ {scanner_term} "
        f"+ C(sample_id)"
    )


    # Ordinary least squares estimates
    base_model = smf.ols(
        formula=formula,
        data=analysis_df,
    ).fit()


    # Cluster-robust covariance:
    #
    # clusters = physical specimens
    #
    # This allows:
    # - heteroscedasticity
    # - correlation between timepoints
    #   from the same physical specimen
    #
    robust_model = (
        base_model
        .get_robustcov_results(
            cov_type="cluster",
            groups=analysis_df[
                "physical_sample_id"
            ],
            use_correction=True,
            df_correction=True,
        )
    )


    # ========================================================
    # PARAMETER INFORMATION
    # ========================================================

    parameter_names = list(
        base_model.params.index
    )

    parameters = pd.Series(
        robust_model.params,
        index=parameter_names,
    )

    covariance = pd.DataFrame(
        robust_model.cov_params(),
        index=parameter_names,
        columns=parameter_names,
    )

    parameter_index = {
        name: i
        for i, name
        in enumerate(parameter_names)
    }


    def scanner_parameter_name(
        scanner,
    ):

        if scanner == REFERENCE_SCANNER:
            return None

        return (
            f"{scanner_term}"
            f"[T.{scanner}]"
        )


    # ========================================================
    # OVERALL SCANNER EFFECT
    #
    # H0:
    #
    #   all non-reference scanner coefficients = 0
    #
    # Robust Wald F-test using cluster df = G - 1
    # ========================================================

    scanner_coefficients = [
        scanner_parameter_name(
            scanner
        )
        for scanner in SCANNER_ORDER
        if scanner != REFERENCE_SCANNER
    ]


    beta_scanner = (
        parameters.loc[
            scanner_coefficients
        ]
        .to_numpy()
    )

    covariance_scanner = (
        covariance.loc[
            scanner_coefficients,
            scanner_coefficients,
        ]
        .to_numpy()
    )


    q = len(
        scanner_coefficients
    )

    covariance_inverse = (
        np.linalg.pinv(
            covariance_scanner
        )
    )


    wald_chi2 = float(
        beta_scanner.T
        @ covariance_inverse
        @ beta_scanner
    )

    robust_F = (
        wald_chi2 / q
    )

    overall_p = (
        f.sf(
            robust_F,
            q,
            cluster_df,
        )
    )


    print(
        "\nOverall scanner effect:"
    )

    print(
        f"  Robust F({q}, {cluster_df}) = "
        f"{robust_F:.3f}"
    )

    print(
        f"  p = {overall_p:.6g}"
    )


    overall_results.append({

        "metric":
        variable,

        "metric_label":
        label,

        "n_observations":
        len(analysis_df),

        "n_sample_ids":
        analysis_df[
            "sample_id"
        ].nunique(),

        "n_physical_samples":
        n_clusters,

        "cluster_df":
        cluster_df,

        "wald_chi2":
        wald_chi2,

        "robust_F":
        robust_F,

        "numerator_df":
        q,

        "denominator_df":
        cluster_df,

        "p_value_scanner_effect":
        overall_p,

        "r_squared":
        base_model.rsquared,

        "adjusted_r_squared":
        base_model.rsquared_adj,

    })


    # ========================================================
    # SAVE MODEL SUMMARY
    # ========================================================

    summary_path = (
        MODEL_ROOT
        / f"{variable}_blocked_cluster_robust_model.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MODEL\n"
        )

        file.write(
            "=" * 80
            + "\n"
        )

        file.write(
            formula
            + "\n\n"
        )

        file.write(
            f"Cluster variable: physical_sample_id\n"
        )

        file.write(
            f"Number of clusters: {n_clusters}\n"
        )

        file.write(
            f"Cluster inference df: {cluster_df}\n\n"
        )

        file.write(
            robust_model
            .summary()
            .as_text()
        )


    # ========================================================
    # PAIRWISE ROBUST SCANNER CONTRASTS
    # ========================================================

    metric_pairwise = []


    for scanner_A, scanner_B in combinations(
        SCANNER_ORDER,
        2,
    ):

        contrast = np.zeros(
            len(parameter_names),
            dtype=float,
        )


        # Difference = B - A

        if scanner_B != REFERENCE_SCANNER:

            name_B = scanner_parameter_name(
                scanner_B
            )

            contrast[
                parameter_index[
                    name_B
                ]
            ] += 1


        if scanner_A != REFERENCE_SCANNER:

            name_A = scanner_parameter_name(
                scanner_A
            )

            contrast[
                parameter_index[
                    name_A
                ]
            ] -= 1


        difference = float(
            contrast
            @ parameters.to_numpy()
        )


        variance = float(
            contrast
            @ covariance.to_numpy()
            @ contrast
        )

        # Protect against very small negative
        # floating-point values.
        variance = max(
            variance,
            0.0,
        )


        standard_error = (
            np.sqrt(
                variance
            )
        )


        if standard_error > 0:

            t_value = (
                difference
                / standard_error
            )

            p_raw = (
                2
                * t.sf(
                    abs(t_value),
                    df=cluster_df,
                )
            )

            t_critical = (
                t.ppf(
                    0.975,
                    df=cluster_df,
                )
            )

            ci_low = (
                difference
                - t_critical
                * standard_error
            )

            ci_high = (
                difference
                + t_critical
                * standard_error
            )

        else:

            t_value = np.nan
            p_raw = np.nan
            ci_low = np.nan
            ci_high = np.nan


        metric_pairwise.append({

            "metric":
            variable,

            "metric_label":
            label,

            "scanner_A":
            scanner_A,

            "scanner_B":
            scanner_B,

            "difference_B_minus_A":
            difference,

            "cluster_robust_SE":
            standard_error,

            "CI95_low":
            ci_low,

            "CI95_high":
            ci_high,

            "t_value":
            t_value,

            "df":
            cluster_df,

            "p_raw":
            p_raw,

        })


    # ========================================================
    # HOLM CORRECTION — SIX PAIRWISE TESTS PER METRIC
    # ========================================================

    valid_rows = [
        row
        for row in metric_pairwise
        if np.isfinite(
            row["p_raw"]
        )
    ]

    if valid_rows:

        raw_p_values = [
            row["p_raw"]
            for row in valid_rows
        ]

        _, corrected_p, _, _ = (
            multipletests(
                raw_p_values,
                method="holm",
            )
        )

        for row, p_holm in zip(
            valid_rows,
            corrected_p,
        ):

            row["p_holm"] = (
                p_holm
            )

            row[
                "significant_holm_0.05"
            ] = (
                p_holm < 0.05
            )


    for row in metric_pairwise:

        if "p_holm" not in row:

            row["p_holm"] = np.nan

            row[
                "significant_holm_0.05"
            ] = False

        pairwise_results.append(
            row
        )


    print(
        "\nPairwise scanner contrasts:"
    )

    print(
        "(positive = scanner B higher than scanner A)"
    )

    for row in metric_pairwise:

        print(
            f"\n{row['scanner_A']} "
            f"vs {row['scanner_B']}"
        )

        print(
            f"  B - A = "
            f"{row['difference_B_minus_A']:.3f}"
        )

        print(
            f"  95% CI = "
            f"[{row['CI95_low']:.3f}, "
            f"{row['CI95_high']:.3f}]"
        )

        print(
            f"  Holm p = "
            f"{row['p_holm']:.6g}"
        )


    # ========================================================
    # ADJUSTED SCANNER MEANS
    #
    # Average predictions across all exact sample IDs.
    #
    # Because each sample gets its own fixed intercept,
    # these are sample-adjusted scanner means.
    # ========================================================

    design_info = (
        base_model
        .model
        .data
        .design_info
    )

    unique_sample_ids = (
        analysis_df[
            "sample_id"
        ]
        .drop_duplicates()
        .tolist()
    )


    for scanner in SCANNER_ORDER:

        prediction_df = pd.DataFrame({

            "sample_id":
            unique_sample_ids,

            "scanner":
            pd.Categorical(
                [scanner]
                * len(
                    unique_sample_ids
                ),
                categories=SCANNER_ORDER,
                ordered=True,
            ),

        })


        X_new = (
            build_design_matrices(
                [design_info],
                prediction_df,
                return_type="dataframe",
            )[0]
        )


        X_new = (
            X_new
            .reindex(
                columns=parameter_names,
                fill_value=0,
            )
            .to_numpy()
        )


        # Mean design vector:
        #
        # gives equal weight to each sample/timepoint ID.

        x_bar = (
            X_new.mean(
                axis=0
            )
        )


        estimate = float(
            x_bar
            @ parameters.to_numpy()
        )


        variance = float(
            x_bar
            @ covariance.to_numpy()
            @ x_bar
        )

        variance = max(
            variance,
            0.0,
        )

        standard_error = (
            np.sqrt(
                variance
            )
        )


        t_critical = (
            t.ppf(
                0.975,
                df=cluster_df,
            )
        )


        adjusted_means_results.append({

            "metric":
            variable,

            "metric_label":
            label,

            "scanner":
            scanner,

            "adjusted_mean":
            estimate,

            "cluster_robust_SE":
            standard_error,

            "CI95_low":
            estimate
            - t_critical
            * standard_error,

            "CI95_high":
            estimate
            + t_critical
            * standard_error,

        })


    # ========================================================
    # MODEL DIAGNOSTICS
    #
    # Robust inference does NOT require equal variance
    # or perfectly normal residuals.
    #
    # These plots are still useful for detecting:
    # - extreme observations
    # - non-linear structure
    # - unusual residual patterns
    # ========================================================

    fitted = (
        base_model
        .fittedvalues
        .to_numpy()
    )

    residuals = (
        base_model
        .resid
        .to_numpy()
    )


    residual_sd = (
        np.std(
            residuals,
            ddof=1,
        )
    )


    if residual_sd > 0:

        standardized_residuals = (
            residuals
            / residual_sd
        )

    else:

        standardized_residuals = (
            np.full_like(
                residuals,
                np.nan,
            )
        )


    diagnostic_df = (
        analysis_df.copy()
    )

    diagnostic_df[
        "fitted"
    ] = fitted

    diagnostic_df[
        "residual"
    ] = residuals

    diagnostic_df[
        "standardized_residual"
    ] = standardized_residuals


    # --------------------------------------------------------
    # Q-Q PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    probplot(
        standardized_residuals[
            np.isfinite(
                standardized_residuals
            )
        ],
        dist="norm",
        plot=ax,
    )

    ax.set_title(
        f"Residual Q-Q plot — {label}"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT
        / f"{variable}_residual_QQ.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


    # --------------------------------------------------------
    # RESIDUALS VS FITTED
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        fitted,
        standardized_residuals,
        alpha=0.6,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Fitted value"
    )

    ax.set_ylabel(
        "Standardized residual"
    )

    ax.set_title(
        f"Residuals vs fitted — {label}"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT
        / f"{variable}_residual_vs_fitted.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


    # --------------------------------------------------------
    # RESIDUALS BY SCANNER
    # --------------------------------------------------------

    residual_groups = [

        diagnostic_df.loc[
            diagnostic_df[
                "scanner"
            ] == scanner,
            "standardized_residual",
        ]
        .dropna()
        .to_numpy()

        for scanner
        in SCANNER_ORDER

    ]


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        residual_groups,
        tick_labels=SCANNER_ORDER,
        showfliers=True,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Scanner"
    )

    ax.set_ylabel(
        "Standardized residual"
    )

    ax.set_title(
        f"Residual distribution by scanner — {label}"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT
        / f"{variable}_residuals_by_scanner.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


    # ========================================================
    # DIAGNOSTIC SUMMARY
    # ========================================================

    finite_residuals = (
        standardized_residuals[
            np.isfinite(
                standardized_residuals
            )
        ]
    )


    if len(finite_residuals) >= 3:

        shapiro_test = (
            shapiro(
                finite_residuals
            )
        )

        shapiro_W = (
            shapiro_test.statistic
        )

        shapiro_p = (
            shapiro_test.pvalue
        )

    else:

        shapiro_W = np.nan
        shapiro_p = np.nan


    scanner_residual_sd = (
        diagnostic_df
        .groupby(
            "scanner",
            observed=True,
        )[
            "standardized_residual"
        ]
        .std()
    )


    valid_sd = (
        scanner_residual_sd
        .dropna()
        .to_numpy()
    )


    if (
        len(valid_sd) > 0
        and np.min(valid_sd) > 0
    ):

        sd_ratio = (
            np.max(valid_sd)
            / np.min(valid_sd)
        )

    else:

        sd_ratio = np.nan


    outlier_mask = (
        np.abs(
            standardized_residuals
        )
        > 3
    )


    diagnostic_results.append({

        "metric":
        variable,

        "metric_label":
        label,

        "n":
        len(
            finite_residuals
        ),

        "shapiro_W":
        shapiro_W,

        "shapiro_p":
        shapiro_p,

        "max_min_scanner_residual_SD_ratio":
        sd_ratio,

        "n_abs_standardized_residual_gt_3":
        int(
            np.sum(
                outlier_mask
            )
        ),

    })


    if np.any(
        outlier_mask
    ):

        metric_outliers = (
            diagnostic_df.loc[
                outlier_mask,
                [
                    "physical_sample_id",
                    "sample_id",
                    "scanner",
                    variable,
                    "fitted",
                    "residual",
                    "standardized_residual",
                ],
            ]
            .copy()
        )

        metric_outliers[
            "metric"
        ] = variable

        metric_outliers[
            "metric_label"
        ] = label

        outlier_results.extend(
            metric_outliers
            .to_dict(
                orient="records"
            )
        )


    # ========================================================
    # NON-PARAMETRIC SENSITIVITY ANALYSIS
    #
    # IMPORTANT:
    #
    # We do NOT use the 48 sample IDs as independent blocks.
    #
    # Instead:
    #
    # 1. Average all timepoints within each physical specimen
    #    separately for each scanner.
    #
    # 2. Each physical specimen contributes ONE value per
    #    scanner.
    #
    # 3. Friedman / Wilcoxon therefore use physical specimen
    #    as the independent unit.
    # ========================================================

    if (
        PHYSICAL_LEVEL_AGGREGATION
        == "mean"
    ):

        physical_long = (
            analysis_df
            .groupby(
                [
                    "physical_sample_id",
                    "scanner",
                ],
                observed=True,
            )[variable]
            .mean()
            .reset_index()
        )

    elif (
        PHYSICAL_LEVEL_AGGREGATION
        == "median"
    ):

        physical_long = (
            analysis_df
            .groupby(
                [
                    "physical_sample_id",
                    "scanner",
                ],
                observed=True,
            )[variable]
            .median()
            .reset_index()
        )

    else:

        raise ValueError(
            "PHYSICAL_LEVEL_AGGREGATION "
            "must be 'mean' or 'median'."
        )


    physical_long[
        "metric"
    ] = variable

    physical_long[
        "metric_label"
    ] = label

    physical_level_rows.extend(
        physical_long
        .to_dict(
            orient="records"
        )
    )


    physical_pivot = (
        physical_long
        .pivot(
            index="physical_sample_id",
            columns="scanner",
            values=variable,
        )
        .reindex(
            columns=SCANNER_ORDER
        )
        .dropna()
    )


    n_physical_complete = (
        len(
            physical_pivot
        )
    )


    print(
        "\nNon-parametric sensitivity analysis:"
    )

    print(
        f"  Complete physical specimens: "
        f"{n_physical_complete}"
    )


    if n_physical_complete > 0:

        # ----------------------------------------------------
        # FRIEDMAN
        # ----------------------------------------------------

        friedman_test = (
            friedmanchisquare(
                *[
                    physical_pivot[
                        scanner
                    ]
                    .to_numpy()

                    for scanner
                    in SCANNER_ORDER
                ]
            )
        )


        kendalls_W = (
            friedman_test.statistic
            /
            (
                n_physical_complete
                * (
                    len(
                        SCANNER_ORDER
                    )
                    - 1
                )
            )
        )


        friedman_results.append({

            "metric":
            variable,

            "metric_label":
            label,

            "aggregation_within_physical_sample":
            PHYSICAL_LEVEL_AGGREGATION,

            "n_physical_samples":
            n_physical_complete,

            "friedman_chi2":
            friedman_test.statistic,

            "df":
            len(
                SCANNER_ORDER
            ) - 1,

            "p_value":
            friedman_test.pvalue,

            "kendalls_W":
            kendalls_W,

        })


        print(
            f"  Friedman χ²"
            f"({len(SCANNER_ORDER)-1}) = "
            f"{friedman_test.statistic:.3f}"
        )

        print(
            f"  p = "
            f"{friedman_test.pvalue:.6g}"
        )

        print(
            f"  Kendall's W = "
            f"{kendalls_W:.3f}"
        )


        # ----------------------------------------------------
        # PAIRWISE WILCOXON
        # ----------------------------------------------------

        metric_wilcoxon = []


        for scanner_A, scanner_B in combinations(
            SCANNER_ORDER,
            2,
        ):

            A = (
                physical_pivot[
                    scanner_A
                ]
                .to_numpy()
            )

            B = (
                physical_pivot[
                    scanner_B
                ]
                .to_numpy()
            )


            differences = (
                B - A
            )


            try:

                result = wilcoxon(
                    A,
                    B,
                    alternative="two-sided",
                    method="auto",
                )

                wilcoxon_statistic = (
                    result.statistic
                )

                p_raw = (
                    result.pvalue
                )

            except ValueError:

                wilcoxon_statistic = (
                    np.nan
                )

                p_raw = 1.0


            rank_biserial = (
                paired_rank_biserial(
                    A,
                    B,
                )
            )


            metric_wilcoxon.append({

                "metric":
                variable,

                "metric_label":
                label,

                "scanner_A":
                scanner_A,

                "scanner_B":
                scanner_B,

                "n_physical_pairs":
                len(A),

                "mean_difference_B_minus_A":
                float(
                    np.mean(
                        differences
                    )
                ),

                "median_difference_B_minus_A":
                float(
                    np.median(
                        differences
                    )
                ),

                "wilcoxon_statistic":
                wilcoxon_statistic,

                "p_raw":
                p_raw,

                "rank_biserial":
                rank_biserial,

            })


        # Holm correction

        p_values = [
            row["p_raw"]
            for row
            in metric_wilcoxon
        ]


        _, corrected_p, _, _ = (
            multipletests(
                p_values,
                method="holm",
            )
        )


        for row, p_holm in zip(
            metric_wilcoxon,
            corrected_p,
        ):

            row[
                "p_holm"
            ] = p_holm

            row[
                "significant_holm_0.05"
            ] = (
                p_holm < 0.05
            )

            wilcoxon_results.append(
                row
            )


# ============================================================
# CONVERT RESULTS TO DATAFRAMES
# ============================================================

overall_df = pd.DataFrame(
    overall_results
)

pairwise_df = pd.DataFrame(
    pairwise_results
)

adjusted_means_df = pd.DataFrame(
    adjusted_means_results
)

diagnostics_df = pd.DataFrame(
    diagnostic_results
)

outliers_df = pd.DataFrame(
    outlier_results
)

friedman_df = pd.DataFrame(
    friedman_results
)

wilcoxon_df = pd.DataFrame(
    wilcoxon_results
)

physical_level_df = pd.DataFrame(
    physical_level_rows
)


# ============================================================
# SAVE TABLES
# ============================================================

overall_df.to_csv(
    TABLE_ROOT
    / "primary_overall_scanner_effect.csv",
    index=False,
)

pairwise_df.to_csv(
    TABLE_ROOT
    / "primary_pairwise_scanner_contrasts.csv",
    index=False,
)

adjusted_means_df.to_csv(
    TABLE_ROOT
    / "primary_adjusted_scanner_means.csv",
    index=False,
)

diagnostics_df.to_csv(
    TABLE_ROOT
    / "primary_model_diagnostics.csv",
    index=False,
)

outliers_df.to_csv(
    TABLE_ROOT
    / "primary_model_outliers.csv",
    index=False,
)

physical_level_df.to_csv(
    TABLE_ROOT
    / "physical_sample_scanner_aggregates.csv",
    index=False,
)

friedman_df.to_csv(
    TABLE_ROOT
    / "sensitivity_friedman_physical_sample.csv",
    index=False,
)

wilcoxon_df.to_csv(
    TABLE_ROOT
    / "sensitivity_wilcoxon_physical_sample.csv",
    index=False,
)


# ============================================================
# PRINT FINAL TABLES
# ============================================================

print("\n")
print("=" * 80)
print("PRIMARY ANALYSIS")
print("BLOCKED FIXED-EFFECT MODEL + CLUSTER-ROBUST SE")
print("=" * 80)

print(
    overall_df[
        [
            "metric_label",
            "n_observations",
            "n_sample_ids",
            "n_physical_samples",
            "robust_F",
            "numerator_df",
            "denominator_df",
            "p_value_scanner_effect",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("PRIMARY PAIRWISE SCANNER CONTRASTS")
print("=" * 80)

print(
    pairwise_df[
        [
            "metric_label",
            "scanner_A",
            "scanner_B",
            "difference_B_minus_A",
            "CI95_low",
            "CI95_high",
            "p_holm",
            "significant_holm_0.05",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("NON-PARAMETRIC SENSITIVITY ANALYSIS")
print("=" * 80)

print(
    friedman_df[
        [
            "metric_label",
            "n_physical_samples",
            "friedman_chi2",
            "df",
            "p_value",
            "kendalls_W",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("FINISHED")
print("=" * 80)

print(
    f"\nAll outputs saved to:\n"
    f"{OUTPUT_ROOT}"
)