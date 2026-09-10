from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    rankdata,
)
from statsmodels.stats.multitest import multipletests
from skimage.color import deltaE_ciede2000


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
    / "step_B_cleaning_effect"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"

for folder in [
    OUTPUT_ROOT,
    FIGURE_ROOT,
    TABLE_ROOT,
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


# Experimental structure
#
# Groups 1–2:
#   T0 = Baseline
#   T1 = Clean
#
# Groups 3–5:
#   T0 = Baseline
#   T1 = Partial
#   T2 = Clean
#
# The script will automatically skip analyses when one or more
# required stages have not yet been collected.

GROUP_STAGE_MAP = {

    1: {
        0: "Baseline",
        1: "Clean",
    },

    2: {
        0: "Baseline",
        1: "Clean",
    },

    3: {
        0: "Baseline",
        1: "Partial",
        2: "Clean",
    },

    4: {
        0: "Baseline",
        1: "Partial",
        2: "Clean",
    },

    5: {
        0: "Baseline",
        1: "Partial",
        2: "Clean",
    },
}


METRICS = {

    "median_L": "Median L*",

    "median_a": "Median a*",

    "median_b": "Median b*",

    "median_chroma": "Median C*ab",

}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STEP B — BIOFILM / CLEANING-STAGE ANALYSIS")
print("=" * 80)

print(f"\nInput:\n{INPUT_FILE}")
print(f"\nRows loaded: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
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

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# CLEAN TYPES
# ============================================================

df = df.dropna(
    subset=[
        "scanner",
        "physical_sample_id",
        "group",
        "timepoint_number",
    ]
).copy()

df["group"] = (
    pd.to_numeric(
        df["group"],
        errors="raise",
    )
    .astype(int)
)

df["timepoint_number"] = (
    pd.to_numeric(
        df["timepoint_number"],
        errors="raise",
    )
    .astype(int)
)


df = df[
    df["scanner"].isin(
        SCANNER_ORDER
    )
].copy()


df["scanner"] = pd.Categorical(
    df["scanner"],
    categories=SCANNER_ORDER,
    ordered=True,
)


# ============================================================
# ADD BIOLOGICAL STAGE LABEL
# ============================================================

def get_stage(row):

    group = int(
        row["group"]
    )

    timepoint = int(
        row["timepoint_number"]
    )

    group_mapping = (
        GROUP_STAGE_MAP.get(
            group,
            {},
        )
    )

    return group_mapping.get(
        timepoint,
        f"T{timepoint}",
    )


df["stage"] = df.apply(
    get_stage,
    axis=1,
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_mask = df.duplicated(
    subset=[
        "physical_sample_id",
        "scanner",
        "timepoint_number",
    ],
    keep=False,
)

if duplicate_mask.any():

    duplicates = (
        df.loc[
            duplicate_mask,
            [
                "physical_sample_id",
                "group",
                "scanner",
                "timepoint_number",
                "stage",
            ],
        ]
        .sort_values(
            [
                "group",
                "physical_sample_id",
                "scanner",
                "timepoint_number",
            ]
        )
    )

    duplicates.to_csv(
        TABLE_ROOT
        / "duplicate_stage_measurements.csv",
        index=False,
    )

    print("\nDuplicate measurements:")
    print(duplicates)

    raise ValueError(
        "Duplicate specimen × scanner × timepoint "
        "measurements were found."
    )


# ============================================================
# DATA AVAILABILITY
# ============================================================

availability = (
    df.groupby(
        [
            "group",
            "stage",
            "scanner",
        ],
        observed=True,
    )
    .agg(
        n_measurements=(
            "sample_id",
            "size",
        ),
        n_physical_samples=(
            "physical_sample_id",
            "nunique",
        ),
    )
    .reset_index()
)


availability.to_csv(
    TABLE_ROOT
    / "data_availability.csv",
    index=False,
)


print("\n")
print("=" * 80)
print("DATA AVAILABILITY")
print("=" * 80)

print(
    availability.to_string(
        index=False
    )
)


# ============================================================
# STAGE-LEVEL DESCRIPTIVE STATISTICS
# ============================================================

stage_summary_rows = []

for (
    group,
    scanner,
    stage
), subset in df.groupby(
    [
        "group",
        "scanner",
        "stage",
    ],
    observed=True,
):

    for variable, label in METRICS.items():

        values = (
            subset[variable]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        q1 = np.percentile(
            values,
            25,
        )

        q3 = np.percentile(
            values,
            75,
        )

        stage_summary_rows.append({

            "group":
            group,

            "scanner":
            scanner,

            "stage":
            stage,

            "metric":
            variable,

            "metric_label":
            label,

            "n":
            len(values),

            "mean":
            np.mean(values),

            "sd":
            (
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else np.nan
            ),

            "median":
            np.median(values),

            "Q1":
            q1,

            "Q3":
            q3,

            "IQR":
            q3 - q1,

            "min":
            np.min(values),

            "max":
            np.max(values),

        })


stage_summary_df = pd.DataFrame(
    stage_summary_rows
)

stage_summary_df.to_csv(
    TABLE_ROOT
    / "stage_descriptive_statistics.csv",
    index=False,
)


# ============================================================
# PAIRED RANK-BISERIAL EFFECT SIZE
# ============================================================

def paired_rank_biserial(
    A,
    B,
):
    """
    Paired rank-biserial effect size.

    Difference is defined as:

        B - A

    Positive effect:
        B tends to be higher than A

    Negative effect:
        B tends to be lower than A
    """

    difference = (
        np.asarray(B)
        - np.asarray(A)
    )

    difference = difference[
        difference != 0
    ]

    if len(difference) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(difference)
    )

    positive = ranks[
        difference > 0
    ].sum()

    negative = ranks[
        difference < 0
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
# STORAGE
# ============================================================

overall_stage_results = []

pairwise_stage_results = []

individual_change_rows = []

deltaE_rows = []

skip_rows = []


# ============================================================
# PAIRED TRAJECTORY PLOT
# ============================================================

def make_paired_stage_plot(
    pivot,
    stages,
    group,
    scanner,
    variable,
    label,
):

    if len(pivot) == 0:
        return

    x = np.arange(
        len(stages)
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    # Individual physical specimens
    ax.plot(
        x,
        pivot[stages]
        .to_numpy()
        .T,
        marker="o",
        linewidth=1,
        alpha=0.25,
    )


    # Mean trajectory
    means = (
        pivot[stages]
        .mean(axis=0)
        .to_numpy()
    )

    ax.plot(
        x,
        means,
        marker="o",
        linewidth=3,
        markersize=8,
        label="Mean",
    )


    ax.set_xticks(x)

    ax.set_xticklabels(
        stages
    )

    ax.set_xlabel(
        "Cleaning stage"
    )

    ax.set_ylabel(
        label
    )

    ax.set_title(
        f"Group {group} — {scanner}\n"
        f"{label}, n = {len(pivot)} paired specimens"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT
        / (
            f"G{group}_"
            f"{scanner}_"
            f"{variable}_"
            f"paired_stages.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN STAGE ANALYSIS
# ============================================================

for group in sorted(
    GROUP_STAGE_MAP.keys()
):

    group_df = df[
        df["group"] == group
    ].copy()

    if group_df.empty:

        skip_rows.append({
            "group":
            group,

            "scanner":
            "ALL",

            "analysis":
            "group",

            "reason":
            "No data collected yet",
        })

        continue


    expected_stages = list(
        GROUP_STAGE_MAP[
            group
        ].values()
    )


    print("\n")
    print("=" * 80)
    print(
        f"GROUP {group}"
    )
    print("=" * 80)

    print(
        "Expected stages:",
        " -> ".join(
            expected_stages
        )
    )


    for scanner in SCANNER_ORDER:

        scanner_df = group_df[
            group_df["scanner"]
            == scanner
        ].copy()

        if scanner_df.empty:

            skip_rows.append({

                "group":
                group,

                "scanner":
                scanner,

                "analysis":
                "all",

                "reason":
                "No scanner data",

            })

            continue


        available_stages = set(
            scanner_df[
                "stage"
            ].unique()
        )


        missing_stages = [
            stage
            for stage
            in expected_stages
            if stage
            not in available_stages
        ]


        if missing_stages:

            print(
                f"\n{scanner}: "
                f"not complete — missing "
                f"{missing_stages}"
            )

            skip_rows.append({

                "group":
                group,

                "scanner":
                scanner,

                "analysis":
                "stage comparison",

                "reason":
                "Missing stages: "
                + ", ".join(
                    missing_stages
                ),

            })

            continue


        # ====================================================
        # ANALYSE EACH METRIC
        # ====================================================

        for variable, label in METRICS.items():

            pivot = (
                scanner_df
                .pivot(
                    index="physical_sample_id",
                    columns="stage",
                    values=variable,
                )
                .reindex(
                    columns=expected_stages
                )
                .dropna()
            )


            n_complete = len(
                pivot
            )


            if n_complete < 2:

                skip_rows.append({

                    "group":
                    group,

                    "scanner":
                    scanner,

                    "analysis":
                    variable,

                    "reason":
                    (
                        "Fewer than 2 complete "
                        "physical specimens"
                    ),

                })

                continue


            # ------------------------------------------------
            # PLOT
            # ------------------------------------------------

            make_paired_stage_plot(
                pivot=pivot,
                stages=expected_stages,
                group=group,
                scanner=scanner,
                variable=variable,
                label=label,
            )


            # =================================================
            # TWO-STAGE GROUP
            #
            # Baseline -> Clean
            # =================================================

            if len(
                expected_stages
            ) == 2:

                stage_A = (
                    expected_stages[0]
                )

                stage_B = (
                    expected_stages[1]
                )

                A = (
                    pivot[
                        stage_A
                    ]
                    .to_numpy()
                )

                B = (
                    pivot[
                        stage_B
                    ]
                    .to_numpy()
                )

                difference = (
                    B - A
                )


                try:

                    test = wilcoxon(
                        A,
                        B,
                        alternative="two-sided",
                        method="auto",
                    )

                    statistic = (
                        test.statistic
                    )

                    p_raw = (
                        test.pvalue
                    )

                except ValueError:

                    statistic = np.nan
                    p_raw = 1.0


                effect_size = (
                    paired_rank_biserial(
                        A,
                        B,
                    )
                )


                result_row = {

                    "group":
                    group,

                    "scanner":
                    scanner,

                    "metric":
                    variable,

                    "metric_label":
                    label,

                    "stage_A":
                    stage_A,

                    "stage_B":
                    stage_B,

                    "n_pairs":
                    n_complete,

                    "mean_difference_B_minus_A":
                    np.mean(
                        difference
                    ),

                    "median_difference_B_minus_A":
                    np.median(
                        difference
                    ),

                    "SD_difference":
                    (
                        np.std(
                            difference,
                            ddof=1,
                        )
                        if len(
                            difference
                        ) > 1
                        else np.nan
                    ),

                    "n_increased":
                    int(
                        np.sum(
                            difference > 0
                        )
                    ),

                    "n_decreased":
                    int(
                        np.sum(
                            difference < 0
                        )
                    ),

                    "n_unchanged":
                    int(
                        np.sum(
                            difference == 0
                        )
                    ),

                    "wilcoxon_statistic":
                    statistic,

                    "p_raw":
                    p_raw,

                    # only one comparison
                    "p_holm":
                    p_raw,

                    "rank_biserial":
                    effect_size,

                    "significant_0.05":
                    p_raw < 0.05,

                }


                pairwise_stage_results.append(
                    result_row
                )


                # ---------------------------------------------
                # SAVE INDIVIDUAL CHANGES
                # ---------------------------------------------

                for physical_id, diff in zip(
                    pivot.index,
                    difference,
                ):

                    individual_change_rows.append({

                        "group":
                        group,

                        "scanner":
                        scanner,

                        "physical_sample_id":
                        physical_id,

                        "metric":
                        variable,

                        "metric_label":
                        label,

                        "stage_A":
                        stage_A,

                        "stage_B":
                        stage_B,

                        "difference_B_minus_A":
                        diff,

                    })


            # =================================================
            # THREE-STAGE GROUP
            #
            # Baseline -> Partial -> Clean
            # =================================================

            elif len(
                expected_stages
            ) == 3:


                arrays = [

                    pivot[
                        stage
                    ]
                    .to_numpy()

                    for stage
                    in expected_stages

                ]


                # ---------------------------------------------
                # FRIEDMAN OVERALL TEST
                # ---------------------------------------------

                friedman = (
                    friedmanchisquare(
                        *arrays
                    )
                )


                kendalls_W = (
                    friedman.statistic
                    /
                    (
                        n_complete
                        * (
                            len(
                                expected_stages
                            )
                            - 1
                        )
                    )
                )


                overall_stage_results.append({

                    "group":
                    group,

                    "scanner":
                    scanner,

                    "metric":
                    variable,

                    "metric_label":
                    label,

                    "n_complete":
                    n_complete,

                    "n_stages":
                    len(
                        expected_stages
                    ),

                    "friedman_chi2":
                    friedman.statistic,

                    "df":
                    (
                        len(
                            expected_stages
                        )
                        - 1
                    ),

                    "p_value":
                    friedman.pvalue,

                    "kendalls_W":
                    kendalls_W,

                })


                # ---------------------------------------------
                # PAIRWISE WILCOXON
                # ---------------------------------------------

                metric_pairwise = []


                for stage_A, stage_B in combinations(
                    expected_stages,
                    2,
                ):

                    A = (
                        pivot[
                            stage_A
                        ]
                        .to_numpy()
                    )

                    B = (
                        pivot[
                            stage_B
                        ]
                        .to_numpy()
                    )

                    difference = (
                        B - A
                    )


                    try:

                        test = wilcoxon(
                            A,
                            B,
                            alternative="two-sided",
                            method="auto",
                        )

                        statistic = (
                            test.statistic
                        )

                        p_raw = (
                            test.pvalue
                        )

                    except ValueError:

                        statistic = np.nan
                        p_raw = 1.0


                    effect_size = (
                        paired_rank_biserial(
                            A,
                            B,
                        )
                    )


                    row = {

                        "group":
                        group,

                        "scanner":
                        scanner,

                        "metric":
                        variable,

                        "metric_label":
                        label,

                        "stage_A":
                        stage_A,

                        "stage_B":
                        stage_B,

                        "n_pairs":
                        n_complete,

                        "mean_difference_B_minus_A":
                        np.mean(
                            difference
                        ),

                        "median_difference_B_minus_A":
                        np.median(
                            difference
                        ),

                        "SD_difference":
                        (
                            np.std(
                                difference,
                                ddof=1,
                            )
                            if len(
                                difference
                            ) > 1
                            else np.nan
                        ),

                        "n_increased":
                        int(
                            np.sum(
                                difference > 0
                            )
                        ),

                        "n_decreased":
                        int(
                            np.sum(
                                difference < 0
                            )
                        ),

                        "n_unchanged":
                        int(
                            np.sum(
                                difference == 0
                            )
                        ),

                        "wilcoxon_statistic":
                        statistic,

                        "p_raw":
                        p_raw,

                        "rank_biserial":
                        effect_size,

                    }


                    metric_pairwise.append(
                        row
                    )


                    # -----------------------------------------
                    # INDIVIDUAL DIFFERENCES
                    # -----------------------------------------

                    for physical_id, diff in zip(
                        pivot.index,
                        difference,
                    ):

                        individual_change_rows.append({

                            "group":
                            group,

                            "scanner":
                            scanner,

                            "physical_sample_id":
                            physical_id,

                            "metric":
                            variable,

                            "metric_label":
                            label,

                            "stage_A":
                            stage_A,

                            "stage_B":
                            stage_B,

                            "difference_B_minus_A":
                            diff,

                        })


                # ---------------------------------------------
                # HOLM CORRECTION
                #
                # Three stage comparisons within
                # one scanner × one outcome.
                # ---------------------------------------------

                p_values = [
                    row["p_raw"]
                    for row
                    in metric_pairwise
                ]


                _, corrected_p, _, _ = (
                    multipletests(
                        p_values,
                        method="holm",
                    )
                )


                for row, p_holm in zip(
                    metric_pairwise,
                    corrected_p,
                ):

                    row[
                        "p_holm"
                    ] = p_holm

                    row[
                        "significant_0.05"
                    ] = (
                        p_holm < 0.05
                    )

                    pairwise_stage_results.append(
                        row
                    )


# ============================================================
# CIEDE2000 BETWEEN CLEANING STAGES
#
# Uses median L*, a*, b* for each ROI.
# ============================================================

for group in sorted(
    GROUP_STAGE_MAP.keys()
):

    group_df = df[
        df["group"] == group
    ].copy()

    if group_df.empty:
        continue


    expected_stages = list(
        GROUP_STAGE_MAP[
            group
        ].values()
    )


    for scanner in SCANNER_ORDER:

        scanner_df = group_df[
            group_df["scanner"]
            == scanner
        ].copy()

        if scanner_df.empty:
            continue


        # Separate pivots for L, a, b
        pivot_L = (
            scanner_df
            .pivot(
                index="physical_sample_id",
                columns="stage",
                values="median_L",
            )
        )

        pivot_a = (
            scanner_df
            .pivot(
                index="physical_sample_id",
                columns="stage",
                values="median_a",
            )
        )

        pivot_b = (
            scanner_df
            .pivot(
                index="physical_sample_id",
                columns="stage",
                values="median_b",
            )
        )


        for stage_A, stage_B in combinations(
            expected_stages,
            2,
        ):

            if (
                stage_A
                not in pivot_L.columns
                or stage_B
                not in pivot_L.columns
            ):
                continue


            physical_ids = (
                pivot_L.index
                .intersection(
                    pivot_a.index
                )
                .intersection(
                    pivot_b.index
                )
            )


            for physical_id in physical_ids:

                lab_A = np.array(
                    [
                        pivot_L.loc[
                            physical_id,
                            stage_A,
                        ],
                        pivot_a.loc[
                            physical_id,
                            stage_A,
                        ],
                        pivot_b.loc[
                            physical_id,
                            stage_A,
                        ],
                    ],
                    dtype=float,
                )


                lab_B = np.array(
                    [
                        pivot_L.loc[
                            physical_id,
                            stage_B,
                        ],
                        pivot_a.loc[
                            physical_id,
                            stage_B,
                        ],
                        pivot_b.loc[
                            physical_id,
                            stage_B,
                        ],
                    ],
                    dtype=float,
                )


                if (
                    np.any(
                        ~np.isfinite(
                            lab_A
                        )
                    )
                    or np.any(
                        ~np.isfinite(
                            lab_B
                        )
                    )
                ):
                    continue


                delta_e = float(
                    deltaE_ciede2000(
                        lab_A.reshape(
                            1,
                            1,
                            3,
                        ),
                        lab_B.reshape(
                            1,
                            1,
                            3,
                        ),
                    )[0, 0]
                )


                deltaE_rows.append({

                    "group":
                    group,

                    "scanner":
                    scanner,

                    "physical_sample_id":
                    physical_id,

                    "stage_A":
                    stage_A,

                    "stage_B":
                    stage_B,

                    "stage_pair":
                    (
                        f"{stage_A} → {stage_B}"
                    ),

                    "deltaE00":
                    delta_e,

                })


# ============================================================
# CREATE RESULT DATAFRAMES
# ============================================================

overall_stage_df = pd.DataFrame(
    overall_stage_results
)

pairwise_stage_df = pd.DataFrame(
    pairwise_stage_results
)

individual_changes_df = pd.DataFrame(
    individual_change_rows
)

deltaE_df = pd.DataFrame(
    deltaE_rows
)

skip_df = pd.DataFrame(
    skip_rows
)


# ============================================================
# DELTA E SUMMARY
# ============================================================

if not deltaE_df.empty:

    deltaE_summary_df = (
        deltaE_df
        .groupby(
            [
                "group",
                "scanner",
                "stage_A",
                "stage_B",
                "stage_pair",
            ],
            observed=True,
        )[
            "deltaE00"
        ]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            min="min",
            max="max",
        )
        .reset_index()
    )


    # Add Q1, Q3 and IQR

    quartiles = (
        deltaE_df
        .groupby(
            [
                "group",
                "scanner",
                "stage_A",
                "stage_B",
                "stage_pair",
            ],
            observed=True,
        )[
            "deltaE00"
        ]
        .quantile(
            [
                0.25,
                0.75,
            ]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.25: "Q1",
                0.75: "Q3",
            }
        )
    )


    deltaE_summary_df = (
        deltaE_summary_df
        .merge(
            quartiles,
            on=[
                "group",
                "scanner",
                "stage_A",
                "stage_B",
                "stage_pair",
            ],
            how="left",
        )
    )


    deltaE_summary_df[
        "IQR"
    ] = (
        deltaE_summary_df[
            "Q3"
        ]
        - deltaE_summary_df[
            "Q1"
        ]
    )


else:

    deltaE_summary_df = (
        pd.DataFrame()
    )


# ============================================================
# DELTA E BOXPLOTS
# ============================================================

if not deltaE_df.empty:

    for (
        group,
        stage_A,
        stage_B
    ), subset in deltaE_df.groupby(
        [
            "group",
            "stage_A",
            "stage_B",
        ],
        observed=True,
    ):

        scanner_data = []

        scanner_labels = []


        for scanner in SCANNER_ORDER:

            values = (
                subset.loc[
                    subset[
                        "scanner"
                    ] == scanner,
                    "deltaE00",
                ]
                .dropna()
                .to_numpy()
            )

            if len(values) > 0:

                scanner_data.append(
                    values
                )

                scanner_labels.append(
                    scanner
                )


        if not scanner_data:
            continue


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        ax.boxplot(
            scanner_data,
            tick_labels=scanner_labels,
            showfliers=True,
        )


        ax.set_xlabel(
            "Scanner"
        )

        ax.set_ylabel(
            "CIEDE2000 ΔE00"
        )

        ax.set_title(
            f"Group {group}: "
            f"{stage_A} → {stage_B}"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()


        safe_A = (
            stage_A
            .replace(
                " ",
                "_",
            )
        )

        safe_B = (
            stage_B
            .replace(
                " ",
                "_",
            )
        )


        fig.savefig(
            FIGURE_ROOT
            / (
                f"G{group}_"
                f"deltaE00_"
                f"{safe_A}_to_{safe_B}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# SAVE EVERYTHING
# ============================================================

stage_summary_df.to_csv(
    TABLE_ROOT
    / "stage_descriptive_statistics.csv",
    index=False,
)

overall_stage_df.to_csv(
    TABLE_ROOT
    / "friedman_overall_stage_effect.csv",
    index=False,
)

pairwise_stage_df.to_csv(
    TABLE_ROOT
    / "wilcoxon_pairwise_stage_effect.csv",
    index=False,
)

individual_changes_df.to_csv(
    TABLE_ROOT
    / "individual_stage_changes.csv",
    index=False,
)

deltaE_df.to_csv(
    TABLE_ROOT
    / "individual_stage_deltaE00.csv",
    index=False,
)

deltaE_summary_df.to_csv(
    TABLE_ROOT
    / "stage_deltaE00_summary.csv",
    index=False,
)

skip_df.to_csv(
    TABLE_ROOT
    / "skipped_incomplete_analyses.csv",
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("TWO-/THREE-STAGE PAIRWISE RESULTS")
print("=" * 80)

if not pairwise_stage_df.empty:

    columns_to_print = [
        "group",
        "scanner",
        "metric_label",
        "stage_A",
        "stage_B",
        "n_pairs",
        "median_difference_B_minus_A",
        "n_increased",
        "n_decreased",
        "rank_biserial",
        "p_holm",
    ]

    print(
        pairwise_stage_df[
            columns_to_print
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No complete paired analyses available."
    )


print("\n")
print("=" * 80)
print("THREE-STAGE FRIEDMAN RESULTS")
print("=" * 80)

if not overall_stage_df.empty:

    print(
        overall_stage_df[
            [
                "group",
                "scanner",
                "metric_label",
                "n_complete",
                "friedman_chi2",
                "p_value",
                "kendalls_W",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No complete three-stage groups available."
    )


print("\n")
print("=" * 80)
print("CIEDE2000 CLEANING-STAGE DIFFERENCES")
print("=" * 80)

if not deltaE_summary_df.empty:

    print(
        deltaE_summary_df[
            [
                "group",
                "scanner",
                "stage_pair",
                "n",
                "median",
                "IQR",
            ]
        ]
        .to_string(
            index=False
        )
    )


print("\n")
print("=" * 80)
print("INCOMPLETE / SKIPPED ANALYSES")
print("=" * 80)

if not skip_df.empty:

    print(
        skip_df.to_string(
            index=False
        )
    )

else:

    print(
        "None."
    )


print("\n")
print("=" * 80)
print("STEP B COMPLETE")
print("=" * 80)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_ROOT}"
)