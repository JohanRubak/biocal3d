"""Matched manuscript layouts for full-ROI color distributions and cleaning responses."""
import numpy as np
from investigate_approved_rois import CHANNELS, COLOURS, LAB_BINS, SCANNERS, plt

ROWS = [("fresh", "baseline"), ("fresh", "clean"), ("mature", "baseline"),
        ("mature", "partial"), ("mature", "clean")]
LABELS = {"raw": "Before harmonization", "corrected": "After harmonization"}


def scanner_legend():
    return [plt.Line2D([], [], color=COLOURS[s], label=s, lw=1.7) for s in SCANNERS]


def distribution_figures(df, hist, figs):
    curves = {}
    for cohort, stage in ROWS:
        for condition in LABELS:
            for scanner in SCANNERS:
                scans = df[(df.cohort == cohort) & (df.stage == stage) &
                           (df.condition == condition) & (df.scanner == scanner)]
                for ch in CHANNELS:
                    if len(scans):
                        curves[cohort, stage, condition, scanner, ch] = np.mean(
                            [hist[sid, condition][ch] for sid in scans.scan_id], axis=0)
    limits = {}
    for j, ch in enumerate(CHANNELS):
        values = [v for key, v in curves.items() if key[-1] == ch]
        active = np.flatnonzero(np.max(values, axis=0) > 0)
        bins = LAB_BINS[j]
        limits[ch] = (max(bins[0], bins[active[0]]-2),
                      min(bins[-1], bins[active[-1]+1]+2),
                      max(v.max() for v in values)/np.diff(bins)[0]*1.08)
    for conditions, filename in [
        (["raw"], "01_CIELAB_before"),
        (["corrected"], "01_CIELAB_after"),

    ]:
        fig, axes = plt.subplots(5, 3*len(conditions),
                                 figsize=(6.8*len(conditions), 9),
                                 squeeze=False, layout="constrained")
        for r, (cohort, stage) in enumerate(ROWS):
            for block, condition in enumerate(conditions):
                for j, ch in enumerate(CHANNELS):
                    ax = axes[r, block*3+j]
                    bins = LAB_BINS[j]
                    for scanner in SCANNERS:
                        curve = curves.get((cohort, stage, condition, scanner, ch))
                        if curve is not None:
                            ax.plot((bins[:-1]+bins[1:])/2, curve/np.diff(bins),
                                    color=COLOURS[scanner], lw=1.15)
                    low, high, maximum = limits[ch]
                    ax.set_xlim(low, high)
                    ax.set_ylim(0, maximum)
                    title = f"{chr(65+r*3+j)}  {cohort.capitalize()} · {stage.capitalize()}"
                    if r == 0:
                        title = LABELS[condition] + "\n" + title
                    ax.set_title(title, loc="left", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=.12)
                    if j == 0:
                        ax.set_ylabel("Density", fontsize=8)
                    if r == 4:
                        ax.set_xlabel(ch+"*", fontsize=9)
        fig.legend(handles=scanner_legend(), loc="outside lower center",
                   ncols=4, frameon=False, fontsize=9)
        figs.save(fig, filename, "", "")


def cleaning_figures(stages, figs):
    columns = ["deltaE00_mean_Lab_raw", "deltaE00_mean_Lab_corrected"]
    maximum = max(1, float(stages[columns].max().max())*1.08)
    for conditions, filename in [
        (["raw"], "02_cleaning_before"),
        (["corrected"], "02_cleaning_after"),
        (["raw", "corrected"], "02_cleaning_before_after"),
    ]:
        rows = [(cohort, condition) for cohort in ["fresh", "mature"] for condition in conditions]
        fig, axes = plt.subplots(len(rows), 4, figsize=(11, 3.2*len(rows)),
                                 sharey=True, squeeze=False, layout="constrained")
        for i, (cohort, condition) in enumerate(rows):
            stage_order = ["baseline", "clean"] if cohort == "fresh" else ["baseline", "partial", "clean"]
            for j, scanner in enumerate(SCANNERS):
                ax = axes[i, j]
                selected = stages[(stages.cohort == cohort) & (stages.scanner == scanner) &
                                  stages.transition.str.startswith("baseline_to_")]
                traces = []
                for _, part in selected.groupby("physical_id"):
                    lookup = part.set_index("transition")[f"deltaE00_mean_Lab_{condition}"]
                    traces.append([0.] + [lookup.get("baseline_to_"+stage, np.nan) for stage in stage_order[1:]])
                color = COLOURS[scanner]
                if traces:
                    values = np.asarray(traces)
                    x = np.arange(len(stage_order))
                    ax.plot(x, values.T, color=color, alpha=.2, lw=.65)
                    low, med, high = np.nanquantile(values, [.25, .5, .75], axis=0)
                    ax.fill_between(x, low, high, color=color, alpha=.12)
                    ax.plot(x, med, "o-", color=color, lw=2, ms=4)
                ax.set_ylim(0, maximum)
                ax.set_xticks(range(len(stage_order)), [s.capitalize() for s in stage_order],
                              rotation=25, ha="right")
                ax.set_title(f"{chr(65+i*4+j)}  {scanner}", loc="left", fontsize=11)
                ax.grid(alpha=.12)
                if j == 0:
                    ax.set_ylabel(f"{cohort.capitalize()} · {LABELS[condition]}\nΔE₀₀ from baseline")
        figs.save(fig, filename, "", "")


def cleaning_overlay(stages, figs):
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.8), sharey=True,
                             squeeze=False, layout="constrained")
    maximum = max(1, float(stages[["deltaE00_mean_Lab_raw", "deltaE00_mean_Lab_corrected"]].max().max())*1.08)
    for i, cohort in enumerate(["fresh", "mature"]):
        stage_order = ["baseline", "clean"] if cohort == "fresh" else ["baseline", "partial", "clean"]
        for j, scanner in enumerate(SCANNERS):
            ax = axes[i, j]
            selected = stages[(stages.cohort == cohort) & (stages.scanner == scanner) &
                              stages.transition.str.startswith("baseline_to_")]
            for condition, style, color in [("raw", "--", ".45"), ("corrected", "-", COLOURS[scanner])]:
                traces = []
                for _, part in selected.groupby("physical_id"):
                    lookup = part.set_index("transition")[f"deltaE00_mean_Lab_{condition}"]
                    traces.append([0.] + [lookup.get("baseline_to_"+stage, np.nan) for stage in stage_order[1:]])
                if traces:
                    x = np.arange(len(stage_order))
                    low, med, high = np.nanquantile(traces, [.25, .5, .75], axis=0)
                    ax.fill_between(x, low, high, color=color, alpha=.1)
                    ax.plot(x, med, linestyle=style, marker="o", color=color, lw=2, ms=4)
            ax.set_ylim(0, maximum)
            ax.set_xticks(range(len(stage_order)), [s.capitalize() for s in stage_order], rotation=25, ha="right")
            ax.set_title(f"{chr(65+i*4+j)}  {scanner}", loc="left")
            ax.grid(alpha=.12)
            if j == 0:
                ax.set_ylabel(f"{cohort.capitalize()}\nΔE₀₀ from baseline")
    handles = [plt.Line2D([], [], color=".45", ls="--", marker="o", label="Before harmonization"),
               plt.Line2D([], [], color="black", ls="-", marker="o", label="After harmonization (scanner colors)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    figs.save(fig, "02_cleaning_before_after_overlay", "", "")


def swd_bar_figure(summary, figs):
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    slots = [(0, slice(0, 2)), (0, slice(4, 6)),
             (1, slice(0, 2)), (1, slice(2, 4)), (1, slice(4, 6))]
    maximum = summary.loc[summary.metric == "sliced_wasserstein", ["raw_ratio", "corrected_ratio"]].max().max()
    maximum = max(1.2, float(maximum)*1.08) if np.isfinite(maximum) else 1.2
    for (cohort, stage), (r, c) in zip(ROWS, slots):
        ax = fig.add_subplot(grid[r, c])
        part = summary[(summary.cohort == cohort) & (summary.scope == stage) &
                       (summary.metric == "sliced_wasserstein")]
        y = np.arange(len(part))[::-1]
        ax.barh(y+.18, part.raw_ratio, height=.33, color="#a5aeb5", label="Before")
        ax.barh(y-.18, part.corrected_ratio, height=.33, color="#008477", label="After")
        ax.axvline(1, color=".3", ls=":", lw=1)
        ax.set_yticks(y, part.scanner_a+" / "+part.scanner_b if c.start == 0 else [""]*len(part),
                      fontsize=8)
        if c.start:
            ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, maximum)
        ax.set_title(f"{cohort.capitalize()} · {stage.capitalize()}", loc="left", fontsize=11)
        ax.grid(axis="x", alpha=.13)
        ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=".3", ls=":", label="Raw within-scanner reference"))
    labels.append("Raw within-scanner reference")
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.supxlabel("Sliced Wasserstein distance / raw within-scanner reference", fontsize=11)
    figs.save(fig, "03_SWD_ratios_fresh_mature", "", "")


def supporting_rgb_tables(summary, output):
    import pandas as pd
    part = summary[summary.scope != "all"].copy()
    keys = ["cohort", "scope", "scanner_a", "scanner_b", "n_specimens"]
    values = ["raw", "corrected", "corrected_minus_raw", "ci_low", "ci_high", "reduction_percent"]
    numerical = part[keys+["metric"]+values]
    numerical.to_csv(output / "Table_RGB_absolute_metrics.csv", index=False)
    swd = part[part.metric == "sliced_wasserstein"]
    swd[keys+["raw", "corrected", "raw_within_reference", "raw_ratio", "corrected_ratio",
              "reduction_percent"]].to_csv(output / "Table_SWD_references_and_ratios.csv", index=False)
    labels = {"mean_rgb_distance": "Mean RGB distance",
              "rgb_sd_distance": "RGB SD distance", "sliced_wasserstein": "SWD"}
    rows = []
    for keys_values, records in numerical.groupby(keys, sort=False):
        cohort, stage, scanner_a, scanner_b, n = keys_values
        item = {"Cohort": cohort.capitalize(), "Stage": stage.capitalize(),
                "Scanner pair": scanner_a+" / "+scanner_b, "n": n}
        for _, row in records.iterrows():
            item[labels[row.metric]+" before / after"] = f"{row.raw:.4f} / {row.corrected:.4f}"
            if row.metric == "sliced_wasserstein":
                item["SWD paired change (95% CI)"] = (
                    f"{row.corrected_minus_raw:.4f} ({row.ci_low:.4f}, {row.ci_high:.4f})")
                item["SWD reduction (%)"] = f"{row.reduction_percent:.1f}"
        rows.append(item)
    html = """<!doctype html><html><head><meta charset="utf-8"><title>ROI color comparison tables</title>
    <style>body{font:14px Arial;margin:30px}table{border-collapse:collapse;margin-bottom:30px}
    th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef4f4}</style></head>
    <body><h1>ROI color comparison tables</h1>
    <p>Absolute RGB distances. Paired change is harmonized minus raw. Confidence intervals
    resample specimens. Each cohort and stage uses its own raw within-scanner reference.
    RGB colors are in [0,1].</p>"""
    formatted = pd.DataFrame(rows)
    formatted.to_csv(output / "Table_RGB_compact.csv", index=False)
    for (cohort, stage), table in formatted.groupby(["Cohort", "Stage"], sort=False):
        html += f"<h2>{cohort} · {stage}</h2>"+table.drop(columns=["Cohort", "Stage"]).to_html(index=False, escape=True)
    html += "</body></html>"
    (output / "Tables_for_manuscript.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pandas as pd
    from investigate_approved_rois import Figures

    parser = argparse.ArgumentParser(description="Regenerate the SWD figure from the existing cohort summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    swd_bar_figure(pd.read_csv(args.summary), Figures(args.summary.parent / "manuscript_figures", args))
