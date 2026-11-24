"""Plot the tabulated in-sample baseline sliced Wasserstein benchmark."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
data = pd.read_csv(base / "thesis_baseline_tables.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharex=True, sharey=True, layout="constrained")
for ax, (region, part) in zip(axes, data.groupby("region", sort=False)):
    positions = np.arange(len(part))[::-1]
    ax.axvline(1, color=".45", ls="--", lw=1)
    ax.hlines(positions, part.swd_ratio_corrected, part.swd_ratio_raw, color=".66", lw=1)
    ax.scatter(part.swd_ratio_raw, positions, facecolors="white", edgecolors="#37566d", label="Raw", zorder=3)
    ax.scatter(part.swd_ratio_corrected, positions, color="#008477", label="Corrected", zorder=3)
    ax.set_yticks(positions, part.scanner_pair)
    ax.set_title("Pseudo-label-defined ROI" if region == "ROI" else "Specimen background")
    ax.set_xlim(.25, 5.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[-1].legend(loc="lower right", frameon=False)
fig.supxlabel("Between-scanner sliced Wasserstein distance / raw within-scanner reference", fontsize=10)
for extension in ("pdf", "png"):
    fig.savefig(base / f"baseline_RGB_benchmark.{extension}", dpi=300)
