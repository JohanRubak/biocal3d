"""Paired analysis of the approved BioCal3D ROI maps before and after harmonization."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from investigate_approved_rois import (
    CHANNELS, COLOURS, LAB_BINS, SCANNERS, STAGES, Figures, deltaE_ciede2000,
    discover, plt, rgb2lab, summarise,
)

LAB = [f"{ch}_mean" for ch in CHANNELS]
DISTANCE = "deltaE00_mean_Lab"
CHANGES = [DISTANCE, *[f"delta_{ch}" for ch in LAB]]


def rgb_values(rgb):
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32) / 255
    if not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1.00001).any():
        raise ValueError("RGB must be finite, with integer values in 0–255 or floats in 0–1.")
    return np.clip(rgb, 0, 1)


def lab_maps(raw_path, corrected_path, stride):
    with np.load(raw_path) as raw, np.load(corrected_path) as corrected:
        valid = raw["valid"].astype(bool)
        area = raw["pixel_surface_area_mm2"]
        rgb_raw, rgb_corrected = raw["rgb"], corrected["rgb"]
        if rgb_raw.shape != valid.shape + (3,) or rgb_corrected.shape != rgb_raw.shape:
            raise ValueError("Raw and corrected maps must share the approved ROI grid.")
        for key in ("valid", "pixel_surface_area_mm2", "fov_mm"):
            if not np.array_equal(raw[key], corrected[key]):
                raise ValueError(f"Corrected map differs in {key} from the approved ROI map.")
        for key in ("vertices", "faces", "height_mm"):
            if key in corrected and not np.array_equal(raw[key], corrected[key], equal_nan=True):
                raise ValueError(f"Corrected map differs in {key} from the approved ROI map.")
        valid = valid[::stride, ::stride]
        weights = area[::stride, ::stride][valid].astype(float)
        if len(weights) < 10 or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid or insufficient approved surface area weights.")
        results = []
        for rgb in (rgb_raw, rgb_corrected):
            lab = rgb2lab(rgb_values(rgb[::stride, ::stride][valid]), illuminant="D65", observer="2")
            means = np.average(lab, weights=weights, axis=0)
            hist = {}
            for i, ch in enumerate(CHANNELS):
                counts = np.histogram(lab[:, i], LAB_BINS[i], weights=weights)[0]
                hist[ch] = counts / counts.sum()
            results.append((dict(zip(LAB, means)), hist))
        return results


def compare(df, hist, mode):
    rows = []
    by_scanner = mode == "stages"
    keys = ["physical_id", "scanner" if by_scanner else "stage"]
    column = "stage" if by_scanner else "scanner"
    pairs = list(itertools.combinations(STAGES if by_scanner else SCANNERS, 2))
    for _, subset in df.groupby(keys):
        lookup = subset.set_index([column, "condition"])
        available = set(subset[column])
        for a, b in pairs:
            if not {a, b} <= available:
                continue
            row = subset.iloc[0]
            result = dict(physical_id=row.physical_id, cohort=row.cohort)
            if by_scanner:
                result.update(scanner=row.scanner, transition=f"{a}_to_{b}")
            else:
                result.update(stage=row.stage, scanner_a=a, scanner_b=b)
            for condition in ("raw", "corrected"):
                first, last = lookup.loc[(a, condition)], lookup.loc[(b, condition)]
                x, y = first[LAB].to_numpy(float), last[LAB].to_numpy(float)
                metrics = dict(zip(CHANGES, [float(deltaE_ciede2000(x, y)), *(y-x)]))
                if not by_scanner:
                    p, q = hist[first.scan_id, condition], hist[last.scan_id, condition]
                    for i, ch in enumerate(CHANNELS):
                        metrics[f"wasserstein_{ch}"] = np.sum(
                            np.abs(np.cumsum(p[ch]) - np.cumsum(q[ch])) * np.diff(LAB_BINS[i])
                        )
                result.update({f"{name}_{condition}": value for name, value in metrics.items()})
            for name in metrics:
                result[f"{name}_corrected_minus_raw"] = result[f"{name}_corrected"] - result[f"{name}_raw"]
            if by_scanner:
                for name in CHANGES[1:]:
                    raw, corrected = result[f"{name}_raw"], result[f"{name}_corrected"]
                    result[f"{name}_sign_retained"] = int(np.sign(raw) == np.sign(corrected))
                    result[f"{name}_negative_raw"] = int(raw < 0)
                    result[f"{name}_negative_corrected"] = int(corrected < 0)
            rows.append(result)
    return pd.DataFrame(rows)


def figures(df, hist, stages, figs):
    for cohort, subset in df.groupby("cohort"):
        present = [stage for stage in STAGES if stage in set(subset.stage)]
        fig, axes = plt.subplots(len(present), 3, figsize=(10, 2.35*len(present)),
                                 squeeze=False, sharex="col", sharey="col", layout="constrained")
        active_by_channel = [set() for _ in CHANNELS]
        for i, stage in enumerate(present):
            for j, ch in enumerate(CHANNELS):
                ax = axes[i, j]
                bins = LAB_BINS[j]
                for (condition, scanner), part in subset[subset.stage == stage].groupby(["condition", "scanner"]):
                    density = np.mean([hist[sid, condition][ch] for sid in part.scan_id], axis=0)
                    active_by_channel[j].update(np.flatnonzero(density > 0))
                    ax.plot((bins[1:]+bins[:-1])/2, density/np.diff(bins), color=COLOURS[scanner],
                            ls="--" if condition == "raw" else "-", lw=1.4)
                ax.set_title(f"{stage.capitalize()} · {ch}*")
                if j == 0:
                    ax.set_ylabel("Density")
        for j, ch in enumerate(CHANNELS):
            bins = LAB_BINS[j]
            active = active_by_channel[j]
            axes[-1, j].set_xlim(max(bins[0], bins[min(active)]-3), min(bins[-1], bins[max(active)+1]+3))
            axes[-1, j].set_xlabel(f"{ch}*")
        handles = [plt.Line2D([], [], color=color, label=name) for name, color in COLOURS.items()]
        handles += [plt.Line2D([], [], color=".3", ls=ls, label=label)
                    for ls, label in [("--", "Raw"), ("-", "Corrected")]
                    if label.lower() in set(subset.condition)]
        fig.suptitle(f"CIELAB distributions · {cohort} specimens")
        axes[0, 2].legend(handles=handles, fontsize=7)
        figs.save(fig, f"Lab_distributions_{cohort}", "", "")

        selected = stages[stages.cohort == cohort]
        if selected.empty:
            continue
        transitions = [f"{a}_to_{b}" for a, b in itertools.combinations(present, 2)]
        fig, axes = plt.subplots(len(transitions), 4, figsize=(12, 2.6*len(transitions)),
                                 squeeze=False, sharex="col", sharey="col", layout="constrained")
        for i, transition in enumerate(transitions):
            part = selected[selected.transition == transition]
            for j, metric in enumerate(CHANGES):
                ax = axes[i, j]
                for k, scanner in enumerate(SCANNERS):
                    x = part[part.scanner == scanner]
                    cols = [f"{metric}_raw", f"{metric}_corrected"]
                    data = x[cols].to_numpy()
                    if not len(data):
                        continue
                    color = COLOURS[scanner]
                    pos = np.array([k-.21, k+.21])
                    ax.plot(pos, data.T, color=color, alpha=.15, lw=.65)
                    q25, median, q75 = np.quantile(data, [.25, .5, .75], axis=0)
                    ax.errorbar(pos, median, yerr=[median-q25, q75-median], color=color, fmt="o-", ms=3)
                ax.axhline(0, color=".65", lw=.7)
                ax.set_xticks(range(len(SCANNERS)), SCANNERS, rotation=25)
                ax.set_title(f"{transition.replace('_to_', ' → ').capitalize()} · {['ΔE₀₀', 'ΔL*', 'Δa*', 'Δb*'][j]}")
        fig.suptitle(f"Within-specimen changes · {cohort} specimens (raw left, corrected right)")
        figs.save(fig, f"cleaning_changes_{cohort}", "", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colour-stride", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.colour_stride < 1 or args.bootstrap < 100 or args.dpi < 72:
        parser.error("Use positive stride, at least 100 bootstrap samples, and DPI ≥ 72.")
    inventory = discover(args.raw_root)
    if len(inventory) != 512:
        print(f"Expected 512 scans for Section 3.1; found {len(inventory)}. Results describe this inventory.")
    rows, hist, missing = [], {}, []
    for row in inventory.to_dict("records"):
        if row["review_status"] in {"pending", "flagged", "rejected"}:
            raise ValueError(f"{row['scan_id']} is not an approved ROI.")
        folder = Path(row.pop("sample_dir"))
        raw_path = folder / "maps.npz"
        corrected = args.corrected_root / folder.relative_to(args.raw_root) / "maps.npz"
        if not corrected.is_file():
            missing.append(str(corrected))
            continue
        if raw_path.resolve() == corrected.resolve():
            raise ValueError("Choose a separate corrected root.")
        try:
            results = lab_maps(raw_path, corrected, args.colour_stride)
        except (ValueError, KeyError, OSError) as error:
            raise ValueError(f"{row['scan_id']}: {error}") from error
        for condition, (means, distributions) in zip(("raw", "corrected"), results):
            rows.append({**row, **means, "condition": condition})
            hist[row["scan_id"], condition] = distributions
    if missing:
        raise ValueError(f"{len(missing)} corrected maps missing. First missing path: {missing[0]}")
    df = pd.DataFrame(rows)
    scanner, stages = compare(df, hist, "scanners"), compare(df, hist, "stages")
    if scanner.empty or stages.empty:
        raise ValueError("Dataset must include matched scanners and repeated cleaning stages.")
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "scan_mean_lab.csv", index=False)
    summarise(df, ["cohort", "stage", "scanner", "condition"], LAB, args).to_csv(
        args.output / "scan_mean_lab_summary.csv", index=False)
    for name, data, keys in [
        ("scanner_agreement", scanner, ["cohort", "stage", "scanner_a", "scanner_b"]),
        ("cleaning_contrast", stages, ["cohort", "scanner", "transition"]),
    ]:
        data.to_csv(args.output / f"{name}_paired.csv", index=False)
        summary_metrics = [c for c in data if c not in [*keys, "physical_id"] and
                           not c.endswith(("_retained", "_negative_raw", "_negative_corrected"))]
        summarise(data, keys, summary_metrics, args).to_csv(args.output / f"{name}_summary.csv", index=False)
    counts = [c for c in stages if c.endswith(("_retained", "_negative_raw", "_negative_corrected"))]
    stages.groupby(["cohort", "scanner", "transition"]).agg(
        n_specimens=("physical_id", "size"), **{c: (c, "sum") for c in counts}
    ).reset_index().to_csv(args.output / "cleaning_direction_counts.csv", index=False)
    from manuscript_figures import distribution_figures, cleaning_figures, cleaning_overlay
    manuscript = Figures(args.output / "manuscript_figures", args)
    distribution_figures(df, hist, manuscript)
    cleaning_figures(stages, manuscript)
    cleaning_overlay(stages, manuscript)
    figures(df, hist, stages, Figures(args.output / "figures", args))
    figures(df[df.condition == "corrected"], hist, stages.iloc[:0],
            Figures(args.output / "corrected_only_figures", args))


if __name__ == "__main__":
    main()
