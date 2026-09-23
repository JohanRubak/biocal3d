"""Compare scanner-relative colour harmonization without reference-chart truth.

Run directly in VS Code. Edit DATA_ROOT below if needed. All estimates use
matched physical sample AND stage across scanners. GroupKFold holds out entire
physical samples, including every scanner and stage of each held-out sample.

Methods (fixed before looking at held-out results): raw, robust offset,
regularized per-feature gain+offset, and regularized 3-channel affine mapping.
For the full affine method, L*, a*, b* use the 3-channel mapping; ROI mean
chroma uses a separate gain+offset because the mean of per-pixel chroma is not
the chroma of the ROI mean Lab vector. No method is calibrated colour.

Every method is fit once per scanner using all available training stages. No
stage-specific transforms, histogram matching, pixel pseudo-replication, or
per-sample normalizations are used. Plots show held-out predictions only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


DATA_ROOT = Path(r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected")
INPUT_FILE = DATA_ROOT / "_pixel_color_analysis" / "precalibration_validation" / "tables" / "analysis_dataset.csv"
FEATURES = ("lab_L_mean", "lab_a_mean", "lab_b_mean", "lab_chroma_mean")
LAB = FEATURES[:3]
STAGES = ("Baseline", "Partial", "Clean")
METHODS = ("raw", "fixed_offset", "ridge_per_feature", "ridge_lab_matrix")
COLOURS = {"raw": "#737373", "fixed_offset": "#238b63",
           "ridge_per_feature": "#df801f", "ridge_lab_matrix": "#4672bf"}


def read_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    necessary = {"physical_sample_id", "scanner", "stage", "analysis_ok", *FEATURES}
    if necessary - set(frame):
        raise ValueError(f"Missing input columns: {sorted(necessary - set(frame))}")
    frame = frame.loc[frame.analysis_ok.astype(str).str.lower().isin(("true", "1", "yes"))].copy()
    for key in ("scanner", "stage", "physical_sample_id"):
        frame[key] = frame[key].astype(str).str.strip()
    frame = frame.loc[frame.stage.isin(STAGES)].copy()
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    dup = frame.duplicated(["physical_sample_id", "stage", "scanner"], keep=False)
    if dup.any():
        raise ValueError("Duplicate (physical sample, stage, scanner) records: fix before fitting.")
    if frame.physical_sample_id.nunique() < 6:
        raise ValueError("At least six physical samples are needed for this comparison.")
    return frame.reset_index(drop=True)


def matched(train: pd.DataFrame, ref_scanner: str) -> pd.DataFrame:
    ref = train.loc[train.scanner.eq(ref_scanner), ["physical_sample_id", "stage", *FEATURES]]
    other = train.loc[train.scanner.ne(ref_scanner), ["physical_sample_id", "stage", "scanner", *FEATURES]]
    return other.merge(ref, on=["physical_sample_id", "stage"], how="inner",
                       suffixes=("_source", "_reference"), validate="many_to_one")


def ridge_residual_model(x: np.ndarray, y: np.ndarray, alpha: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit y ≈ x + intercept + centered(x) @ slopes; penalize slopes only.

    Inputs scaled by training IQR so one regularization value works across Lab
    channels; the intercept stays free. Imputing is intentionally disallowed.
    """
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (x - center) / scale
    target = y - x
    z0 = z - z.mean(axis=0)
    target0 = target - target.mean(axis=0)
    slopes = np.linalg.solve(z0.T @ z0 + alpha * np.eye(x.shape[1]), z0.T @ target0)
    intercept = target.mean(axis=0) - z.mean(axis=0) @ slopes
    return center, scale, np.vstack([intercept, slopes])


def fit_models(train: pd.DataFrame, reference: str, alpha: float,
               min_pairs: int) -> dict:
    joined = matched(train, reference)
    scanners = sorted(set(train.scanner) - {reference})
    fitted: dict = {}
    for scanner in scanners:
        group = joined.loc[joined.scanner.eq(scanner)]
        if len(group) < min_pairs:
            raise ValueError(f"{scanner}: {len(group)} matched training observations < {min_pairs}")
        models = {"fixed_offset": {}, "ridge_per_feature": {}, "ridge_lab_matrix": {}}
        for feature in FEATURES:
            available = group.dropna(subset=[feature + "_source", feature + "_reference"])
            if len(available) < min_pairs:
                raise ValueError(f"{scanner}/{feature}: fewer than {min_pairs} valid matched pairs")
            # Same fixed-offset definition as the existing exploratory script:
            # median per available stage, then median of the stage medians.
            stage_offsets = [np.median(part[feature + "_source"] - part[feature + "_reference"])
                             for _, part in available.groupby("stage")]
            models["fixed_offset"][feature] = float(np.median(stage_offsets))
            x = available[[feature + "_source"]].to_numpy(float)
            y = available[[feature + "_reference"]].to_numpy(float)
            models["ridge_per_feature"][feature] = ridge_residual_model(x, y, alpha)
        all_cols = [f + side for side in ("_source", "_reference") for f in LAB]
        complete = group.dropna(subset=all_cols)
        if len(complete) < min_pairs:
            raise ValueError(f"{scanner}: too few complete Lab vectors")
        x = complete[[f + "_source" for f in LAB]].to_numpy(float)
        y = complete[[f + "_reference" for f in LAB]].to_numpy(float)
        models["ridge_lab_matrix"]["lab"] = ridge_residual_model(x, y, alpha)
        models["ridge_lab_matrix"]["lab_chroma_mean"] = models["ridge_per_feature"]["lab_chroma_mean"]
        fitted[scanner] = models
    return fitted


def predict_ridge(x: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, scale, coefficients = parameters
    return x + coefficients[0] + ((x - center) / scale) @ coefficients[1:]


def apply_models(test: pd.DataFrame, models: dict, reference: str) -> pd.DataFrame:
    chunks = []
    for method in METHODS:
        result = test[["physical_sample_id", "scanner", "stage", *FEATURES]].copy()
        result["method"] = method
        for scanner, fitted in models.items():
            where = result.scanner.eq(scanner)
            if method == "fixed_offset":
                for feature in FEATURES:
                    result.loc[where, feature] -= fitted[method][feature]
            elif method == "ridge_per_feature":
                for feature in FEATURES:
                    subset = result.loc[where, [feature]].to_numpy(float)
                    result.loc[where, feature] = predict_ridge(subset, fitted[method][feature])[:, 0]
            elif method == "ridge_lab_matrix":
                subset = result.loc[where, list(LAB)].to_numpy(float)
                result.loc[where, list(LAB)] = predict_ridge(subset, fitted[method]["lab"])
                subset = result.loc[where, ["lab_chroma_mean"]].to_numpy(float)
                result.loc[where, "lab_chroma_mean"] = predict_ridge(
                    subset, fitted[method]["lab_chroma_mean"])[:, 0]
        chunks.append(result)
    return pd.concat(chunks, ignore_index=True)


def matched_stage_rows(predictions: pd.DataFrame, reference: str) -> pd.DataFrame:
    results = []
    for method, part in predictions.groupby("method", sort=False):
        pairs = matched(part, reference)
        for feature in FEATURES:
            feature_rows = pairs[["physical_sample_id", "stage", "scanner",
                                 feature + "_source", feature + "_reference"]].dropna()
            for sample, stage, scanner, measured, target in feature_rows.itertuples(index=False, name=None):
                results.append(dict(method=method, physical_sample_id=sample, stage=stage,
                                    scanner=scanner, feature=feature, scanner_value=measured,
                                    reference_value=target, difference=measured-target))
    return pd.DataFrame(results)


def make_stage_changes(predictions: pd.DataFrame, reference: str) -> pd.DataFrame:
    changes = []
    for (method, sample, scanner), subset in predictions.groupby(["method", "physical_sample_id", "scanner"]):
        series = subset.set_index("stage")
        for first, last in (("Baseline", "Partial"), ("Partial", "Clean"), ("Baseline", "Clean")):
            if first not in series.index or last not in series.index:
                continue
            for feature in FEATURES:
                a, b = series.loc[first, feature], series.loc[last, feature]
                if pd.notna(a) and pd.notna(b):
                    changes.append(dict(method=method, physical_sample_id=sample, scanner=scanner,
                                        first=first, last=last, feature=feature, change=float(b-a)))
    changes = pd.DataFrame(changes)
    reference_changes = changes.loc[changes.scanner.eq(reference),
                                    ["method", "physical_sample_id", "first", "last", "feature", "change"]]
    return changes.merge(reference_changes, on=["method", "physical_sample_id", "first", "last", "feature"],
                         how="left", suffixes=("", "_reference"), validate="many_to_one")


def summarize(stage_pairs: pd.DataFrame, changes: pd.DataFrame,
              reference: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute = []
    for (method, scanner, feature), group in stage_pairs.groupby(["method", "scanner", "feature"]):
        values = group.difference.to_numpy(float)
        absolute.append(dict(method=method, scanner=scanner, feature=feature,
                             n_sample_stage_pairs=len(values), n_physical_samples=group.physical_sample_id.nunique(),
                             median_absolute_error=float(np.median(abs(values))),
                             median_bias=float(np.median(values)),
                             mean_bias=float(np.mean(values)),
                             difference_sd=float(np.std(values, ddof=1))))
    clean = changes.loc[changes["first"].eq("Baseline") & changes["last"].eq("Clean") &
                        changes.scanner.ne(reference) & changes.change_reference.notna()].copy()
    agreement = []
    for (method, scanner, feature), group in clean.groupby(["method", "scanner", "feature"]):
        error = (group.change - group.change_reference).to_numpy(float)
        x = group.change.to_numpy(float)
        y = group.change_reference.to_numpy(float)
        rho = float(spearmanr(x, y).statistic) if len(group) >= 4 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        agreement.append(dict(method=method, scanner=scanner, feature=feature,
                              n_physical_samples=len(group), mean_change_error=float(error.mean()),
                              change_error_sd=float(error.std(ddof=1)),
                              limit_of_agreement_low=float(error.mean()-1.96*error.std(ddof=1)),
                              limit_of_agreement_high=float(error.mean()+1.96*error.std(ddof=1)),
                              median_absolute_change_error=float(np.median(abs(error))),
                              spearman_change_rho=rho))
    preservation = []
    original = changes.loc[changes.method.eq("raw"),
                           ["physical_sample_id", "scanner", "first", "last", "feature", "change"]]
    comparison = changes.merge(original, on=["physical_sample_id", "scanner", "first", "last", "feature"],
                               suffixes=("", "_raw"), validate="many_to_one")
    for (method, feature), group in comparison.groupby(["method", "feature"]):
        diffs = (group.change - group.change_raw).to_numpy(float)
        nonzero = group.change_raw.ne(0)
        reversals = (group.loc[nonzero, "change"] * group.loc[nonzero, "change_raw"] < 0).sum()
        preservation.append(dict(method=method, feature=feature, n_stage_pairs=len(group),
                                 n_direction_reversals=int(reversals),
                                 max_absolute_change_distortion=float(max(abs(diffs))),
                                 median_absolute_change_distortion=float(np.median(abs(diffs)))))
    return pd.DataFrame(absolute), pd.DataFrame(agreement), pd.DataFrame(preservation)


def monotonic_trajectory(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, sample, scanner), subset in predictions.groupby(["method", "physical_sample_id", "scanner"]):
        by_stage = subset.set_index("stage")
        if not set(STAGES).issubset(by_stage.index):
            continue
        for feature, direction in (("lab_a_mean", "decrease"),
                                   ("lab_chroma_mean", "decrease"),
                                   ("lab_L_mean", "increase")):
            vals = by_stage.loc[list(STAGES), feature].to_numpy(float)
            if not np.isfinite(vals).all():
                continue
            step = np.diff(vals)
            records.append(dict(method=method, sample=sample, scanner=scanner, feature=feature,
                                expected_direction=direction,
                                monotonic=bool(np.all(step < 0) if direction == "decrease" else np.all(step > 0))))
    return pd.DataFrame(records)


def parameters_for_record(models: dict) -> dict:
    serialized = {}
    for scanner, methods in models.items():
        serialized[scanner] = {}
        for method, features in methods.items():
            serialized[scanner][method] = {}
            for name, values in features.items():
                if isinstance(values, tuple):
                    center, scale, beta = values
                    serialized[scanner][method][name] = dict(center=center.tolist(), scale=scale.tolist(),
                                                             intercept=beta[0].tolist(),
                                                             slopes=beta[1:].tolist())
                else:
                    serialized[scanner][method][name] = values
    return serialized


def plots(stage_pairs: pd.DataFrame, changes: pd.DataFrame, absolute: pd.DataFrame,
          preservation: pd.DataFrame, out: Path, reference: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    selected = ("lab_a_mean", "lab_chroma_mean", "lab_L_mean")
    labels = ("Redness a*", "Chroma C*ab", "Lightness L*")
    scanners = sorted(stage_pairs.scanner.unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, feature, label in zip(axes, selected, labels):
        for method_i, method in enumerate(METHODS):
            d = absolute.loc[absolute.method.eq(method) & absolute.feature.eq(feature)].set_index("scanner")
            axis.plot(np.arange(len(scanners)), d.loc[scanners, "median_absolute_error"], "o-",
                      label=method, color=COLOURS[method], lw=1.8)
        axis.set_title(label)
        axis.set_xticks(range(len(scanners)), scanners, rotation=22, ha="right")
        axis.set_ylabel("Median absolute difference to " + reference)
        axis.grid(axis="y", alpha=.25)
    axes[2].legend(fontsize=8)
    fig.suptitle("Absolute colour agreement on held-out physical samples")
    fig.savefig(out / "heldout_absolute_agreement.png", dpi=180)
    plt.close(fig)

    nonref = changes.loc[changes.scanner.ne(reference) & changes["first"].eq("Baseline") &
                         changes["last"].eq("Clean") & changes.feature.eq("lab_a_mean") &
                         changes.change_reference.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    limits = [float(np.nanmin(nonref[["change", "change_reference"]].to_numpy())),
              float(np.nanmax(nonref[["change", "change_reference"]].to_numpy()))]
    for axis, scanner in zip(axes, scanners):
        part = nonref.loc[nonref.scanner.eq(scanner)]
        for method in METHODS:
            subset = part.loc[part.method.eq(method)]
            axis.scatter(subset.change_reference, subset.change, s=22, alpha=.65,
                         label=method, color=COLOURS[method])
        axis.plot(limits, limits, "k--", lw=1)
        axis.set_title(scanner)
        axis.set_xlabel(reference + " Clean − Baseline a*")
        axis.set_ylabel(scanner + " Clean − Baseline a*")
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Did normalization improve agreement in each sample's cleaning change?")
    fig.savefig(out / "heldout_cleaning_change_agreement.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, feature, title in zip(axes, selected[:2], labels[:2]):
        d = preservation.loc[preservation.feature.eq(feature)].set_index("method").loc[list(METHODS)]
        axis.bar(range(len(METHODS)), d.median_absolute_change_distortion,
                 color=[COLOURS[method] for method in METHODS])
        axis.set_xticks(range(len(METHODS)), list(METHODS), rotation=22, ha="right")
        axis.set_ylabel("Median |adjusted change − original change|")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("How much did each method alter the within-scanner plaque effect?")
    fig.savefig(out / "plaque_change_distortion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, feature, title in zip(axes, selected, labels):
        for scanner in scanners:
            subset = stage_pairs.loc[stage_pairs.scanner.eq(scanner) & stage_pairs.feature.eq(feature) &
                                    stage_pairs.method.eq("raw")]
            median = subset.groupby("stage").difference.median()
            axis.plot(range(len(STAGES)), [median.get(stage, np.nan) for stage in STAGES], "o-", label=scanner)
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(range(len(STAGES)), STAGES)
        axis.set_title(title)
        axis.set_ylabel("Median raw scanner − " + reference)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Diagnostic: scanner differences depend on biological stage")
    fig.savefig(out / "stage_dependent_scanner_difference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference", default="TRIOS3")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=25.0,
                        help="Prespecified ridge penalty toward identity, same for all folds (default: 25)")
    parser.add_argument("--min-pairs", type=int, default=8)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}. Edit INPUT_FILE or use --input.")
    if args.folds < 2 or args.min_pairs < 2 or args.alpha < 0:
        parser.error("folds and min-pairs must be >=2 and alpha >=0")
    data = read_data(args.input)
    if args.reference not in set(data.scanner):
        parser.error(f"Reference scanner {args.reference!r} not found.")
    folds = min(args.folds, data.physical_sample_id.nunique())
    predictions = []
    fold_records = []
    fold_models = {}
    cv = GroupKFold(n_splits=folds)
    for fold, (train_i, test_i) in enumerate(cv.split(data, groups=data.physical_sample_id), start=1):
        train, test = data.iloc[train_i], data.iloc[test_i]
        assert set(train.physical_sample_id).isdisjoint(test.physical_sample_id)
        models = fit_models(train, args.reference, args.alpha, args.min_pairs)
        fold_models[str(fold)] = parameters_for_record(models)
        fold_records.append(dict(fold=fold, training_samples=train.physical_sample_id.nunique(),
                                 heldout_samples=test.physical_sample_id.nunique(),
                                 training_rows=len(train), heldout_rows=len(test)))
        predictions.append(apply_models(test, models, args.reference).assign(fold=fold))
    crossfit = pd.concat(predictions, ignore_index=True)
    stage_pairs = matched_stage_rows(crossfit, args.reference)
    changes = make_stage_changes(crossfit, args.reference)
    absolute, agreement, preservation = summarize(stage_pairs, changes, args.reference)
    monotonic = monotonic_trajectory(crossfit)
    # The raw and offset methods must leave *all* within-scanner changes intact.
    audit = preservation.loc[preservation.method.isin(("raw", "fixed_offset"))]
    if (audit.max_absolute_change_distortion > 1e-9).any():
        raise AssertionError("A fixed offset altered a within-scanner plaque change.")
    output = args.output or args.input.parent / "normalization_strategy_comparison"
    output.mkdir(parents=True, exist_ok=True)
    crossfit.to_csv(output / "crossfit_roi_values.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "physical_sample_folds.csv", index=False)
    stage_pairs.to_csv(output / "heldout_matched_stage_pairs.csv", index=False)
    changes.to_csv(output / "heldout_stage_changes.csv", index=False)
    absolute.to_csv(output / "heldout_absolute_agreement.csv", index=False)
    agreement.to_csv(output / "heldout_cleaning_change_agreement.csv", index=False)
    preservation.to_csv(output / "plaque_change_preservation.csv", index=False)
    monotonic.to_csv(output / "mature_monotonic_trajectories.csv", index=False)
    (output / "fitted_models_by_fold.json").write_text(json.dumps(fold_models, indent=2) + "\n", encoding="utf-8")
    plots(stage_pairs, changes, absolute, preservation, output / "figures", args.reference)
    config = dict(input=str(args.input.resolve()), reference=args.reference, folds=folds,
                  alpha=args.alpha, min_pairs=args.min_pairs, methods=METHODS,
                  physical_samples=int(data.physical_sample_id.nunique()), n_qc_rows=len(data),
                  interpretation="Relative harmonization only; no chart-based colour truth.",
                  selection_rule="Compare absolute agreement, paired change agreement and change preservation on held-out samples. Do not choose solely by absolute colour error.",
                  note="L*a*b* and mean chroma are ROI endpoints. Chroma is modeled separately, not recomputed from mean a* and b*. This script does not modify pixel maps.")
    (output / "analysis_protocol.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output.resolve()}")
    print(f"{folds} sample-grouped folds, {config['physical_samples']} physical samples, {len(data)} QC-passing scans")
    print("Raw and fixed-offset paired plaque changes: unchanged by construction (verified).")
    print("Check heldout_cleaning_change_agreement.csv before selecting a method.")


if __name__ == "__main__":
    main()
