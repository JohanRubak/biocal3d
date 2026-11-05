"""Explain the actual standardized logistic-regression stage heads.

cached: exact feature contributions and frozen-representation region deletion.
input: edit original encoder inputs, re-encode, and measure the same head's response.
"""
import argparse
import csv
import html
from pathlib import Path
import re
from types import SimpleNamespace

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image

from make_feature_pseudomasks import render
from stage_ablation import MODES, STAGES, RESULT_SUBDIR, feature_path, read_json
from study_data import digest, make_surface, signature, write_csv, write_json

VERSION = "stage-decision-explanation-1.1"


def choose_scans(rows, args):
    """Explicit IDs preserve input order; batches rotate across scanner/stage groups."""
    lookup = {r["scan_id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate scan IDs in cohort")
    if args.scan:
        if len(set(args.scan)) != len(args.scan): raise ValueError("Repeated --scan IDs")
        unknown = set(args.scan) - lookup.keys()
        if unknown: raise ValueError(f"Unknown scan IDs: {sorted(unknown)}")
        selected = [lookup[s] for s in args.scan]
    else:
        selected = list(rows)
    partition = args.partition or ("all" if args.scan else "test")
    if partition != "all": selected = [r for r in selected if r["partition"] == partition]
    if args.stage: selected = [r for r in selected if r["stage"] in args.stage]
    if args.scanner: selected = [r for r in selected if r["scanner"] in args.scanner]
    if args.specimen:
        unknown = set(args.specimen) - {r["physical_id"] for r in rows}
        if unknown: raise ValueError(f"Unknown physical specimen IDs: {sorted(unknown)}")
        selected = [r for r in selected if r["physical_id"] in args.specimen]
    if args.scan and {r["scan_id"] for r in selected} != set(args.scan):
        raise ValueError("Some requested scan IDs conflict with the other selection filters")
    if not args.scan:
        groups = {}
        for row in sorted(selected,key=lambda r:(r["scanner"],r["stage"],r["physical_id"],r["scan_id"])):
            groups.setdefault((row["scanner"],row["stage"]),[]).append(row)
        selected = [group[i] for i in range(max((len(v) for v in groups.values()),default=0))
                    for group in groups.values() if i < len(group)]
    limit = args.limit if args.limit is not None else (None if args.all or args.scan or args.list_scans else 12)
    if limit is not None: selected=selected[:limit]
    if not selected: raise ValueError("No scans match the selection; check partition/stage/scanner/specimen filters")
    return selected


def comparison_for(row, args):
    target = args.target or row["stage"]
    against = args.against or ("partial" if target == "clean" else "clean")
    if target == against: raise ValueError(f"Target and comparison are both {target} for {row['scan_id']}")
    return target, against


def gallery(root, entries, failures):
    esc=lambda value:html.escape(str(value),quote=True)
    cards=[]
    for item in entries:
        folder=item["folder"]
        image=folder / item["figure"]
        rel=image.relative_to(root).as_posix()
        links=' | '.join(f'<a href="{esc(p.relative_to(root).as_posix())}">{esc(p.parent.name)}: enhanced mean/max view</a>' for p in sorted(folder.glob('*/allocation_detail.png')))
        cards.append(f'<article data-scanner="{esc(item["scanner"])}" data-stage="{esc(item["stage"])}"><h2>{esc(item["scan_id"])}</h2><p>{esc(item["partition"])} | {esc(item["target"])} versus {esc(item["against"])} | specimen {esc(item["physical_id"])}</p><p>{links}</p><a href="{esc(rel)}"><img loading="lazy" src="{esc(rel)}" alt="Classifier explanation for {esc(item["scan_id"])}"></a></article>')
    scanner_options=''.join(f'<option>{esc(s)}</option>' for s in sorted({i["scanner"] for i in entries}))
    stage_options=''.join(f'<option>{esc(s)}</option>' for s in STAGES if any(i["stage"]==s for i in entries))
    issues=''.join(f'<li>{esc(f["scan_id"])}: {esc(f["error"])}</li>' for f in failures)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Stage classifier explanations</title>
<style>body{font:15px system-ui;background:#f4f6f8;color:#182332;margin:0}main{max-width:1600px;margin:24px auto;padding:0 20px}header{position:sticky;top:0;background:#f4f6f8ed;padding:12px 0;z-index:1}h1{font-size:23px;margin:0 0 10px}h2{font-size:17px;margin:0}select{padding:8px;margin-right:12px}article{background:white;border:1px solid #dce2e9;border-radius:10px;padding:16px;margin:20px 0}article img{width:100%;height:auto}p{color:#526176}a{color:#175d98}</style><main><header><h1>Stage classifier explanations</h1><label>Scanner <select id="scanner"><option value="">All scanners</option>SCANNERS</select></label><label>Recorded stage <select id="stage"><option value="">All stages</option>STAGES</select></label><a href="batch_decisions.csv">Combined decision summary</a><p>Click a figure to open it at full resolution. Numeric colour scales can differ between models and scans.</p></header>CARDS<ul>ISSUES</ul></main>
<script>function filter(){document.querySelectorAll('article').forEach(a=>{a.hidden=(scanner.value&&a.dataset.scanner!==scanner.value)||(stage.value&&a.dataset.stage!==stage.value)})}const scanner=document.getElementById('scanner'),stage=document.getElementById('stage');scanner.addEventListener('change',filter);stage.addEventListener('change',filter);</script></html>'''
    page=page.replace('SCANNERS',scanner_options).replace('STAGES',stage_options).replace('CARDS',''.join(cards)).replace('ISSUES',issues)
    (root / "index.html").write_text(page,encoding="utf-8")


def pool(features, weights, max_eligible, keep=None):
    if keep is None:
        keep = np.ones(len(features), bool)
    w = weights * keep
    eligible = max_eligible & keep
    if w.sum() <= 0 or not eligible.any():
        raise ValueError("Region removal leaves no valid pooling support")
    return np.concatenate((np.average(features, axis=0, weights=w), features[eligible].max(axis=0)))


def load_cache(run, row, mode):
    config = read_json(run / "features" / f"{mode.split('_')[0]}_config.json")
    file = feature_path(run, mode, row)
    with np.load(file, allow_pickle=False) as z:
        if z["key"].item() != signature([signature(config), row["geometry_sha256"], row["row_id"]]):
            raise ValueError(f"Feature provenance mismatch: {file}")
        if "local" not in z:
            raise ValueError(f"Local features missing for {mode}. Repeat original extraction with --spatial.")
        local = z["local"].astype(np.float64)
        if mode.startswith("dinov3"):
            coverage = z["coverage"].astype(np.float64)
            x = local.reshape(local.shape[0], -1).T
            w = coverage.ravel(); eligible = w >= .05
            meta = {"coverage": coverage, "shape": coverage.shape}
        else:
            x = local; w = np.ones(len(x)); eligible = np.ones(len(x), bool)
            meta = {k: z[k] for k in ("xyz_scaled", "scale_mm", "rotation", "center_mm")}
            meta["xyz_mm"] = (meta["xyz_scaled"] * meta["scale_mm"]) @ meta["rotation"].T + meta["center_mm"]
        full = z["pooled"].astype(np.float64)
    if not np.allclose(pool(x, w, eligible), full, atol=2e-5, rtol=2e-5):
        raise ValueError("Local features do not reproduce the saved stage-classifier input")
    return dict(x=x, w=w, eligible=eligible, pooled=full, meta=meta, config=config, cache_sha256=digest(file))


def load_head(run, mode, source):
    path = run / source / RESULT_SUBDIR / f"{mode}_all_development_head.joblib"
    head = joblib.load(path)
    clf = head.named_steps["logisticregression"]
    if not np.array_equal(clf.classes_, np.arange(len(STAGES))):
        raise ValueError("Unexpected stage-head class order")
    return head, path


def margin(head, pooled, target, against):
    logits = head.decision_function(np.asarray(pooled)[None])[0]
    return float(logits[target] - logits[against])


def decomposition(head, data, target, against):
    scaler = head.named_steps["standardscaler"]
    clf = head.named_steps["logisticregression"]
    coefficient = clf.coef_[target] - clf.coef_[against]
    standardized = scaler.transform(data["pooled"][None])[0]
    terms = coefficient * standardized
    intercept = float(clf.intercept_[target] - clf.intercept_[against])
    dimensions = data["x"].shape[1]
    gradients = coefficient / scaler.scale_
    mean_gradient, max_gradient = gradients[:dimensions], gradients[dimensions:]
    mean_ref = scaler.mean_[:dimensions]
    mean_map = data["w"] / data["w"].sum() * ((data["x"] - mean_ref) @ mean_gradient)
    maxima = data["pooled"][dimensions:]
    winners = (data["x"] == maxima) & data["eligible"][:, None]
    count = winners.sum(axis=0)
    if not count.all():
        raise ValueError("Max-pooling winners could not be located")
    max_map = winners @ (terms[dimensions:] / count)
    actual = margin(head, data["pooled"], target, against)
    if not np.isclose(terms.sum() + intercept, actual, atol=2e-4, rtol=2e-5):
        raise ValueError("Feature contributions do not reconstruct the classifier margin")
    if not np.isclose(mean_map.sum() + max_map.sum() + intercept, actual, atol=2e-4, rtol=2e-5):
        raise ValueError("Spatial allocation does not reconstruct the classifier margin")
    return dict(terms=terms, standardized=standardized, coefficient=coefficient,
                gradients=gradients, intercept=intercept, margin=actual,
                mean_map=mean_map, max_map=max_map, combined=mean_map + max_map)


def native_xy(data, image_shape, center, field):
    h, w = image_shape
    if "shape" in data["meta"]:
        gh, gw = data["meta"]["shape"]
        yy, xx = np.mgrid[:gh, :gw]
        return np.column_stack(((xx.ravel() + .5) * w / gw - .5, (yy.ravel() + .5) * h / gh - .5))
    xyz = data["meta"]["xyz_mm"]
    return np.column_stack(((xyz[:, 0] - center[0] + field / 2) * w / field - .5,
                            (center[1] + field / 2 - xyz[:, 1]) * h / field - .5))


def tile_ids(xy, shape, grid):
    h, w = shape
    cols = np.clip(np.floor((xy[:, 0] + .5) * grid / w).astype(int), 0, grid - 1)
    rows = np.clip(np.floor((xy[:, 1] + .5) * grid / h).astype(int), 0, grid - 1)
    return rows * grid + cols


def image_values(data, scores, valid, center, field):
    from make_stage_heatmaps import patch_overlay, point_overlay
    if "shape" in data["meta"]:
        return patch_overlay(scores.reshape(data["meta"]["shape"]), data["meta"]["coverage"], valid.shape[1], valid.shape[0])
    return point_overlay(data["meta"]["xyz_mm"], scores, center, field, valid)


def plot_signed(ax, original, valid, values, title, limit=None):
    shown = np.where(valid, values, np.nan)
    finite = np.abs(shown[np.isfinite(shown)])
    limit = max(1e-8, float(np.quantile(finite, .99))) if limit is None else max(limit, 1e-8)
    ax.imshow(original)
    im = ax.imshow(np.ma.masked_invalid(shown), cmap="coolwarm", vmin=-limit, vmax=limit, alpha=.8)
    ax.set_title(title, fontsize=9); ax.set_axis_off()
    return im


def allocation_detail(original, valid, maps, result, mode, target, against, folder):
    """Display-only contrast enhancement; preserve all original evidence values."""
    fig, axes=plt.subplots(1,3,figsize=(12,4.4),layout="constrained")
    gray=np.asarray(Image.fromarray(original).convert("L"))
    fig.suptitle(f"{mode} | {STAGES[target]} versus {STAGES[against]}\nSeparate colour scales for visibility; compare numeric colourbars and summed contributions",fontsize=11)
    names=["Combined allocation","Mean-pooling allocation","Max-pooling allocation"]
    sums=[result["combined"].sum(),result["mean_map"].sum(),result["max_map"].sum()]
    for ax,values,name,total in zip(axes,maps,names,sums):
        values=np.where(valid,values,np.nan)
        finite=np.abs(values[np.isfinite(values)])
        limit=max(1e-10,float(np.quantile(finite,.99)))
        ax.imshow(gray,cmap="gray",vmin=0,vmax=255)
        im=ax.imshow(np.ma.masked_invalid(values),cmap="coolwarm",vmin=-limit,vmax=limit,alpha=.94)
        ax.set_title(f"{name}\nNative contributions sum to {total:+.3f}",fontsize=10)
        ax.set_axis_off();fig.colorbar(im,ax=ax,shrink=.72,label="Allocated logit contribution")
    fig.savefig(folder / "allocation_detail.png",dpi=170,bbox_inches="tight");plt.close(fig)


def region_deletions(head, data, native_tiles, grid, target, against):
    baseline = margin(head, data["pooled"], target, against)
    heat = np.full(grid * grid, np.nan)
    rows = []
    for region in np.unique(native_tiles[data["w"] > 0]):
        removed = native_tiles == region
        fraction = float(data["w"][removed].sum() / data["w"].sum())
        if fraction >= .95: continue
        changed = pool(data["x"], data["w"], data["eligible"], ~removed)
        after = margin(head, changed, target, against)
        heat[region] = baseline - after
        prob = head.predict_proba(changed[None])[0]
        rows.append(dict(tile=int(region), removed_support_fraction=fraction, baseline_margin=baseline,
                         after_margin=after, margin_drop=baseline-after,
                         **{f"p_{s}": float(prob[i]) for i, s in enumerate(STAGES)}))
    return heat.reshape(grid, grid), rows


def feature_controls(head, data, evidence, target, against, seed=123):
    baseline = margin(head, data["pooled"], target, against)
    active = np.flatnonzero(data["w"] > 0)
    rank = active[np.argsort(evidence[active])[::-1]]
    rows = []
    for fraction in (.05, .1, .2, .4):
        quota = fraction * data["w"].sum()
        n = min(len(rank) - 1, 1 + int(np.searchsorted(np.cumsum(data["w"][rank]), quota)))
        ids = rank[:n]
        keep = np.ones(len(data["x"]), bool); keep[ids] = False
        changed = pool(data["x"], data["w"], data["eligible"], keep)
        rows.append(dict(policy="largest positive allocated contributions", requested_fraction=fraction,
                         actual_fraction=float(data["w"][ids].sum()/data["w"].sum()),
                         margin_drop=baseline-margin(head, changed, target, against)))
        rng = np.random.default_rng(seed)
        for repeat in range(10):
            random_rank = rng.permutation(active)
            n = min(len(random_rank)-1, 1 + int(np.searchsorted(np.cumsum(data["w"][random_rank]), quota)))
            ids = random_rank[:n]
            keep = np.ones(len(data["x"]), bool); keep[ids] = False
            changed = pool(data["x"], data["w"], data["eligible"], keep)
            rows.append(dict(policy="random locations", repeat=repeat, requested_fraction=fraction,
                             actual_fraction=float(data["w"][ids].sum()/data["w"].sum()),
                             margin_drop=baseline-margin(head, changed, target, against)))
    return rows


def feature_histograms(args, row, mode, head, indices, out):
    scaler = head.named_steps["standardscaler"]
    rows = [r for r in read_json(args.run / "cohort.json")["rows"] if r["partition"] == "development" and r["scanner"] == row["scanner"]]
    observations = []
    for r in rows:
        path = feature_path(args.run, mode, r)
        if not path.is_file(): continue
        with np.load(path, allow_pickle=False) as z:
            config = signature(read_json(args.run / "features" / f"{mode.split('_')[0]}_config.json"))
            if z["key"].item() != signature([config, r["geometry_sha256"], r["row_id"]]):
                raise ValueError("A development cache has mismatched provenance")
            x = scaler.transform(z["pooled"][None])[0]
        observations.append((r, x))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    fig.suptitle(f"{mode}: top decision features across development scans ({row['scanner']})\nDescriptive distributions; repeated scans of specimens are not independent", fontsize=11)
    records = []
    width = head.named_steps["logisticregression"].coef_.shape[1] // 2
    for ax, index in zip(axes.flat, indices[:6]):
        for i, stage in enumerate(STAGES):
            values = [x[index] for r, x in observations if r["stage"] == stage]
            if values: ax.hist(values, bins=12, histtype="step", density=True, label=stage)
            records.extend(dict(scan_id=r["scan_id"], physical_id=r["physical_id"], stage=stage,
                                branch="mean" if index < width else "max", channel=int(index % width),
                                standardized_value=float(x[index])) for r, x in observations if r["stage"] == stage)
        ax.set_title(f"{'mean' if index < width else 'max'} channel {index % width}", fontsize=10)
        ax.set_xlabel("Training-standardized pooled feature")
        ax.legend(fontsize=8)
    fig.savefig(out / "feature_stage_distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    write_csv(out / "feature_stage_distributions.csv", records)


def cached(args, row, out):
    original, valid, center, field = render(args.run, row)
    target = STAGES.index(args.target or row["stage"]); against = STAGES.index(args.against)
    if target == against: raise ValueError("Target and comparison stage must differ")
    overview, axes = plt.subplots(len(args.modes), 5, figsize=(16, 3.1*len(args.modes)), squeeze=False, layout="constrained")
    provenance = "development specimen used by the original all-development head" if row["partition"] == "development" else "held-out test specimen"
    overview.suptitle(f"{row['scan_id']} | {STAGES[target]} versus {STAGES[against]} | {provenance}\nExact stage-head allocation and frozen-feature region deletion (not input occlusion)", fontsize=11)
    summary = []
    for axrow, mode in zip(axes, args.modes):
        head, head_path = load_head(args.run, mode, args.head_source)
        data = load_cache(args.run, row, mode)
        result = decomposition(head, data, target, against)
        folder = out / mode; folder.mkdir()
        probabilities = head.predict_proba(data["pooled"][None])[0]
        dimensions = data["x"].shape[1]
        order = np.argsort(np.abs(result["terms"]))[::-1]
        top = order[:args.top]
        records = []
        for index in order:
            replacement = data["pooled"].copy()
            replacement[index] = head.named_steps["standardscaler"].mean_[index]
            after = margin(head, replacement, target, against)
            records.append(dict(pooled_index=int(index), channel=int(index % dimensions),
                branch="mean" if index < dimensions else "max", raw_feature=float(data["pooled"][index]),
                standardized_feature=float(result["standardized"][index]),
                contrast_coefficient=float(result["coefficient"][index]),
                contribution=float(result["terms"][index]),
                margin_after_replacing_feature_with_training_mean=after))
        write_csv(folder / "feature_contributions.csv", records)
        xy = native_xy(data, valid.shape, center, field)
        tiles = tile_ids(xy, valid.shape, args.grid)
        drop_map, drops = region_deletions(head, data, tiles, args.grid, target, against)
        write_csv(folder / "frozen_feature_region_deletion.csv", drops)
        controls = feature_controls(head, data, result["combined"], target, against)
        write_csv(folder / "frozen_feature_deletion_controls.csv", controls)
        axrow[0].imshow(original); axrow[0].set_axis_off()
        axrow[0].set_title(f"{mode}\nPredicted {STAGES[int(probabilities.argmax())]}; margin {result['margin']:+.2f}", fontsize=9)
        maps = [image_values(data, result[key], valid, center, field) for key in ("combined", "mean_map", "max_map")]
        allocation_detail(original,valid,maps,result,mode,target,against,folder)
        limit = max(1e-8, *(float(np.nanquantile(np.abs(m[valid]), .99)) for m in maps))
        for ax, values, name in zip(axrow[1:4], maps, ["Combined allocation", "Mean-pooling allocation", "Max-pooling winner allocation"]):
            im = plot_signed(ax, original, valid, values, name, limit)
            overview.colorbar(im, ax=ax, shrink=.65)
        up = np.array(Image.fromarray(drop_map.astype(np.float32)).resize((valid.shape[1], valid.shape[0]), Image.Resampling.NEAREST))
        im = plot_signed(axrow[4], original, valid, up, "Repool after regional feature deletion\nRed: removal reduces target-vs-comparison margin")
        overview.colorbar(im, ax=axrow[4], shrink=.65)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        names = [f"{'mean' if j < dimensions else 'max'} ch {j % dimensions}" for j in top]
        values = result["terms"][top]
        axs[0].barh(np.arange(len(top)), values, color=np.where(values >= 0, "#bd4147", "#376cc1"))
        axs[0].set_yticks(np.arange(len(top)), names); axs[0].invert_yaxis(); axs[0].axvline(0,color="gray",lw=.7)
        axs[0].set_xlabel("Exact contribution to target-versus-comparison logit margin")
        other = result["terms"].sum() - values.sum()
        axs[0].set_title(f"{mode}: strongest channels\nIntercept {result['intercept']:+.2f}; other channels {other:+.2f}; total {result['margin']:+.2f}")
        fractions = sorted({r["requested_fraction"] for r in controls})
        for policy in ("largest positive allocated contributions", "random locations"):
            vals = [[r["margin_drop"] for r in controls if r["policy"] == policy and r["requested_fraction"] == f] for f in fractions]
            means = np.array([np.mean(v) for v in vals]); lo=np.array([np.min(v) for v in vals]); hi=np.array([np.max(v) for v in vals])
            axs[1].plot(np.array(fractions)*100, means, marker="o", label=policy)
            axs[1].fill_between(np.array(fractions)*100,lo,hi,alpha=.15)
        axs[1].axhline(0,color="gray",lw=.7); axs[1].legend(fontsize=8)
        axs[1].set_xlabel("Requested removal of native feature support (%)"); axs[1].set_ylabel("Drop in classifier logit margin")
        axs[1].set_title("Frozen-feature deletion; no encoder pass\nRandom band: range of 10 repeats")
        fig.savefig(folder / "decision_features.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        feature_histograms(args, row, mode, head, order, folder)
        channel_fig, ca = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
        channel_fig.suptitle(f"{mode}: locations associated with top channels\nContextual features; heat colours are not plaque labels", fontsize=11)
        for ax, j in zip(ca.flat, top[:6]):
            ch = int(j % dimensions)
            if j < dimensions:
                local = data["w"] / data["w"].sum() * (data["x"][:, ch]-head.named_steps["standardscaler"].mean_[j]) * result["gradients"][j]
            else:
                winners = (data["x"][:, ch] == data["pooled"][j]) & data["eligible"]
                local = winners * result["terms"][j] / winners.sum()
            image = image_values(data, local, valid, center, field)
            im = plot_signed(ax, original, valid, image, f"{'Mean' if j<dimensions else 'Max'} channel {ch}: {result['terms'][j]:+.2f}")
            channel_fig.colorbar(im, ax=ax, shrink=.7)
        channel_fig.savefig(folder / "top_channel_locations.png", dpi=150, bbox_inches="tight"); plt.close(channel_fig)
        np.savez_compressed(folder / "exact_evidence.npz", **result, native_xy_pixels=xy, native_weights=data["w"],
                            feature_region_drop=drop_map, **data["meta"])
        summary.append(dict(scan_id=row["scan_id"],physical_id=row["physical_id"],scanner=row["scanner"],mode=mode,recorded_stage=row["stage"],
            partition=row["partition"], target_stage=STAGES[target],comparison_stage=STAGES[against],
            predicted_stage=STAGES[int(probabilities.argmax())],margin=result["margin"],intercept=result["intercept"],
            mean_branch_contribution=float(result["terms"][:dimensions].sum()),
            max_branch_contribution=float(result["terms"][dimensions:].sum()),
            head_sha256=digest(head_path), feature_sha256=data["cache_sha256"],
            **{f"p_{s}":float(probabilities[i]) for i,s in enumerate(STAGES)}))
        print(f"{mode}: exact margin={result['margin']:+.3f}; saved {folder}", flush=True)
    overview.savefig(out / "classifier_evidence.png",dpi=160,bbox_inches="tight"); plt.close(overview)
    write_csv(out / "decision_summary.csv", summary)
    return summary


def make_reencoder(args, row, data, mode):
    import torch
    from models_2d import Encoder, commit
    config = data["config"]
    if digest(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the stage-classification extraction")
    if config.get("upstream_commit") != commit(args.repo):
        raise ValueError("Encoder repository commit differs from original extraction")
    if args.encoder == "dinov3":
        encoder = Encoder("dinov3", {"dinov3":dict(repo=str(args.repo),checkpoint=str(args.checkpoint))}, args.device)
        if encoder.meta["encoder_state_sha256"] != config["encoder_state_sha256"]:
            raise ValueError("Loaded DINO encoder state differs")
        image_dir = args.run / "renders" / row["partition"] / "images" / row["scanner"] / row["sample_name"]
        image = np.array(Image.open(image_dir / ("gray.png" if mode.endswith("gray") else "rgb.png")).convert("RGB"))
        mask = np.array(Image.open(image_dir / "valid_mask.png")) > 0
        def encode(region=None, fill="median"):
            changed = image.copy()
            if region is not None:
                color = np.median(image[mask],axis=0) if fill == "median" else np.array([127,127,127])
                changed[region] = np.rint(color).astype(np.uint8)
            return encoder.extract(changed, mask)[0]
        return encode
    from concerto_backend import ConcertoEncoder
    copy = dict(row)
    if args.snapshot:
        copy["mesh_path"] = str(args.snapshot / row["relative_mesh_path"])
    if digest(copy["mesh_path"]) != row["geometry_sha256"]:
        raise ValueError("Source geometry changed")
    settings = config["surface"]
    surface = make_surface(copy, settings, "concerto")
    if not np.allclose(surface["coord"], data["meta"]["xyz_scaled"], atol=1e-7, rtol=1e-6):
        raise ValueError("Recreated point order/coordinates differ from original features")
    encoder = ConcertoEncoder(SimpleNamespace(repo=args.repo,checkpoint=args.checkpoint,
        grid_size=config["voxel_mm"] / settings["scale_mm"], attention_patch=config["attention_patch"]), args.device)
    def encode(region=None, fill="median"):
        arrays = {k:surface[k].copy() for k in ("coord","color","normal","grid_origin")}
        if region is not None:
            if args.intervention == "remove-points":
                for k in ("coord","color","normal"): arrays[k] = arrays[k][~region]
            else:
                arrays["color"][region] = np.median(surface["color"],axis=0) if fill == "median" else 127/255
        tensors = {k:torch.from_numpy(v)[None] for k,v in arrays.items()}
        with torch.inference_mode(): local,_ = encoder(tensors, use_color=mode == "concerto_rgb")
        local = local[0].float().cpu().numpy()
        return np.concatenate((local.mean(axis=0), local.max(axis=0)))
    return encode


def input_occlusion(args, row, out):
    mode = args.mode
    target=STAGES.index(args.target or row["stage"]); against=STAGES.index(args.against)
    if target == against: raise ValueError("Target and comparison must differ")
    if mode.split('_')[0] != args.encoder: raise ValueError("Mode and encoder differ")
    if args.encoder == "dinov3" and args.intervention == "remove-points": raise ValueError("Point removal requires Concerto")
    head, head_path = load_head(args.run, mode, args.head_source)
    data = load_cache(args.run,row,mode)
    original,valid,center,field = render(args.run,row)
    encode = make_reencoder(args,row,data,mode)
    baseline = encode()
    baseline_margin = margin(head,baseline,target,against)
    cached_margin = margin(head,data["pooled"],target,against)
    baseline_p = head.predict_proba(baseline[None])[0]
    cache_p = head.predict_proba(data["pooled"][None])[0]
    if not np.allclose(baseline,data["pooled"],atol=1e-3,rtol=1e-3) or abs(baseline_margin-cached_margin)>.02 or not np.allclose(baseline_p,cache_p,atol=2e-3):
        raise ValueError(f"Baseline re-encoding differs from saved stage prediction (margin {baseline_margin:+.4f} versus {cached_margin:+.4f}); inspect environments/settings before perturbing inputs")
    if args.encoder == "dinov3":
        yy,xx=np.mgrid[:valid.shape[0],:valid.shape[1]]
        tile=tile_ids(np.column_stack((xx.ravel(),yy.ravel())),valid.shape,args.grid).reshape(valid.shape)
        eligible=valid
    else:
        tile=tile_ids(native_xy(data,valid.shape,center,field),valid.shape,args.grid)
        eligible=np.ones(len(tile),bool)
    fills = ["remove"] if args.intervention == "remove-points" else ["median","neutral"]
    results=[]; maps={}
    for fill in fills:
        heat=np.full(args.grid*args.grid,np.nan)
        for region in np.unique(tile[eligible]):
            selected=(tile==region)&eligible
            fraction=float(selected.sum()/eligible.sum())
            if fraction>=.95:continue
            changed=encode(selected,fill)
            after=margin(head,changed,target,against)
            prob=head.predict_proba(changed[None])[0]
            heat[region]=baseline_margin-after
            results.append(dict(tile=int(region),intervention=args.intervention,fill=fill,
                removed_or_replaced_fraction=fraction,baseline_margin=baseline_margin,after_margin=after,
                margin_drop=baseline_margin-after,**{f"p_{s}":float(prob[i]) for i,s in enumerate(STAGES)}))
            print(f"{mode} {fill}: tile {region}, margin drop {baseline_margin-after:+.3f}",flush=True)
        maps[fill]=heat.reshape(args.grid,args.grid)
    write_csv(out / "input_occlusion.csv",results)
    np.savez_compressed(out / "input_occlusion.npz",**maps,baseline_probabilities=baseline_p,
                        baseline_margin=baseline_margin,cached_margin=cached_margin)
    fig,axes=plt.subplots(1,len(fills)+1,figsize=(5*(len(fills)+1),4),layout="constrained")
    axes[0].imshow(original);axes[0].set_title(f"Original\nP({STAGES[target]})={baseline_p[target]:.3f}");axes[0].set_axis_off()
    common_limit=max(1e-8,*(float(np.nanquantile(np.abs(m),.99)) for m in maps.values()))
    for ax,fill in zip(axes[1:],fills):
        values=np.array(Image.fromarray(maps[fill].astype(np.float32)).resize((valid.shape[1],valid.shape[0]),Image.Resampling.NEAREST))
        im=plot_signed(ax,original,valid,values,f"Input intervention: {fill}\nPositive: edit reduces target-vs-comparison margin", common_limit)
        fig.colorbar(im,ax=ax,shrink=.7,label="Drop in logit margin")
    fig.suptitle(f"{mode} | {row['scan_id']} | {STAGES[target]} vs {STAGES[against]}\nOriginal input changed and frozen encoder rerun; grid {args.grid} x {args.grid}",fontsize=11)
    fig.savefig(out / "input_occlusion.png",dpi=160,bbox_inches="tight");plt.close(fig)
    example_fig,example_axes=plt.subplots(2,3,figsize=(10,7),layout="constrained")
    example_fig.suptitle("Regions with largest absolute effects on the same classifier\nOutlined area is edited; compare signs, fill styles and edited fractions",fontsize=11)
    from matplotlib.patches import Rectangle
    for ax,item in zip(example_axes.flat,sorted(results,key=lambda r:abs(r["margin_drop"]),reverse=True)[:6]):
        ax.imshow(original)
        y,x=divmod(item["tile"],args.grid)
        width,height=valid.shape[1]/args.grid,valid.shape[0]/args.grid
        ax.add_patch(Rectangle((x*width-.5,y*height-.5),width,height,fill=False,edgecolor="yellow",linewidth=2))
        ax.set_title(f"Tile {item['tile']} ({item['fill']}): margin drop {item['margin_drop']:+.2f}\nP({STAGES[target]}) {baseline_p[target]:.3f} -> {item['p_'+STAGES[target]]:.3f}",fontsize=9)
        ax.set_axis_off()
    example_fig.savefig(out / "input_top_regions.png",dpi=150,bbox_inches="tight");plt.close(example_fig)
    write_json(out / "input_baseline.json",dict(head_sha256=digest(head_path),
        feature_sha256=data["cache_sha256"],baseline_margin=baseline_margin,cached_margin=cached_margin,
        baseline_probabilities=baseline_p.tolist(),cached_probabilities=cache_p.tolist()))
    return [dict(scan_id=row["scan_id"],physical_id=row["physical_id"],scanner=row["scanner"],mode=mode,
        recorded_stage=row["stage"],partition=row["partition"],target_stage=STAGES[target],comparison_stage=STAGES[against],
        margin=baseline_margin,predicted_stage=STAGES[int(baseline_p.argmax())],
        **{f"p_{s}":float(baseline_p[i]) for i,s in enumerate(STAGES)})]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("cached","input"):
        sub=commands.add_parser(name)
        sub.add_argument("--run",type=Path,required=True)
        sub.add_argument("--scan",nargs="+",help="One or more exact scan IDs; omit for a filtered batch")
        sub.add_argument("--partition",choices=["test","development","all"],help="Batches default to test; explicit IDs can come from either partition")
        sub.add_argument("--stage",nargs="+",choices=STAGES,help="Filter by recorded stage")
        sub.add_argument("--scanner",nargs="+",choices=["LABscanner","TRIOS3","TRIOS5","iTERO"])
        sub.add_argument("--specimen",nargs="+",help="Physical specimen IDs, e.g. BCG4-2, keeping its scanner views/stages")
        count=sub.add_mutually_exclusive_group()
        count.add_argument("--limit",type=int,help="Maximum number of scans, default 12 for automatic selection")
        count.add_argument("--all",action="store_true",help="Process every scan matching the filters")
        sub.add_argument("--list-scans",action="store_true",help="List matches without loading features or writing output")
        sub.add_argument("--out",type=Path)
        sub.add_argument("--target",choices=STAGES)
        sub.add_argument("--against",choices=STAGES,help="Default clean; for a clean target, default partial")
        sub.add_argument("--head-source",choices=["locked_test","development_cv"],default="locked_test")
        sub.add_argument("--grid",type=int,default=8)
        if name=="cached":
            sub.add_argument("--modes",nargs="+",choices=MODES,default=["dinov3_rgb","concerto_rgb"])
            sub.add_argument("--top",type=int,default=12)
        else:
            sub.add_argument("--encoder",choices=["dinov3","concerto"],required=True)
            sub.add_argument("--mode",choices=MODES,required=True)
            sub.add_argument("--repo",type=Path,required=True)
            sub.add_argument("--checkpoint",type=Path,required=True)
            sub.add_argument("--snapshot",type=Path,help="Same approved snapshot root at a different Windows/WSL path")
            sub.add_argument("--device",default="cuda",choices=["cuda","cpu"])
            sub.add_argument("--intervention",choices=["appearance","remove-points"],default="appearance")
    args=parser.parse_args()
    if args.grid<2 or (args.command=="cached" and args.top<6): parser.error("Require grid>=2 and top>=6")
    if args.limit is not None and args.limit<1: parser.error("Require a positive --limit")
    if not args.list_scans and args.out is None: parser.error("--out is required unless listing scans")
    try:
        rows=read_json(args.run / "cohort.json")["rows"]
        selected=choose_scans(rows,args)
        if args.list_scans:
            for row in selected:print(row["scan_id"],row["scanner"],row["stage"],row["partition"],row["physical_id"],sep="\t")
            return
        for row in selected:comparison_for(row,args)
        if args.out.exists(): raise ValueError("Output exists; choose a new --out")
        args.out.mkdir(parents=True)
        entries,failures,summary=[],[],[]
        write_json(args.out / "selection.json",dict(version=VERSION,cohort_sha256=digest(args.run / "cohort.json"),
            command=args.command,scans=[r["scan_id"] for r in selected],head_source=args.head_source,grid=args.grid))
        for index,row in enumerate(selected,1):
            target,against=comparison_for(row,args)
            folder=args.out if len(selected)==1 else args.out / re.sub(r"[^A-Za-z0-9_.-]","_",row["scan_id"]+"_"+str(row["row_id"]))
            per_scan=SimpleNamespace(**vars(args));per_scan.target=target;per_scan.against=against
            try:
                folder.mkdir(parents=True,exist_ok=True)
                write_json(folder / "analysis.json",dict(version=VERSION,scan_id=row["scan_id"],partition=row["partition"],
                    cohort_sha256=digest(args.run / "cohort.json"),command=args.command,head_source=args.head_source,
                    target=target,against=against,grid=args.grid,in_sample_head_specimen=row["partition"]=="development",
                    meaning="Classifier explanation. Cached deletion holds encoder features fixed; input interventions rerun the encoder. Neither is validated plaque segmentation."))
                print(f"[{index}/{len(selected)}] {row['scan_id']}: {target} versus {against}",flush=True)
                summary.extend((cached if args.command=="cached" else input_occlusion)(per_scan,row,folder))
                entries.append(dict(row,folder=folder,target=target,against=against,
                                    figure="classifier_evidence.png" if args.command=="cached" else "input_occlusion.png"))
            except (OSError,ValueError,KeyError,ImportError,RuntimeError) as exc:
                failures.append(dict(scan_id=row["scan_id"],error=str(exc)))
                print(f"Failed {row['scan_id']}: {exc}",flush=True)
            finally:plt.close("all")
        write_csv(args.out / "batch_decisions.csv",summary)
        if failures:write_csv(args.out / "failures.csv",failures)
        gallery(args.out,entries,failures)
        print(f"Saved classifier explanations: {args.out}",flush=True)
        if failures:raise ValueError(f"{len(failures)} scan(s) failed; successful figures are saved. See failures.csv.")
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":main()
