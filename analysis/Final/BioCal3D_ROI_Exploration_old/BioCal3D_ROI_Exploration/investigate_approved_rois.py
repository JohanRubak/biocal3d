#!/usr/bin/env python3
"""BioCal3D approved-ROI exploration. Read-only inputs; no labels required.

Run with --help. Methods and the output guide are in README.md.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import html
import io
import itertools
import json
import os
import platform
import re
import sys
import time
from importlib.metadata import version as package_version
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")  # Never loads Qt; safe in the existing Windows conda environment.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image
try:
    from skimage.color import rgb2lab, deltaE_ciede2000
except ImportError:
    raise SystemExit("Missing scikit-image. Run: python -m pip install -r requirements.txt")

VERSION = "1.0.0"
SCANNERS = ["iTERO", "TRIOS3", "TRIOS5", "LABscanner"]
COLOURS = dict(zip(SCANNERS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]))
STAGES = ["baseline", "partial", "clean"]
LAB_BINS = [np.linspace(0, 100, 101), np.linspace(-128, 128, 257), np.linspace(-128, 128, 257)]
CHANNELS = ["L", "a", "b"]
PATTERN = re.compile(r"BCG(?P<g>\d+)T(?P<t>\d+)-(?P<n>\d+)", re.I)
NATIVE = ["vertex_count", "face_count", "geometric_vertex_count", "surface_area_mm2",
          "vertices_per_mm2", "geometric_vertices_per_mm2", "faces_per_mm2", "edge_median_um",
          "triangle_area_median_um2", "height_mean_abs_um", "height_rms_um",
          "height_range90_um", "slope_median_deg", "slope_p95_deg"]


def write_json(path, obj):
    def default(v):
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else {}


def qweighted(values, weights, quantiles):
    x, w = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    take = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[take], w[take]
    if not len(x):
        return np.full(np.shape(quantiles), np.nan)
    order = np.argsort(x, kind="stable"); x, w = x[order], w[order]
    cdf = (np.cumsum(w) - .5*w)/w.sum()
    return np.interp(quantiles, cdf, x)


def normalise_physical_id(value):
    value = str(value).strip()
    m = re.fullmatch(r"(?:BCG|G)(\d+)(?:-|_S)(\d+)", value, re.I)
    if not m:
        raise ValueError(f"Unrecognised specimen ID {value!r}; use BCG1-1 or G1_S1.")
    return f"BCG{int(m[1])}-{int(m[2])}"


def canonical_scanner(value):
    aliases = {"itero": "iTERO", "lab": "LABscanner", "labscanner": "LABscanner",
               "trios3": "TRIOS3", "trios5": "TRIOS5"}
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if key not in aliases:
        raise ValueError(f"Unknown scanner {value!r}; this protocol supports {SCANNERS}.")
    return aliases[key]


def discover(root, split_file=None):
    """Only prepared sample folders; never annotations or original full scans."""
    sample_root = root / "samples" if (root / "samples").is_dir() else root
    files = sorted(sample_root.rglob("geometry.npz"))
    if not files:
        raise ValueError("No geometry.npz found. Choose the rebuilt Approved ROIs dataset root.")
    index = read_json(root / "dataset_index.json")
    indexed = {}
    scans = index.get("scans", []) if isinstance(index, dict) else []
    if isinstance(scans, list):
        for row in scans:
            if isinstance(row, dict) and row.get("scan_id"):
                indexed[str(row["scan_id"])] = row
    rows = []
    for path in files:
        rel = path.relative_to(sample_root)
        if any(p.startswith(("_", ".")) or p.lower() == "annotations" for p in rel.parts):
            continue
        m = PATTERN.search(path.parent.name)
        if not m:
            raise ValueError(f"Cannot determine group/stage/specimen from {path.parent}.")
        g, t, n = (int(m[k]) for k in ("g", "t", "n"))
        if g not in range(1, 7) or t not in ([0, 1] if g <= 2 else [0, 1, 2]):
            raise ValueError(f"Unexpected G{g}/T{t}; update the explicit protocol before analysis.")
        meta = read_json(path.parent / "metadata.json")
        scanner = meta.get("scanner")
        if not scanner:
            for part in reversed(path.parts[:-1]):
                try:
                    scanner = canonical_scanner(part); break
                except ValueError:
                    pass
        scanner = canonical_scanner(scanner)
        sample = f"BCG{g}T{t}-{n}"; sid = f"{scanner}__{sample}"
        stage = ("baseline" if t == 0 else "clean") if g <= 2 else STAGES[t]
        if meta.get("stage") and meta["stage"] != stage:
            raise ValueError(f"Stage metadata disagrees with protocol for {sid}.")
        old = indexed.get(sid, {})
        split = str(meta.get("split", old.get("split", "unassigned"))).lower()
        rows.append(dict(scan_id=sid, sample_id=sample, physical_id=f"BCG{g}-{n}",
                         scanner=scanner, group=g, specimen_number=n, timepoint=f"T{t}",
                         stage=stage, cohort="fresh" if g <= 2 else "mature", split=split,
                         sample_dir=str(path.parent), review_status=meta.get("geometry_review_status", "not_recorded")))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No eligible G1–G6 approved sample folders found.")
    if df.scan_id.duplicated().any():
        raise ValueError("Duplicate scan identities: " + ", ".join(df.loc[df.scan_id.duplicated(False), "scan_id"]))
    # A supplied split is authoritative, including when preparation metadata is older.
    if split_file:
        splits = pd.read_csv(split_file, dtype=str).fillna("")
        if not {"physical_id", "split"}.issubset(splits):
            raise ValueError("Split CSV must contain physical_id and split columns.")
        splits["physical_id"] = splits.physical_id.map(normalise_physical_id)
        splits["split"] = splits.split.str.strip().str.lower().replace({"validation": "val", "valid": "val"})
        if splits.physical_id.duplicated().any() or not set(splits.split) <= {"train", "val", "test"}:
            raise ValueError("Split CSV requires one unique row per specimen and train/val/test values.")
        assignment = splits.set_index("physical_id").split.to_dict()
        df["split"] = df.physical_id.map(assignment).fillna("unassigned")
    if (df.groupby("physical_id").split.nunique() > 1).any():
        raise ValueError("A physical specimen appears in multiple splits. All its scanners/stages must stay together.")
    return df


def mesh_measurements(vertices, faces, n_samples=50000, seed=42):
    """Area-integrated best-fit plane; exact RMS; uniform-area samples for quantiles/absolute height."""
    v, f = np.asarray(vertices, float), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected vertices [N,3] and triangular faces [M,3].")
    if not np.issubdtype(f.dtype, np.integer) or not np.isfinite(v).all() or not len(f):
        raise ValueError("Invalid vertices or face indices.")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("Face index outside vertex array.")
    tri = v[f]; cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = np.linalg.norm(cross, axis=1)/2
    good = area > 1e-14
    if not good.any():
        raise ValueError("No nondegenerate surface triangles.")
    a, t = area[good], tri[good]; total = a.sum()
    origin = np.sum(t.mean(1)*a[:, None], axis=0)/total
    centred = t-origin; summed = centred.sum(1)
    second = (np.einsum("ni,nj->nij", summed, summed) +
              np.einsum("nki,nkj->nij", centred, centred))/12
    cov = np.einsum("n,nij->ij", a, second)/total
    eig, axes = np.linalg.eigh(cov); normal = axes[:, 0]
    if normal[2] < 0:
        normal = -normal
    if abs(normal[2]) < .5:
        raise ValueError("ROI is not approximately aligned to canonical XY; inspect preparation coordinates.")
    u = np.array([1., 0., 0.])-normal[0]*normal; u /= np.linalg.norm(u)
    basis = np.column_stack([u, np.cross(normal, u), normal])
    xyz = (v-origin)@basis
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(t), int(n_samples), p=a/total)
    uv = rng.random((int(n_samples), 2)); reflect = uv.sum(1) > 1; uv[reflect] = 1-uv[reflect]
    hverts = (t-origin)@normal
    h = ((1-uv.sum(1))*hverts[ids, 0]+uv[:, 0]*hverts[ids, 1]+uv[:, 1]*hverts[ids, 2])
    # RMS of linear height over a triangle: integral(h^2)/area = ((sum h)^2 + sum h^2)/12.
    rms2 = np.sum(a*((hverts.sum(1))**2+(hverts*hverts).sum(1))/12)/total
    normals = cross[good]/(2*a[:, None])
    slope = np.degrees(np.arccos(np.clip(np.abs(normals@normal), 0, 1)))
    _, inverse = np.unique(v, axis=0, return_inverse=True)
    gf = inverse[f]
    edges = np.sort(np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]]), axis=1)
    unique_vertices = np.unique(v, axis=0)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(unique_vertices[edges[:, 1]]-unique_vertices[edges[:, 0]], axis=1)
    lengths = lengths[lengths > 0]
    row = dict(vertex_count=len(v), geometric_vertex_count=len(unique_vertices), face_count=len(f),
               degenerate_faces=int((~good).sum()), duplicated_coordinate_vertices=len(v)-len(unique_vertices),
               surface_area_mm2=float(total), vertices_per_mm2=len(v)/total,
               geometric_vertices_per_mm2=len(unique_vertices)/total, faces_per_mm2=len(f)/total,
               triangle_area_median_um2=float(np.median(a)*1e6), triangle_area_p10_um2=float(np.quantile(a, .1)*1e6),
               triangle_area_p90_um2=float(np.quantile(a, .9)*1e6), edge_median_um=float(np.median(lengths)*1000),
               edge_p10_um=float(np.quantile(lengths, .1)*1000), edge_p90_um=float(np.quantile(lengths, .9)*1000),
               height_mean_abs_um=float(np.mean(np.abs(h))*1000), height_rms_um=float(np.sqrt(max(0, rms2))*1000),
               height_range90_um=float(np.diff(np.quantile(h, [.05, .95]))[0]*1000),
               slope_median_deg=float(qweighted(slope, a, .5)), slope_p95_deg=float(qweighted(slope, a, .95)))
    return row, xyz, area


@lru_cache(maxsize=32)
def rebin_operator(size, fov, pitch):
    """Fractional pixel/cell overlap, cached for datasets with a shared FOV."""
    mmpp = fov/size
    edges = np.r_[np.arange(0, fov-1e-10, pitch), fov]
    source = np.linspace(0, fov, size+1)
    overlap = np.maximum(0., np.minimum(edges[1:, None], source[None, 1:])-
                         np.maximum(edges[:-1, None], source[None, :-1]))/mmpp
    return csr_matrix(overlap), np.diff(edges), (edges[:-1]+edges[1:])/2


def common_grid(height, valid, area, fov, pitch, min_coverage=.75):
    """Box-average existing projected surface heights into cells of fixed width in mm. No hole filling."""
    h = np.asarray(height, float); valid = np.asarray(valid, bool)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or valid.shape != h.shape or area.shape != h.shape:
        raise ValueError("Height, mask and pixel-area maps must have the same square shape.")
    mmpp = fov/h.shape[0]
    if pitch < mmpp-1e-9:
        raise ValueError(f"Analysis pitch {pitch:g} mm is finer than source spacing {mmpp:g} mm.")
    operator, widths, centres = rebin_operator(h.shape[0], float(fov), float(pitch))
    def rebin(a):
        return (operator@(operator@a).T).T
    good = valid & np.isfinite(h) & np.isfinite(area) & (area > 0)
    source_area = np.where(good, area, 0.)
    support = rebin(source_area)
    projected_area = rebin(good.astype(float))*mmpp**2
    coverage = projected_area/np.outer(widths, widths)
    value = rebin(source_area*np.where(good, h, 0.))/np.maximum(support, 1e-30)
    use = (coverage >= min_coverage) & (support > 0)
    if use.sum() < 9:
        raise ValueError(f"Fewer than 9 supported cells at pitch {pitch:g} mm.")
    source_centres = (np.arange(h.shape[0])+.5)*mmpp
    # Use actual weighted source coordinates when levelling, so partial pixel/cell
    # overlaps cannot introduce false roughness into an affine plane.
    xx = rebin(source_area*(source_centres[None, :]-fov/2))/np.maximum(support, 1e-30)
    yy = rebin(source_area*(fov/2-source_centres[:, None]))/np.maximum(support, 1e-30)
    design = np.column_stack([xx[use], yy[use], np.ones(use.sum())])
    sw = np.sqrt(support[use]); coef = np.linalg.lstsq(design*sw[:, None], value[use]*sw, rcond=None)[0]
    levelled = value-(coef[0]*xx+coef[1]*yy+coef[2]); levelled[~use] = np.nan
    weights = support[use]; z = levelled[use]; p05, p95 = qweighted(z, weights, [.05, .95])
    gy, gx = np.gradient(levelled, centres, centres)
    angle = np.degrees(np.arctan(np.hypot(gx, gy))); interior = use & np.isfinite(angle)
    row = dict(pitch_mm=pitch, supported_cells=int(use.sum()),
               visible_projected_area_mm2=float(projected_area.sum()),
               height_mean_abs_um=float(np.average(np.abs(z), weights=weights)*1000),
               height_rms_um=float(np.sqrt(np.average(z*z, weights=weights))*1000),
               height_range90_um=float((p95-p05)*1000),
               slope_median_deg=float(qweighted(angle[interior], support[interior], .5)))
    return row, levelled.astype(np.float32)


def load_and_measure(row, args):
    folder = Path(row["sample_dir"]); meta = read_json(folder/"metadata.json")
    if row["review_status"] in {"pending", "flagged", "rejected"}:
        raise ValueError("This is not an approved rebuilt ROI. Select the final Approved ROIs folder.")
    with np.load(folder/"geometry.npz", allow_pickle=False) as g:
        vertices, faces = np.asarray(g["vertices"], float), np.asarray(g["faces"])
    seed = int(hashlib.sha256(row["scan_id"].encode()).hexdigest()[:8], 16)^args.seed
    native, xyz, areas = mesh_measurements(vertices, faces, args.surface_samples, seed)
    maps_path = folder/"maps.npz"
    if not maps_path.is_file():
        raise ValueError("maps.npz is missing. Rebuild reviewed geometry/maps; external textures are not silently replaced by vertex RGB.")
    if args.verify_hashes:
        for name, field in [("geometry.npz", "geometry_sha256"), ("maps.npz", "maps_sha256")]:
            expected = meta.get(field)
            if expected and hashlib.file_digest((folder/name).open("rb"), "sha256").hexdigest() != expected:
                raise ValueError(f"{name} does not match its saved SHA256.")
    with np.load(maps_path, allow_pickle=False) as m:
        needed = {"vertices", "faces", "rgb", "valid", "height_mm", "pixel_surface_area_mm2", "fov_mm"}
        if not needed <= set(m.files):
            raise ValueError(f"Map export lacks {sorted(needed-set(m.files))}; use the reviewed-data rebuild.")
        if not np.array_equal(m["faces"], faces) or not np.array_equal(m["vertices"], vertices):
            raise ValueError("Maps do not match the approved mesh. Regenerate maps after ROI corrections.")
        rgb = np.asarray(m["rgb"]); valid = np.asarray(m["valid"], bool)
        height = np.asarray(m["height_mm"], float); area = np.asarray(m["pixel_surface_area_mm2"], float)
        fov = float(m["fov_mm"])
    if not np.isfinite(fov) or fov <= 0 or rgb.shape != valid.shape+(3,) or height.shape != valid.shape or area.shape != valid.shape:
        raise ValueError("Invalid FOV or map dimensions.")
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32)/255
    if not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0) or np.any(rgb[valid] > 1.00001):
        raise ValueError("Saved RGB values are not finite values in [0,1].")
    if not np.isfinite(area[valid]).all() or np.any(area[valid] <= 0) or not np.isfinite(height[valid]).all():
        raise ValueError("Invalid surface weights or heights inside the saved valid mask.")
    stride = args.colour_stride
    mask = valid[::stride, ::stride]; weights = area[::stride, ::stride][mask]
    if len(weights) < 10:
        raise ValueError("Too few valid colour pixels.")
    lab = rgb2lab(np.clip(rgb[::stride, ::stride][mask], 0, 1), illuminant="D65", observer="2")
    hist = []
    for k, channel in enumerate(CHANNELS):
        mean = np.average(lab[:, k], weights=weights)
        native[f"{channel}_mean"] = float(mean)
        native[f"{channel}_sd"] = float(np.sqrt(np.average((lab[:, k]-mean)**2, weights=weights)))
        for quantile, value in zip(["p05", "median", "p95"], qweighted(lab[:, k], weights, [.05, .5, .95])):
            native[f"{channel}_{quantile}"] = float(value)
        counts = np.histogram(lab[:, k], LAB_BINS[k], weights=weights)[0]
        if counts.sum() < weights.sum()*.999:
            raise ValueError("CIELAB values fall outside the histogram range.")
        hist.append(counts/counts.sum())
    native.update(fov_mm=fov, image_size=valid.shape[0], mm_per_pixel=fov/valid.shape[0],
                  valid_pixels=int(valid.sum()), colour_pixels_analysed=len(weights),
                  visible_surface_area_mm2=float(area[valid].sum()),
                  visible_projected_area_mm2=float(valid.sum()*(fov/valid.shape[0])**2),
                  color_source=meta.get("color_source", "saved_rgb_map"))
    scales, finest = [], None
    for pitch in args.pitches_mm:
        item, grid = common_grid(height, valid, area, fov, pitch, args.min_cell_coverage)
        scales.append({**row, **item})
        if finest is None:
            finest = grid
        native[f"grid_{round(pitch*1000):03d}um_height_rms_um"] = item["height_rms_um"]
    small = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255)); small.thumbnail((320, 320), Image.Resampling.BOX)
    small_valid = Image.fromarray(valid.astype(np.uint8)*255).resize(small.size, Image.Resampling.NEAREST)
    return {**row, **native}, dict(hist_L=hist[0], hist_a=hist[1], hist_b=hist[2],
                                 view_rgb=np.asarray(small), view_valid=np.asarray(small_valid)>0,
                                 view_height=finest, fov_mm=np.array(fov),
                                 scale_rows_json=np.array(json.dumps(scales)))


def fingerprint(row, args):
    settings = {k: getattr(args, k) for k in ["surface_samples", "seed", "colour_stride", "pitches_mm", "min_cell_coverage", "verify_hashes"]}
    files = []
    for name in ["geometry.npz", "maps.npz", "metadata.json"]:
        p = Path(row["sample_dir"])/name
        files.append((name, p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else (name, None, None))
    return hashlib.sha256(json.dumps([VERSION, settings, files, Path(__file__).read_text(encoding="utf-8")], sort_keys=True).encode()).hexdigest()


def bootstrap_median(values, seed=42, repetitions=1000):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    if len(x) < 2:
        return float(x[0]), np.nan, np.nan
    boot = np.median(np.random.default_rng(seed).choice(x, (repetitions, len(x))), axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(np.median(x)), float(lo), float(hi)


def summarise(df, groups, metrics, args):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, part in df.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric in metrics:
            if metric not in part:
                continue
            # One independent contribution per physical specimen in each comparison.
            x = part.groupby("physical_id")[metric].mean().dropna().to_numpy()
            med, lo, hi = bootstrap_median(x, args.seed, args.bootstrap)
            rows.append({**dict(zip(groups, keys)), "metric": metric, "n_specimens": len(x),
                         "median": med, "ci_low": lo, "ci_high": hi,
                         "q25": float(np.quantile(x, .25)) if len(x) else np.nan,
                         "q75": float(np.quantile(x, .75)) if len(x) else np.nan})
    return pd.DataFrame(rows)


def paired_tables(df, cache, args):
    scanner_rows, stage_rows = [], []
    metrics = [f"{x}_mean" for x in CHANNELS]+NATIVE+[c for c in df if re.fullmatch(r"grid_\d+um_height_rms_um", c)]
    for _, part in df.groupby(["physical_id", "timepoint"]):
        for a, b in itertools.combinations(SCANNERS, 2):
            aa, bb = part[part.scanner == a], part[part.scanner == b]
            if aa.empty or bb.empty:
                continue
            arow, brow = aa.iloc[0], bb.iloc[0]
            item = {k: arow[k] for k in ["physical_id", "group", "cohort", "stage", "timepoint"]}
            item.update(scanner_a=a, scanner_b=b, scan_a=arow.scan_id, scan_b=brow.scan_id)
            item["deltaE00_mean_Lab"] = float(deltaE_ciede2000(arow[["L_mean", "a_mean", "b_mean"]].to_numpy(float)[None], brow[["L_mean", "a_mean", "b_mean"]].to_numpy(float)[None])[0])
            for metric in metrics:
                item[f"delta_{metric}"] = float(brow[metric]-arow[metric])
            for k, ch in enumerate(CHANNELS):
                pa, pb = cache[arow.scan_id][f"hist_{ch}"], cache[brow.scan_id][f"hist_{ch}"]
                item[f"wasserstein_{ch}"] = float(np.sum(np.abs(np.cumsum(pa)-np.cumsum(pb))*np.diff(LAB_BINS[k])))
            scanner_rows.append(item)
    transitions = [("baseline", "partial"), ("partial", "clean"), ("baseline", "clean")]
    for _, part in df.groupby(["physical_id", "scanner"]):
        for first, last in transitions:
            aa, bb = part[part.stage == first], part[part.stage == last]
            if aa.empty or bb.empty:
                continue
            arow, brow = aa.iloc[0], bb.iloc[0]
            item = {k: arow[k] for k in ["physical_id", "group", "cohort", "scanner"]}
            item.update(stage_from=first, stage_to=last, transition=f"{first}_to_{last}", scan_from=arow.scan_id, scan_to=brow.scan_id)
            item["deltaE00_mean_Lab"] = float(deltaE_ciede2000(arow[["L_mean", "a_mean", "b_mean"]].to_numpy(float)[None], brow[["L_mean", "a_mean", "b_mean"]].to_numpy(float)[None])[0])
            for metric in metrics:
                item[f"delta_{metric}"] = float(brow[metric]-arow[metric])
            item["face_count_ratio"] = float(brow.face_count/arow.face_count)
            item["face_density_ratio"] = float(brow.faces_per_mm2/arow.faces_per_mm2)
            stage_rows.append(item)
    return pd.DataFrame(scanner_rows), pd.DataFrame(stage_rows)


def response_table(stage):
    rows = []
    if stage.empty:
        return pd.DataFrame()
    for _, part in stage.groupby(["physical_id", "transition"]):
        for a, b in itertools.combinations(SCANNERS, 2):
            aa, bb = part[part.scanner == a], part[part.scanner == b]
            if aa.empty or bb.empty:
                continue
            aa, bb = aa.iloc[0], bb.iloc[0]
            item = {k: aa[k] for k in ["physical_id", "cohort", "group", "transition"]}
            item.update(scanner_a=a, scanner_b=b)
            for c in stage:
                if c.startswith("delta_") or c == "deltaE00_mean_Lab":
                    item[f"difference_{c}"] = float(bb[c]-aa[c])
            rows.append(item)
    return pd.DataFrame(rows)


class Figures:
    def __init__(self, out, args):
        self.out, self.args, self.items = out, args, []
        out.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "savefig.facecolor": "white"})

    def save(self, fig, name, title, caption):
        # Buffer and atomically replace: avoids partial files after interrupted/shared-drive writes.
        for extension in ["png", "pdf"]:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=self.args.dpi, bbox_inches="tight")
            data = buffer.getvalue()
            if extension == "png":
                Image.open(io.BytesIO(data)).verify()
            elif not data.startswith(b"%PDF-") or b"%%EOF" not in data[-32:]:
                raise OSError("Incomplete PDF figure export.")
            target = self.out/f"{name}.{extension}"; temp = target.with_suffix(".tmp")
            with temp.open("wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            temp.replace(target)
        plt.close(fig)
        self.items.append(dict(name=name, title=title, caption=caption))
        print(f"  Figure: {name}", flush=True)


def stage_axes(ax, part, metric, label, log=False):
    stages = [s for s in STAGES if s in set(part.stage)]
    for j, scanner in enumerate(SCANNERS):
        sub = part[part.scanner == scanner]
        if sub.empty:
            continue
        off = (j-1.5)*.10; med, lo, hi = [], [], []
        for i, stage in enumerate(stages):
            x = sub[sub.stage == stage].sort_values("physical_id")[metric].dropna().to_numpy()
            if len(x):
                jitter = np.linspace(-.035, .035, len(x))
                ax.scatter(i+off+jitter, x, s=9, color=COLOURS[scanner], alpha=.26, zorder=2)
            med.append(np.median(x) if len(x) else np.nan)
            lo.append(np.quantile(x, .25) if len(x) else np.nan)
            hi.append(np.quantile(x, .75) if len(x) else np.nan)
        xx = np.arange(len(stages))+off
        ax.plot(xx, med, "o-", ms=4, lw=1.7, color=COLOURS[scanner], label=scanner)
        ax.vlines(xx, lo, hi, color=COLOURS[scanner], lw=2)
    ax.set_xticks(range(len(stages)), [s.capitalize() for s in stages])
    ax.set_ylabel(label); ax.grid(axis="y", alpha=.17)
    if log and np.any(part[metric].to_numpy() > 0):
        ax.set_yscale("log")


def make_colour_figures(df, paired, stages, cache, figs):
    for cohort, part in df.groupby("cohort", sort=False):
        present = [s for s in STAGES if s in set(part.stage)]
        channel_limits = []
        for k, ch in enumerate(CHANNELS):
            mixture = np.stack([cache[sid][f"hist_{ch}"] for sid in part.scan_id]).mean(0)
            centres = (LAB_BINS[k][1:]+LAB_BINS[k][:-1])/2
            low, high = np.interp([.001, .999], np.cumsum(mixture)/mixture.sum(), centres)
            pad = max(3, (high-low)*.07)
            channel_limits.append((max(LAB_BINS[k][0], low-pad), min(LAB_BINS[k][-1], high+pad)))
        fig, axes = plt.subplots(len(present), 3, figsize=(12, 2.8*len(present)), squeeze=False, constrained_layout=True)
        for i, stage in enumerate(present):
            for k, ch in enumerate(CHANNELS):
                ax = axes[i, k]; centres = (LAB_BINS[k][1:]+LAB_BINS[k][:-1])/2
                all_hist = []
                for scanner in SCANNERS:
                    ids = part[(part.stage == stage)&(part.scanner == scanner)].scan_id
                    if not len(ids):
                        continue
                    hs = np.stack([cache[sid][f"hist_{ch}"] for sid in ids]); mean = hs.mean(0)
                    ax.plot(centres, mean/np.diff(LAB_BINS[k]), label=scanner, color=COLOURS[scanner], lw=1.7)
                    all_hist.append(mean)
                ax.set_xlim(channel_limits[k])
                ax.set_xlabel(f"{ch}*"); ax.set_title(f"{stage.capitalize()} — {ch}*")
                if k == 0:
                    ax.set_ylabel("Area-weighted density")
                ax.grid(alpha=.15)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle(f"CIELAB distributions — {cohort} specimens")
        figs.save(fig, f"01_Lab_distributions_{cohort}", f"CIELAB distributions: {cohort}",
                  "Within each scan, colour histograms are weighted by visible 3D surface area and normalised. "
                  "Curves average scan histograms with one equal contribution per specimen. Assumed sRGB/D65/2°; "
                  "these describe exported appearance. Background pixels are excluded.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False, constrained_layout=True)
    markers = dict(baseline="o", partial="^", clean="s")
    for ax, cohort in zip(axes[0], ["fresh", "mature"]):
        part = df[df.cohort == cohort]
        for scanner in SCANNERS:
            for stage in STAGES:
                sub = part[(part.scanner == scanner)&(part.stage == stage)]
                ax.scatter(sub.a_mean, sub.b_mean, marker=markers[stage], color=COLOURS[scanner], alpha=.65, s=25,
                           label=f"{scanner}: {stage}" if len(sub) else None)
        ax.set(xlabel="Area-weighted mean a*", ylabel="Area-weighted mean b*", title=cohort.capitalize())
        ax.grid(alpha=.15)
    axes[0, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    figs.save(fig, "02_Lab_scan_centroids", "Colour centroids by scanner and stage",
              "Each point is one scan's surface-area-weighted mean CIELAB colour. Colours identify scanners; "
              "symbols identify experimental stages. Separation is descriptive, not a classification result.")
    if not paired.empty:
        vmax = max(1., float(paired.deltaE00_mean_Lab.quantile(.98)))
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        image = None
        for i, cohort in enumerate(["fresh", "mature"]):
            for j, stage in enumerate(STAGES):
                ax = axes[i, j]; sub = paired[(paired.cohort == cohort)&(paired.stage == stage)]
                if sub.empty:
                    message = "Not part of the\nfresh protocol" if cohort == "fresh" and stage == "partial" else "No paired scans\nfor this cohort/stage"
                    ax.axis("off"); ax.text(.5, .5, message, ha="center", va="center"); continue
                mat = np.full((4, 4), np.nan); ns = np.zeros((4, 4), int); np.fill_diagonal(mat, 0)
                for (a, b), part in sub.groupby(["scanner_a", "scanner_b"]):
                    x, y = SCANNERS.index(a), SCANNERS.index(b)
                    mat[x, y] = mat[y, x] = part.deltaE00_mean_Lab.median(); ns[x, y] = ns[y, x] = len(part)
                image = ax.imshow(mat, vmin=0, vmax=vmax, cmap="magma")
                for x in range(4):
                    for y in range(4):
                        if x != y and np.isfinite(mat[x, y]):
                            ax.text(y, x, f"{mat[x,y]:.1f}\nn={ns[x,y]}", ha="center", va="center",
                                    color="white" if mat[x, y] < vmax*.6 else "black", fontsize=8)
                ax.set_xticks(range(4), SCANNERS, rotation=25, ha="right"); ax.set_yticks(range(4), SCANNERS)
                ax.set_title(f"{cohort.capitalize()} — {stage}")
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75, label="Median paired ΔE₀₀ between scan mean Lab colours")
        figs.save(fig, "03_paired_scanner_colour", "Paired colour differences between scanners",
                  "Each comparison matches the same physical specimen and experimental stage. Cells show median ΔE₀₀ "
                  "between area-weighted scan mean Lab colours and the number of paired specimens. This is a centroid "
                  "difference, not a pointwise ΔE map or a colour-accuracy measurement.")
    if not stages.empty:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
        for i, cohort in enumerate(["fresh", "mature"]):
            order = ["baseline", "clean"] if cohort == "fresh" else STAGES
            for ax, scanner in zip(axes[i], SCANNERS):
                sub = stages[(stages.cohort == cohort)&(stages.scanner == scanner)&(stages.stage_from == "baseline")]
                if sub.empty:
                    ax.axis("off"); continue
                traces = []
                for _, rows in sub.groupby("physical_id"):
                    values = [0.]+[float(rows.loc[rows.stage_to == s, "deltaE00_mean_Lab"].iloc[0])
                                  if (rows.stage_to == s).any() else np.nan for s in order[1:]]
                    ax.plot(range(len(order)), values, color=COLOURS[scanner], alpha=.23, lw=.8); traces.append(values)
                ax.plot(range(len(order)), np.nanmedian(traces, axis=0), "o-", color=COLOURS[scanner], lw=2)
                ax.set_xticks(range(len(order)), [s.capitalize() for s in order], rotation=20)
                ax.set_title(f"{scanner} — {cohort}"); ax.set_ylabel("ΔE₀₀ from own baseline"); ax.grid(alpha=.15)
        figs.save(fig, "04_paired_colour_stage_changes", "Within-specimen colour changes across stages",
                  "Thin lines follow individual physical specimens within each scanner; thick lines show medians. "
                  "Colour change is measured relative to each specimen's own baseline scan. Fresh specimens have "
                  "two stages; mature specimens have three. No plaque labels are inferred from stage names.")


def make_geometry_figures(df, scales, stage, figs):
    mesh_metrics = [("vertex_count", "Encoded vertices", True), ("geometric_vertex_count", "Unique geometric vertices", True),
                    ("face_count", "Triangles", True), ("faces_per_mm2", "Triangles / mm² of surface", True),
                    ("edge_median_um", "Median unique-edge length (µm)", True),
                    ("triangle_area_median_um2", "Median triangle area (µm²)", True)]
    geom_metrics = [("height_mean_abs_um", "Mean absolute height (µm)", False),
                    ("height_rms_um", "RMS height (µm)", False), ("height_range90_um", "Height P95–P5 (µm)", False),
                    ("slope_median_deg", "Median native face slope (°)", False),
                    ("slope_p95_deg", "P95 native face slope (°)", False)]
    gridcols = [c for c in df if re.fullmatch(r"grid_\d+um_height_rms_um", c)]
    if gridcols:
        key = min(gridcols, key=lambda s: abs(int(re.search(r"\d+", s)[0])-100))
        pitch_label = int(key.split("_")[1][:-2])
        geom_metrics.append((key, f"RMS after {pitch_label} µm cell averaging (µm)", False))
    for cohort, part in df.groupby("cohort", sort=False):
        for number, title, metrics in [("05", "Mesh sampling", mesh_metrics), ("06", "Surface topography", geom_metrics)]:
            fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
            for ax, (metric, label, log) in zip(axes.ravel(), metrics):
                stage_axes(ax, part, metric, label, log)
            for ax in axes.ravel()[len(metrics):]:
                ax.axis("off")
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{title} — {cohort} specimens")
            caption = ("Dots represent individual scans; connected markers show scanner medians and vertical bars show "
                       "the interquartile range. Vertex counts include encoded texture-seam duplicates; unique geometric "
                       "vertices weld exact coordinate duplicates for counting only. Source meshes are not changed.") if number == "05" else (
                       "Plane-detrended topographic descriptors. Native RMS uses exact triangle-area integration; mean "
                       "absolute height and height quantiles use deterministic uniform-area samples. Native slopes are "
                       "area-weighted. These are scanner-derived, unfiltered descriptors, not independently validated "
                       "standard roughness measurements. Dots are scans; markers/IQRs summarise specimens.")
            figs.save(fig, f"{number}_{title.lower().replace(' ', '_')}_{cohort}", f"{title}: {cohort}", caption)
        sub = scales[scales.cohort == cohort]
        present = [s for s in STAGES if s in set(sub.stage)]
        fig, axes = plt.subplots(1, max(1, len(present)), figsize=(4.5*max(1, len(present)), 4), squeeze=False, constrained_layout=True)
        for ax, name in zip(axes[0], present):
            for scanner in SCANNERS:
                ss = sub[(sub.stage == name)&(sub.scanner == scanner)]
                if ss.empty:
                    continue
                gr = ss.groupby("pitch_mm").height_rms_um
                x = np.array(sorted(ss.pitch_mm.unique())); med = gr.median().reindex(x)
                ax.plot(x*1000, med, "o-", color=COLOURS[scanner], label=scanner)
                ax.fill_between(x*1000, gr.quantile(.25).reindex(x), gr.quantile(.75).reindex(x), color=COLOURS[scanner], alpha=.12)
            ax.set(title=name.capitalize(), xlabel="Common cell width (µm)", ylabel="RMS height after cell averaging (µm)")
            ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        figs.save(fig, f"07_common_scale_roughness_{cohort}", f"Roughness at shared physical scales: {cohort}",
                  "Stored height maps are surface-area-weighted box-averaged into cells of the indicated physical width, "
                  "then plane-detrended again. Lines show medians and shading the specimen IQR. Cells below the coverage "
                  "threshold are excluded; holes are not filled. Equal grid spacing does not establish equal scanner "
                  "resolution. No Gaussian or standardised roughness-band filtering is applied.")
    if not stage.empty:
        for cohort, part in stage.groupby("cohort", sort=False):
            transitions = [s for s in ["baseline_to_partial", "partial_to_clean", "baseline_to_clean"] if s in set(part.transition)]
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
            columns = [("deltaE00_mean_Lab", "Colour-change magnitude (ΔE₀₀)"), ("delta_L_mean", "Change in L*"),
                       ("delta_a_mean", "Change in a*"), ("delta_b_mean", "Change in b*"),
                       ("delta_height_rms_um", "Change in native RMS height (µm)"), ("face_density_ratio", "Final / initial triangle density")]
            for ax, (metric, label) in zip(axes.ravel(), columns):
                pp = part.copy(); pp["stage"] = pp.transition
                # Use the transition sequence explicitly rather than the stage-name helper.
                for j, scanner in enumerate(SCANNERS):
                    ss = part[part.scanner == scanner]; xx = np.arange(len(transitions))+(j-1.5)*.1
                    ys = [ss.loc[ss.transition == t, metric].median() for t in transitions]
                    ax.plot(xx, ys, "o-", label=scanner, color=COLOURS[scanner])
                    for x, tr in zip(xx, transitions):
                        values = ss.loc[ss.transition == tr, metric].dropna().to_numpy()
                        if len(values):
                            ax.scatter(x+np.linspace(-.025, .025, len(values)), values, color=COLOURS[scanner], s=9, alpha=.25)
                ax.set_xticks(range(len(transitions)), [t.replace("_to_", " → ") for t in transitions], rotation=25, ha="right")
                ax.set_ylabel(label); ax.axhline(1 if "ratio" in metric else 0, color="grey", lw=.8, alpha=.5); ax.grid(alpha=.15)
            axes[0, 0].legend(fontsize=8)
            figs.save(fig, f"08_scanner_stage_response_{cohort}", f"Scanner-dependent stage changes: {cohort}",
                      "All changes are paired within a physical specimen and scanner. Curves show scanner medians and "
                      "dots show individual paired changes. Signed changes are later minus earlier; density ratios are "
                      "later divided by earlier. Differences indicate stage associations, without identifying plaque pixels.")


def choose_examples(df, requested):
    expected_scanners = set(df.scanner)
    rows = []
    for physical, part in df.groupby("physical_id"):
        stages = ["baseline", "clean"] if part.cohort.iloc[0] == "fresh" else STAGES
        complete = all(set(part.loc[part.stage == stage, "scanner"]) == expected_scanners for stage in stages)
        item = dict(physical_id=physical, cohort=part.cohort.iloc[0], complete=complete)
        for metric in ["L_mean", "a_mean", "b_mean", "faces_per_mm2", "height_rms_um"]:
            item[metric] = float(part[metric].mean())
        rows.append(item)
    ranks = pd.DataFrame(rows); ranks["median_profile_distance"] = np.nan
    for _, part in ranks.groupby("cohort"):
        use = part[part.complete]
        if use.empty:
            continue
        cols = ["L_mean", "a_mean", "b_mean", "faces_per_mm2", "height_rms_um"]
        values = use[cols].to_numpy(); med = np.nanmedian(values, axis=0)
        scale = np.nanquantile(values, .75, axis=0)-np.nanquantile(values, .25, axis=0); scale[scale < 1e-10] = 1.
        ranks.loc[use.index, "median_profile_distance"] = np.sqrt(np.mean(((values-med)/scale)**2, axis=1))
    if requested:
        chosen = [normalise_physical_id(x) for x in requested]
        missing = set(chosen)-set(df.physical_id)
        if missing:
            raise ValueError(f"Requested example specimens are absent from the selected scope: {sorted(missing)}")
    else:
        chosen = []
        for _, part in ranks.dropna(subset=["median_profile_distance"]).groupby("cohort"):
            chosen.append(part.sort_values(["median_profile_distance", "physical_id"]).physical_id.iloc[0])
    ranks["selected"] = ranks.physical_id.isin(chosen)
    return chosen, ranks


def make_examples(df, cache, examples, figs, args):
    # One shared height limit across every selected scan, not auto-contrast per scanner.
    if args.height_limit_um:
        limit = args.height_limit_um
    else:
        samples = [np.abs(cache[sid]["view_height"][np.isfinite(cache[sid]["view_height"])])*1000 for sid in df.scan_id]
        limit = max(.1, float(np.quantile(np.concatenate(samples), .98)))
    for physical in examples:
        part = df[df.physical_id == physical]
        scanners = [s for s in SCANNERS if s in set(part.scanner)]
        present = [s for s in STAGES if s in set(part.stage)]
        fov = max(part.fov_mm); extent = [-fov/2, fov/2, -fov/2, fov/2]
        fig, axes = plt.subplots(len(present), len(scanners), figsize=(3.2*len(scanners), 3.2*len(present)), squeeze=False, constrained_layout=True)
        for i, stage in enumerate(present):
            for j, scanner in enumerate(scanners):
                ax = axes[i, j]; sub = part[(part.stage == stage)&(part.scanner == scanner)]
                if sub.empty:
                    ax.axis("off"); continue
                row = sub.iloc[0]
                with np.load(Path(row.sample_dir)/"maps.npz", allow_pickle=False) as m:
                    rgb, mask = m["rgb"].copy(), m["valid"].astype(bool)
                if np.issubdtype(rgb.dtype, np.integer):
                    rgb = rgb.astype(float)/255
                rgb[~mask] = 1
                ef = row.fov_mm/2
                ax.imshow(rgb, extent=[-ef, ef, -ef, ef]); ax.set(xlim=extent[:2], ylim=extent[2:])
                if i == 0:
                    ax.set_title(scanner)
                if j == 0:
                    ax.set_ylabel(f"{stage.capitalize()}\ny (mm)")
                ax.set_xlabel("x (mm)")
        fig.suptitle(f"Matched specimen {physical}: colour across scanners and stages")
        figs.save(fig, f"09_stage_montage_{physical}", f"Matched stage series: {physical}",
                  "Original saved unlit RGB projections; columns are scanners and rows are experimental stages. "
                  "Images share physical display limits. Scanner-specific canonical frames do not provide exact "
                  "pointwise correspondence between scanners.")
        stage = "partial" if "partial" in present else "baseline"
        selected = part[part.stage == stage]
        fig, axes = plt.subplots(4, len(scanners), figsize=(3.4*len(scanners), 11.5), squeeze=False, constrained_layout=True)
        image = None
        for j, scanner in enumerate(scanners):
            sub = selected[selected.scanner == scanner]
            if sub.empty:
                for ax in axes[:, j]:
                    ax.axis("off")
                continue
            row = sub.iloc[0]; folder = Path(row.sample_dir)
            with np.load(folder/"geometry.npz", allow_pickle=False) as g:
                v, faces = g["vertices"].copy(), g["faces"].copy()
            _, xyz, _ = mesh_measurements(v, faces, 1000, args.seed)
            with np.load(folder/"maps.npz", allow_pickle=False) as m:
                rgb, mask = m["rgb"].copy(), m["valid"].astype(bool)
                if {"origin", "basis"} <= set(m.files):
                    # Align visual mesh panels with the saved image coordinates within each scan.
                    xyz = (v-np.asarray(m["origin"]))@np.asarray(m["basis"])
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(float)/255
            rgb[~mask] = 1; ef = row.fov_mm/2
            axes[0, j].imshow(rgb, extent=[-ef, ef, -ef, ef])
            ex = xyz.copy(); ex[:, 2] *= args.relief_exaggeration
            tri = ex[faces]; normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            normals[normals[:, 2] < 0] *= -1
            light = np.array([.4, -.5, 1.]); light /= np.linalg.norm(light)
            shade = .25+.75*np.clip(normals@light, 0, 1)
            axes[1, j].tripcolor(xyz[:, 0], xyz[:, 1], faces, facecolors=shade, cmap="gray", vmin=0, vmax=1, shading="flat", rasterized=True)
            # Only exact coordinate duplicates are welded for drawing edge lines.
            points, inverse = np.unique(xyz, axis=0, return_inverse=True); ff = inverse[faces]
            edges = np.unique(np.sort(np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1), axis=0)
            segments = points[edges, :2]
            intersects = (segments[:, :, 0].min(1) <= args.mesh_zoom_mm/2)&(segments[:, :, 0].max(1) >= -args.mesh_zoom_mm/2)&(segments[:, :, 1].min(1) <= args.mesh_zoom_mm/2)&(segments[:, :, 1].max(1) >= -args.mesh_zoom_mm/2)
            axes[2, j].add_collection(LineCollection(segments[intersects], colors="#34546a", linewidths=.35, rasterized=True))
            h = cache[row.scan_id]["view_height"]*1000
            image = axes[3, j].imshow(h, extent=[-ef, ef, -ef, ef], cmap="coolwarm", vmin=-limit, vmax=limit)
            for i, ax in enumerate(axes[:, j]):
                half = args.mesh_zoom_mm/2 if i == 2 else fov/2
                ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="x (mm)")
                if j == 0:
                    names = ["Unlit RGB", f"Neutral geometry\n{args.relief_exaggeration:g}× relief", f"Native mesh edges\n{args.mesh_zoom_mm:g} × {args.mesh_zoom_mm:g} mm", "Detrended height"]
                    ax.set_ylabel(names[i]+"\ny (mm)")
            axes[0, j].set_title(scanner)
        if image is not None:
            fig.colorbar(image, ax=axes[3].tolist(), shrink=.8, label="Height (µm)")
        figs.save(fig, f"10_colour_geometry_mesh_{physical}_{stage}", f"Colour, geometry and native mesh: {physical}, {stage}",
                  f"Matched scans of one specimen. Rows show saved RGB, neutral-shaded native geometry with a fixed "
                  f"{args.relief_exaggeration:g}× vertical exaggeration for display only, native triangle edges in a "
                  f"{args.mesh_zoom_mm:g} mm square, and area-averaged/detrended height at {min(args.pitches_mm):g} mm cells. "
                  f"The height colour limit is shared across all selected scans (±{limit:.3g} µm). Lighting and relief "
                  "exaggeration are identical across scanners; quantitative measurements use unexaggerated geometry. "
                  "These views illustrate appearance and structure without asserting plaque location.")
        fig, axes = plt.subplots(3, len(scanners), figsize=(3.4*len(scanners), 8.5), squeeze=False, constrained_layout=True)
        limits = [(0, 100), (-80, 80), (-80, 80)]
        cmaps = ["gray", "PiYG_r", "coolwarm"]
        for j, scanner in enumerate(scanners):
            sub = selected[selected.scanner == scanner]
            if sub.empty:
                for ax in axes[:, j]:
                    ax.axis("off")
                continue
            row = sub.iloc[0]
            with np.load(Path(row.sample_dir)/"maps.npz", allow_pickle=False) as m:
                rgb, mask = m["rgb"], m["valid"].astype(bool)
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(float)/255
            lab = rgb2lab(rgb, illuminant="D65", observer="2")
            for k, ax in enumerate(axes[:, j]):
                values = np.where(mask, lab[..., k], np.nan)
                image = ax.imshow(values, extent=[-row.fov_mm/2, row.fov_mm/2]*2, cmap=cmaps[k], vmin=limits[k][0], vmax=limits[k][1])
                ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x (mm)")
                if j == 0:
                    ax.set_ylabel(f"{CHANNELS[k]}*\ny (mm)")
                if k == 0:
                    ax.set_title(scanner)
                if j == len(scanners)-1:
                    fig.colorbar(image, ax=axes[k].tolist(), shrink=.75, label=f"{CHANNELS[k]}*")
        figs.save(fig, f"11_Lab_maps_{physical}_{stage}", f"CIELAB maps: {physical}, {stage}",
                  "CIELAB channels derived from the saved RGB projection under assumed sRGB/D65/2°. Each channel uses "
                  "identical display limits across scanners; limits for a* and b* are ±80 and may clip display extremes. "
                  "All numeric summaries use the full, unclipped values.")


def report(out, inventory, df, failed, figs, args, timings):
    rows = []
    for item in figs.items:
        stem = item["name"]; title = html.escape(item["title"]); caption = html.escape(item["caption"])
        rows.append(f'<section><h2>{title}</h2><a href="figures/{stem}.png"><img loading="lazy" src="figures/{stem}.png" alt="{title}"></a>'
                    f'<p>{caption}</p><p><a href="figures/{stem}.png">PNG</a> · <a href="figures/{stem}.pdf">PDF</a></p></section>')
    tables = "".join(f'<li><a href="tables/{p.name}">{html.escape(p.name)}</a></li>' for p in sorted((out/"tables").glob("*.csv")))
    scope = ("All-data descriptive analysis. Use a separate --scope train run with the actual specimen split for "
             "decisions that influence preprocessing or model evaluation.") if args.scope == "all" else (
             "Training-only exploration. All scanners and stages from a physical specimen retain the same split.")
    counts = df.groupby(["cohort", "scanner", "stage"]).size().rename("analysed_scans").reset_index().to_html(index=False)
    error_table = failed.to_html(index=False, escape=True) if len(failed) else "<p>No processing failures.</p>"
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>BioCal3D ROI exploration</title>
<style>body{{max-width:1250px;margin:35px auto;padding:0 22px;font:16px/1.55 system-ui;color:#20323d}}h1,h2{{line-height:1.25}}section{{margin:38px 0;border-top:1px solid #ccd6dc;padding-top:22px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px 10px;border:1px solid #d4dce1}}.note{{background:#eef5f8;padding:16px}}a{{color:#086b91}}</style>
<h1>Approved ROI exploration</h1><p>Version {VERSION} · {html.escape(str(out))}</p>
<div class="note"><strong>{html.escape(scope)}</strong><br>No plaque labels, masks, plaque percentages or plaque–clean separability estimates are used.</div>
<p>Discovered {len(inventory)} scans / {inventory.physical_id.nunique()} physical specimens; analysed {len(df)} scans / {df.physical_id.nunique()} specimens. Failures: {len(failed)}. LABscanner is a laboratory scanner.</p>
<h2>How to interpret these outputs</h2><ul>
<li>Colour: assumed sRGB, D65, 2° observer. Visible surface-area weighting within scans; equal specimen contributions to group histograms. Exported appearance, not calibrated colour accuracy.</li>
<li>Paired scanner differences match specimen and stage. ΔE₀₀ compares mean Lab colours; distribution differences are also provided. No pointwise interscanner correspondence is assumed.</li>
<li>Fresh G1–G2: baseline T0, clean T1. Mature G3–G6: baseline T0, partial T1, clean T2. Every group is included.</li>
<li>Mesh counts and edge lengths describe sampling. Duplicate coordinates at texture seams are counted separately and also geometrically deduplicated for counts. Source geometry is preserved.</li>
<li>Roughness descriptors: unfiltered, plane-detrended scanner topography. Native RMS is area-integrated; common-scale values use physical cell averaging. These are not standardised Sa/Sq measurements or independently verified physical roughness.</li>
<li>Bootstrap intervals resample physical specimens within each displayed comparison. No pixel-wise significance tests or causal claims are made.</li>
<li>Grid spacing and vertical exaggeration do not establish scanner resolution or geometric accuracy.</li></ul>
<h2>Analysed scans</h2>{counts}<h2>Tables</h2><ul>{tables}</ul><h2>Figures</h2>{''.join(rows)}
<h2>Processing failures</h2>{error_table}<p>Runtime: {timings:.1f} seconds. See run_config.json for settings, assumptions and software versions.</p></html>'''
    (out/"report.html").write_text(text, encoding="utf-8")
    write_json(out/"figures.json", figs.items)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="Root of rebuilt Approved ROIs; contains samples/ and run_config.json.")
    p.add_argument("--output", type=Path, help="Separate output folder. Default: timestamped sibling of data root.")
    p.add_argument("--scope", choices=["all", "train"], default="all", help="All-data description or training-only decision-making exploration.")
    p.add_argument("--splits", type=Path, help="Authoritative CSV with physical_id,split; accepts BCG1-1 and G1_S1 specimen IDs.")
    p.add_argument("--examples", nargs="*", help="Representative specimen IDs, e.g. BCG1-2 BCG6-4. Default: one complete median-profile specimen per cohort.")
    p.add_argument("--pitches-mm", nargs="+", type=float, default=[.05, .1, .2], help="Common physical cell widths for height-map averaging.")
    p.add_argument("--min-cell-coverage", type=float, default=.75)
    p.add_argument("--surface-samples", type=int, default=50000, help="Uniform-area samples per native mesh for absolute height and quantiles; RMS is exact.")
    p.add_argument("--colour-stride", type=int, default=1, help="1 uses all saved valid pixels; 2 uses every second row and column.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300, help="PNG figure resolution; PDF is also saved.")
    p.add_argument("--mesh-zoom-mm", type=float, default=4.)
    p.add_argument("--relief-exaggeration", type=float, default=20., help="Fixed vertical multiplier for neutral geometry views only.")
    p.add_argument("--height-limit-um", type=float, help="Symmetric shared display limit; default P98 absolute common-grid height over selected scans.")
    p.add_argument("--verify-hashes", action="store_true", help="Verify metadata SHA256 for source geometry/maps (extra disk reads).")
    p.add_argument("--resume", action="store_true", help="Reuse valid per-scan caches in an existing output folder; regenerate figures/tables.")
    p.add_argument("--tables-only", action="store_true", help="Calculate measurements and comparisons without rendering figures.")
    args = p.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.pitches_mm = sorted(set(args.pitches_mm))
    if not args.data_root.is_dir():
        p.error("--data-root is not a directory.")
    if min(args.pitches_mm) <= 0 or not 0 < args.min_cell_coverage <= 1 or args.surface_samples < 1000 or args.colour_stride < 1 or args.bootstrap < 100:
        p.error("Positive pitches, coverage in (0,1], >=1000 surface samples, stride>=1 and >=100 bootstrap repetitions are required.")
    if args.dpi < 72 or args.mesh_zoom_mm <= 0 or args.relief_exaggeration <= 0 or (args.height_limit_um is not None and args.height_limit_um <= 0):
        p.error("Invalid figure settings.")
    return args


def main(argv=None):
    args = parse_args(argv); started = time.perf_counter()
    out = (args.output or args.data_root.parent/("ROI_Exploration_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out == args.data_root or args.data_root in out.parents or out in args.data_root.parents:
        raise ValueError("Output must be a separate sibling directory, not inside or above the source dataset.")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty. Use a new folder or --resume to reuse compatible caches.")
    out.mkdir(parents=True, exist_ok=True)
    for name in ["tables", "cache", "figures"]:
        (out/name).mkdir(exist_ok=True)
    print("Reading approved ROI inventory...", flush=True)
    inventory = discover(args.data_root, args.splits)
    selected = inventory if args.scope == "all" else inventory[inventory.split == "train"]
    if args.scope == "train" and (set(inventory.split)-{"train", "val", "test"} or selected.empty):
        raise ValueError("Training-only analysis requires complete specimen split assignments. Supply --splits specimen_splits.csv. No split is generated automatically.")
    if selected.empty:
        raise ValueError("No scans selected.")
    previous = read_json(out/"run_config.json") if args.resume else {}
    if previous and (previous.get("scope") != args.scope or previous.get("input_root") != str(args.data_root)):
        raise ValueError("Resume cannot change the dataset or analysis scope; choose a new output folder.")
    old_selection = out/"tables"/"selected_scans.csv"
    if args.resume and old_selection.is_file() and set(pd.read_csv(old_selection).scan_id) != set(selected.scan_id):
        raise ValueError("Selected specimens changed. Use a new output folder rather than mixing analyses.")
    inventory.to_csv(out/"tables"/"inventory.csv", index=False, encoding="utf-8-sig")
    inventory.groupby(["cohort", "group", "scanner", "stage", "split"]).size().rename("scan_count").reset_index().to_csv(out/"tables"/"inventory_counts.csv", index=False)
    selected.to_csv(out/"tables"/"selected_scans.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(version=VERSION, input_root=str(args.data_root), output=str(out), started=datetime.now().astimezone().isoformat(),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                  scikit_image=package_version("scikit-image"), pillow=package_version("Pillow"),
                  colour_encoding="Assumed sRGB/D65/2-degree; no calibration", native_roughness="Plane-detrended unfiltered scanner topography",
                  common_scale="Surface-area-weighted physical cell averaging; no hole filling; re-level at each scale",
                  decision_scope="Use training-only outputs for preprocessing/model decisions.",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/"run_config.json", config)
    measurements, cache, scale_rows, failures = [], {}, [], []
    print(f"Discovered {len(inventory)} scans. Analysing {len(selected)} ({args.scope}); output: {out}", flush=True)
    for i, row in enumerate(selected.to_dict("records"), 1):
        path = out/"cache"/(row["scan_id"]+".npz")
        print(f"[{i}/{len(selected)}] {row['scan_id']}", flush=True)
        try:
            fp = fingerprint(row, args); reused = False
            if args.resume and path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    if str(data["fingerprint"]) == fp:
                        stats = {**json.loads(str(data["statistics_json"])), **row}
                        arrays = {k: data[k].copy() for k in data.files if k not in {"fingerprint", "statistics_json"}}
                        reused = True
            if not reused:
                stats, arrays = load_and_measure(row, args)
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **arrays, fingerprint=np.array(fp), statistics_json=np.array(json.dumps(stats)))
                temp.replace(path)
            measurements.append(stats); cache[row["scan_id"]] = arrays
            scale_rows.extend([{**x, **row} for x in json.loads(str(arrays["scale_rows_json"]))])
        except Exception as exc:
            failures.append({**row, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
        pd.DataFrame(failures).to_csv(out/"tables"/"failures.csv", index=False)
    df = pd.DataFrame(measurements); failed = pd.DataFrame(failures)
    df.to_csv(out/"tables"/"scan_measurements.csv", index=False, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"No scans could be processed. See {out/'tables'/'failures.csv'}")
    scales = pd.DataFrame(scale_rows); scales.to_csv(out/"tables"/"common_scale_measurements.csv", index=False)
    paired, stage = paired_tables(df, cache, args)
    paired.to_csv(out/"tables"/"paired_scanner_differences.csv", index=False)
    stage.to_csv(out/"tables"/"paired_stage_changes.csv", index=False)
    responses = response_table(stage); responses.to_csv(out/"tables"/"scanner_stage_response_differences.csv", index=False)
    metrics = [f"{c}_{s}" for c in CHANNELS for s in ["mean", "sd", "median", "p05", "p95"]]+NATIVE+[c for c in df if re.fullmatch(r"grid_\d+um_height_rms_um", c)]
    summarise(df, ["cohort", "scanner", "stage"], metrics, args).to_csv(out/"tables"/"scanner_stage_summary.csv", index=False)
    summarise(df, ["cohort", "group", "scanner", "stage"], metrics, args).to_csv(out/"tables"/"group_scanner_stage_summary.csv", index=False)
    summarise(scales, ["cohort", "scanner", "stage", "pitch_mm"], ["height_mean_abs_um", "height_rms_um", "height_range90_um", "slope_median_deg"], args).to_csv(out/"tables"/"common_scale_summary.csv", index=False)
    if not paired.empty:
        cols = [c for c in paired if c.startswith(("delta_", "wasserstein_")) or c == "deltaE00_mean_Lab"]
        summarise(paired, ["cohort", "stage", "scanner_a", "scanner_b"], cols, args).to_csv(out/"tables"/"paired_scanner_summary.csv", index=False)
        summarise(paired, ["cohort", "group", "stage", "scanner_a", "scanner_b"], cols, args).to_csv(out/"tables"/"group_paired_scanner_summary.csv", index=False)
    if not stage.empty:
        cols = [c for c in stage if c.startswith("delta_") or c in {"deltaE00_mean_Lab", "face_count_ratio", "face_density_ratio"}]
        summarise(stage, ["cohort", "scanner", "transition"], cols, args).to_csv(out/"tables"/"paired_stage_summary.csv", index=False)
        summarise(stage, ["cohort", "group", "scanner", "transition"], cols, args).to_csv(out/"tables"/"group_paired_stage_summary.csv", index=False)
    if not responses.empty:
        summarise(responses, ["cohort", "transition", "scanner_a", "scanner_b"], [c for c in responses if c.startswith("difference_")], args).to_csv(out/"tables"/"scanner_response_summary.csv", index=False)
    examples, ranks = choose_examples(df, args.examples); ranks.to_csv(out/"tables"/"representative_candidates.csv", index=False)
    print("Measurements complete. Creating figures...", flush=True)
    figs = Figures(out/"figures", args)
    if not args.tables_only:
        make_colour_figures(df, paired, stage, cache, figs)
        make_geometry_figures(df, scales, stage, figs)
        make_examples(df, cache, examples, figs, args)
    seconds = time.perf_counter()-started
    config.update(finished=datetime.now().astimezone().isoformat(), seconds=seconds, discovered_scans=len(inventory),
                  analysed_scans=len(df), analysed_specimens=df.physical_id.nunique(), failed_scans=len(failures),
                  fov_values_mm=sorted(df.fov_mm.unique().tolist()), examples=examples,
                  status="complete" if not failures else "complete_with_processing_failures")
    write_json(out/"run_config.json", config)
    report(out, inventory, df, failed, figs, args, seconds)
    print(f"Finished: {len(df)} scans, {len(failures)} failures, {len(figs.items)} figures.\nOpen: {out/'report.html'}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
