#!/usr/bin/env python3
"""BioCal3D 3D stage-classification benchmark (not plaque segmentation).

Official PointNeXt-S, official TSGCNet with a BioCal3D classification adapter,
and official pretrained Sonata/Concerto with frozen-encoder head training.
See README.md for model installation and reproducibility details.

python biocal3d_train_3d_models.py --data-root "C:/path/to/Data collected" --live
python biocal3d_train_3d_models.py --output runs/3d_models --report-only

No substitute backbone is used when an official implementation is unavailable.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.spatial import cKDTree

STAGES = ["baseline", "partial", "clean"]
MODELS = ["pointnext_s", "tsgcnet", "sonata", "concerto"]
VERSION = 1


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, default=str, allow_nan=False)


def signature(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(path, obj):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def identify(path):
    """The timepoint is excluded from physical_id; scanner is excluded too."""
    matches = [re.search(r"BCG(\d+)T(\d+)[_-](\d+)", p, re.I) for p in path.parts]
    matches = [m for m in matches if m]
    if not matches:
        raise ValueError(f"Cannot infer BCGxTy-n identity: {path}. Supply --manifest.")
    m = matches[-1]
    g, t, n = map(int, m.groups())
    part_index = next(i for i in range(len(path.parts)-1, -1, -1)
                      if re.search(r"BCG\d+T\d+[_-]\d+", path.parts[i], re.I))
    # Expected processed tree: scanner/Ggroup/BCGxTy-n/roi.vtp.
    if part_index < 2:
        raise ValueError(f"Cannot infer scanner from {path}; supply --manifest.")
    scanner = path.parts[part_index - 2]
    if t == 0:
        stage = "baseline"
    elif g in (1, 2) and t == 1:
        stage = "clean"
    elif g in (3, 4, 5, 6) and t == 1:
        stage = "partial"
    elif g in (3, 4, 5, 6) and t == 2:
        stage = "clean"
    else:
        raise ValueError(f"Unknown group/timepoint combination G{g}T{t}; supply explicit stage.")
    sample = f"BCG{g}T{t}-{n}"
    return dict(scanner=scanner, sample_name=sample, physical_id=f"BCG{g}-{n}",
                group=str(g), stage=stage)


def specimen_identity(value):
    """Recognize both preprocessing BCG3-4 and DINO G3_S4 identity conventions."""
    m = re.fullmatch(r"(?:BCG(\d+)[_-](\d+)|G(\d+)_S(\d+))", value, re.I)
    if m:
        return tuple(int(x) for x in m.groups() if x is not None)
    return value


def validate_records(records):
    ids = set()
    specimens = {}
    if not records:
        raise ValueError("No ROI meshes found.")
    for r in records:
        for key in ("scanner", "sample_name", "physical_id", "stage", "mesh_path"):
            if not r.get(key):
                raise ValueError(f"Missing {key}: {r}")
        if r["stage"] not in STAGES:
            raise ValueError(f"Unknown stage {r['stage']}; expected {STAGES}")
        if not Path(r["mesh_path"]).is_file():
            raise FileNotFoundError(r["mesh_path"])
        key = (r["scanner"].casefold(), r["sample_name"].upper())
        if key in ids:
            raise ValueError(f"Duplicate scan identity: {key}")
        ids.add(key)
        if r.get("split"):
            if r["split"] not in ("train", "val", "test"):
                raise ValueError(f"Unknown split: {r['split']}")
            specimens.setdefault(r["physical_id"], set()).add(r["split"])
    if any(len(s) > 1 for s in specimens.values()):
        raise ValueError("Physical specimen appears in multiple splits (scanner/stage leakage).")
    same_stage = {}
    same_specimen = {}
    same_group = {}
    for r in records:
        key = (r["physical_id"], r["sample_name"])
        same_stage.setdefault(key, set()).add(r["stage"])
        same_specimen.setdefault(r["sample_name"].upper(), set()).add(r["physical_id"])
        same_group.setdefault(r["physical_id"], set()).add(str(r["group"]))
    if any(len(s) > 1 for s in same_stage.values()):
        raise ValueError("Matched acquisitions have inconsistent stage labels.")
    if any(len(s) > 1 for s in same_specimen.values()):
        raise ValueError("Matched scanner acquisitions have different physical specimen IDs.")
    if any(len(s) > 1 for s in same_group.values()):
        raise ValueError("Physical specimen ID conflates different experimental groups.")


def make_split(records, seed):
    rng = np.random.default_rng(seed)
    strata = {}
    for r in records:
        strata.setdefault(r["group"], set()).add(r["physical_id"])
    assignment = {}
    for group, specimens in sorted(strata.items()):
        specimens = sorted(specimens)
        rng.shuffle(specimens)
        if len(specimens) < 3:
            raise ValueError(f"Group {group} has fewer than 3 specimens. Supply an explicit split CSV.")
        nv = max(1, int(round(len(specimens) * .1)))
        nt = max(1, int(round(len(specimens) * .1)))
        for i, specimen in enumerate(specimens):
            assignment[specimen] = "val" if i < nv else "test" if i < nv + nt else "train"
    for r in records:
        r["split"] = assignment[r["physical_id"]]


def load_records(args):
    if args.manifest:
        records = read_csv(args.manifest)
        for r in records:
            r["sample_name"] = r.get("sample_name") or r.get("sample_id", "")
            raw = r.get("mesh_path") or r.get("roi_vtp") or ""
            if not raw:
                raise ValueError("--manifest requires mesh_path or roi_vtp.")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = args.manifest.parent / path
            r["mesh_path"] = str(path.resolve())
            if not r.get("physical_id") or not r.get("group") or not r.get("stage"):
                inferred = identify(path)
                for k, v in inferred.items():
                    if not r.get(k):
                        r[k] = v
            r["stage"] = r["stage"].strip().lower()
    else:
        root = args.roi_root or args.data_root / "_processed_ROI"
        if not root.is_dir():
            raise FileNotFoundError(f"No {root}; use --roi-root or --manifest.")
        paths = sorted(root.rglob("roi.vtp")) + sorted(root.rglob("roi.npz"))
        records = [dict(identify(p), mesh_path=str(p.resolve())) for p in paths]
    for r in records:
        r["row_id"] = signature([r["scanner"].casefold(), r["sample_name"].upper()])[:16]
    split_path = args.split_manifest
    auto = args.data_root / "_dino_preparation_80_10_10" / "scan_manifest.csv"
    if split_path is None and auto.exists() and not args.make_split:
        split_path = auto
    if split_path:
        rows = read_csv(split_path)
        splits = {}
        for r in rows:
            sample = r.get("sample_name") or r.get("sample_id")
            key = (r["scanner"].casefold(), sample.upper())
            if key in splits:
                raise ValueError(f"Duplicate entry in split CSV: {key}")
            splits[key] = r
        for r in records:
            key = (r["scanner"].casefold(), r["sample_name"].upper())
            if key not in splits:
                raise ValueError(f"Existing split does not cover {key}. Update it or use --make-split.")
            s = splits[key]
            if s.get("physical_id") and specimen_identity(s["physical_id"]) != specimen_identity(r["physical_id"]):
                raise ValueError(f"Physical identity differs from split CSV: {key}")
            if s.get("physical_id"):
                r["physical_id"] = s["physical_id"]
            if s.get("stage") and s["stage"].lower() != r["stage"]:
                raise ValueError(f"Stage differs from split CSV: {key}")
            r["split"] = s["split"].lower()
    if args.make_split:
        make_split(records, args.split_seed)
    elif not all(r.get("split") for r in records):
        if any(r.get("split") for r in records):
            raise ValueError("Only part of the manifest has split assignments.")
        make_split(records, args.split_seed)
    validate_records(records)
    for split in ("train", "val"):
        if {r["stage"] for r in records if r["split"] == split} != set(STAGES):
            raise ValueError(f"{split} must contain all three stages.")
    records.sort(key=lambda r: r["row_id"])
    args.full_records = records
    for split in ("train", "val", "test"):
        selected = [r for r in records if r["split"] == split]
        print(f"{split}: {len(selected)} scans / {len({r['physical_id'] for r in selected})} specimens", flush=True)
    # Test meshes are never opened without explicit final-evaluation flag.
    return [r for r in records if r["split"] != "test" or args.evaluate_test]


def normalize_rgb(rgb, count):
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[0] != count or rgb.shape[1] < 3:
        raise ValueError("RGB array must be N x 3 or N x 4.")
    rgb = rgb[:, :3].astype(np.float32)
    if rgb.max() > 1.001:
        rgb /= 255.
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1.001:
        raise ValueError("Invalid RGB; expected [0,1] or [0,255].")
    return np.clip(rgb, 0, 1)


def pv_arrays(mesh):
    rgb = uv = None
    for key in mesh.point_data:
        a = np.asarray(mesh.point_data[key])
        if key.casefold() in ("rgb", "rgba", "colors", "colours", "color", "colour") or "rgb" in key.casefold():
            rgb = normalize_rgb(a, mesh.n_points)
            break
    for key in mesh.point_data:
        a = np.asarray(mesh.point_data[key])
        if a.shape == (mesh.n_points, 2) and any(x in key.casefold() for x in ("tcoord", "texture", "uv")):
            uv = a.astype(float)
            break
    if uv is None and mesh.active_texture_coordinates is not None:
        uv = np.asarray(mesh.active_texture_coordinates)
    return rgb, uv


def texture_colors(uv, path):
    if uv is None or path is None:
        return None
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.
    x = np.rint(np.clip(uv[:, 0], 0, 1) * (im.shape[1] - 1)).astype(int)
    y = np.rint((1 - np.clip(uv[:, 1], 0, 1)) * (im.shape[0] - 1)).astype(int)
    return im[y, x]


def existing_path(value, relative):
    if not value:
        return None
    p = Path(str(value))
    if p.is_file():
        return p
    p = relative / str(value)
    return p if p.is_file() else None


def source_metadata(row, mesh_path):
    meta = dict(row)
    geometry = mesh_path.parent / "geometry.npz"
    if geometry.exists():
        with np.load(geometry, allow_pickle=False) as z:
            for k in ("source_ply", "source_texture", "original_to_canonical"):
                if k in z:
                    meta[k] = z[k].copy() if k == "original_to_canonical" else str(z[k].item())
    source = existing_path(meta.get("source_ply"), mesh_path.parent)
    texture = existing_path(meta.get("source_texture"), mesh_path.parent)
    if texture is None and source is not None:
        images = [p for p in source.parent.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
        matched = [p for p in images if p.stem.lower() in (source.stem.lower(), source.stem.lower() + "_texture")]
        if len(matched) == 1:
            texture = matched[0]
        elif len(images) == 1:
            texture = images[0]
    return meta, source, texture


def load_mesh(row, args):
    path = Path(row["mesh_path"])
    source = texture = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            vertices = z["vertices"].astype(float)
            faces = z["faces"].astype(np.int64)
            rgb = normalize_rgb(z["rgb"] if "rgb" in z else None, len(vertices))
    else:
        import pyvista as pv
        mesh = pv.read(path).extract_surface().triangulate()
        vertices = np.asarray(mesh.points, dtype=float)
        packed = np.asarray(mesh.faces)
        if packed.size % 4 or not np.all(packed.reshape(-1, 4)[:, 0] == 3):
            raise ValueError(f"Nontriangular surface: {path}")
        faces = packed.reshape(-1, 4)[:, 1:].astype(np.int64)
        rgb, uv = pv_arrays(mesh)
        if args.inputs == "geometry_rgb":
            meta, source, texture = source_metadata(row, path)
            if rgb is None:
                rgb = texture_colors(uv, texture)
            if rgb is None and source is not None:
                original = pv.read(source).extract_surface().triangulate()
                original_rgb, original_uv = pv_arrays(original)
                if original_rgb is None:
                    original_rgb = texture_colors(original_uv, texture)
                transform = np.asarray(meta.get("original_to_canonical", []))
                if original_rgb is not None and transform.shape == (4, 4):
                    op = np.asarray(original.points)
                    mapped = (np.column_stack([op, np.ones(len(op))]) @ transform.T)[:, :3]
                    distance, index = cKDTree(mapped).query(vertices)
                    spacing = np.median(cKDTree(vertices).query(vertices, k=2)[0][:, 1])
                    tolerance = max(.001, .15 * spacing)
                    # Training cannot quietly accept unmatched color vertices.
                    if np.all(distance <= tolerance):
                        rgb = original_rgb[index]
                    else:
                        raise ValueError(f"Source color mapping has unmatched vertices in {path}; repair ROI colors first.")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Invalid faces: {path}")
    if args.inputs == "geometry_rgb" and rgb is None:
        raise ValueError(f"No usable RGB for {path}. Restore vertex RGB/UV+texture or use --inputs geometry.")
    dependencies = [path]
    geometry = path.parent / "geometry.npz"
    dependencies.extend(p for p in (geometry, source, texture) if p is not None and p.is_file())
    return vertices, faces, rgb, {str(p): file_hash(p) for p in dependencies}


def mesh_samples(vertices, faces, rgb, args, row_id):
    vertices = np.asarray(vertices, dtype=float)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    valid = area2 > 1e-12
    ids = np.flatnonzero(valid)
    if len(ids) < args.neighbors + 1:
        raise ValueError("Too few nondegenerate triangles for TSGCNet neighborhood size.")
    face_normal = cross / np.maximum(area2[:, None], 1e-12)
    vertex_normal = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vertex_normal, faces[:, k], cross)
    vertex_normal /= np.maximum(np.linalg.norm(vertex_normal, axis=1, keepdims=True), 1e-12)
    centers = tri.mean(axis=1)
    center = np.average(centers[valid], axis=0, weights=area2[valid])
    # A single fixed physical scale for all samples and axes; no Z stretching.
    vertices = (vertices - center) / args.scale_mm
    rng = np.random.default_rng(int(row_id[:8], 16) ^ args.sampling_seed)
    prob = area2[valid] / area2[valid].sum()
    chosen = rng.choice(ids, size=args.points, replace=len(ids) < args.points, p=prob)
    corners = vertices[faces[chosen]]
    coord = corners.mean(axis=1)
    corner_normals = vertex_normal[faces[chosen]]
    normal = face_normal[chosen]
    # Original TSGCNet expects 12 coord and 12 normal channels per face.
    tsgc = np.concatenate([corners.reshape(-1, 9), coord,
                           corner_normals.reshape(-1, 9), normal], axis=1)
    color = np.zeros_like(coord) if rgb is None or args.inputs == "geometry" else rgb[faces[chosen]].mean(axis=1)
    return {"coord": coord.astype(np.float32), "normal": normal.astype(np.float32),
            "color": color.astype(np.float32), "tsgc": tsgc.astype(np.float32),
            "face_index": chosen.astype(np.int64)}


def prepare_data(records, args):
    folder = args.output / "surface_cache"
    folder.mkdir(exist_ok=True, parents=True)
    qc = []
    for i, row in enumerate(records, 1):
        vertices, faces, rgb, hashes = load_mesh(row, args)
        spec = {"version": VERSION, "dependencies": hashes, "points": args.points,
                "scale_mm": args.scale_mm, "sampling_seed": args.sampling_seed,
                "inputs": args.inputs, "neighbors": args.neighbors}
        key = signature(spec)
        dest = folder / (row["row_id"] + ".npz")
        reuse = False
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                reuse = str(z["signature"].item()) == key
        if not reuse:
            data = mesh_samples(vertices, faces, rgb, args, row["row_id"])
            np.savez_compressed(dest, **data, signature=np.array(key))
        row["surface_cache"] = str(dest)
        row["surface_signature"] = key
        qc.append(dict(row_id=row["row_id"], scanner=row["scanner"], sample_name=row["sample_name"],
                       vertices=len(vertices), faces=len(faces), sampled_faces=args.points,
                       rgb_available=rgb is not None, replacement_sampling=len(faces) < args.points,
                       mesh_path=row["mesh_path"]))
        print(f"ROI {i}/{len(records)}: {row['scanner']} {row['sample_name']}", flush=True)
    write_csv(args.output / "data_qc.csv", qc)


class SurfaceDataset(Dataset):
    def __init__(self, records, cached=None):
        self.records = records
        self.cached = cached

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        if self.cached:
            with np.load(self.cached / (r["row_id"] + ".npz"), allow_pickle=False) as z:
                data = {"embedding": torch.from_numpy(z["embedding"].copy())}
        else:
            with np.load(r["surface_cache"], allow_pickle=False) as z:
                data = {k: torch.from_numpy(z[k].copy()) for k in ("coord", "normal", "color", "tsgc")}
        return data, STAGES.index(r["stage"]), index


def batch_to(data, device):
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}


def repo_info(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-checkout"


class ClassificationHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 128),
                                 nn.GELU(), nn.Dropout(.25), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x)


def pooled(features):
    return torch.cat([features.mean(1), features.amax(1)], dim=1)


class TSGCEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        path = args.backends_root / "TSGCNet" / "TSGCNet.py"
        text = path.read_text(encoding="utf-8")
        # Compatibility patches for the official code's fixed CUDA/16k assumptions.
        replacements = {"device = torch.device('cuda')": "device = coor.device",
                        ".reshape(1,1,16000)": ".unsqueeze(1)",
                        "avgSum = avgSum_coor+avgSum_nor":
                        "avgSum = avgSum_coor+avgSum_nor\n        avgSum = torch.where(avgSum.abs() < 1e-6, torch.where(avgSum >= 0, torch.full_like(avgSum, 1e-6), torch.full_like(avgSum, -1e-6)), avgSum)"}
        for before, after in replacements.items():
            if before not in text:
                raise RuntimeError(f"Official TSGCNet source changed; compatibility patch missing: {before}")
            text = text.replace(before, after)
        module = types.ModuleType("biocal3d_official_tsgcnet")
        exec(compile(text, str(path), "exec"), module.__dict__)
        self.model = module.TSGCNet(k=args.neighbors, in_channels=12, output_channels=3)
        for p in self.model.pred4.parameters():
            p.requires_grad = False  # Original segmentation output is not the stage head.
        self.model.pred3.register_forward_hook(self._capture)
        self.latest = None
        self.channels = 128
        self.provenance = dict(source=str(path), source_sha256=file_hash(path),
                               commit=repo_info(path.parent), adaptation="pred3 local features + stage head; fixed-device/fixed-shape/numerical patches")

    def _capture(self, module, inputs, output):
        self.latest = output

    def forward(self, data):
        self.model(data["tsgc"].transpose(1, 2).contiguous())
        x = self.latest.transpose(1, 2)
        self.latest = None
        return x, data["coord"]


class PointNextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        root = args.backends_root / "PointNeXt"
        sys.path.insert(0, str(root))
        from openpoints.utils import EasyConfig
        from openpoints.models import build_model_from_cfg
        cfg = EasyConfig()
        cfg.load(str(root / "cfgs" / "scanobjectnn" / "pointnext-s.yaml"), recursive=False)
        encoder_cfg = copy.deepcopy(cfg.model.encoder_args)
        encoder_cfg.in_channels = 9 if args.inputs == "geometry_rgb" else 6
        self.input_channels = encoder_cfg.in_channels
        self.model = build_model_from_cfg(encoder_cfg)
        self.channels = self.model.out_channels
        self.provenance = dict(commit=repo_info(root), openpoints_commit=repo_info(root / "openpoints"),
                               encoder_config=dict(encoder_cfg), adaptation="official S encoder + mean/max stage head; XYZ/normals/optional RGB")

    def forward(self, data):
        parts = [data["coord"], data["normal"]]
        if self.input_channels == 9:
            parts.append(data["color"] * 2 - 1)
        feat = torch.cat(parts, dim=-1).transpose(1, 2).contiguous()
        positions, features = self.model.forward_seg_feat(data["coord"].contiguous(), feat)
        return features[-1].transpose(1, 2), positions[-1]


class PretrainedEncoder(nn.Module):
    def __init__(self, name, args, device):
        super().__init__()
        sys.path.insert(0, str(args.backends_root / name))
        package = importlib.import_module(name)
        checkpoint = args.sonata_checkpoint if name == "sonata" else args.concerto_checkpoint
        model_name = str(checkpoint) if checkpoint else ("sonata" if name == "sonata" else args.concerto_variant)
        self.model = package.model.load(model_name, custom_config={"enable_flash": args.flash,
            "enc_patch_size": [args.attention_patch] * 5}).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Explicit voxel grid replaces scene-scale defaults and fragment voting.
        self.package = package
        self.normalize_color = package.transform.NormalizeColor()
        self.grid_size = args.grid_size
        self.provenance = dict(commit=repo_info(args.backends_root / name), model=model_name,
            checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
            initialization="official pretrained", grid_size=args.grid_size, flash=args.flash,
            adaptation="frozen encoder; deterministic voxel representatives; original-face mean/max pooling")

    def forward(self, data):
        if data["coord"].shape[0] != 1:
            raise ValueError("Pretrained feature extraction uses one scan at a time.")
        coord = data["coord"][0].detach().cpu().numpy()
        color = data["color"][0].detach().cpu().numpy()
        # The official transform owns normalization. Sonata currently uses
        # [0,255] -> [0,1]; do not assume the commonly used [-1,1] convention.
        color = self.normalize_color({"color": color * 255.})["color"]
        normal = data["normal"][0].detach().cpu().numpy()
        grid = np.floor((coord - coord.min(0)) / self.grid_size).astype(np.int32)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        device = next(self.model.parameters()).device
        point = dict(coord=torch.from_numpy(coord[first]).to(device),
            grid_coord=torch.from_numpy(grid[first]).to(device),
            feat=torch.from_numpy(np.concatenate([coord[first], color[first], normal[first]], axis=1)).to(device),
            offset=torch.tensor([len(first)], dtype=torch.int64, device=device))
        point = self.model(point)
        # Official encoder hierarchy upcast: concatenate the first two parent
        # feature levels, then propagate to the encoder's input resolution.
        for _ in range(2):
            if "pooling_parent" not in point or "pooling_inverse" not in point:
                raise RuntimeError("Unexpected pretrained hierarchy; official API changed.")
            parent = point.pop("pooling_parent")
            idx = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[idx]], dim=-1)
            point = parent
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            idx = point.pop("pooling_inverse")
            parent.feat = point.feat[idx]
            point = parent
        feat = point.feat[torch.from_numpy(inverse).to(device)]
        return feat[None], data["coord"]


class StageModel(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.use_color = args.inputs == "geometry_rgb"
        # All candidates get a small direct RGB route after spatial extraction;
        # TSGCNet retains its geometric streams instead of silently modifying them.
        self.head = ClassificationHead(2 * (encoder.channels + (3 if self.use_color else 0)))

    def embedding(self, data):
        feat, pos = self.encoder(data)
        if self.use_color:
            # RGB is pooled separately so hierarchical downsampling does not
            # change the direct appearance route between candidates.
            return torch.cat([pooled(feat), pooled(data["color"] * 2 - 1)], dim=1)
        return pooled(feat)

    def forward(self, data):
        return self.head(self.embedding(data))


def make_model(name, args, device):
    if name == "tsgcnet":
        encoder = TSGCEncoder(args)
    elif name == "pointnext_s":
        encoder = PointNextEncoder(args)
    else:
        encoder = PretrainedEncoder(name, args, device)
    return encoder


def cache_pretrained(name, encoder, records, args, device):
    folder = args.output / "feature_cache" / name
    folder.mkdir(parents=True, exist_ok=True)
    provenance = dict(encoder.provenance)
    # Exact pretrained state fingerprint, not the checkpoint filename alone.
    h = hashlib.sha256()
    for k, v in encoder.model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    provenance["state_sha256"] = h.hexdigest()
    encoder_key = signature(provenance)
    write_json(folder / "encoder.json", provenance)
    total_time = 0.
    encoder.eval()
    for i, row in enumerate(records, 1):
        path = folder / (row["row_id"] + ".npz")
        spec = signature([encoder_key, row["surface_signature"], args.inputs, args.export_local_features])
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["signature"].item()) == spec:
                    with np.load(path, allow_pickle=False) as cached:
                        total_time += float(cached["encoder_seconds"])
                    continue
        data, _, _ = SurfaceDataset([row])[0]
        data = batch_to({k: v[None] for k, v in data.items()}, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            local, pos = encoder(data)
            vector = pooled(local)
            if args.inputs == "geometry_rgb":
                vector = torch.cat([vector, pooled(data["color"] * 2 - 1)], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        total_time += seconds
        result = dict(embedding=vector[0].cpu().numpy(), signature=np.array(spec), encoder_seconds=np.array(seconds))
        if args.export_local_features:
            result.update(local_features=local[0].cpu().numpy(), coord=pos[0].cpu().numpy())
            with np.load(row["surface_cache"]) as z:
                result["face_index"] = z["face_index"]
        np.savez_compressed(path, **result)
        print(f"{name}: features {i}/{len(records)} ({seconds:.2f}s)", flush=True)
    with np.load(folder / (records[0]["row_id"] + ".npz")) as z:
        channels = len(z["embedding"])
    return folder, channels, provenance, total_time


def metrics(truth, probability, specimens):
    truth = np.asarray(truth)
    prediction = np.asarray(probability).argmax(1)
    cm = confusion_matrix(truth, prediction, labels=range(3))
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(3, np.nan), where=support != 0)
    # Undefined BA when a class is absent; do not silently inflate per-scanner BA.
    ba = float(recall.mean()) if np.all(support > 0) else None
    correct = prediction == truth
    specimen_acc = [float(correct[np.asarray(specimens) == s].mean()) for s in sorted(set(specimens))]
    specimen_class_recalls = []
    for c in range(3):
        values = [float(correct[(np.asarray(specimens) == s) & (truth == c)].mean())
                  for s in sorted(set(specimens)) if np.any((np.asarray(specimens) == s) & (truth == c))]
        specimen_class_recalls.append(float(np.mean(values)) if values else None)
    specimen_ba = float(np.mean(specimen_class_recalls)) if all(x is not None for x in specimen_class_recalls) else None
    auc = None
    if np.all(support > 0):
        auc = float(roc_auc_score(truth, probability, multi_class="ovr", average="macro"))
    return dict(accuracy=float(correct.mean()), balanced_accuracy=ba,
        specimen_macro_accuracy=float(np.mean(specimen_acc)), specimen_balanced_accuracy=specimen_ba,
        macro_f1=float(f1_score(truth, prediction, labels=range(3), average="macro", zero_division=0)),
        macro_auc=auc, class_support=support.tolist(), scans=len(truth), specimens=len(set(specimens)))


def evaluate(model, loader, device):
    model.eval()
    probs, targets, order = [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for data, label, idx in loader:
            data = batch_to(data, device)
            logit = model(data["embedding"]) if "embedding" in data else model(data)
            if not torch.isfinite(logit).all():
                raise FloatingPointError("Nonfinite evaluation logits.")
            probs.extend(logit.softmax(1).cpu().numpy())
            targets.extend(label.numpy())
            order.extend(idx.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    probs = np.asarray(probs)
    records = [loader.dataset.records[i] for i in order]
    scores = metrics(targets, probs, [r["physical_id"] for r in records])
    scores["loss"] = float(-np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-12)).mean())
    scores["evaluation_seconds"] = time.perf_counter() - start
    rows = [dict(row_id=r["row_id"], physical_id=r["physical_id"], scanner=r["scanner"],
                 sample_name=r["sample_name"], truth=int(y), prediction=int(p.argmax()),
                 p_baseline=float(p[0]), p_partial=float(p[1]), p_clean=float(p[2]))
            for r, y, p in zip(records, targets, probs)]
    return scores, rows


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def run_training(model, dataset, validation, args, run, device, run_signature):
    if len(dataset) < 2:
        raise ValueError("Need at least 2 training scans.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        drop_last=len(dataset) % args.batch_size == 1,
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    count = np.bincount([STAGES.index(r["stage"]) for r in dataset.records], minlength=3)
    weights = torch.tensor(count.sum() / (3 * count), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best, stale, history, train_seconds = 0, -math.inf, 0, [], 0.
    checkpoint = run / "last.pt"
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"Run already exists: {run}. Use --resume or a new --output.")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["signature"] != run_signature:
            raise ValueError("Run/data/model settings changed. Use a new output directory.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best, stale = saved["epoch"], saved["best"], saved["stale"]
        history, train_seconds = saved["history"], saved["train_seconds"]
        restore_rng(saved["rng"])
    interrupted = False
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            if stale >= args.patience:
                break
            model.train()
            total_loss, total = 0., 0
            start = time.perf_counter()
            for data, label, idx in loader:
                optimizer.zero_grad(set_to_none=True)
                data = batch_to(data, device)
                label = label.to(device)
                logit = model(data["embedding"]) if "embedding" in data else model(data)
                loss = criterion(logit, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(label)
                total += len(label)
            scores, _ = evaluate(model, val_loader, device)
            selection = scores["specimen_balanced_accuracy"]
            if selection is None:
                raise ValueError("Validation lacks a class.")
            seconds = time.perf_counter() - start
            train_seconds += seconds
            improved = selection > best + 1e-8
            if improved:
                best, stale = selection, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=total_loss / total, val_loss=scores["loss"],
                                val_balanced_accuracy=scores["balanced_accuracy"],
                                val_specimen_balanced_accuracy=selection, seconds=seconds))
            saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                         best=best, stale=stale, history=history, train_seconds=train_seconds,
                         signature=run_signature, rng=rng_state())
            save_checkpoint(checkpoint, saved)
            if improved:
                save_checkpoint(run / "best.pt", saved)
            write_csv(run / "history.csv", history)
            write_json(run / "status.json", dict(status="training", completed_epochs=epoch))
            print(f"{run.name} epoch {epoch:03d}: loss {total_loss/total:.4f}, val loss {scores['loss']:.4f}, specimen BA {selection:.3f} ({seconds:.1f}s)", flush=True)
            if args.live:
                report(args.output)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. Last completed epoch is saved; generating comparison from best checkpoints.", flush=True)
    if not (run / "best.pt").exists():
        raise RuntimeError("No completed epoch yet. Resume after the first epoch can finish.")
    best_saved = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_saved["model"])
    return history, train_seconds, interrupted, best_saved["epoch"]


def scanner_metrics(rows):
    result = []
    for scanner in sorted({r["scanner"] for r in rows}):
        selected = [r for r in rows if r["scanner"] == scanner]
        m = metrics([r["truth"] for r in selected], [[r["p_baseline"], r["p_partial"], r["p_clean"]] for r in selected],
                    [r["physical_id"] for r in selected])
        result.append(dict(scanner=scanner, **m))
    return result


def bootstrap_ba(rows, repeats=500, seed=123):
    # Resample whole specimens, keeping stages and scanners together.
    specimens = sorted({r["physical_id"] for r in rows})
    by_specimen = {s: [r for r in rows if r["physical_id"] == s] for s in specimens}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = []
        for i, s in enumerate(rng.choice(specimens, len(specimens), replace=True)):
            sample.extend(dict(r, physical_id=str(i)) for r in by_specimen[s])
        value = metrics([r["truth"] for r in sample], [[r["p_baseline"], r["p_partial"], r["p_clean"]] for r in sample],
                        [r["physical_id"] for r in sample])["specimen_balanced_accuracy"]
        if value is not None:
            values.append(value)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else [None, None]


def export_focus(model, name, records, cache, args, run, device):
    # Occlusion attribution works for all backbones, including frozen probes.
    # These are sensitivity maps, not plaque labels or thickness maps.
    import matplotlib.pyplot as plt
    folder = run / "focus"
    folder.mkdir(exist_ok=True)
    model.eval()
    subset = []
    for scanner in sorted({r["scanner"] for r in records}):
        for stage in STAGES:
            selected = [r for r in records if r["scanner"] == scanner and r["stage"] == stage]
            if selected:
                subset.append(selected[0])
    if cache is not None:
        encoder = PretrainedEncoder(name, args, device)
    else:
        encoder = model.encoder
    for row in subset[:args.focus_examples]:
        data, _, _ = SurfaceDataset([row])[0]
        original = batch_to({k: v[None] for k, v in data.items()}, device)
        coords = data["coord"].numpy()
        # Partition XY into a small regular grid, removing surface patches.
        xy = coords[:, :2]
        extent = np.ptp(xy, axis=0)
        cells = np.floor((xy - xy.min(0)) / np.maximum(extent, 1e-8) * 4).astype(int).clip(0, 3)
        region = cells[:, 0] + 4 * cells[:, 1]
        def forward_score(d):
            if cache is None:
                return model(d)
            f, _ = encoder(d)
            emb = pooled(f)
            if args.inputs == "geometry_rgb":
                emb = torch.cat([emb, pooled(d["color"] * 2 - 1)], dim=1)
            return model(emb)
        with torch.no_grad():
            logits = forward_score(original)
            target = int(logits.argmax(1))
            baseline = float(logits.softmax(1)[0, target])
            sensitivity = np.zeros(len(coords), dtype=np.float32)
            for cell in np.unique(region):
                keep = np.flatnonzero(region != cell)
                if len(keep) < args.neighbors + 1:
                    continue
                # Fill to fixed N by deterministic resampling of the retained patch.
                idx = np.resize(keep, len(coords))
                d = {k: v[:, idx] for k, v in original.items()}
                p = float(forward_score(d).softmax(1)[0, target])
                sensitivity[region == cell] = baseline - p
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].scatter(xy[:, 0], xy[:, 1], c=data["color"].numpy() if args.inputs == "geometry_rgb" else coords[:, 2], s=3)
        limit = max(float(np.abs(sensitivity).max()), 1e-4)
        artist = axes[1].scatter(xy[:, 0], xy[:, 1], c=sensitivity, s=4, cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(artist, ax=axes[1], label="Probability drop after region removal")
        axes[0].set_title("Input surface")
        axes[1].set_title(f"Occlusion sensitivity: {STAGES[target]}")
        for ax in axes:
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(f"{row['scanner']} {row['sample_name']} | {name}")
        fig.tight_layout()
        fig.savefig(folder / (row["row_id"] + ".png"), dpi=140)
        plt.close(fig)
        np.savez_compressed(folder / (row["row_id"] + ".npz"), coord=coords,
                            sensitivity=sensitivity, face_index=np.load(row["surface_cache"])["face_index"],
                            explained_class=target, original_probability=baseline)
        print(f"Focus example: {name} {row['sample_name']}", flush=True)


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    runs = sorted((output / "runs").glob("*")) if (output / "runs").exists() else []
    summaries, cards, errors = [], [], []
    for run in runs:
        result_path = run / "metrics_val.json"
        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        if status.get("status") == "failed":
            errors.append(f"<li>{html.escape(run.name)}: {html.escape(status.get('error', 'Failed'))}</li>")
            continue
        history_path = run / "history.csv"
        if history_path.exists():
            h = read_csv(history_path)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            epochs = [int(r["epoch"]) for r in h]
            axes[0].plot(epochs, [float(r["train_loss"]) for r in h], label="train (class-weighted)")
            axes[0].plot(epochs, [float(r["val_loss"]) for r in h], label="validation (unweighted)")
            axes[0].legend(fontsize=7)
            axes[0].set_ylabel("Cross entropy")
            axes[1].plot(epochs, [float(r["val_specimen_balanced_accuracy"]) for r in h])
            axes[1].set_ylabel("Specimen balanced accuracy")
            axes[1].set_ylim(0, 1)
            for ax in axes:
                ax.set_xlabel("Epoch")
            fig.tight_layout()
            fig.savefig(run / "training.png", dpi=140)
            plt.close(fig)
        if not result_path.exists():
            cards.append(f"<section><h2>{html.escape(run.name)}</h2><p>{html.escape(status.get('status','Pending'))}</p>" +
                         (f'<img src="runs/{run.name}/training.png">' if history_path.exists() else '') + "</section>")
            continue
        scores = json.loads(result_path.read_text())
        rows = read_csv(run / "predictions_val.csv")
        for r in rows:
            for k in ("truth", "prediction"):
                r[k] = int(r[k])
            for k in ("p_baseline", "p_partial", "p_clean"):
                r[k] = float(r[k])
        ci_path = run / "ci_val.json"
        prediction_hash = file_hash(run / "predictions_val.csv")
        old_ci = json.loads(ci_path.read_text()) if ci_path.exists() else {}
        if old_ci.get("predictions_sha256") == prediction_hash:
            ci = old_ci["interval"]
        else:
            ci = bootstrap_ba(rows)
            write_json(ci_path, dict(predictions_sha256=prediction_hash, interval=ci))
        info = json.loads((run / "config.json").read_text())
        summary = dict(run=run.name, model=info["model"], seed=info["seed"],
                       training_mode=info["training_mode"], completed_epochs=status.get("completed_epochs"),
                       best_epoch=scores["best_epoch"], specimen_ba=scores["specimen_balanced_accuracy"],
                       specimen_ba_ci_low=ci[0], specimen_ba_ci_high=ci[1],
                       balanced_accuracy=scores["balanced_accuracy"], accuracy=scores["accuracy"],
                       macro_f1=scores["macro_f1"], macro_auc=scores["macro_auc"],
                       train_seconds=scores["train_seconds"], feature_extraction_seconds=scores["feature_extraction_seconds"],
                       trainable_parameters=scores["trainable_parameters"], total_encoder_parameters=scores["total_encoder_parameters"])
        summaries.append(summary)
        cm = confusion_matrix([r["truth"] for r in rows], [r["prediction"] for r in rows], labels=range(3))
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks(range(3), STAGES, rotation=25)
        ax.set_yticks(range(3), STAGES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Experimental stage")
        fig.tight_layout()
        fig.savefig(run / "confusion_matrix.png", dpi=140)
        plt.close(fig)
        figures = [f'<img src="runs/{run.name}/{p}">' for p in ("training.png", "confusion_matrix.png")]
        figures += [f'<img src="{p.relative_to(output).as_posix()}">' for p in sorted((run / "focus").glob("*.png"))]
        cards.append(f"<section><h2>{html.escape(run.name)}</h2><p>Validation specimen BA: {scores['specimen_balanced_accuracy']:.3f}; "
                     f"{scores['specimens']} specimens. {html.escape(info['training_mode'])}.</p>" + ''.join(figures) + "</section>")
    summaries.sort(key=lambda r: r["specimen_ba"], reverse=True)
    write_csv(output / "comparison.csv", summaries)
    table = ""
    if summaries:
        keys = ["run", "training_mode", "specimen_ba", "balanced_accuracy", "macro_f1", "macro_auc", "best_epoch", "train_seconds"]
        table = '<table><tr>' + ''.join(f'<th>{k}</th>' for k in keys) + '</tr>'
        for r in summaries:
            table += '<tr>' + ''.join(f'<td>{html.escape(str(round(r[k],4) if isinstance(r[k],float) else r[k]))}</td>' for k in keys) + '</tr>'
        table += '</table>'
        fig, ax = plt.subplots(figsize=(max(7, len(summaries)*1.2), 4))
        y = [r["specimen_ba"] for r in summaries]
        ax.bar(range(len(y)), y, color="#326f9f")
        low = [max(0, r["specimen_ba"]-r["specimen_ba_ci_low"]) for r in summaries]
        high = [max(0, r["specimen_ba_ci_high"]-r["specimen_ba"]) for r in summaries]
        ax.errorbar(range(len(y)), y, yerr=[low, high], fmt="none", color="black", capsize=4)
        ax.set_xticks(range(len(y)), [r["run"] for r in summaries], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Validation specimen balanced accuracy")
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
        scanners = sorted({s["scanner"] for r in summaries for s in json.loads(
            (output / "runs" / r["run"] / "per_scanner_val.json").read_text())})
        matrix = np.full((len(summaries), len(scanners)), np.nan)
        for i, r in enumerate(summaries):
            for s in json.loads((output / "runs" / r["run"] / "per_scanner_val.json").read_text()):
                if s["specimen_balanced_accuracy"] is not None:
                    matrix[i, scanners.index(s["scanner"])] = s["specimen_balanced_accuracy"]
        fig, ax = plt.subplots(figsize=(max(5, len(scanners)*1.4), max(3, len(summaries)*.5+1.5)))
        artist = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(scanners)), scanners, rotation=25, ha="right")
        ax.set_yticks(range(len(summaries)), [r["run"] for r in summaries])
        for i in range(len(summaries)):
            for j in range(len(scanners)):
                ax.text(j,i, f"{matrix[i,j]:.2f}" if np.isfinite(matrix[i,j]) else "N/A", ha="center", va="center")
        fig.colorbar(artist, ax=ax, label="Specimen balanced accuracy")
        fig.tight_layout()
        fig.savefig(output / "per_scanner_comparison.png", dpi=150)
        plt.close(fig)
    # Ensure report comparisons refer to the same held-out acquisitions.
    cohorts = []
    for run in runs:
        if (run / "predictions_val.csv").exists() and any(s["run"] == run.name for s in summaries):
            cohorts.append({r["row_id"] for r in read_csv(run / "predictions_val.csv")})
    if cohorts and any(c != cohorts[0] for c in cohorts):
        raise ValueError("Cannot compare models evaluated on different validation cohorts.")
    grouped = []
    for model in sorted({r["model"] for r in summaries}):
        selected = [r for r in summaries if r["model"] == model]
        values = [r["specimen_ba"] for r in selected]
        grouped.append(dict(model=model, seeds=len(values), mean_specimen_ba=float(np.mean(values)),
                            sd_across_seeds=float(np.std(values, ddof=1)) if len(values)>1 else None))
    write_csv(output / "comparison_by_model.csv", grouped)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>BioCal3D 3D comparison</title>
<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#17324a;background:#f7f9fb}section{background:white;padding:20px;margin:20px 0;border-radius:12px}img{max-width:100%;max-height:430px}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{background:#e5eff8;padding:18px;border-radius:10px}</style></head><body>
<h1>BioCal3D · 3D stage classification</h1><p class="note">Experimental stages are weak sample-level labels, not plaque masks.
Validation ranking is exploratory: these specimens also select the best checkpoint.
Error bars resample physical specimens and do not account for model selection or training uncertainty.
Scratch training and frozen pretrained probes have different prior data and compute.
Occlusion maps explain sensitivity to region removal; they are not plaque segmentation.
CPU/GPU results and head-only versus full-encoder timings must be interpreted separately.</p>"""
    if errors:
        page += "<h2>Models requiring attention</h2><ul>" + ''.join(errors) + "</ul>"
    page += table
    if summaries:
        page += '<p><img src="comparison.png"></p><p><img src="per_scanner_comparison.png"></p>'
    page += ''.join(cards) + "</body></html>"
    (output / "comparison.html").write_text(page, encoding="utf-8")


def run_model(name, seed, records, args, device):
    seed_all(seed)
    run = args.output / "runs" / f"{name}__seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "status.json", dict(status="loading"))
    encoder = make_model(name, args, device)
    total_parameters = sum(p.numel() for p in encoder.parameters())
    cache, extraction_seconds = None, 0.
    if name in ("sonata", "concerto"):
        cache, channels, provenance, extraction_seconds = cache_pretrained(name, encoder, records, args, device)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_all(seed)
        model = ClassificationHead(channels).to(device)
        training_mode = "frozen pretrained encoder + trained stage head"
    else:
        provenance = encoder.provenance
        model = StageModel(encoder, args).to(device)
        training_mode = "scratch encoder + trained stage head"
    config = dict(model=name, seed=seed, inputs=args.inputs, points=args.points, scale_mm=args.scale_mm,
        sampling_seed=args.sampling_seed, grid_size=args.grid_size, neighbors=args.neighbors,
        lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay, patience=args.patience,
        training_mode=training_mode, device=str(device), provenance=provenance,
        cohort=[dict(row_id=r["row_id"], stage=r["stage"], split=r["split"], physical_id=r["physical_id"],
                     surface_signature=r["surface_signature"]) for r in records if r["split"] != "test"])
    key = signature(config)
    config_path = run / "config.json"
    if config_path.exists() and signature(json.loads(config_path.read_text())) != key:
        raise ValueError(f"Settings changed for {run}; use a new output.")
    write_json(config_path, config)
    train = SurfaceDataset([r for r in records if r["split"] == "train"], cache)
    val_records = [r for r in records if r["split"] == "val"]
    validation = SurfaceDataset(val_records, cache)
    history, seconds, interrupted, best_epoch = run_training(model, train, validation, args, run, device, key)
    for split in (["val", "test"] if args.evaluate_test else ["val"]):
        selected = [r for r in records if r["split"] == split]
        if not selected:
            continue
        dataset = SurfaceDataset(selected, cache)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        scores, predictions = evaluate(model, loader, device)
        scores.update(best_epoch=best_epoch, train_seconds=seconds,
                      feature_extraction_seconds=extraction_seconds,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      total_encoder_parameters=total_parameters, device=str(device))
        write_json(run / f"metrics_{split}.json", scores)
        write_csv(run / f"predictions_{split}.csv", predictions)
        write_json(run / f"per_scanner_{split}.json", scanner_metrics(predictions))
        write_csv(run / f"per_scanner_{split}.csv", scanner_metrics(predictions))
    write_json(run / "status.json", dict(status="interrupted" if interrupted else "complete",
        completed_epochs=history[-1]["epoch"], best_epoch=best_epoch))
    report(args.output)
    if args.focus_examples and not interrupted:
        export_focus(model, name, val_records, cache, args, run, device)
        report(args.output)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return interrupted


def self_test():
    seed_all(7)
    head = ClassificationHead(12)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.02)
    x = torch.randn(24, 12)
    y = (x[:, 0] > 0).long()
    before = float(nn.functional.cross_entropy(head.eval()(x), y).detach())
    for _ in range(20):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head.train()(x), y)
        loss.backward()
        optimizer.step()
    after = float(nn.functional.cross_entropy(head.eval()(x), y).detach())
    assert after < before
    rows = [dict(scanner="TRIOS3", sample_name=f"BCG{g}T{t}-{n}", physical_id=f"BCG{g}-{n}",
                 group=str(g), stage="baseline" if t == 0 else "clean", mesh_path=__file__)
            for g in (1, 2) for n in range(1, 9) for t in (0, 1)]
    make_split(rows, 42)
    validate_records(rows)
    bad = copy.deepcopy(rows)
    bad[1]["split"] = "val" if bad[0]["split"] != "val" else "train"
    try:
        validate_records(bad)
        raise AssertionError("Leakage was not rejected")
    except ValueError:
        pass
    m = metrics([0, 1, 2], np.eye(3), ["a", "b", "c"])
    assert m["specimen_balanced_accuracy"] == 1 and m["macro_auc"] == 1
    assert metrics([0, 1], np.eye(3)[:2], ["a", "b"])["balanced_accuracy"] is None
    assert identify(Path("root/TRIOS3/G3/BCG3T1-4/roi.vtp"))["physical_id"] == "BCG3-4"
    print("Self-test passed: head learning, grouping/leakage rejection, identity, and metrics.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path.home() / "repos/biocal3d/data/02-09-2026-Data collected")
    p.add_argument("--roi-root", type=Path)
    p.add_argument("--manifest", type=Path, help="CSV with mesh_path/scanner/sample_name/physical_id/group/stage and optional split")
    p.add_argument("--split-manifest", type=Path, help="Reuse your DINO scan_manifest.csv split")
    p.add_argument("--make-split", action="store_true", help="Generate a new specimen split; do not use when keeping an established split")
    p.add_argument("--output", type=Path, default=Path("runs/3d_models"))
    p.add_argument("--backends-root", type=Path, default=Path(__file__).resolve().parent / "third_party")
    p.add_argument("--models", choices=MODELS, nargs="+", default=MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--sampling-seed", type=int, default=123)
    p.add_argument("--inputs", choices=["geometry", "geometry_rgb"], default="geometry_rgb")
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--scale-mm", type=float, default=10., help="Fixed coordinate divisor; input meshes must be in mm")
    p.add_argument("--grid-size", type=float, default=.01, help="In scaled coordinates: .01*10mm = .1mm")
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=.0001)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-cpu", action="store_true", help="Allow explicit slow CPU fallback when CUDA is absent")
    p.add_argument("--flash", action="store_true", help="Enable optional FlashAttention if installed")
    p.add_argument("--attention-patch", type=int, default=128)
    p.add_argument("--sonata-checkpoint", type=Path)
    p.add_argument("--concerto-checkpoint", type=Path)
    p.add_argument("--concerto-variant", default="concerto_small")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate-test", action="store_true", help="Final test evaluation using validation-selected checkpoint")
    p.add_argument("--live", action="store_true", help="Update comparison.html and curves after every epoch")
    p.add_argument("--report-only", action="store_true", help="Rebuild report from saved runs, including interrupted runs")
    p.add_argument("--preflight", action="store_true", help="Load each actual backend and check a forward pass without training")
    p.add_argument("--prepare-only", action="store_true", help="Validate ROI data and save common samples without loading models")
    p.add_argument("--export-local-features", action="store_true", help="Save pretrained per-face features for later fusion")
    p.add_argument("--focus-examples", type=int, default=0, help="Optional costly occlusion examples; e.g. 6")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if min(args.points, args.epochs, args.patience, args.neighbors, args.attention_patch, args.scale_mm, args.grid_size, args.lr) <= 0:
        p.error("Sizes, rates and epoch counts must be positive.")
    if args.batch_size < 2:
        p.error("Use batch size >=2 because official TSGCNet contains BatchNorm.")
    if args.points <= args.neighbors:
        p.error("--points must exceed --neighbors.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.self_test:
        self_test()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.output)
        print(f"Report: {(args.output / 'comparison.html').resolve()}")
        return
    records = load_records(args)
    # Preserve experiment-level settings even for runs split across environments.
    experiment = dict(version=VERSION, inputs=args.inputs, points=args.points, scale_mm=args.scale_mm,
        sampling_seed=args.sampling_seed, grid_size=args.grid_size,
        cohort=[{k: r[k] for k in ("row_id", "physical_id", "split", "stage")} for r in records if r["split"] != "test"])
    path = args.output / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != experiment:
        raise ValueError("Experiment settings/split changed; use a new --output directory.")
    write_json(path, experiment)
    write_csv(args.output / "scan_manifest.csv", args.full_records)
    prepare_data(records, args)
    if args.prepare_only:
        print("Prepared common surface samples. No model training performed.")
        return
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu" and args.device == "auto" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch; use --device cpu or --allow-cpu only for deliberate CPU testing.")
    print(f"Device: {device}; PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    failures = []
    for name in args.models:
        for seed in args.seeds:
            run = args.output / "runs" / f"{name}__seed{seed}"
            try:
                if args.preflight:
                    encoder = make_model(name, args, device).to(device).eval()
                    data, _, _ = SurfaceDataset([records[0]])[0]
                    data = batch_to({k: v[None] for k, v in data.items()}, device)
                    with torch.no_grad():
                        local, coords = encoder(data)
                    assert local.ndim == 3 and torch.isfinite(local).all()
                    print(f"PASS {name}: local features {tuple(local.shape)}", flush=True)
                    del encoder
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                if run_model(name, seed, records, args, device):
                    return
            except KeyboardInterrupt:
                report(args.output)
                return
            except Exception as exc:
                failures.append(dict(model=name, seed=seed, error=f"{type(exc).__name__}: {exc}"))
                error_dir = args.output / "preflight" if args.preflight else run
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / (f"{name}_error.txt" if args.preflight else "error.txt")
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                if not args.preflight:
                    write_json(run / "status.json", dict(status="failed", error=str(exc)))
                print(f"FAILED {name}: {exc}\nSee {error_path}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                report(args.output)
    write_json(args.output / ("preflight_failures.json" if args.preflight else "failures.json"), failures)
    report(args.output)
    print(f"Comparison: {(args.output / 'comparison.html').resolve()}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} model run(s) failed. The report lists them; no substitute models were trained.")


if __name__ == "__main__":
    main()
